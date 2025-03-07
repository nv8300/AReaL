# Copyright 2025 Ant Group Inc.

import asyncio
import dataclasses
import os
import time
import uuid
from collections import defaultdict
from typing import Dict, Hashable, List, Set, Tuple

import wandb

import realhf.api.core.config as config_api
import realhf.api.core.data_api as data_api
import realhf.api.core.dfg as dfg
import realhf.api.core.system_api as config_pkg
import realhf.base.recover as recover
import realhf.system.request_reply_stream as request_reply_stream
from realhf.api.core.config import ModelName
from realhf.api.core.model_api import ReaLModelConfig
from realhf.base import constants, logging, topology
from realhf.system.buffer import AsyncIOSequenceBuffer
from realhf.system.flops_counter import FlopsCounter

logger = logging.getLogger(__name__, "system")
blogger = logging.getLogger("benchmark")


@dataclasses.dataclass
class RPCCorountineControl:
    # for counting the number of finished training steps
    # one training step corresponds to traversal of the whole DFG
    train_count: asyncio.Queue
    # For flushing requests
    topo_level_count: asyncio.Queue

    # for training data management and data cleaning after each step
    ids_to_clear: Set[Hashable] = dataclasses.field(default_factory=set)
    flops_counter: FlopsCounter = dataclasses.field(default_factory=FlopsCounter)

    data_owner: Dict[Tuple[int, str], Tuple[ModelName, int]] = dataclasses.field(
        default_factory=dict
    )

    should_save: bool = False
    should_eval: bool = False
    should_ckpt: bool = False
    step_info: recover.StepInfo = dataclasses.field(default_factory=recover.StepInfo)

    # recover information
    used_hash_vals_this_epoch: List[int] = dataclasses.field(default_factory=list)
    hash_vals_to_ignore_in_recover: List[int] = dataclasses.field(default_factory=list)


class FunctionCall:
    def __init__(
        self,
        rpc: dfg.MFCDef,
        src_rpc: dfg.MFCDef,
        stream: request_reply_stream.NameResolvingRequestClient,
        msid2mwid: Dict[config_pkg.ModelShardID, int],
        model_topos: Dict[str, topology.PipeModelDataParallelTopology],
        model_configs: Dict[str, None | ReaLModelConfig],
        ctrl: RPCCorountineControl,
        buffer: AsyncIOSequenceBuffer,
    ):

        self.rpc = rpc
        self.src_rpc = src_rpc
        self.stream = stream

        self.msid2mwid = msid2mwid
        self.model_topos = model_topos
        self.model_configs = model_configs

        self.model_save_root = os.path.join(
            constants.MODEL_SAVE_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
        )

        self.rpc_ctrl = ctrl

        self.buffer = buffer

    @property
    def dp_size(self):
        return self.model_topos[self.rpc.model_name].get_dim("data")

    @property
    def pp_size(self):
        return self.model_topos[self.rpc.model_name].get_dim("pipe")

    def attach_payloads_with_hooks(
        self,
        payloads: Dict[config_api.ModelShardID, request_reply_stream.Payload],
        mwids: List[int],
        main_handlers: List[config_pkg.ModelShardID],
        hook_type: str,
    ) -> Tuple[Dict[config_api.ModelShardID, request_reply_stream.Payload], List[int]]:
        assert hook_type in ["pre", "post"], hook_type

        rpc = self.rpc
        model_topos = self.model_topos
        model_configs = self.model_configs

        main_mwids = set([self.msid2mwid[h] for h in main_handlers])
        for hook in getattr(rpc, f"_{hook_type}_hooks"):
            if isinstance(hook, dfg.ParamReallocHook):
                assert (hook.source is None) != (hook.target is None), hook
                if hook.source is None:
                    src_topo = model_topos[rpc.model_name]
                    dst_topo = model_topos[hook.target]
                    dst_config = model_configs[hook.target]
                    src_model_name, dst_model_name = rpc.model_name, hook.target
                    other_model_name = hook.target
                    other_topo = dst_topo
                else:
                    src_topo = model_topos[hook.source]
                    dst_topo = model_topos[rpc.model_name]
                    dst_config = model_configs[rpc.model_name]
                    src_model_name, dst_model_name = hook.source, rpc.model_name
                    other_model_name = hook.source
                    other_topo = src_topo

                ps_data = {
                    "from_model_name": src_model_name,
                    "to_model_name": dst_model_name,
                    "from_topo": src_topo,
                    "to_topo": dst_topo,
                    "to_model_config": dst_config,
                    "eta": hook.eta,
                }
                for h in main_handlers:
                    getattr(payloads[h], f"{hook_type}_hooks").append("param_realloc")
                    getattr(payloads[h], f"{hook_type}_hook_data").append(ps_data)
                other_handlers = [
                    config_api.ModelShardID.from_parallelism_rank(
                        other_model_name, other_topo, j
                    )
                    for j in range(other_topo.world_size())
                ]
                for h in other_handlers:
                    if self.msid2mwid[h] not in mwids:
                        payloads[h] = request_reply_stream.Payload(
                            handler=h,
                            handle_name="empty",
                        )
                        setattr(payloads[h], f"{hook_type}_hooks", ["param_realloc"])
                        setattr(payloads[h], f"{hook_type}_hook_data", [ps_data])
                        mwids.append(self.msid2mwid[h])
                    elif self.msid2mwid[h] not in main_mwids:
                        hh = next(
                            hh
                            for hh in payloads
                            if self.msid2mwid[hh] == self.msid2mwid[h]
                        )
                        getattr(payloads[hh], f"{hook_type}_hooks").append(
                            "param_realloc"
                        )
                        getattr(payloads[hh], f"{hook_type}_hook_data").append(ps_data)

            elif isinstance(hook, dfg.OffloadHook):
                for h in main_handlers:
                    getattr(payloads[h], f"{hook_type}_hooks").append("offload")
                    getattr(payloads[h], f"{hook_type}_hook_data").append(
                        dict(model_name=h.model_name)
                    )
            else:
                raise NotImplementedError(f"Unknown hook type: {hook}")
        return payloads, mwids

    def request(
        self,
        producer_names: Dict[str, str],
        producer_name2producer_handlers: Dict[str, List[config_pkg.ModelShardID]],
        producer_mappings: Dict[str, Dict[str, List[int]]],
        target_mapping: Dict[str, List[int]],
        meta_sample: data_api.SequenceSample,
        handlers: List[config_pkg.ModelShardID],
    ) -> Tuple[List[uuid.UUID], List[uuid.UUID]]:

        rpc = self.rpc
        ctrl = self.rpc_ctrl

        dt_data = {
            "keys": rpc.input_keys,
            "target": rpc.model_name,
            "producer_names": producer_names,
            "producer_mappings": producer_mappings,
            "target_mapping": target_mapping,
            "handle_name": rpc.interface_type.value,
            "rpc_name": rpc.name,
            "meta_sample": meta_sample,
        }

        payloads = {
            handler: request_reply_stream.Payload(
                handler=handler,
                handle_name=rpc.interface_type.value,
                pre_hooks=["data_transfer"],
                pre_hook_data=[dt_data],
                data=rpc.name,
            )
            for handler in handlers
        }
        if ctrl.should_eval:
            for p in payloads.values():
                p.post_hooks.append("evaluate")
                p.post_hook_data.append(dict(model_name=rpc.model_name))
        if (
            ctrl.should_save or ctrl.should_ckpt
        ) and rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP:
            for p in payloads.values():
                p.post_hooks.append("save")
                save_dir = os.path.join(
                    self.model_save_root,
                    rpc.model_name.role,
                    f"epoch{ctrl.step_info.epoch + 1}"
                    f"epochstep{ctrl.step_info.epoch_step + 1}"
                    f"globalstep{ctrl.step_info.global_step + 1}",
                )
                p.post_hook_data.append(
                    dict(
                        model_name=rpc.model_name,
                        save_dir=save_dir,
                        recover_only=not ctrl.should_save,
                    )
                )
        mwids = [self.msid2mwid[h] for h in handlers]
        assert len(mwids) == len(set(mwids))

        for producer_name in producer_names.values():
            for h in producer_name2producer_handlers[producer_name]:
                if self.msid2mwid[h] not in mwids:
                    payloads[h] = request_reply_stream.Payload(
                        handler=h,
                        handle_name="empty",
                        pre_hooks=["data_transfer"],
                        pre_hook_data=[dt_data],
                    )
                    mwids.append(self.msid2mwid[h])

        payloads, mwids = self.attach_payloads_with_hooks(
            payloads,
            mwids,
            main_handlers=handlers,
            hook_type="pre",
        )
        payloads, mwids = self.attach_payloads_with_hooks(
            payloads,
            mwids,
            main_handlers=handlers,
            hook_type="post",
        )
        main_payloads = [p for h, p in payloads.items() if h in handlers]
        other_payloads = [p for h, p in payloads.items() if h not in handlers]
        all_req_ids = self.stream.request(
            payloads=main_payloads + other_payloads,
        )
        return all_req_ids[: len(main_payloads)], all_req_ids[len(main_payloads) :]

    def data_parallel_dispatch(
        self, buf_indices: List[int], sample: data_api.SequenceSample
    ) -> Tuple[List[int], data_api.SequenceSample, List[Tuple[int, int]]]:
        # Dispatch data to different data parallel ranks.
        if self.rpc.is_generate():
            # The workload of generation is decided by batch size, instead of the generated length.
            samples, forward_indices, _ = sample.split_with_lengths(
                mb_spec=data_api.MicroBatchSpec(n_mbs=self.dp_size),
                lens=[1 for _ in range(sample.bs)],
            )
        else:
            samples, forward_indices, _ = sample.split(
                data_api.MicroBatchSpec(n_mbs=self.dp_size)
            )
        blogger.info(
            f"DP split (DP size {self.dp_size}) for RPC {self.rpc.name}: "
            f"#seqs: {[s.bs for s in samples]}, "
            f"#tokens: {[sum([sum(lens) for lens in s.seqlens[s._get_split_key()]]) for s in samples]}"
        )
        sample = data_api.SequenceSample.gather(samples)
        buf_indices = [buf_indices[i] for i in forward_indices]

        partitions = data_api.SequenceSplitSpec(
            sizes=[s.bs for s in samples]
        ).partitions
        return buf_indices, sample, partitions

    async def run_step(self, buf_indices, sample):
        rpc = self.rpc
        topo = self.model_topos[rpc.model_name]
        ctrl = self.rpc_ctrl

        handlers = [
            config_pkg.ModelShardID.from_parallelism_rank(rpc.model_name, topo, j)
            for j in range(topo.world_size())
        ]

        producer_names = {}  # data key -> model name
        for k in rpc.input_keys:
            if k in rpc.data_producers:
                producer_names[k] = rpc.data_producers[k]
            else:
                producer_names[k] = self.src_rpc.model_name
        keys_to_send = defaultdict(list)  # model name -> List[keys] to send
        for k in producer_names:
            keys_to_send[producer_names[k]].append(k)

        # convert producer model name to ModelShardID
        producer_name2producer_handlers = {}
        for producer_name in keys_to_send:
            producer_name2producer_handlers[producer_name] = [
                config_pkg.ModelShardID.from_parallelism_rank(
                    producer_name, self.model_topos[producer_name], j
                )
                for j in range(self.model_topos[producer_name].world_size())
            ]

        dp_head_indices = [
            topo.get_rank(data=i, pipe=topo.get_dim("pipe") - 1, model=0)
            for i in range(self.dp_size)
        ]

        ctrl.flops_counter.add_rpc(rpc, sample, self.model_configs[rpc.model_name])

        # logger.info(f"Model rpc {rpc.name} requesting.")

        # Sample may be reordered here.
        buf_indices, sample, partitions = self.data_parallel_dispatch(
            buf_indices, sample
        )
        target_mapping = {i: list(range(v[0], v[1])) for i, v in enumerate(partitions)}

        # Set data owner of produced data by this RPC, such that downstream RPCs can know
        # where to fetch these data.
        for dp_idx, (st, ed) in enumerate(partitions):
            for i in range(st, ed):
                for k in rpc.output_keys:
                    self.rpc_ctrl.data_owner[sample.ids[i], k] = (
                        rpc.model_name,
                        dp_idx,
                    )

        # Get the data owner of this RPC's input data.
        # We use it to determine the source of data transfer.
        producer_mappings = {}
        for k in rpc.input_keys:
            names, dp_indices = [], []
            for sample_id in sample.ids:
                owner_name, dp_idx = self.rpc_ctrl.data_owner[(sample_id, k)]
                names.append(owner_name)
                dp_indices.append(dp_idx)
            assert len(set(names)) == 1
            producer_mapping = defaultdict(list)
            for i, dp_idx in enumerate(dp_indices):
                producer_mapping[dp_idx].append(i)
            producer_mapping = {k: sorted(v) for k, v in producer_mapping.items()}
            producer_mappings[names[0], k] = producer_mapping

        # send partitioned data to model workers
        req_ids, other_req_ids = self.request(
            producer_names=producer_names,
            producer_name2producer_handlers=producer_name2producer_handlers,
            producer_mappings=producer_mappings,
            target_mapping=target_mapping,
            meta_sample=sample,
            handlers=handlers,
        )
        tik = time.perf_counter()

        await ctrl.topo_level_count.put(1)
        logger.info(f"Model rpc {rpc.name} requested.")

        # Then, wait for all main requests to finish.
        responses = await self.stream.gather_async(request_ids=req_ids)
        # logger.info(f"rpc {rpc.name} received responses {req_ids}")

        # Filter out responses other than DP heads.
        # Other repsonses are duplicated or None.
        responses = [responses[i] for i in dp_head_indices]

        # If the returned data is a SequenceSample, it is the data returned by
        # model function calls. The data shoulbe be amended into buffer.
        # Otherwise, it's the train statistics and should be reduced and logged.
        if isinstance(responses[-1], data_api.SequenceSample):
            res = data_api.SequenceSample.gather(responses)
        else:
            res = data_api.gather_stat(responses)

        if rpc.log_return_value:
            logger.info(f"RPC name {rpc.name} returns {res}")

            if isinstance(res, Dict):
                wandb.log(res, step=ctrl.step_info.global_step)

        logger.info(
            f"Model rpc {rpc.name} finished. "
            f"Run time {time.perf_counter() - tik:.4f}s."
        )

        # If this RPC is the final node in the dataflow graph,
        # update the train counter.
        # Otherwise, amend data in the buffer.
        if rpc.is_dst:
            ctrl.ids_to_clear = ctrl.ids_to_clear.union(sample.ids)
            await ctrl.train_count.put(1)
        else:
            logger.info(f"Amending RPC {rpc.name} output keys: {res.keys}")
            await self.buffer.amend_batch(buf_indices, res.unpack())

        # Wait for all side-effect requests to finish.
        # Side-effect or empty requests are required for data transfer
        # and parameter synchronization.
        # Wait them after the main request to log the oorrect MFC time.
        await self.stream.gather_async(other_req_ids)

    async def run(self):
        rpc = self.rpc
        topo = self.model_topos[rpc.model_name]
        ctrl = self.rpc_ctrl

        logger.info(
            f"Running Model RPC, interface_type=#{rpc.interface_type}# "
            f"(dp,mp,pp) = *({topo.get_dim('data')},{topo.get_dim('model')},{topo.get_dim('pipe')})*"
        )

        consumed = 0
        while True:
            buf_indices, sample = await self.buffer.get_batch_for_rpc(rpc)

            await self.run_step(buf_indices, sample)
            consumed += sample.bs

            # Ensure that parent RPCs will not be over-consumed.
            if all(consumed >= c.n_seqs for c in rpc.all_successors()):
                break
