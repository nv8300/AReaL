# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import collections
import copy
import dataclasses
import gc
import getpass
import itertools
import os
import pprint
import random
import re
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import colorama
import networkx as nx
import numpy as np
import torch
import torch.distributed
import wandb

import realhf.api.core.config as config_api
import realhf.api.core.data_api as data_api
import realhf.api.core.dfg as dfg
import realhf.api.core.model_api as model_api
import realhf.api.core.system_api as config_pkg
import realhf.base.recover as recover
import realhf.system.request_reply_stream as request_reply_stream
import realhf.system.worker_base as worker_base
from realhf.api.core.config import ModelName
from realhf.api.core.model_api import ReaLModelConfig
from realhf.base import (
    constants,
    datapack,
    logging,
    name_resolve,
    names,
    seeding,
    timeutil,
    topology,
)
from realhf.base.asyncio_utils import (
    raise_asyncio_exception,
    setup_run_until_complete,
    teardown_run_util_complete,
)
from realhf.system.buffer import AsyncIOSequenceBuffer
from realhf.system.flops_counter import FlopsCounter

logger = logging.getLogger("master worker", "system")
blogger = logging.getLogger("benchmark")


def _attach_param_realloc_hooks(
    payload: request_reply_stream.Payload,
    msid2mwid: Dict[config_pkg.ModelShardID, int],
    from_model_name: ModelName,
    to_model_name: ModelName,
    from_topo: topology.PipeModelDataParallelTopology,
    to_topo: topology.PipeModelDataParallelTopology,
    to_model_config: ReaLModelConfig,
    pre: bool,
) -> request_reply_stream.Payload:

    model_name = from_model_name
    target = to_model_name
    # Prioritize handlers of `from_model`, then handlers of `to_model`.
    # As a result, if both `from_model` and `to_model` reside in a model worker,
    # the handler in the received request will be `from_model`. Layers will also built in `from_model`.
    # After that, we assign layers of the `from_model` to `to_model`.
    handlers = [
        config_pkg.ModelShardID.from_parallelism_rank(model_name, from_topo, j)
        for j in range(from_topo.world_size())
    ]
    all_handler_mwids = set([msid2mwid[h] for h in handlers])
    dst_handlers = [
        config_pkg.ModelShardID.from_parallelism_rank(target, to_topo, j)
        for j in range(to_topo.world_size())
    ]
    for h in dst_handlers:
        if msid2mwid[h] not in all_handler_mwids:
            handlers.append(h)
            all_handler_mwids.add(msid2mwid[h])

    ps_data = {
        "from_model_name": model_name,
        "to_model_name": target,
        "from_topo": from_topo,
        "to_topo": to_topo,
        "to_model_config": to_model_config,
        "eta": 1.0,
    }
    if pre:
        payload.pre_hooks.append("param_realloc")
        payload.pre_hook_data.append(ps_data)
    else:
        payload.post_hooks.append("param_realloc")
        payload.post_hook_data.append(ps_data)
    return payload


@dataclasses.dataclass
class RPCCorountineControl:
    ## Shared resources ##
    stop: asyncio.Event
    # for counting the number of finished training steps
    # one training step corresponds to traversal of the whole DFG
    train_count: asyncio.Queue
    # For flushing requests
    topo_level_count: asyncio.Queue

    ## Per-coroutine resources ##
    # Used for counting the number of concurrent calls.
    rpc_concurrency: Dict[str, asyncio.Semaphore]
    rpc_traversal: Dict[str, int]
    # for synchronizing req ids between req and reply coroutines
    request_queues: Dict[str, asyncio.Queue]

    # for training data management and data cleaning after each step
    ids_to_clear: Set[int] = dataclasses.field(default_factory=set)
    flops_counter: FlopsCounter = dataclasses.field(default_factory=FlopsCounter)

    should_save: bool = False
    should_eval: bool = False
    should_ckpt: bool = False
    step_info: recover.StepInfo = dataclasses.field(default_factory=recover.StepInfo)

    # recover information
    used_hash_vals_this_epoch: List[int] = dataclasses.field(default_factory=list)
    hash_vals_to_ignore_in_recover: List[int] = dataclasses.field(default_factory=list)


def _attach_payloads_with_hooks(
    rpc: dfg.MFCDef,
    payloads: Dict[config_api.ModelShardID, request_reply_stream.Payload],
    mwids: List[int],
    msid2mwid: Dict[config_pkg.ModelShardID, int],
    model_configs: Dict[str, None | ReaLModelConfig],
    model_topos: Dict[str, topology.PipeModelDataParallelTopology],
    main_handlers: List[config_pkg.ModelShardID],
    hook_type: str,
) -> Tuple[Dict[config_api.ModelShardID, request_reply_stream.Payload], List[int]]:
    assert hook_type in ["pre", "post"], hook_type

    main_mwids = set([msid2mwid[h] for h in main_handlers])
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
                if msid2mwid[h] not in mwids:
                    payloads[h] = request_reply_stream.Payload(
                        handler=h,
                        handle_name="empty",
                    )
                    setattr(payloads[h], f"{hook_type}_hooks", ["param_realloc"])
                    setattr(payloads[h], f"{hook_type}_hook_data", [ps_data])
                    mwids.append(msid2mwid[h])
                elif msid2mwid[h] not in main_mwids:
                    hh = next(hh for hh in payloads if msid2mwid[hh] == msid2mwid[h])
                    getattr(payloads[hh], f"{hook_type}_hooks").append("param_realloc")
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


def _request_model_function_call(
    rpc: dfg.MFCDef,
    stream: request_reply_stream.NameResolvingRequestClient,
    msid2mwid: Dict[config_pkg.ModelShardID, int],
    model_topos: Dict[str, topology.PipeModelDataParallelTopology],
    model_configs: Dict[str, None | ReaLModelConfig],
    producer_names: Dict[str, str],
    producer_name2producer_handlers: Dict[str, List[config_pkg.ModelShardID]],
    producer_mappings: Dict[str, Dict[str, List[int]]],
    target_mapping: Dict[str, List[int]],
    meta_sample: data_api.SequenceSample,
    handlers: List[config_pkg.ModelShardID],
    ctrl: RPCCorountineControl,
    model_save_root: str,
) -> List[uuid.UUID]:

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
                model_save_root,
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
    mwids = [msid2mwid[h] for h in handlers]
    assert len(mwids) == len(set(mwids))

    for producer_name in producer_names.values():
        for h in producer_name2producer_handlers[producer_name]:
            if msid2mwid[h] not in mwids:
                payloads[h] = request_reply_stream.Payload(
                    handler=h,
                    handle_name="empty",
                    pre_hooks=["data_transfer"],
                    pre_hook_data=[dt_data],
                )
                mwids.append(msid2mwid[h])

    payloads, mwids = _attach_payloads_with_hooks(
        rpc,
        payloads,
        mwids,
        msid2mwid=msid2mwid,
        model_configs=model_configs,
        model_topos=model_topos,
        main_handlers=handlers,
        hook_type="pre",
    )
    payloads, mwids = _attach_payloads_with_hooks(
        rpc,
        payloads,
        mwids,
        msid2mwid=msid2mwid,
        model_configs=model_configs,
        model_topos=model_topos,
        main_handlers=handlers,
        hook_type="post",
    )
    main_payloads = [p for h, p in payloads.items() if h in handlers]
    other_payloads = [p for h, p in payloads.items() if h not in handlers]
    all_req_ids = stream.request(
        payloads=main_payloads + other_payloads,
    )
    return all_req_ids[: len(main_payloads)], all_req_ids[len(main_payloads) :]


async def model_rpc_request_func(
    rpc: dfg.MFCDef,
    msid2mwid: Dict[config_pkg.ModelShardID, int],
    src_rpc_model_name: ModelName,
    stream: request_reply_stream.NameResolvingRequestClient,
    buffer: AsyncIOSequenceBuffer,
    data_owner: Dict[Tuple[int, str], Tuple[ModelName, int]],
    model_topos: Dict[str, topology.PipeModelDataParallelTopology],
    model_configs: Dict[str, None | ReaLModelConfig],
    model_save_root: str,
    ctrl: RPCCorountineControl,
):
    """The corountine for sending requests to model workers."""
    topo = model_topos[rpc.model_name]
    logger.info(
        f"Requesting Model RPC, interface_type=#{rpc.interface_type}# "
        f"(dp,mp,pp) = *({topo.get_dim('data')},{topo.get_dim('model')},{topo.get_dim('pipe')})*"
    )

    topo = model_topos[rpc.model_name]
    handlers = [
        config_pkg.ModelShardID.from_parallelism_rank(rpc.model_name, topo, j)
        for j in range(topo.world_size())
    ]

    producer_names = {}  # data key -> model name
    for k in rpc.input_keys:
        if k in rpc.data_producers:
            producer_names[k] = rpc.data_producers[k]
        else:
            producer_names[k] = src_rpc_model_name
    keys_to_send = defaultdict(list)  # model name -> List[keys] to send
    for k in producer_names:
        keys_to_send[producer_names[k]].append(k)

    # convert producer model name to ModelShardID
    producer_name2producer_handlers = {}
    for producer_name in keys_to_send:
        producer_name2producer_handlers[producer_name] = [
            config_pkg.ModelShardID.from_parallelism_rank(
                producer_name, model_topos[producer_name], j
            )
            for j in range(model_topos[producer_name].world_size())
        ]

    request_queue = ctrl.request_queues[rpc.name]
    rpc_concurrency = ctrl.rpc_concurrency[rpc.name]

    this_rpc_consumed_seqs = 0
    while not ctrl.stop.is_set():

        await rpc_concurrency.acquire()

        # Ensure that parent RPCs will not be over-consumed.
        while any(
            this_rpc_consumed_seqs >= (ctrl.rpc_traversal[c.name] + 1) * c.n_seqs
            for c in rpc.all_successors()
        ):
            await asyncio.sleep(0.1)

        buf_indices, sample = await buffer.get_batch_for_rpc(rpc)

        ctrl.flops_counter.add_rpc(rpc, sample, model_configs[rpc.model_name])

        this_rpc_consumed_seqs += sample.bs

        # logger.info(f"Model rpc {rpc.name} requesting.")

        # Dispatch data to different data parallel ranks.
        dp_size = topo.get_dim("data")
        if rpc.is_generate():
            # The workload of generation is decided by batch size, instead of the generated length.
            samples, forward_indices, _ = sample.split_with_lengths(
                mb_spec=data_api.MicroBatchSpec(n_mbs=dp_size),
                lens=[1 for _ in range(sample.bs)],
            )
        else:
            samples, forward_indices, _ = sample.split(
                data_api.MicroBatchSpec(n_mbs=dp_size)
            )
        blogger.info(
            f"DP split (DP size {dp_size}) for RPC {rpc.name}: "
            f"#seqs: {[s.bs for s in samples]}, "
            f"#tokens: {[sum([sum(lens) for lens in s.seqlens[s._get_split_key()]]) for s in samples]}"
        )
        sample = data_api.SequenceSample.gather(samples)
        buf_indices = [buf_indices[i] for i in forward_indices]

        partitions = data_api.SequenceSplitSpec(
            sizes=[s.bs for s in samples]
        ).partitions
        target_mapping = {i: list(range(v[0], v[1])) for i, v in enumerate(partitions)}

        # Set data owner of produced data by this RPC, such that downstream RPCs can know
        # where to fetch these data.
        for dp_idx, (st, ed) in enumerate(partitions):
            for i in range(st, ed):
                for k in rpc.output_keys:
                    data_owner[sample.ids[i], k] = (rpc.model_name, dp_idx)

        # Get the data owner of this RPC's input data.
        # We use it to determine the source of data transfer.
        producer_mappings = {}
        for k in rpc.input_keys:
            names, dp_indices = [], []
            for sample_id in sample.ids:
                owner_name, dp_idx = data_owner[(sample_id, k)]
                names.append(owner_name)
                dp_indices.append(dp_idx)
            assert len(set(names)) == 1
            producer_mapping = defaultdict(list)
            for i, dp_idx in enumerate(dp_indices):
                producer_mapping[dp_idx].append(i)
            producer_mapping = {k: sorted(v) for k, v in producer_mapping.items()}
            producer_mappings[names[0], k] = producer_mapping

        # send partitioned data to model workers
        req_ids, other_req_ids = _request_model_function_call(
            rpc=rpc,
            stream=stream,
            msid2mwid=msid2mwid,
            model_topos=model_topos,
            model_configs=model_configs,
            producer_names=producer_names,
            producer_name2producer_handlers=producer_name2producer_handlers,
            producer_mappings=producer_mappings,
            target_mapping=target_mapping,
            meta_sample=sample,
            handlers=handlers,
            ctrl=ctrl,
            model_save_root=model_save_root,
        )
        await request_queue.put(
            (buf_indices, sample.ids, req_ids, other_req_ids, time.perf_counter())
        )
        await ctrl.topo_level_count.put(1)
        logger.info(f"Model rpc {rpc.name} requested.")


async def model_rpc_reply_func(
    rpc: dfg.MFCDef,
    stream: request_reply_stream.NameResolvingRequestClient,
    buffer: AsyncIOSequenceBuffer,
    model_topos: Dict[str, topology.PipeModelDataParallelTopology],
    ctrl: RPCCorountineControl,
):
    topo = model_topos[rpc.model_name]
    dp_size = topo.get_dim("data")
    dp_head_indices = [
        topo.get_rank(data=i, pipe=topo.get_dim("pipe") - 1, model=0)
        for i in range(dp_size)
    ]

    request_queue = ctrl.request_queues[rpc.name]
    rpc_concurrency = ctrl.rpc_concurrency[rpc.name]

    while not ctrl.stop.is_set():
        # Wait for master worker's request.
        buf_indices, ids, req_ids, other_req_ids, tik = await request_queue.get()

        # Then, wait for all main requests to finish.
        responses = await stream.gather_async(request_ids=req_ids)
        # logger.info(f"rpc {rpc.name} received responses {req_ids}")

        # Filter out responses other than DP heads.
        # Other repsonses are duplicated or None.
        responses: List[request_reply_stream.Payload] = [
            responses[i] for i in dp_head_indices
        ]

        # If the returned data is a SequenceSample, it is the data returned by
        # model function calls. The data shoulbe be amended into buffer.
        # Otherwise, it's the train statistics and should be reduced and logged.
        if isinstance(responses[-1], data_api.SequenceSample):
            res = data_api.SequenceSample.gather(responses)
        else:
            res = _gather_stat(responses)

        if rpc.log_return_value:
            logger.info(f"RPC name {rpc.name} returns {res}")

            if isinstance(res, Dict):
                wandb.log(res, step=ctrl.step_info.global_step)

        logger.info(
            f"Model rpc {rpc.name} finished. Run time {time.perf_counter() - tik:.4f}s."
        )

        # Release the semaphore to let the request corountine continue running.
        rpc_concurrency.release()
        ctrl.rpc_traversal[rpc.name] += 1

        # If this RPC is the final node in the dataflow graph,
        # update the train counter.
        # Otherwise, amend data in the buffer.
        if rpc.is_dst:
            ctrl.ids_to_clear = ctrl.ids_to_clear.union(ids)
            await ctrl.train_count.put(1)
        else:
            logger.info(f"Amending RPC {rpc.name} output keys: {res.keys}")
            await buffer.amend_batch(buf_indices, res.unpack())

        # Wait for all side-effect requests to finish.
        # Side-effect or empty requests are required for data transfer
        # and parameter synchronization.
        # Wait them after the main request to log the oorrect MFC time.
        await stream.gather_async(other_req_ids)


def _gather_stat(src: List[Dict]) -> Dict:
    cnt, stats = {}, {}
    for reply in src:
        for k, v in reply.items():
            cnt[k] = cnt.get(k, 0) + 1
            stats[k] = stats.get(k, 0) + v
    res = {k: v / cnt for k, v, cnt in zip(stats.keys(), stats.values(), cnt.values())}
    for k, c in cnt.items():
        if c != len(src):
            logger.warning(f"Gathered `{k}` is not present in every returned stats.")
    for k, v in res.items():
        if any(abs(v - x.get(k, None)) > 1e-4 for x in src):
            logger.warning(
                f"Gathered `{k}` is not all-reduced "
                f"before returning: ({[x.get(k, None) for x in src]}, {v})."
            )
    return res


class MasterWorker(worker_base.Worker):
    os.makedirs(constants.MODEL_SAVE_ROOT, exist_ok=True)
    global_exp_tik = time.perf_counter()

    def _configure(self, config: config_pkg.MasterWorker):
        self.config = config

        seeding.set_random_seed(self.config.base_seed + self.config.n_model_workers)

        self.__model_topos: Dict[ModelName, topology.PipeModelDataParallelTopology] = (
            config.model_topos
        )

        # Build execution graph and initialize concurrency utilities.
        self.__model_rpcs = config.model_rpcs

        # Sort all MFCs in the topological order and
        # calculate the width of each level.
        # These numbers will determine when to flush MFC requests.
        self.__topo_widths = []
        for generation in nx.topological_generations(self.__model_rpcs[0]._G):
            self.__topo_widths.append(len(generation))
        logger.info("Topological widths: " + str(self.__topo_widths))

        self.__mwid2msids = defaultdict(list)
        for msid, mwid in self.config.msid2mwid.items():
            self.__mwid2msids[mwid].append(msid)

        self.__rpc_srcs = list(filter(lambda rpc: rpc.is_src, self.__model_rpcs))
        self.__rpc_dsts = list(filter(lambda rpc: rpc.is_dst, self.__model_rpcs))
        self.__n_rpc_srcs = len(self.__rpc_srcs)
        self.__n_rpc_dsts = len(self.__rpc_dsts)

        # Save and eval control.
        self.__total_train_epochs = config.exp_ctrl.total_train_epochs
        self.__save_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.exp_ctrl.save_freq_epochs,
            freq_step=config.exp_ctrl.save_freq_steps,
            freq_sec=config.exp_ctrl.save_freq_secs,
        )
        if (
            config.exp_ctrl.ckpt_freq_epochs is None
            and config.exp_ctrl.ckpt_freq_steps is None
            and config.exp_ctrl.ckpt_freq_secs is None
        ):
            self.__ckpt_ctl = self.__save_ctl
        else:
            self.__ckpt_ctl = timeutil.EpochStepTimeFreqCtl(
                freq_epoch=config.exp_ctrl.ckpt_freq_epochs,
                freq_step=config.exp_ctrl.ckpt_freq_steps,
                freq_sec=config.exp_ctrl.ckpt_freq_secs,
            )
        self.__eval_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.exp_ctrl.eval_freq_epochs,
            freq_step=config.exp_ctrl.eval_freq_steps,
            freq_sec=config.exp_ctrl.eval_freq_secs,
        )

        self.MODEL_SAVE_ROOT = os.path.join(
            constants.MODEL_SAVE_ROOT,
            config.worker_info.experiment_name,
            config.worker_info.trial_name,
        )
        os.makedirs(self.MODEL_SAVE_ROOT, exist_ok=True)

        self.__initialized = False
        self.__recover_run, self.__recover_info = recover.load_recover_info()
        if self.__recover_info is not None:
            logger.info(
                f"Loaded recover info: recover_start={self.__recover_info.recover_start}, "
                f"last_step_info={self.__recover_info.last_step_info}."
            )
            logger.info(
                f"Number of used data in recover info: {len(self.__recover_info.hash_vals_to_ignore)}. "
                f"The previous experiment probably ran for {len(self.__recover_info.hash_vals_to_ignore) // self.__rpc_srcs[0].n_seqs} steps in the epoch."
            )

        # Create corountine control objects for running the dataflow graph.
        self.__rpc_ctrl = RPCCorountineControl(
            stop=asyncio.Event(),
            train_count=asyncio.Queue(maxsize=len(self.__rpc_dsts)),
            topo_level_count=asyncio.Queue(maxsize=sum(self.__topo_widths)),
            rpc_traversal={rpc.name: 0 for rpc in self.__model_rpcs},
            request_queues={rpc.name: asyncio.Queue(1) for rpc in self.__model_rpcs},
            rpc_concurrency={
                rpc.name: asyncio.Semaphore(1) for rpc in self.__model_rpcs
            },
            # NOTE: We should accumulate the used data hashes in the same epoch
            # to prevent loading data used before.
            used_hash_vals_this_epoch=(
                copy.deepcopy(self.__recover_info.hash_vals_to_ignore)
                if self.__recover_run
                else list()
            ),
            hash_vals_to_ignore_in_recover=(
                copy.deepcopy(self.__recover_info.hash_vals_to_ignore)
                if self.__recover_run
                else list()
            ),
        )

        if self.__recover_run:
            self.__rpc_ctrl.step_info = copy.deepcopy(self.__recover_info.recover_start)

            self.__eval_ctl.load_state_dict(self.__recover_info.eval_ctl_info)
            self.__save_ctl.load_state_dict(self.__recover_info.save_ctl_info)
            self.__ckpt_ctl.load_state_dict(self.__recover_info.ckpt_ctl_info)

            logger.info(
                f"Recovering from previous run. "
                f"Epoch: {self.__rpc_ctrl.step_info.epoch + 1}, "
                f"Epoch Step: {self.__rpc_ctrl.step_info.epoch_step + 1} "
                f"Global Step: {self.__rpc_ctrl.step_info.global_step + 1}."
            )

        # for benchmark
        self.e2e_time_history = []
        self.__benchmark_steps = config.exp_ctrl.benchmark_steps

        return config.worker_info

    def __lazy_init(self):
        # Set up streams.
        handler_routing = copy.deepcopy(self.config.msid2mwid)
        src_rpc = self.__rpc_srcs[0]
        src_rpc_topo = self.config.model_topos[src_rpc.model_name]
        src_rpc_dp_size = src_rpc_topo.get_dim("data")
        src_rpc_pp_size = src_rpc_topo.get_dim("pipe")
        for i in range(src_rpc_dp_size):
            rank = src_rpc_topo.get_rank(data=i, pipe=src_rpc_pp_size - 1, model=0)
            handler_routing[f"__data{i}__"] = self.config.msid2mwid[
                config_pkg.ModelShardID.from_parallelism_rank(
                    model_name=src_rpc.model_name,
                    topo=src_rpc_topo,
                    parallelism_rank=rank,
                )
            ]
        self.__stream = request_reply_stream.make_master_stream(
            self.config.worker_info,
            n_subscribers=self.config.n_model_workers,
            handler_routing=handler_routing,
        )
        self.__stream: request_reply_stream.NameResolvingRequestClient

        self.__src_rpc = src_rpc = [
            rpc for rpc in self.config.model_rpcs if rpc.is_src
        ][0]
        src_rpc_model_name = src_rpc.model_name
        self.__src_rpc_dp_size = src_rpc_dp_size = self.config.model_topos[
            src_rpc.model_name
        ].get_dim("data")

        # Request training specification from data workers.
        all_data = sum(
            self.__stream.call(
                handlers=[f"__data{i}__" for i in range(src_rpc_dp_size)],
                datas=[None for i in range(src_rpc_dp_size)],
                handle_type="spec",
            ),
            [],
        )

        # NOTE: For dynamic datasets, we still count epoch according to the initial number of data,
        # such that the learning rate decay is not affected.
        seqlens = [max(sum(v[0]) for v in x.seqlens.values()) for x in all_data]
        self._dataset_size = len(all_data)
        self._steps_per_epoch = self._dataset_size // src_rpc.n_seqs
        self._avg_tokens_per_batch = sum(seqlens) / self._steps_per_epoch
        self._dataset_ids = [copy.deepcopy(x.ids[0]) for x in all_data]

        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

        # Build some data required for subsequent model function calls.
        self.__all_model_handlers: List[config_pkg.ModelShardID] = []
        self.__all_mw_handlers: List[config_pkg.ModelShardID] = []
        _covered_mws = set()
        self.__dp0_model_handlers: List[config_pkg.ModelShardID] = []
        self.__trainable_model_handlers: List[config_pkg.ModelShardID] = []
        for model_name, topo in self.config.model_topos.items():
            for j in range(topo.world_size()):
                h = config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                _mw_id = self.config.msid2mwid[h]
                if _mw_id not in _covered_mws:
                    _covered_mws.add(_mw_id)
                    self.__all_mw_handlers.append(h)
            num_dp = topo.get_dim("data")
            self.__all_model_handlers += [
                config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                for j in range(topo.world_size())
            ]

            if any(
                rpc.model_name == model_name
                and rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP
                for rpc in self.__model_rpcs
            ):
                self.__trainable_model_handlers += [
                    config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                    for j in range(topo.world_size())
                ]
            self.__dp0_model_handlers += [
                config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                for j in topo.filter_match(data=0)
            ]

        # Request model configs from model workers.
        # Return None if the model is not a ReaLModel.
        self.__model_configs: Dict[ModelName, None | ReaLModelConfig] = {}
        for model_name in self.config.model_topos:
            h = config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, 0)
            self.__model_configs[model_name] = self.__stream.call(
                handlers=[h],
                datas=[None],
                handle_type="model_config",
            )[0]

        # Initialize model backends.
        # For models with the same role, they share the same model parameters.
        # Therefore, we must call reallocate parameters from A to B
        # before we send requests to initialize B.
        _param_senders = [v[0] for v in self.config.sync_param_pairs]
        _param_receivers = [v[1] for v in self.config.sync_param_pairs]

        # The parameters are by default held by the trainable model.
        # If all replicas are not trainable, the parameters are held in replica 0.
        _model_is_trainable = collections.defaultdict(list)
        for rpc in self.__model_rpcs:
            _model_is_trainable[rpc.model_name].append(
                rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP
            )

        _model_is_trainable = {
            model_name: any(values)
            for model_name, values in _model_is_trainable.items()
        }

        _roles = set([rpc.model_name.role for rpc in self.__model_rpcs])
        _role_cnt = {
            role: len(
                set(
                    [
                        rpc.model_name
                        for rpc in self.__model_rpcs
                        if rpc.model_name.role == role
                    ]
                )
            )
            for role in _roles
        }
        _reordered_model_names = []
        for role in sorted(_roles):
            if _role_cnt[role] == 1:
                _reordered_model_names.append(ModelName(role, 0))
                continue
            _indices = list(range(_role_cnt[role]))
            _trainable_this_role = [
                _model_is_trainable[ModelName(role, i)] for i in range(_role_cnt[role])
            ]
            if any(_trainable_this_role):
                assert (
                    sum(_trainable_this_role) == 1
                ), "only one train for each model is allowed"
                _trainable_idx = _trainable_this_role.index(True)
                _reordered_model_names.append(ModelName(role, _trainable_idx))
                _indices.remove(_trainable_idx)
            for i in _indices:
                _reordered_model_names.append(ModelName(role, i))

        # Send initialization requests.
        self.logger.info(
            f"Initialize model backends with order: {_reordered_model_names}."
        )
        train_rpcs = list(
            filter(
                lambda rpc: rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP,
                self.__model_rpcs,
            )
        )
        assert all(rpc.n_seqs == train_rpcs[0].n_seqs for rpc in train_rpcs)
        if len(train_rpcs) > 0:
            ft_spec = model_api.FinetuneSpec(
                total_train_epochs=self.config.exp_ctrl.total_train_epochs,
                dataset_size=self._dataset_size,
                train_batch_size=train_rpcs[0].n_seqs,
            )
        else:
            ft_spec = model_api.FinetuneSpec(
                total_train_epochs=self.config.exp_ctrl.total_train_epochs,
                dataset_size=self._dataset_size,
                train_batch_size=self.__src_rpc.n_seqs,
            )
        _initialized_roles = []
        for model_name in _reordered_model_names:
            topo = self.config.model_topos[model_name]
            # Build FinetuneSpec, which is required to initialize backends.
            _handlers = [
                config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                for j in range(topo.world_size())
            ]

            init_payloads = [
                request_reply_stream.Payload(
                    handler=_h,
                    handle_name="initialize",
                    data=ft_spec,
                )
                for _h in _handlers
            ]

            # Reallocate parameters if necessary.
            if model_name.role in _initialized_roles and model_name in _param_receivers:
                _param_realloc_src = _param_senders[_param_receivers.index(model_name)]

                # Update handler and payloads to send empty requests
                # with param realloc hooks to source workers.
                src_topo = self.config.model_topos[_param_realloc_src]
                other_handlers = [
                    config_pkg.ModelShardID.from_parallelism_rank(
                        _param_realloc_src, src_topo, j
                    )
                    for j in range(src_topo.world_size())
                ]
                main_mw_ids = [self.config.msid2mwid[_h] for _h in _handlers]
                _other_hanlders = []
                for other_h in other_handlers:
                    if self.config.msid2mwid[other_h] not in main_mw_ids:
                        _other_hanlders.append(other_h)
                init_payloads += [
                    request_reply_stream.Payload(
                        handler=_h,
                        handle_name="empty",
                    )
                    for _h in _other_hanlders
                ]
                _handlers = _handlers + _other_hanlders

                for i, p in enumerate(init_payloads):
                    p = _attach_param_realloc_hooks(
                        payload=p,
                        msid2mwid=self.config.msid2mwid,
                        from_model_name=_param_realloc_src,
                        to_model_name=model_name,
                        from_topo=self.config.model_topos[_param_realloc_src],
                        to_topo=self.config.model_topos[model_name],
                        to_model_config=self.__model_configs[model_name],
                        pre=True,
                    )
                    init_payloads[i] = _attach_param_realloc_hooks(
                        payload=p,
                        msid2mwid=self.config.msid2mwid,
                        from_model_name=model_name,
                        to_model_name=_param_realloc_src,
                        to_topo=self.config.model_topos[_param_realloc_src],
                        from_topo=self.config.model_topos[model_name],
                        to_model_config=self.__model_configs[_param_realloc_src],
                        pre=False,
                    )

            # Send initialization requests then immediately flush them.
            self.__stream.request(
                payloads=init_payloads,
            )
            self.__stream.request(
                handlers=_handlers,
                handle_type="flush",
                no_syn=True,
            )

            _initialized_roles.append(model_name.role)

        self._ft_spec = ft_spec
        logger.info("Initializations of models and backends complete.")

        self.__seqbuffer = AsyncIOSequenceBuffer(
            self.__model_rpcs,
            max_size=int(os.getenv("REAL_MASTER_BUFFER_SIZE", str(int(1e7)))),
        )

        self.__data_owner = {}

        logger.info(f"Creating asyncio coroutines...")

        # Create coroutines for model RPCs.
        coroutine_tasks = []
        for rpc in self.__model_rpcs:
            request_task = event_loop.create_task(
                model_rpc_request_func(
                    rpc=rpc,
                    msid2mwid=self.config.msid2mwid,
                    src_rpc_model_name=src_rpc_model_name,
                    data_owner=self.__data_owner,
                    stream=self.__stream,
                    buffer=self.__seqbuffer,
                    model_topos=self.__model_topos,
                    model_configs=self.__model_configs,
                    ctrl=self.__rpc_ctrl,
                    model_save_root=self.MODEL_SAVE_ROOT,
                )
            )
            reply_task = event_loop.create_task(
                model_rpc_reply_func(
                    rpc=rpc,
                    stream=self.__stream,
                    buffer=self.__seqbuffer,
                    model_topos=self.__model_topos,
                    ctrl=self.__rpc_ctrl,
                )
            )
            coroutine_tasks += [request_task, reply_task]

        # Set up a run context of EventLoop.run_util_complete, baiscally copy-paste from cpython.
        # With this context, we can call the non-block EventLoop._run_once (similar to worker._poll).
        self.__asyncio_tasks: List[asyncio.Task] = coroutine_tasks
        self.__asyncio_ctx = setup_run_until_complete(
            event_loop, asyncio.gather(*coroutine_tasks)
        )

        # wandb init, connect to remote wandb host
        wandb.login()
        wandb.init(
            mode=self.wandb_config.mode,
            entity=self.wandb_config.entity,
            project=self.wandb_config.project or constants.experiment_name(),
            name=self.wandb_config.name or constants.trial_name(),
            job_type=self.wandb_config.job_type,
            group=self.wandb_config.group,
            notes=self.wandb_config.notes,
            tags=self.wandb_config.tags,
            config=self.wandb_config.config,
            dir=os.path.join(
                constants.LOG_ROOT, constants.experiment_name(), constants.trial_name()
            ),
            force=True,
            resume="allow",
            settings=wandb.Settings(start_method="fork"),
        )

        logger.info(f"Coroutines created. The master worker is ready to run.")

        self.__initialized = True
        self._train_start_time = time.perf_counter()

        self.__last_step_info = recover.StepInfo(
            epoch=-1,
            epoch_step=-1,
            global_step=-1,
        )

    def _poll(self):
        is_new_epoch = False
        first_poll = not self.__initialized

        if not self.__initialized:
            self.__lazy_init()
            self._maybe_request_load_data(first_poll=first_poll)

        # Main execution steps. The graph runs under-the-hood in RPC & stream threads.
        # Wait for the finish of the traversal of the execution graph.
        execution_start = time.perf_counter()
        logger.info("Master worker is waiting for the finish of the execution graph.")
        if self.__rpc_ctrl.ids_to_clear:
            # Send clear cache requests to model workers.
            # Clearing the data used in the last step.
            self._clear_gpu_cache()

        for _ in range(10):
            self._maybe_request_load_data(first_poll=first_poll)
        if self.__seqbuffer.size < self.__src_rpc.n_seqs:
            raise RuntimeError(
                f"Buffer size {self.__seqbuffer.size} smaller than "
                f"required batch size {self.__src_rpc.n_seqs} after loading data. "
                "This should not happen, but we raise an error to stop the experiment. "
                "Is your dataset size larger than the configured batch size?"
            )

        is_new_epoch = self._ft_spec.is_new_epoch(self.__rpc_ctrl.step_info)
        is_epoch_last_step = self._ft_spec.is_epoch_last_step(self.__rpc_ctrl.step_info)

        # Check whether we should evaluate or save models.
        self.__rpc_ctrl.should_eval = self.__eval_ctl.check(
            epochs=int(is_epoch_last_step), steps=1
        )
        self.__rpc_ctrl.should_save = self.__save_ctl.check(
            epochs=int(is_epoch_last_step), steps=1
        )
        self.__rpc_ctrl.should_ckpt = self.__ckpt_ctl.check(
            epochs=int(is_epoch_last_step), steps=1
        )

        _rpc_dst_cnt = 0
        _topo_level_cnt, _topo_level_idx = 0, 0
        while _rpc_dst_cnt < self.__n_rpc_dsts:
            try:
                self.__rpc_ctrl.train_count.get_nowait()
                _rpc_dst_cnt += 1
                continue
            except asyncio.QueueEmpty:
                pass
            try:
                self.__rpc_ctrl.topo_level_count.get_nowait()
                _topo_level_cnt += 1
                if _topo_level_cnt >= self.__topo_widths[_topo_level_idx]:
                    logger.info(
                        f"Flushing the current level of the DFG with {self.__topo_widths[_topo_level_idx]} vertices."
                    )
                    self.__stream.request(
                        handlers=self.__all_mw_handlers,
                        handle_type="flush",
                    )
                    _topo_level_idx += 1
                    _topo_level_cnt = 0
                continue
            except asyncio.QueueEmpty:
                pass
            try:
                # Similar to worker._poll. Run multiple times until a train step is finished.
                self.__asyncio_ctx.loop._run_once()
                # NOTE: The following line will propagate errors in corountines back to the main thread.
                # It raises asyncio.exceptions.InvalidStateError if the result is not ready.
                # (In our use cases, the result will never be ready because corountines run while-loops.)
                # We just ignore this error and continue running.
                self.__asyncio_ctx.future.result()
            except asyncio.exceptions.InvalidStateError:
                # Catch the exception when future.result() is not ready.
                pass
            except KeyboardInterrupt as e:
                raise_asyncio_exception(self.__asyncio_ctx, raise_error=False)
                raise e
            except:
                raise_asyncio_exception(self.__asyncio_ctx)
        logger.info("Execution finished!")

        if self.__rpc_ctrl.should_save or self.__rpc_ctrl.should_ckpt:
            self.__last_step_info = copy.deepcopy(self.__rpc_ctrl.step_info)

        self.__rpc_ctrl.used_hash_vals_this_epoch += list(self.__rpc_ctrl.ids_to_clear)

        if is_epoch_last_step:
            self.__rpc_ctrl.used_hash_vals_this_epoch = (
                self.__rpc_ctrl.used_hash_vals_this_epoch[self._dataset_size :]
            )

        if is_new_epoch:
            self.__rpc_ctrl.step_info.epoch += 1
            self.__rpc_ctrl.step_info.epoch_step = 0

        # Logging.
        time_since_configure = time.perf_counter() - self._train_start_time
        e2e_time = time.perf_counter() - execution_start
        self.e2e_time_history.append(e2e_time)

        self._log_training_stats(e2e_time, time_since_configure)

        # Updata counters.
        self.__rpc_ctrl.step_info.epoch_step += 1
        self.__rpc_ctrl.step_info.global_step += 1

        if self.__rpc_ctrl.should_save or self.__rpc_ctrl.should_ckpt:
            self.__recover_save()

        # Pause the worker if experiment or system-wise benchmark completes.
        if (
            self.__benchmark_steps is not None
            and self.__rpc_ctrl.step_info.global_step >= self.__benchmark_steps
        ) or (
            self.__rpc_ctrl.step_info.global_step * self.__src_rpc.n_seqs
            >= self.__total_train_epochs * self._dataset_size
        ):
            # We don't know whether it is the last step of the current epoch,
            # so we exit at the first step of the next epoch.
            if self.__benchmark_steps is not None:
                logger.info(
                    f"Finished benchmark {self.__benchmark_steps}. "
                    f"Time consumption of this setup: {time_since_configure:.3f}"
                )
                logger.info(f"avg #e2e# time *{np.mean(self.e2e_time_history):.3f}*")
            return self.experiment_complete_exit()

        return worker_base.PollResult(sample_count=1, batch_count=1)

    def _maybe_request_load_data(self, first_poll: bool):
        should_load_data = self.__seqbuffer.size < self.__src_rpc.n_seqs
        if not should_load_data:
            return
        blogger.info(
            f"Current buffer size {self.__seqbuffer.size}/{self.__seqbuffer.max_size}. "
            f"The batch size of the source MFC is {self.__src_rpc.n_seqs}."
        )

        src_rpc_dp_size = self.__src_rpc_dp_size
        src_rpc = self.__src_rpc
        src_rpc_model_name = src_rpc.model_name
        data_owner = self.__data_owner
        buffer = self.__seqbuffer
        stream = self.__stream
        ctrl = self.__rpc_ctrl

        # fetch data from dataloader to fill the sequence buffer
        blogger.info(f"Filling data into the buffer in a new epoch.")
        fetch_data_start = time.perf_counter()

        # NOTE: PyTorch dataloader will shuffle data for us.
        all_data: List[data_api.SequenceSample] = []
        received_ids = set()

        # NOTE: Currently we send dataloading requests until iterating
        # over the entire dataset. This may lead to a huge memory waste
        # with super-large datasets. Empirically, it's fine.
        is_final_batch = [False for _ in range(src_rpc_dp_size)]
        is_first_batch = [True for _ in range(src_rpc_dp_size)]
        while not all(is_final_batch):
            # Send request to model workers to get the specification of data.
            # Data itself is not transferred to the master worker.
            data_batches: List[data_api.DataBatchMeta | None] = [
                None for _ in range(src_rpc_dp_size)
            ]
            for i in range(src_rpc_dp_size):
                if is_final_batch[i]:
                    data_batches[i] = None
                    continue
                data_batches[i] = stream.call(
                    handlers=[f"__data{i}__"],
                    handle_type="fetch",
                    datas=[dict(first_batch=is_first_batch[i], first_poll=first_poll)],
                    verbose=False,
                )[0]
                is_final_batch[i] = data_batches[i].is_final_batch
                is_first_batch[i] = False

            # Unpack batched sequences into individual sequences.
            for dp_rank, x in enumerate(data_batches):
                if x is None:
                    continue
                if x.meta_sample is None:
                    continue
                for xx in x.meta_sample.unpack():
                    if xx.ids[0] in received_ids:
                        raise ValueError(
                            f"Duplicate data id {xx.ids[0]}. Is the final batch? {is_final_batch}."
                        )
                    received_ids.add(xx.ids[0])
                    # Store the owner information of the data.
                    # RPCs corountines will use this information to
                    # determine the src and dst of data transfer.
                    for k in xx.keys:
                        data_owner[(xx.ids[0], k)] = (src_rpc_model_name, dp_rank)
                all_data += x.meta_sample.unpack()

        filtered_data = []
        for x in all_data:
            if x.ids[0] in ctrl.hash_vals_to_ignore_in_recover:
                ctrl.hash_vals_to_ignore_in_recover.remove(x.ids[0])
                ctrl.ids_to_clear.add(x.ids[0])
            else:
                filtered_data.append(x)
        all_data = filtered_data

        # We load data in a round-robin manner across different DP ranks,
        # so we also need to shuffle the data to fuse different dataset splits.
        random.shuffle(all_data)

        blogger.info(
            f"Master worker loaded {len(all_data)} pieces of data. "
            f"Training epoch {self.__rpc_ctrl.step_info.epoch + 1} approximately has {self._steps_per_epoch} steps. "
            f"Each batch has {self._avg_tokens_per_batch:.2f} tokens in average. "
            f"Remaining number of data to ignore: {len(self.__rpc_ctrl.hash_vals_to_ignore_in_recover)}."
        )

        # Store into buffer!
        buffer_indices = buffer.put_batch_synced(all_data)
        assert len(buffer_indices) == len(all_data)

        blogger.info(
            f"Filling data finished. Time consumption: "
            f"{time.perf_counter() - fetch_data_start:.3f}s."
        )

        # We should let model workers clear the data ignored during recover.
        if ctrl.ids_to_clear:
            self._clear_gpu_cache()

    def _log_training_stats(self, e2e_time: float, time_since_configure: float):
        # calculate flops
        #########################################
        if not all(
            isinstance(v, ReaLModelConfig) for v in self.__model_configs.values()
        ):
            logger.warning(
                f"Not all models are ReaLModels. Unable to calculate FLOP/s."
            )
            flops = None
            tflops_per_gpu = float("inf")
        else:
            flops = self.__rpc_ctrl.flops_counter.get_flops()
            tflops = flops / (e2e_time * (10**12))
            tflops_per_gpu = flops / (e2e_time * self.config.n_model_workers * (10**12))
        self.__rpc_ctrl.flops_counter.clear()
        #########################################

        epoch = self.__rpc_ctrl.step_info.epoch + 1
        epoch_step = self.__rpc_ctrl.step_info.epoch_step + 1
        global_step = self.__rpc_ctrl.step_info.global_step + 1
        s = f"Epoch {epoch}/{self.config.exp_ctrl.total_train_epochs} "
        s += f"step {epoch_step}/{self._steps_per_epoch} "
        s += f"(global step {global_step}) finishes. "
        s += f"Average #tokens per batch is {self._avg_tokens_per_batch:.0f}. "
        s += f"#End to end# execution time: *{e2e_time:.3f}*s. "
        s += f"Total time consumption: {time_since_configure:.3f}s. "
        if len(self.e2e_time_history) > 2:
            remaining_steps = self._steps_per_epoch - epoch_step
            remaining_epochs = self.__total_train_epochs - epoch
            avg_t = np.mean(self.e2e_time_history[2:])
            remain_t = avg_t * remaining_steps
            remain_t += avg_t * self._steps_per_epoch * remaining_epochs
            s += f"Estimated remaining time: {remain_t:.3f}s. "
        if flops is not None:
            s += f"TFLOP/s per GPU: {tflops_per_gpu:.2f}, total TFLOP/s: {tflops:.2f}."
        logger.info(s)
        logger.info(
            f"Time taken so far across all configurations: {time.perf_counter() - self.global_exp_tik:.2f}s"
        )

    def _clear_gpu_cache(self):
        self.__stream.request(
            handlers=self.__all_mw_handlers,
            handle_type="clear_data_cache",
            datas=[self.__rpc_ctrl.ids_to_clear for _ in self.__all_mw_handlers],
            no_syn=True,
        )
        self.__rpc_ctrl.ids_to_clear.clear()

    def experiment_complete_exit(self):
        self.__rpc_ctrl.stop.set()
        for task in self.__asyncio_tasks:
            task.cancel()
        self.__asyncio_ctx.future.set_result(None)
        # NOTE: stopping the loop immediately after cancelling tasks may
        # raise warnings sometimes, but it doesn't matter.
        self.__asyncio_ctx.loop.stop()
        teardown_run_util_complete(self.__asyncio_ctx)
        logger.info(
            colorama.Style.RESET_ALL
            + colorama.Fore.YELLOW
            + colorama.Style.BRIGHT
            + "\033[1m"
            + "Experiment Completes! Yeah!!!!!!!!"
            + colorama.Style.RESET_ALL
        )

        # Send requests to pause model workers.
        # Model workers will not respond to this message.
        self.__stream.request(
            handlers=self.__all_mw_handlers,
            handle_type="reset",
            datas=[None for _ in self.__all_mw_handlers],
        )
        self.__stream.close()
        constants.reset_run()
        # Reset names used for distributed training.
        # The next round of training will set up a new distributed environment.
        name_resolve.clear_subtree(
            names.distributed_root(constants.experiment_name(), constants.trial_name())
        )
        name_resolve.clear_subtree(
            names.request_reply_stream_root(
                constants.experiment_name(), constants.trial_name()
            )
        )

        wandb.finish()
        gc.collect()
        self.__initialized = False
        self.pause()
        return worker_base.PollResult(0, 0)

    def __recover_save(self):
        # save step info for recover
        if os.getenv("REAL_SAVE_RECOVER_STATES", "0") != "1":
            return
        # save step info for recover
        this_step_info = copy.deepcopy(self.__rpc_ctrl.step_info)
        recover_info = recover.RecoverInfo(
            recover_start=this_step_info,
            last_step_info=self.__last_step_info,
            save_ctl_info=self.__save_ctl.state_dict(),
            ckpt_ctl_info=self.__ckpt_ctl.state_dict(),
            eval_ctl_info=self.__eval_ctl.state_dict(),
            hash_vals_to_ignore=self.__rpc_ctrl.used_hash_vals_this_epoch,
        )

        recover.dump_recover_info(recover_info)
        logger.info("Dumped recover info to file.")
        logger.info(f"Will recover from: {recover_info.recover_start}")
        logger.info(
            f"Number of data used in this epoch: {len(recover_info.hash_vals_to_ignore)}"
        )

    # def _exit_hook(self, exit_status: worker_base.WorkerServerStatus):
    #     logger.info(f"Master worker exits with {exit_status}.")
    #     if os.getenv("REAL_SAVE_RECOVER_STATES", "0") != "1":
    #         return
    #     if exit_status == worker_base.WorkerServerStatus.ERROR:
    #         try:
    #             sleep_time = 600
    #             current_sleep_time = 0
    #             while current_sleep_time < sleep_time:
    #                 logger.info(
    #                     f"ERROR exit, waited {current_sleep_time} s for interruption ..."
    #                 )
    #                 time.sleep(10)
    #                 current_sleep_time += 10
    #         except KeyboardInterrupt:
    #             logger.info("Received SIGINT, starting recover save")

    #     self.__recover_save()
    #                 )
    #                 time.sleep(10)
    #                 current_sleep_time += 10
    #         except KeyboardInterrupt:
    #             logger.info("Received SIGINT, starting recover save")

    #     self.__recover_save()
