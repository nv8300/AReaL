# Copyright 2025 Ant Group Inc.
import asyncio
import random
from typing import *

import networkx as nx

from realhf.api.core.config import ModelName, ModelShardID
from realhf.api.core.data_api import DataBatchMeta, SequenceSample
from realhf.api.core.dfg import MFCDef
from realhf.api.core.model_api import ReaLModelConfig
from realhf.base import logging
from realhf.base.topology import PipeModelDataParallelTopology
from realhf.system.buffer import AsyncIOSequenceBuffer
from realhf.system.request_reply_stream import NameResolvingRequestClient
from realhf.system.v2.function_call import FunctionCall, RPCCorountineControl

logger = logging.getLogger(__name__, "system")
blogger = logging.getLogger("benchmark")


class FunctionExecutor:
    def __init__(
        self,
        rpcs: List[MFCDef],
        msid2mwid: Dict[ModelShardID, int],
        stream: NameResolvingRequestClient,
        buffer: AsyncIOSequenceBuffer,
        model_topos: Dict[str, PipeModelDataParallelTopology],
        model_configs: Dict[str, None | ReaLModelConfig],
        ctrl: RPCCorountineControl,
    ):

        self.func_calls: Dict[str, FunctionCall] = {}
        self.ctrl = ctrl

        self.n_model_workers = len(set(msid2mwid.values()))

        self.rpcs = rpcs
        self.src_rpc = list(filter(lambda rpc: rpc.is_src, rpcs))[0]
        self.src_dp_size = model_topos[self.src_rpc.model_name].get_dim("data")

        # Create model function calls.
        for rpc in self.rpcs:
            func_call = FunctionCall(
                rpc=rpc,
                src_rpc=self.src_rpc,
                stream=stream,
                msid2mwid=msid2mwid,
                model_topos=model_topos,
                model_configs=model_configs,
                ctrl=ctrl,
                buffer=buffer,
            )
            self.func_calls[rpc.name] = func_call

        self.stream = stream
        self.buffer = buffer

        # Sort all MFCs in the topological order and
        # calculate the width of each level.
        # These numbers will determine when to flush MFC requests.
        self.topo_widths = []
        for generation in nx.topological_generations(rpcs[0]._G):
            self.topo_widths.append(len(generation))

    def get_leaf_tasks(self) -> List[str]:
        dst_rpcs = list(filter(lambda rpc: rpc.is_dst, self.rpcs))
        return [rpc.name for rpc in dst_rpcs]

    async def flush_calls(self):
        for level, w in enumerate(self.topo_widths):
            for _ in range(w):
                await self.ctrl.topo_level_count.get()
            logger.info(f"DFG level {level}. Flushing {w} function calls.")
            self.stream.request(
                handlers=list(range(self.n_model_workers)), handle_type="flush"
            )

    async def finish_traverse(self):
        for _ in range(len(self.get_leaf_tasks())):
            await self.ctrl.train_count.get()

    async def load_data(self):
        src_rpc = self.src_rpc
        src_rpc_model_name = src_rpc.model_name
        buffer = self.buffer
        ctrl = self.ctrl

        dp_idx = -1
        received_ids = set()

        while self.buffer.size < max(rpc.n_seqs for rpc in self.rpcs):

            all_data = []

            dp_idx += 1
            dp_idx %= self.src_dp_size

            resps = await self.stream.call_async(
                handlers=[f"__data{dp_idx}__"],
                handle_type="fetch",
                datas=[None],
                verbose=False,
            )
            x: DataBatchMeta | None = resps[0]

            if x is None:
                continue
            if x.meta_sample is None:
                continue

            # Store the owner information of the data.
            # RPCs corountines will use this information to
            # determine the src and dst of data transfer.
            for xx in x.meta_sample.unpack():
                if xx.ids[0] in received_ids:
                    raise ValueError(
                        f"Duplicate data id {xx.ids[0]}. Is the final batch? {is_final_batch}."
                    )
                received_ids.add(xx.ids[0])
                for k in xx.keys:
                    self.ctrl.data_owner[(xx.ids[0], k)] = (
                        src_rpc_model_name,
                        dp_idx,
                    )
                all_data += x.meta_sample.unpack()

            filtered_data = []
            for xx in x.meta_sample.unpack():
                if xx.ids[0] in ctrl.hash_vals_to_ignore_in_recover:
                    ctrl.hash_vals_to_ignore_in_recover.remove(xx.ids[0])
                    ctrl.ids_to_clear.add(xx.ids[0])
                else:
                    filtered_data.append(xx)
            all_data = filtered_data

            # We load data in a round-robin manner across different DP ranks,
            # so we also need to shuffle the data to fuse different dataset splits.
            random.shuffle(all_data)

            # Store into buffer!
            buffer_indices = await buffer.put_batch(all_data)
            assert len(buffer_indices) == len(all_data)

            blogger.info(
                f"Master worker loaded {len(all_data)} pieces of data. "
                f"Remaining number of data to ignore: {len(self.ctrl.hash_vals_to_ignore_in_recover)}. "
                f"Current buffer size: {buffer.size}/{buffer.max_size}. "
            )

    def execute_step(self):
        logger.info("Waiting for the finish of the execution graph.")
        loop = asyncio.get_event_loop()

        tasks = [loop.create_task(fc.run()) for fc in self.func_calls.values()] + [
            loop.create_task(self.flush_calls()),
            loop.create_task(self.load_data()),
        ]

        completion_future = loop.create_task(self.finish_traverse())

        loop.run_until_complete(completion_future)

        for task in tasks:
            loop.run_until_complete(task)

        logger.info("Execution finished!")
