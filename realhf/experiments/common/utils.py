# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import collections
import dataclasses
import enum
import itertools
import re
from typing import *

import numpy as np
from omegaconf import DictConfig, OmegaConf

from realhf.api.core.config import (
    ModelBackendAbstraction,
    ModelInterfaceType,
    ModelName,
)
from realhf.api.core.dfg import OffloadHook, ParamReallocHook
from realhf.api.quickstart.device_mesh import RPCAllocation
from realhf.api.quickstart.model import (
    ModelTrainEvalConfig,
    ParallelismConfig,
    parallelism_eq,
)
from realhf.base import logging
from realhf.base.topology import PipeModelDataParallelTopology

logger = logging.getLogger("Experiment Common Utils", "benchmark")


def get_topo(
    parallel: ParallelismConfig,
    gradient_checkpointing: bool,
    gradient_accumulation_fusion: bool,
    max_prompt_len: Optional[int] = None,
) -> PipeModelDataParallelTopology:
    return PipeModelDataParallelTopology(
        num_mp=parallel.model_parallel_size,
        num_pp=parallel.pipeline_parallel_size,
        num_dp=parallel.data_parallel_size,
        sequence_parallel=parallel.use_sequence_parallel,
        gradient_checkpointing=gradient_checkpointing,
        max_prompt_len=max_prompt_len,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
    )


def get_world_size(parallel: ParallelismConfig) -> int:
    return (
        parallel.model_parallel_size
        * parallel.pipeline_parallel_size
        * parallel.data_parallel_size
    )


def make_train_backend_config(
    model_cfg: ModelTrainEvalConfig, parallel_cfg: ParallelismConfig
):
    if model_cfg.backend == "deepspeed":
        return ModelBackendAbstraction(
            "deepspeed",
            args=dict(
                optimizer_name="adam",
                optimizer_config=dict(
                    lr=model_cfg.optimizer.lr,
                    weight_decay=model_cfg.optimizer.weight_decay,
                    eps=model_cfg.optimizer.eps,
                    betas=(
                        model_cfg.optimizer.beta1,
                        model_cfg.optimizer.beta2,
                    ),
                ),
                lr_scheduler_type=model_cfg.optimizer.lr_scheduler_type,
                warmup_steps_proportion=model_cfg.optimizer.warmup_steps_proportion,
                min_lr_ratio=model_cfg.optimizer.min_lr_ratio,
                zero_stage=(
                    model_cfg.zero_stage
                    if parallel_cfg.pipeline_parallel_size == 1
                    else min(model_cfg.zero_stage, 1)
                ),
                offload_optimizer_state=model_cfg.optimizer.offload,
                offload_param=model_cfg.offload,
                enable_bf16=model_cfg.enable_bf16,
                enable_fp16=model_cfg.enable_fp16,
            ),
        )
    elif model_cfg.backend == "megatron":
        if model_cfg.optimizer.offload or model_cfg.offload:
            raise ValueError("Offload is not supported in Megatron backend.")
        if model_cfg.zero_stage == 3:
            raise ValueError("Zero stage 3 is not supported in Megatron backend.")
        if model_cfg.zero_stage == 2:
            logger.warning(
                "Megatron does not support ZeRO stage 2. Degenerates to stage 1."
            )
            model_cfg.zero_stage = 1
        megatron_args: Dict[str, Any] = OmegaConf.to_container(model_cfg.megatron)
        return ModelBackendAbstraction(
            "megatron",
            args=dict(
                enable_bf16=model_cfg.enable_bf16,
                enable_fp16=model_cfg.enable_fp16,
                zero_stage=model_cfg.zero_stage,
                optimizer=model_cfg.optimizer,
                **megatron_args,
            ),
        )
    elif model_cfg.backend == "mock_train":
        return ModelBackendAbstraction(
            "mock_train",
            args=dict(
                optimizer_name="adam",
                optimizer_config=dict(
                    lr=model_cfg.optimizer.lr,
                    weight_decay=model_cfg.optimizer.weight_decay,
                    eps=model_cfg.optimizer.eps,
                    betas=(
                        model_cfg.optimizer.beta1,
                        model_cfg.optimizer.beta2,
                    ),
                ),
            ),
        )
    else:
        raise NotImplementedError(f"Backend {model_cfg.backend} is not supported.")


def make_inf_backend_config(
    model_cfg: ModelTrainEvalConfig, parallel_cfg: ParallelismConfig
):
    return ModelBackendAbstraction("inference")


def resolve_replica_ids(
    rpc_allocs: List[RPCAllocation], models: Dict[str, ModelTrainEvalConfig]
):
    role_cnt = collections.defaultdict(int)
    first_device_mesh = dict()
    first_parallel = dict()
    first_rpc = dict()
    for alloc in rpc_allocs:
        rpc = alloc.rpc
        if rpc.role not in first_device_mesh:
            first_device_mesh[rpc.role] = alloc.device_mesh
            first_parallel[rpc.role] = alloc.parallel
            first_rpc[rpc.role] = rpc
            continue
        model_cfg = models[rpc.role]
        if (rpc.is_train() and first_rpc[rpc.role].is_generate()) or (
            rpc.is_generate() and first_rpc[rpc.role].is_train()
        ):
            if model_cfg.vllm.hybrid_train:
                role_cnt[rpc.role] += 1
                rpc.model_name = ModelName(rpc.role, role_cnt[rpc.role])
                continue
        if alloc.device_mesh != first_device_mesh[rpc.role] or not parallelism_eq(
            alloc.parallel, first_parallel[rpc.role]
        ):
            role_cnt[rpc.role] += 1
            rpc.model_name = ModelName(rpc.role, role_cnt[rpc.role])
            continue
        assert rpc.model_name.replica_id == 0


def resolve_rpc_hooks(
    rpc_allocs: List[RPCAllocation], model_configs: Dict[str, ModelTrainEvalConfig]
):
    role_interface_types = collections.defaultdict(set)
    for rpc_alloc in rpc_allocs:
        role_interface_types[rpc_alloc.rpc.role].add(rpc_alloc.rpc.interface_type)

    for i, rpc_alloc in enumerate(rpc_allocs):
        rpc = rpc_alloc.rpc
        parallel = rpc_alloc.parallel
        device_mesh = rpc_alloc.device_mesh
        # check param realloc hooks for train_step rpcs
        if rpc.interface_type == ModelInterfaceType.TRAIN_STEP:
            for j, other in enumerate(rpc_allocs):
                if rpc.name == other.rpc.name:
                    continue
                if rpc.role != other.rpc.role:
                    continue
                if (
                    parallelism_eq(parallel, other.parallel)
                    and device_mesh == other.device_mesh
                    and not (
                        model_configs[rpc.role].vllm.hybrid_train
                        and other.rpc.is_generate()
                    )
                ):
                    continue
                self_config = model_configs[rpc.model_name.role]
                other_config = model_configs[other.rpc.model_name.role]
                if (
                    self_config.backend == "deepspeed"
                    or other_config.backend == "deepspeed"
                ):
                    raise ValueError(
                        "Param realloc hooks are not supported in DeepSpeed backend."
                    )
                other.rpc.add_pre_hook(ParamReallocHook(source=rpc.model_name))
                other.rpc.add_post_hook(ParamReallocHook(target=rpc.model_name))
                logger.info(
                    f"Add param sync hooks between "
                    f"{rpc.name} and {other.rpc.name} for role {rpc.role}"
                )

        # Add offload hooks for inference and generate rpcs.
        # Add the offload hook only if the role will not be trained (e.g., reward model)
        # and its allocation is overlapped with at least one other RPCs.
        # As a result, a single inference/generate RPC will not be offloaded.
        overlapped_with_other = False
        for other in rpc_allocs:
            if rpc.name == other.rpc.name:
                continue
            if np.any(np.logical_and(other.device_mesh.mapping, device_mesh.mapping)):
                overlapped_with_other = True
                break
        if (
            ModelInterfaceType.TRAIN_STEP not in role_interface_types[rpc.role]
            and overlapped_with_other
        ):
            rpc.add_post_hook(OffloadHook())
            logger.info(f"Add offload hook for rpc {rpc.name} for role {rpc.role}")


class AllocationType(enum.Enum):
    DECOUPLED = 1
    GLOBAL_HYBRID = 2
    MANUAL = 3
    SEARCH = 4


@dataclasses.dataclass
class AllocationMode:
    type_: AllocationType
    parallel_strat: Dict[str, Dict[str, int]]

    def is_decoupled(self):
        return self.type_ == AllocationType.DECOUPLED

    def is_global_hybrid(self):
        return self.type_ == AllocationType.GLOBAL_HYBRID

    @classmethod
    def from_str(cls, allocation_mode: str):
        if allocation_mode == "manual":
            return cls(AllocationType.MANUAL, None)
        if allocation_mode == "search":
            return cls(AllocationType.SEARCH, None)
        alloc_3d = AllocationMode.extract_3d_alloc(allocation_mode)
        alloc_hybrid = AllocationMode.extract_key_value_alloc(allocation_mode)
        alloc_decoupled = AllocationMode.extract_decoupled_alloc(allocation_mode)
        if alloc_decoupled:
            return cls(AllocationType.DECOUPLED, alloc_decoupled)
        if alloc_3d:
            return cls(AllocationType.GLOBAL_HYBRID, alloc_3d)
        if alloc_hybrid:
            return cls(AllocationType.GLOBAL_HYBRID, alloc_hybrid)
        raise NotImplementedError(f"Failed to parse allocation: {allocation_mode}")

    @staticmethod
    def extract_3d_alloc(allocation_mode: str) -> Dict | None:
        for x, y, z in itertools.permutations(["d", "m", "p"]):
            pattern = rf"{x}(\d+){y}(\d+){z}(\d+)"
            m = re.match(pattern, allocation_mode)
            if not m:
                continue
            a, b, c = map(int, m.groups())
            # to be consistent with the key-value pattern
            return {
                "*": {
                    x: a,
                    y: b,
                    z: c,
                }
            }

    @staticmethod
    def extract_decoupled_alloc(allocation_mode: str) -> Dict | None:
        pattern = re.compile(
            r"(?:(?:vllm|sglang)\.(.+?)\+(.+))|(?:(.+?)\+(?:vllm|sglang)\.(.+))"
        )
        m = pattern.match(allocation_mode)
        if not m:
            return
        if m.group(1):
            gen_alloc = m.group(1)
            other_alloc = m.group(2)
        else:
            gen_alloc = m.group(4)
            other_alloc = m.group(3)
        gen_alloc = AllocationMode.extract_3d_alloc(gen_alloc)
        if not gen_alloc:
            return
        other_alloc = AllocationMode.extract_3d_alloc(
            other_alloc
        ) or AllocationMode.extract_key_value_alloc(other_alloc)
        if not other_alloc:
            return
        other_alloc.update({"gen": gen_alloc["*"]})
        return other_alloc

    @staticmethod
    def extract_key_value_alloc(
        allocation_mode: str,
    ) -> Dict[str, Dict[str, int]] | None:
        def parse_key_value_pairs(s: str):
            pattern = re.compile(r"([^:,]+):([^:,]+)")
            matches = pattern.findall(s)
            if not matches:
                return None
            return {key: value for key, value in matches}

        allocs = parse_key_value_pairs(allocation_mode)
        if not allocs:
            return
        for k, v in allocs.items():
            v = AllocationMode.extract_3d_alloc(v)
            if not v:
                return
            allocs[k] = v["*"]
        return allocs


def asdict(cfg):
    if isinstance(cfg, (OmegaConf, DictConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return dataclasses.asdict(cfg)
