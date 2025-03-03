# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").
import os
from typing import List

import realhf.api.core.model_api as model_api
from realhf.api.quickstart.device_mesh import RPCAllocation
from realhf.api.quickstart.model import ModelTrainEvalConfig, vLLMConfig
from realhf.base import logging

logger = logging.getLogger(__name__)


def check_is_realhf_native_impl(_cls):
    return _cls.__module__.startswith("realhf")


def check_is_realhf_native_model_interface(name):
    # NOTE: we should not import iterfaces here,
    # such that we can avoid CUDA initialization.
    return name in ["ppo_actor", "ppo_critic", "sft"]


def check_valid_vllm(role: str, vllm: vLLMConfig, rpc_allocs: List[RPCAllocation]):
    rpcs = [alloc.rpc for alloc in rpc_allocs if alloc.rpc.role == role]
    if vllm.hybrid_train and not any(rpc.is_train() for rpc in rpcs):
        logger.warning(
            "vLLM hybrid_train is enabled, but no training RPCs are found. Set it to False."
        )
        vllm.hybrid_train = False
    if vllm.hybrid_train and not vllm.enforce_eager:
        raise ValueError("vLLM hybrid_train requires eager mode to be enabled.")


def check_valid_optimizer(model: ModelTrainEvalConfig):
    if model.optimizer.min_lr_ratio < 0.0 or model.optimizer.min_lr_ratio > 1.0:
        raise ValueError(f"Invalid min_lr_ratio: {model.optimizer.min_lr_ratio}")
    if (
        model.optimizer.warmup_steps_proportion < 0.0
        or model.optimizer.warmup_steps_proportion > 1.0
    ):
        raise ValueError(
            f"Invalid warmup_steps_proportion: {model.optimizer.warmup_steps_proportion}"
        )


def check_valid_backend(role: str, model: ModelTrainEvalConfig):
    if (model.offload or model.optimizer.offload) and model.backend != "deepspeed":
        raise ValueError(
            f"For model `{role}`, offload is only" " valid for the deepspeed backend."
        )
    if model.backend == "megatron" and model.zero_stage in [3]:
        raise ValueError(
            f"For model `{role}`, the Megatron backend"
            " only supports zero stage 0, 1 or 2."
        )


def check_valid_model_and_path(role: str, model: ModelTrainEvalConfig):
    if model.enable_bf16 and model.enable_fp16:
        raise ValueError(
            f"For model `{role}`, enable_bf16 and" " enable_fp16 cannot be both True."
        )

    if not os.path.exists(model.path):
        raise FileNotFoundError(
            f"The model path `{model.path}` for `{role}` does not exist locally. "
            "You must download the HuggingFace checkpoint before loading it."
        )


def check_valid_parallel_batch_size(rpc_alloc: RPCAllocation):
    try:
        rpc = rpc_alloc.rpc
        mb_spec = rpc.mb_spec

        dp_size = rpc_alloc.parallel.data_parallel_size
        tp_size = rpc_alloc.parallel.model_parallel_size
        pp_size = rpc_alloc.parallel.pipeline_parallel_size

        factor = 1
        if rpc.is_train() and rpc_alloc.parallel.pipeline_parallel_size > 1:
            factor = 2

        assert (
            rpc.n_seqs
            >= factor * dp_size * pp_size * rpc.min_n_seqs_per_pass * mb_spec.n_mbs
        ), (
            rpc.name,
            rpc.n_seqs,
            mb_spec,
            rpc.min_n_seqs_per_pass,
            factor,
            dp_size,
            pp_size,
        )
    except AssertionError as e:
        raise ValueError(
            f"Invalid parallel batch size and batch size configuration."
        ) from e
