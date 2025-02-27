# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
from typing import *

import realhf.base.logging as logging
from realhf.api.core.config import (
    ModelAbstraction,
    ModelFamily,
    ModelWrapperAbstraction,
)

logger = logging.getLogger("Quickstart Model Config")


@dataclasses.dataclass(unsafe_hash=True)
class ParallelismConfig:
    """Configuration for 3D parallelism.

    :param model_parallel_size: Size of tensor-model parallelism.
    :type model_parallel_size: int
    :param pipeline_parallel_size: Number of pipeline parallelism
        stages.
    :type pipeline_parallel_size: int
    :param data_parallel_size: Data parallelism size for ZeRO
        optimization.
    :type data_parallel_size: int
    :param use_sequence_parallel: Whether to use sequence parallelism in
        Megatron in combination with tensor-model parallelism.
    :type use_sequence_parallel: bool
    """

    model_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    use_sequence_parallel: bool = False

    def __str__(self):
        return (
            f"Parallel(mp={self.model_parallel_size},"
            f"pp={self.pipeline_parallel_size},"
            f"dp={self.data_parallel_size})"
        )


def parallelism_eq(this, other):
    # NOTE: We write this function because
    # 1) we don't want to compare sequence_parallelism (it's irrelevant to parameter reallocation)
    # 2) implementing this function as a method of ParallelismConfig would cause a OmegaConf bug
    return (
        (this.model_parallel_size == other.model_parallel_size)
        and (this.pipeline_parallel_size == other.pipeline_parallel_size)
        and (this.data_parallel_size == other.data_parallel_size)
    )


@dataclasses.dataclass
class OptimizerConfig:
    """Configuration for the optimizer.

    For models that will not be trained, the optimizer type should be
    set to "empty".

    :param type: Type of optimizer. Currently, only "adam" and "empty"
        optimizers are supported.
    :type type: str
    :param lr: Learning rate.
    :type lr: float
    :param weight_decay: Weight decay.
    :type weight_decay: float
    :param beta1: Adam beta1 parameter.
    :type beta1: float
    :param beta2: Adam beta2 parameter.
    :type beta2: float
    :param eps: Adam epsilon parameter in the denominator.
    :type eps: float
    :param min_lr_ratio: Minimum learning rate ratio after learning rate
        annealing. Should be in the interval [0.0, 1.0].
    :type min_lr_ratio: float
    :param lr_scheduler_type: Type of learning rate scheduler. One of
        "linear", "cosine", or "constant".
    :type lr_scheduler_type: str
    :param warmup_steps_proportion: Proportion of total training steps
        allocated for warming up. Should be in the interval [0.0, 1.0].
    :type warmup_steps_proportion: float
    :param offload: Whether to offload the optimizer to CPU. Only valid
        for the DeepSpeed backend.
    :type offload: bool
    """

    type: str = dataclasses.field(
        metadata={"choices": ["adam", "empty"]},
        default="adam",
    )
    lr: float = 1e-5
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-5
    min_lr_ratio: float = 0.0
    lr_scheduler_type: str = dataclasses.field(
        metadata={"choices": ["linear", "cosine", "constant"]},
        default="cosine",
    )
    warmup_steps_proportion: float = 0.02
    offload: bool = False
    initial_loss_scale: float = 2**32
    min_loss_scale: float = 1.0
    loss_scale_window: float = 5
    hysteresis: int = 2
    gradient_clipping: float = 1.0


@dataclasses.dataclass
class vLLMConfig:
    max_num_seqs: int = 256
    kv_cache_type: str = "auto"
    num_scheduler_steps: int = 1
    multi_step_stream_outputs: bool = True
    block_size: int = 16
    swap_space: int = 4
    cpu_offload_gb: float = 0
    max_seq_len_to_capture: int = 8192
    max_model_len: Optional[int] = None
    # Setting it to False because it will reuse the block after
    # model weights are updated. Using v0.7.2 reset_prefix_cache
    # will fix this issue.
    enable_prefix_caching: bool = False
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    hybrid_train: bool = False
    additional_engine_args: Dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class MegatronConfig:
    """When using the DistributedOptimizer of Megatron, parameters and
    gradients will not be splitted across DP ranks, but optimizer states will
    be. In other words, Megatron only supports ZeRO-1.

    Megatron DDP will split the whole flattend parameter into buckets.
    Buckets do not respect parameter boundaries and are dispatched to different DP ranks.
    The optimizer on a specific DP rank will only manage its own bucket,
    but parameters and gradients are held by all ranks and will not be further splitted.
    (That's why only optimizer states are partitioned.) During backward, bucket gradients
    will be scatter-reduced (controlled by the `use_distributed_optimizer` option
    in Megatron DDP, otherwise all-reduce will be issued), and parameters will then
    be updated locally. At this point, the parameters are not synced across DP ranks.
    The DistributedOptimizer will then call all-gather on parameters.

    Since Megatron allocates static tensors for scatter-reducing parameter gradients,
    it does not decrease memory usage just as DeepSpeed ZeRO-2. To be more specific,
    with dynamic allocation, we can allocate gradient memory layer-by-layer. When the
    backward finishes at layer N, we can scatter-reduce gradients and release the memory
    after scattering. As a result, given DP size K, layer number L, and parameter size P
    for each layer, dynamic allocation requires P * (1 + L/K) memory for gradients,
    but Megatron requires P * L. Memory is not freed after scattering in Megatron.

    'use_distributed_optimizer' enables bucketing and scatter-reduce gradients.
    When setting to False, optimizer states will not be partitioned.

    'overlap_grad_reduce' enables issuing all-reduce/scatter-reduce on the fly
    during bacwkard once the gradient is ready, which should usually be enabled.

    'overlap_param_gather' overlaps param all-gather with the next forward pass.
    It creates a forward hook that waits for the previous parameter all-gather
    after the optimizer step. While this sounds good, it can be problematic with
    parameter reallocation, because the reallocated parameters do not have the hook.
    Can be enabled for SFT, but should be disabled for PPO.

    As a final note, Megatron is in an awkward place for PPO with param-realloc.
    First, it does not minimize the memory usage of gradients (i.e., ZeRO-2).
    Second, for functional correctness, we can't enable `overlap_param_gather`,
    and a parameter update will be scatter-reduce grad + all-gather param, instead
    of an all-reduce (running all-reduce requires setting `use_distributed_optimizer`
    to False, but that will not partition optimizer states!), so it is not that
    efficient, either. We use Megatron because it is the only backend that we can
    make it functionally correct. The DeepSpeed code is too hard to read and modify.
    """

    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = False
    accumulate_allreduce_grads_in_fp32: bool = False

    # addtional args
    additional_config: Dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelTrainEvalConfig:
    """Runtime configuration for models (or LLMs) in ReaL.

    We use a customized model class instead of HuggingFace's. This customized model has
    the following highlights:

    1. Support for 3D parallelism and sequence parallelism.

    2. Support for flash attention during both training and generation.

    3. Input sequences are packed into a single 1D tensor to save GPU memory and improve efficiency.

    Consequently, each HuggingFace model of interest needs to be manually converted to this
    customized model. Implemented models can be found in the ``realhf/api/from_hf/`` directory.

    :param type: Model family type, e.g., llama, qwen2, etc.
    :type type: ModelFamily
    :param backend: Backend for training. Currently, only "megatron" and "deepspeed" are supported.
        Use "deepspeed" for offloading parameters or optimizer states, and "megatron" for
        parameter reallocation.
    :type backend: str
    :param path: Path of the HuggingFace checkpoint.
    :type path: str
    :param gradient_checkpointing: Whether to use gradient checkpointing to save memory.
    :type gradient_checkpointing: bool
    :param enable_fp16: Whether to use fp16 precision.
    :type enable_fp16: bool
    :param enable_bf16: Whether to use bf16 precision. Mutually exclusive with fp16.
    :type enable_bf16: bool
    :param offload: Whether to offload model parameters to CPU. Only valid for the DeepSpeed backend.
    :type offload: bool
    :param parallel: Configuration for parallelism.
    :type parallel: ParallelismConfig
    :param zero_stage: Stage of ZeRO optimization. Should be one of 0, 1, 2, or 3.
    :type zero_stage: int
    :param optimizer: Configuration for the optimizer.
    :type optimizer: Optional[OptimizerConfig]
    :param init_critic_from_actor: Whether to initialize a critic/reward model from a saved LM checkpoint.
    :type init_critic_from_actor: bool
    """

    type: ModelFamily = dataclasses.field(default=ModelFamily("llama", 7, False))
    backend: str = dataclasses.field(
        default="megatron", metadata={"choices": ["megatron", "deepspeed"]}
    )
    path: str = ""
    gradient_checkpointing: bool = True
    enable_fp16: bool = True
    enable_bf16: bool = False
    offload: bool = False
    zero_stage: int = dataclasses.field(
        metadata={"choices": [0, 1, 2, 3]},
        default=2,
    )
    optimizer: Optional[OptimizerConfig] = dataclasses.field(
        default_factory=OptimizerConfig
    )
    megatron: MegatronConfig = dataclasses.field(default_factory=MegatronConfig)
    vllm: vLLMConfig = dataclasses.field(default_factory=vLLMConfig)
    init_from_scratch: bool = False
    init_critic_from_actor: bool = False


def get_real_model_config(
    model_path: str,
    hf_model_family: str,
    is_critic: bool,
    init_from_scratch: bool,
    init_critic_from_actor: bool,
    dtype: Optional[str] = None,
) -> ModelAbstraction:
    """Make a configuration to build model."""
    model = ModelAbstraction(
        "real_model",
        args=dict(
            model_path=model_path,
            is_critic=is_critic,
            init_critic_from_actor=init_critic_from_actor,
            dtype=dtype,
            hf_model_family=hf_model_family,
            init_from_scratch=init_from_scratch,
        ),
    )
    return model
