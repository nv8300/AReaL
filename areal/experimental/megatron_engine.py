import dataclasses
import functools
import gc
import os
from datetime import datetime
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.optimizer import OptimizerConfig as MCoreOptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from tensordict import TensorDict

from areal.api.cli_args import MicroBatchSpec
from areal.api.engine_api import FinetuneSpec, TrainEngine
from areal.experimental.api.cli_args import (
    ExperimentalTrainEngineConfig as TrainEngineConfig,
)
from areal.experimental.api.io_struct import (
    MegatronParallelStrategy,
    ParallelStrategy,
    ParamSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.experimental.model.registry import (
    load_from_hf,
    make_hf_and_mcore_config,
    make_mcore_model,
    save_to_hf,
)
from areal.experimental.utils.mcore.packed_context_parallel import (
    packed_context_parallel_forward,
)
from areal.experimental.utils.megatron_checkpointer import MegatronCheckpointManager
from areal.utils import logging, name_resolve, names, pkg_version
from areal.utils.data import (
    MicroBatchList,
    amend_position_ids,
    broadcast_tensor,
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    pad_mb_list,
    reorder_list,
    split_padded_tensor_dict_into_mb_list,
    unpack_sequence,
    unpad_logits,
)
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.model import disable_dropout_in_model
from areal.utils.nccl import NCCL_DEFAULT_TIMEOUT

USE_MBRIDGE = False
if pkg_version.is_available("mbridge"):
    import mbridge

    USE_MBRIDGE = True
else:
    USE_MBRIDGE = False


logger = logging.getLogger("MegatronEngine")


class MegatronEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.hf_config = None
        self.tf_config = None
        self.model = None
        self.dtype = getattr(torch, self.config.dtype)
        self.device = None
        self.optimizer_config = config.optimizer
        self.mcore_config = config.megatron
        self.parallel_strategy = None
        self.optimizer = None
        self.lr_scheduler = None
        self.bridge = None
        self.initialized = False
        self._version: int = 0
        self.rank = None
        self.world_size = None
        self.rank_generator = None
        self.context_and_model_parallel_group = None
        self.checkpointer = None
        self.seed = 0

    def initialize(
        self,
        addr: str | None,
        ft_spec: FinetuneSpec | None,
        parallel_strategy: ParallelStrategy,
        seed: int = 0,
    ):
        # TODO: add parallel_strategy & seed in engine api when moving out of experimental
        self.parallel_strategy = self._make_parallel_strategy(parallel_strategy)
        self.seed = seed

        assert addr is None, "FSDPEngine does not support remote initialization."
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        self.create_process_group()
        self.tokenizer = load_hf_tokenizer(self.config.path)
        if USE_MBRIDGE:
            self.bridge = mbridge.AutoBridge.from_pretrained(self.config.path)
            self.bridge.dtype = self.dtype
            logger.info(
                "Using mbridge to create models and hf model save/load in MegatronEngine."
            )

        self.hf_config, self.tf_config = make_hf_and_mcore_config(
            self.config.path, dtype=self.dtype, bridge=self.bridge
        )
        # initialize mcore (DDP Wrapped) GPTModel
        with self.device:
            self.model = make_mcore_model(
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                mcore_config=self.mcore_config,
                bridge=self.bridge,
            )
            self._load_model_from_hf(self.config.path)

        if self.config.disable_dropout:
            disable_dropout_in_model(self.model)

        self.create_optimizer(ft_spec)
        self.initialized = True

    def _make_parallel_strategy(
        self, parallel_strategy: ParallelStrategy
    ) -> MegatronParallelStrategy:
        return MegatronParallelStrategy(
            use_sequence_parallel=parallel_strategy.tensor_parallel_size > 1,
            **dataclasses.asdict(parallel_strategy),
        )

    def create_process_group(self):
        # Required by NCCL weight update group for SGLang
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        if not dist.is_initialized():
            # TODO: Handle the condition when WORLD_SIZE and RANK is not set in launcher
            # NOTE: device_id **SHOULD NOT** be passed into init_process_group,
            # otherwise initializing the NCCL weight update group will be wrong!
            dist.init_process_group(
                backend="nccl",
                timeout=NCCL_DEFAULT_TIMEOUT,
            )
            # Initialize Megatron parallel states
            # NOTE: we assume all MegatronEngine has the same parallel strategy.
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.parallel_strategy.tensor_parallel_size,
                pipeline_model_parallel_size=self.parallel_strategy.pipeline_parallel_size,
                virtual_pipeline_model_parallel_size=self.parallel_strategy.virtual_pipeline_parallel_size,
                use_sharp=False,
                order="tp-cp-ep-dp-pp",
                context_parallel_size=self.parallel_strategy.context_parallel_size,
                expert_model_parallel_size=self.parallel_strategy.expert_parallel_size,
                expert_tensor_parallel_size=self.parallel_strategy.expert_tensor_parallel_size,
                distributed_timeout_minutes=int(NCCL_DEFAULT_TIMEOUT.seconds / 60),
            )
            self.own_global_group = True

        # Initialize context and model parallel groups, which are only used in AReaL
        # for data distribution
        rank_generator = mpu.RankGenerator(
            tp=self.parallel_strategy.tensor_parallel_size,
            ep=1,
            dp=self.parallel_strategy.data_parallel_size,
            pp=self.parallel_strategy.pipeline_parallel_size,
            cp=self.parallel_strategy.context_parallel_size,
            order="tp-cp-ep-dp-pp",
            rank_offset=0,
        )
        context_and_model_parallel_ranks = rank_generator.get_ranks("tp-cp-pp")
        # create context and model_parallel_groups
        for dp_rank, ranks in enumerate(context_and_model_parallel_ranks):
            group = mpu.create_group(
                ranks,
                timeout=NCCL_DEFAULT_TIMEOUT,
                pg_options=mpu.get_nccl_options("tp-cp-pp", {}),
                group_desc="CONTEXT_AND_MODEL_PARALLEL_GROUP",
            )
            if dp_rank == mpu.get_data_parallel_rank():
                self.context_and_model_parallel_group = group
        self._parallelism_group = dist.new_group()
        # Set megatron model parallel seed
        tensor_parallel.model_parallel_cuda_manual_seed(self.seed)

    def create_optimizer(self, ft_spec: FinetuneSpec):
        if self.optimizer_config is None:
            return
        assert self.model is not None

        assert (
            self.optimizer_config.type == "adam"
        ), "Only AdamW optimizer is supported in this engine."

        # Make megatron optimizer config, from legacy MegatronEngine
        # TODO: add DDP options
        # TODO: check if there is more options in mcore v0.13.1
        mcore_opt_config = MCoreOptimizerConfig(
            optimizer="adam",
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
            bf16=self.dtype is torch.bfloat16,
            fp16=self.dtype is torch.float16,
            adam_beta1=self.optimizer_config.beta1,
            adam_beta2=self.optimizer_config.beta2,
            adam_eps=self.optimizer_config.eps,
            use_distributed_optimizer=self.mcore_config.ddp.use_distributed_optimizer,
            params_dtype=self.dtype,
        )
        mcore_opt_config.overlap_param_gather_with_optimizer_step = (
            self.mcore_config.overlap_param_gather_with_optimizer_step
        )
        mcore_opt_config.use_precision_aware_optimizer = (
            self.mcore_config.use_precision_aware_optimizer
        )
        mcore_opt_config.main_grads_dtype = getattr(
            torch, self.mcore_config.main_grads_dtype
        )
        mcore_opt_config.main_params_dtype = getattr(
            torch, self.mcore_config.main_params_dtype
        )
        mcore_opt_config.exp_avg_dtype = getattr(torch, self.mcore_config.exp_avg_dtype)
        mcore_opt_config.exp_avg_sq_dtype = getattr(
            torch, self.mcore_config.exp_avg_sq_dtype
        )

        self.optimizer = get_megatron_optimizer(
            mcore_opt_config,
            [self.model],
            no_weight_decay_cond=lambda n, p: any(
                k in n for k in ["bias", "ln.weight", "ln_f.weight"]
            ),
            scale_lr_cond=None,
            lr_mult=1.0,
        )

        warmup_steps_proportion = self.optimizer_config.warmup_steps_proportion
        warmup_steps = int(warmup_steps_proportion * ft_spec.total_train_steps)
        lr_scheduler = OptimizerParamScheduler(
            self.optimizer,
            init_lr=0.0 if warmup_steps_proportion > 0 else self.optimizer_config.lr,
            max_lr=self.optimizer_config.lr,
            min_lr=self.optimizer_config.min_lr_ratio * self.optimizer_config.lr,
            lr_warmup_steps=warmup_steps,
            lr_decay_steps=ft_spec.total_train_steps - warmup_steps,
            lr_decay_style=self.optimizer_config.lr_scheduler_type,
            start_wd=self.optimizer_config.weight_decay,
            end_wd=self.optimizer_config.weight_decay,
            wd_incr_steps=ft_spec.total_train_steps,
            wd_incr_style="constant",
        )
        self.lr_scheduler = lr_scheduler

        self.checkpointer = MegatronCheckpointManager(
            model=[self.model],
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.mcore_config.ddp.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.mcore_config.use_checkpoint_opt_param_scheduler,
            async_save=self.mcore_config.async_save,
        )

        self.checkpointer = MegatronCheckpointManager(
            model=[self.model],
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.mcore_config.ddp.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.mcore_config.use_checkpoint_opt_param_scheduler,
            async_save=self.mcore_config.async_save,
        )

    @property
    def parallelism_group(self) -> dist.ProcessGroup:
        assert self.initialized
        return self._parallelism_group

    def destroy(self):
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        dist.destroy_process_group(self.parallelism_group)
        if self.own_global_group:
            mpu.destroy_model_parallel()
            dist.destroy_process_group()
        self.initialized = False

    def train(self, mode: bool = True):
        self.model.train(mode=mode)
        return self

    def upload_weights(self, meta: WeightUpdateMeta):
        if meta.type == "nccl":
            raise NotImplementedError(
                "NCCL weight update is not yet supported in MegatronEngine."
            )
        elif meta.type == "disk":
            self._save_model_to_hf(meta.path, self.tokenizer, None)
            # dist.barrier() are called when _save_model_to_hf finished
            if dist.get_rank() == 0:
                update_name = names.update_weights_from_disk(
                    self.config.experiment_name,
                    self.config.trial_name,
                    self.get_version(),
                )
                name_resolve.add(
                    update_name, str(datetime.now().timestamp()), keepalive_ttl=120
                )
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def get_param_specs(
        self, weight_chunked_mem_mb: int = 1024
    ) -> List[List[ParamSpec]]:
        raise NotImplementedError()

    def set_version(self, version: int):
        self._version = version

    def get_version(self) -> int:
        return self._version

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            assert (
                not meta.with_optim
            ), "HF format does not support optimizer state saving, please use DCP format instead."
            self._save_model_to_hf(meta.path, meta.tokenizer, meta.processor)
        elif meta.weight_format == "dcp":
            self.checkpointer.save_checkpoint(meta.path, with_optimizer=meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def _save_model_to_hf(
        self, path: str, tokenizer: Any | None = None, processor: Any | None = None
    ):
        assert self.model is not None, "Model is not initialized."
        os.makedirs(path, exist_ok=True)

        # Save model weights
        save_to_hf(
            hf_config=self.hf_config,
            save_path=path,
            model=self.model,
            bridge=self.bridge,
        )

        if dist.get_rank() == 0:
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            if processor is not None:
                processor.save_pretrained(path)
        dist.barrier(device_ids=[self.device.index])

    def load(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            assert (
                not meta.with_optim
            ), "HF format does not support optimizer state loading, please use DCP format instead."
            self._load_model_from_hf(meta.path)
        elif meta.weight_format == "dcp":
            self.checkpointer.load_checkpoint(meta.path, with_optimizer=meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def _load_model_from_hf(self, path: str):
        assert self.model is not None, "Model is not initialized."
        load_from_hf(
            hf_config=self.hf_config,
            load_path=path,
            model=self.model,
            bridge=self.bridge,
        )

    # TODO: clean code, merge with base_hf_engine.py
    def prepare_mb_list(
        self, input_: TensorDict, forward_only: bool = True
    ) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        if isinstance(input_, dict):
            input_ = TensorDict(input_, batch_size=[input_["input_ids"].shape[0]])
        input_ = amend_position_ids(input_)
        # Parallel sizes
        pp_size = self.parallel_strategy.pipeline_parallel_size
        cp_size = self.parallel_strategy.context_parallel_size
        tp_size = self.parallel_strategy.tensor_parallel_size
        # Split the input tensor dict into micro-batches
        min_n_mbs = (
            1 if forward_only else 2 * pp_size
        )  # avoid pipeline bubbles in training
        # NOTE: self.config.mb_spec.max_tokens_per_gpu determines
        # the expected number of tokens per micro-batch **in the forward pass on a GPU**.
        # The micro batch list splitted here will be splitted to each
        # context parallel rank, so the total number of tokens per
        # micro-batch here will be `max_tokens_per_gpu * cp_size`.
        mb_spec = MicroBatchSpec.new(
            self.config.mb_spec,
            n_mbs=max(min_n_mbs, self.config.mb_spec.n_mbs),
            _context_parallel_size=cp_size,
        )
        mb_list = split_padded_tensor_dict_into_mb_list(
            input_,
            mb_spec,
            group=mpu.get_data_parallel_group(),
        )
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        # NOTE: Pad micro-batches to:
        # 1. Reduce GPU memory fragmentation, pad memory usage to GPU page size or
        #    max_tokens_per_gpu
        # 2. Align sequence lengths to integer multiples of `align_to_multiple_of=tp_size*cp_size*2`
        #    to satisfy the requirement of Megatron parallelism.
        align_to_multiple_of = tp_size * cp_size * 2 if cp_size > 1 else tp_size
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
            align_sequences=True,
            align_to_multiple_of=align_to_multiple_of,
        )
        logger.info(
            f"Microbatch #tokens (rank {dist.get_rank()}): {mb_list.group_lens}, aligned to: {mb_list.align_to_lengths}, "
            f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}."
        )
        # FIXME: the resulting max_seqlen is a tensor rather than an integer
        # TODO: remove the usage of tensordict
        # Modern model implementations takes a dict as the input.
        # This eliminates a bug of Qwen2.5-VL for transformers<=4.53.1
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)
        for mb in mb_list.mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
        for mb in mb_list.padded_mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
        return mb_list

    def step_lr_scheduler(self):
        assert self.lr_scheduler is not None, "LR Scheduler is not initialized."
        self.lr_scheduler.step(1)

    def train_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> Dict[str, float]:
        # TODO: simple training for testing, no parallelism and packing
        assert self.model is not None, "Model is not initialized."
        assert self.optimizer is not None, "Optimizer is not initialized."
        input_ = amend_position_ids(input_)
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(**input_)
        loss = loss_fn(output, input_)
        loss.backward()

        # Update optimizer
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "update_successful": update_successful,
        }

    @torch.no_grad()
    def eval_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> torch.Tensor | None:
        raise NotImplementedError()

    def _current_data_parallel_head(self) -> int:
        """Get the rank of the head of the current data parallel group."""
        assert self.initialized
        ranks = dist.get_process_group_ranks(self.context_and_model_parallel_group)
        return ranks[0]

    def is_data_parallel_head(self) -> bool:
        assert self.initialized
        ranks = dist.get_process_group_ranks(self.context_and_model_parallel_group)
        return ranks[0] == self.rank

    @torch.no_grad()
    def forward(
        self,
        input_: TensorDict,
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, TensorDict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        # TODO: simple forward for testing, no parallelism and packing
        assert self.model is not None, "Model is not initialized."
        mb_list = None
        cu_seqlens = None
        to_broadcast = [None, None]

        if self.is_data_parallel_head():
            cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
            mb_list = self.prepare_mb_list(input_)
            to_broadcast = [mb_list, cu_seqlens]

        torch.cuda.synchronize()
        dist.barrier()
        # broadcast mb_list to model+context parallel group
        # TODO: write a function to broadcast structure with nested tensors for
        # better performance
        dp_head_rank = self._current_data_parallel_head()
        dist.broadcast_object_list(
            to_broadcast, src=dp_head_rank, group=self.context_and_model_parallel_group
        )

        mb_list, cu_seqlens = to_broadcast
        # NOTE: Move tensors to correct device, since dist.broadcast_object_list does not
        # ensure the device of tensors in the object list
        cu_seqlens: torch.Tensor = cu_seqlens.to(self.device)
        mb_list: MicroBatchList = mb_list.to(self.device)

        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()

        max_total_len = max(m["max_seqlen"] for m in mb_list.padded_mbs)
        micro_batch_generator = iter(mb_list.padded_mbs)
        forward_step_count = 0

        def forward_step(batch_iter, model):
            nonlocal forward_step_count
            batch = next(batch_iter)
            padding_length = mb_list.padding_lengths[forward_step_count]
            cu_seqlens = batch["cu_seqlens"]
            old_cu_seqlens = mb_list.old_cu_seqlens_list[forward_step_count]

            forward_step_count += 1
            output = packed_context_parallel_forward(model, batch)

            if mpu.is_pipeline_last_stage():
                output = unpad_logits(
                    output,
                    padding_length=padding_length,
                    cu_seqlens=cu_seqlens,
                    old_cu_seqlens=old_cu_seqlens,
                )

            def _post_process_fn(input_, output):
                loss = torch.tensor(1.0, device=output.device)
                if post_hook is not None:
                    output = post_hook(output, input_)
                return loss, {"output": output}

            return output, functools.partial(_post_process_fn, batch)

        forward_backward_func = get_forward_backward_func()
        output_list = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=micro_batch_generator,
            model=self.model,
            num_microbatches=len(mb_list.padded_mbs),
            seq_length=max_total_len,  # max # tokens across all micro-batches
            micro_batch_size=1,  # should be 1 when using packed input
            forward_only=True,
        )

        result = None
        if mpu.is_pipeline_last_stage():
            res = aggregate_fn([o["output"] for o in output_list])
            output_seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
            unpacked = unpack_sequence(res, lens=output_seqlens, dim=0)
            reordered = reorder_list(unpacked, mb_list.backward_indices)
            result = pad_and_stack_tensors_along_first_dim(reordered)

        # Broadcast the shape of the result tensor
        result = broadcast_tensor(
            result,
            src_rank=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
            device=self.device,
        )
        return result
