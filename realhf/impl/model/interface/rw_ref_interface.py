# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import dataclasses
import functools
import itertools
import time
from typing import Dict, Literal, Optional, Tuple

import torch

import realhf.api.core.model_api as model_api
import realhf.base.logging as logging
import realhf.impl.model.utils.ppo_functional as ppo_functional
from realhf.api.core.data_api import MicroBatchSpec, SequenceSample
from realhf.base.datapack import flat2d
from realhf.impl.model.interface.rw_interface import PackedRewardInterface
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.utils.functional import (
    gather_packed_shifted_log_probs,
    masked_normalization,
)

logger = logging.getLogger("RefRwInterface")

TASK_TYPE_REF: Literal["ref"] = "ref"
TASK_TYPE_RW_MATH: Literal["rw_math"] = "rw_math"
TASK_TYPE_RW_CODE: Literal["rw_code"] = "rw_code"


@dataclasses.dataclass
class RefRwInterface(model_api.ModelInterface):
    n_minibatches: int = 4

    # Use dict here to allow argument passing through commandline.
    generation_config: Dict = dataclasses.field(default_factory=dict)

    kl_ctl: float = 0.1

    adv_norm: bool = True
    discount: float = 1.0
    gae_lambda: float = 1.0

    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0

    disable_value: bool = False

    early_stop_kl: Optional[float] = None  # e.g. 0.1
    early_stop_imp_ratio: Optional[float] = None  # e.g., 10.0

    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000

    enable_save: bool = True

    value_norm: bool = False
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp"
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5

    group_size: int = 1
    generation_size: Optional[int] = None
    mask_no_eos_with_zero: bool = False
    group_adv_norm: bool = False
    mask_too_long: bool = False
    use_dense_reward: bool = False
    reward_delta: bool = True
    token_normalize_scope: Literal["global", "dp"] = "global"
    rew_inf_args: Dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(
                self.kl_ctl, self.adaptive_kl_target, self.adaptive_kl_horizon
            )
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.kl_ctl)
        if self.value_norm:
            from realhf.impl.model.modules import (
                ExponentialRunningMeanStd,
                MovingAverageRunningMeanStd,
            )

            if self.value_norm_type == "exp":
                self.rms = ExponentialRunningMeanStd(
                    beta=self.value_norm_beta, epsilon=self.value_norm_eps
                )
            elif self.value_norm_type == "ma":
                self.rms = MovingAverageRunningMeanStd()
            else:
                raise ValueError(f"Unknown value_norm_type {self.value_norm_type}")
        self.kl_ctl = None

        self.gconfig = model_api.GenerationHyperparameters(**self.generation_config)
        if self.generation_size is not None:
            assert self.generation_size >= self.group_size
        else:
            self.generation_size = self.group_size
        self.gconfig.n = self.generation_size

    def save(self, model: model_api.Model, save_dir: str):
        if not self.enable_save:
            return
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        module.save_to_hf(
            tokenizer=model.tokenizer,
            save_dir=save_dir,
        )

    def _dispatch_tasks(self, data):
        math_data, code_data, rlhf_data, ref_data = data, data, data, data
        return math_data, code_data, rlhf_data, ref_data

    def _gather_tasks(self, data_map):
        # merge SequenceSamples from math_data, code_data, rlhf_data, ref_data
        return data_map.get(TASK_TYPE_REF, None)

    @torch.no_grad()
    def ref_inference(
        self, model: model_api.Model, input_: SequenceSample, mb_spec: MicroBatchSpec
    ):
        module = model.module
        module.eval()

        # This post_hook will gather log probabilities in mini-batches,
        # reducing peak memory usage.
        def calc_logprobs(logits, input_):
            logits /= self.gconfig.temperature

            input_lens = torch.tensor(input_.seqlens["packed_input_ids"]).view(-1)
            cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()

            logprobs = gather_packed_shifted_log_probs(
                logits, cu_seqlens, input_.data["packed_input_ids"]
            )
            return logprobs

        input_flattend = SequenceSample.from_default(
            ids=list(range(input_.bs * self.group_size)),
            seqlens=flat2d(input_.seqlens["packed_input_ids"]),
            data=dict(packed_input_ids=input_.data["packed_input_ids"]),
        )
        # add posthook to avoid storing full logits
        logprobs = module.forward(
            input_=input_flattend,
            post_hook=calc_logprobs,
            output_seqlens=[
                [x - 1 for x in slens]
                for slens in input_flattend.seqlens["packed_input_ids"]
            ],
            mb_spec=mb_spec,
        )

        res = SequenceSample(
            keys=["packed_ref_logprobs"],
            ids=input_.ids,
            dtypes=dict(packed_ref_logprobs=model.module.dtype),
            trailing_shapes=dict(packed_ref_logprobs=()),
            data=dict(packed_ref_logprobs=logprobs),
            seqlens=dict(
                packed_ref_logprobs=[
                    [x - 1 for x in slen] for slen in input_.seqlens["packed_input_ids"]
                ]
            ),
        )

        return res

    def inference(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> SequenceSample:
        math_data, code_data, rlhf_data, ref_data = self._dispatch_tasks(input_)

        if not hasattr(self, "rew_inf_args") or not isinstance(self.rew_inf_args, dict):
            raise ValueError("Invalid rew_inf_args. Expected a dictionary.")
        rewardInterface = PackedRewardInterface(**self.rew_inf_args)
        logger.info(f"self.rew_inf_args: {self.rew_inf_args}, input_: {input_}")

        task_map = {
            TASK_TYPE_REF: (self.ref_inference, ref_data),
            TASK_TYPE_RW_MATH: (rewardInterface.inference, math_data),
            TASK_TYPE_RW_CODE: (rewardInterface.inference, code_data),
        }

        def _task_func(func, task_type: str):
            def _wrapped_func(*args, **kwargs):
                start_time = time.perf_counter()
                logger.info(f"[{task_type}] ref_rw task start @ {start_time:.4f}")
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"{task_type} ref_rw task failed: {e}")
                finally:
                    duration = time.perf_counter() - start_time
                    logger.info(
                        f"[{task_type}] ref_rw task cost: {duration:.4f}s, start @ {start_time:.4f}"
                    )
                return result

            return _wrapped_func

        async def _run_tasks() -> dict:
            tasks = []
            for task_type, (func, data) in task_map.items():
                if not data:
                    continue
                task_func = _task_func(func, task_type)
                task_args = (model, data, mb_spec)
                task = asyncio.create_task(asyncio.to_thread(task_func, *task_args))
                tasks.append((task_type, task))

            results = {}
            for task_type, task in tasks:
                try:
                    results[task_type] = await task
                except Exception as e:
                    logger.error(f"{task_type} task failed: {e}")
                    results[task_type] = None
            return results

        task_results = asyncio.run(_run_tasks())
        final_result = self._gather_tasks(task_results)
        return final_result

    # Mock methods for profiling only.
    def _mock_inference(
        self,
        model: model_api.Model,
        dataset_input: SequenceSample,
    ) -> SequenceSample:
        prompt_lens = flat2d(dataset_input.seqlens["packed_prompts"])
        seqlens = [x + self.gconfig.max_new_tokens for x in prompt_lens]
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        mconfig = module.config
        packed_input_ids = torch.randint(
            0,
            mconfig.vocab_size,
            (sum(seqlens),),
            dtype=torch.long,
            device=model.device,
        )

        return SequenceSample.from_default(
            seqlens=seqlens,
            ids=dataset_input.ids,
            data=dict(packed_input_ids=packed_input_ids),
        )


model_api.register_interface("ref_rw", RefRwInterface)
