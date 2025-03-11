# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").
import copy
import dataclasses
import math
import os
import pprint
from typing import *

import numpy as np
from omegaconf import DictConfig, OmegaConf

import realhf.base.logging as logging
from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
)
from realhf.api.core.dfg import MFCDef, ParamReallocHook
from realhf.api.core.model_api import GenerationHyperparameters
from realhf.api.core.system_api import ExperimentConfig
from realhf.api.quickstart.dataset import PromptOnlyDatasetConfig
from realhf.api.quickstart.device_mesh import DeviceMesh, MFCConfig, RPCAllocation
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig, ParallelismConfig
from realhf.experiments.common.common import CommonExperimentConfig
from realhf.experiments.common.utils import resolve_replica_ids, resolve_rpc_hooks

logger = logging.getLogger("PPO Code exp", "colored")


@dataclasses.dataclass
class PPOHyperparameters:
    """Configuration of PPO hyperparameters.

    :param gen: Generation hyperparameters.
    :type gen: GenerationHyperparameters
    :param ppo_n_minibatches: Number of minibatches in each PPO update.
    :type ppo_n_minibatches: int
    :param kl_ctl: Coefficient of KL divergence rewards.
    :type kl_ctl: float
    :param discount: Discount factor.
    :type discount: float
    :param gae_lambda: Lambda factor in GAE.
    :type gae_lambda: float
    :param eps_clip: PPO actor probability ratio clipping factor.
    :type eps_clip: float
    :param value_eps_clip: PPO value clipping factor.
    :type value_eps_clip: float
    :param max_reward_clip: Maximum reward value.
    :type max_reward_clip: float
    :param reward_output_scaling: Scaling factor of the reward model output.
    :type reward_output_scaling: float
    :param reward_output_bias: Bias of the reward model output.
        The number outputed by the reward model will be
        CLIP((x - bias) * scaling, -max_reward_clip, max_reward_clip).
    :type reward_output_bias: float
    :param early_stop_imp_ratio: PPO update will be early stopped if importance ratio
        exceeds this maximum value.
    :type early_stop_imp_ratio: float
    :param use_adaptive_kl_ctl: Whether to use adaptive KL divergence coefficient.
    :type use_adaptive_kl_ctl: bool
    :param adv_norm: Whether to use advantage normalization.
    :type adv_norm: bool
    :param value_norm: Whether to denormalize valued and normalize return predictions.
    :type value_norm: bool
    :param value_norm_type: Type of value normalization.
        Either exponential moving average ("exp") or moving average ("ma").
    :type value_norm_type: str
    :param value_norm_beta: Exponential decay factor
        in exponential moving average.
    :type value_norm_beta: float
    :param value_norm_eps: Epsilon factor in the
        denominator of exponential moving average.
    :type value_norm_eps: float
    :param disable_value: A shortcut option to disable the critic model.
    :type disable_value: bool
    """

    gen: GenerationHyperparameters = dataclasses.field(
        default_factory=GenerationHyperparameters
    )
    ppo_n_minibatches: int = 4
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 1.0
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    disable_value: bool = False
    max_reward_clip: float = 20.0
    reward_output_scaling: float = 1.0
    reward_output_bias: float = 0.0
    early_stop_imp_ratio: float = 5.0
    use_adaptive_kl_ctl: bool = False
    adv_norm: bool = True
    value_norm: bool = True
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp"
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5


@dataclasses.dataclass
class PPOCODEConfig(CommonExperimentConfig):
    """PPO experiment configuration.

    It is a subclass of :class:`CommonExperimentConfig`,
    so all CLI options in the base class are available.

    We don't implement runtime evaluation for PPO.

    We identify that the RLHF process is composed of four
    distinct models with independent parameters and six
    *model function calls* upon these models.

    The four models are\:

    - Actor\: The primary LLM that generates text.
    - Critic\: The value function that estimates the value of a state.
    - Ref\: The reference LLM that provides KL regularization.
    - Rew\: The reward model that provides reward signals.

    The four model function calls and their dependencies are\:

    - Rollout\: Generate text from the actor model.
    - InfReward\: Infer rewards from the reward model given generated text.
    - InfRef\: Infer log probabilities from the reference model given generated text.
    - InfValues\: Infer values from the critic model given generated text.
    - TrainActor\: Train the actor model given generated text, rewards, values, and reference log probabilities.
    - TrainCritic\: Train the critic model given generated text, rewards, values, and reference log probabilities.

    This class resolves these dependencies under the hood.
    What the users should specify are the runtime configurations
    of models and allocations of *each model function call*.

    :param actor: Runtime configuration of the primary LLM.
    :type actor: ModelTrainEvalConfig
    :param critic: Runtime configuration of the critic model of PPO.
    :type critic: ModelTrainEvalConfig
    :param ref: Runtime configuration of the reference LLM.
    :type ref: ModelTrainEvalConfig
    :param rew: Runtime configuration of the reward LLM.
    :type rew: ModelTrainEvalConfig
    :param actor_train: :class:`MFCConfig` for TrainActor.
    :type actor_train: MFCConfig
    :param critic_train: :class:`MFCConfig` for TrainCritic.
    :type critic_train: MFCConfig
    :param actor_gen: :class:`MFCConfig` for Rollout.
    :type actor_gen: MFCConfig
    :param critic_inf: :class:`MFCConfig` for InfValues.
    :type critic_inf: MFCConfig
    :param rew_inf: :class:`MFCConfig` for InfReward.
    :type rew_inf: MFCConfig
    :param ref_inf: :class:`MFCConfig` for InfRef.
    :type ref_inf: MFCConfig
    :param dataset: Dataset configuration.
    :type dataset: PromptOnlyDatasetConfig
    :param ppo: Configuration for the PPO algorithm.
    :type ppo: PPOHyperparameters
    :param group_size: The number of answers remained for each prompt.
    :type group_size: int
    :param generation_size: The number of answers sampled for each prompt.
        Among them, only `group_size` samples are remained according to
        the reward score, aka best-of-n sampling. If None, use `group_size`.
    :type generation_size: Optional[int]
    :param mask_no_eos_with_zero: Whether to mask out the reward if an
        answer is truncated due to exceeding the length limit.
    :type mask_no_eos_with_zero: bool
    :param mask_too_long: Whether to mask out the PPO loss if an
        answer is truncated due to exceeding the length limit.
    :type mask_too_long: bool
    :param check_verifier_status: If True, raise an error
        when the reward is all-zero. This usually indicates a bug
        of the verifier.
    :type check_verifier_status: bool
    :param group_adv_norm: Whther to use grouped advantage
        normaliztion in GRPO.
    :type group_adv_norm: bool
    """

    actor: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    critic: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    ref: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    rew: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)

    # for manual allocation only
    actor_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    critic_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    actor_gen: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    critic_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    rew_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    ref_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)

    dataset: PromptOnlyDatasetConfig = dataclasses.field(
        default_factory=PromptOnlyDatasetConfig
    )

    ppo: PPOHyperparameters = dataclasses.field(default_factory=PPOHyperparameters)

    group_size: int = 1
    generation_size: Optional[int] = None
    mask_no_eos_with_zero: bool = False
    rm_output_scaling: Optional[float] = None
    ref_ema_eta: Optional[float] = None
    group_adv_norm: bool = False
    mask_too_long: bool = False
    rw_type: Optional[str] = "sparse"
    task: str = "code"
    check_xml_format: bool = False
    use_dense_reward: bool = False
    reward_delta: bool = True

    check_verifier_status: bool = False

    dataset_filter_threshold: float = 100.0
    dataset_max_filter_percentage: float = 0.0

    def __post_init__(self):

        self.ppo_kwargs = dict(
            n_minibatches=self.ppo.ppo_n_minibatches,
            kl_ctl=self.ppo.kl_ctl,
            discount=self.ppo.discount,
            gae_lambda=self.ppo.gae_lambda,
            eps_clip=self.ppo.eps_clip,
            value_eps_clip=self.ppo.value_eps_clip,
            max_reward_clip=self.ppo.max_reward_clip,
            adaptive_kl_ctl=self.ppo.use_adaptive_kl_ctl,
            value_norm=self.ppo.value_norm,
            value_norm_type=self.ppo.value_norm_type,
            value_norm_beta=self.ppo.value_norm_beta,
            value_norm_eps=self.ppo.value_norm_eps,
            disable_value=self.ppo.disable_value,
            mask_no_eos_with_zero=self.mask_no_eos_with_zero,
        )

        if self.rm_output_scaling is None:
            self.rm_output_scaling = self.ppo.reward_output_scaling

    @property
    def models(self) -> Dict[str, ModelTrainEvalConfig]:
        # role to config
        if self.ppo.disable_value:
            return {
                "actor": self.actor,
                # "critic": self.critic,
                "ref": self.ref,
                "reward": self.rew,
            }
        else:
            return {
                "actor": self.actor,
                "critic": self.critic,
                "ref": self.ref,
                "reward": self.rew,
            }

    @property
    def rpcs(self):
        if (
            self.dataset.max_prompt_len + self.ppo.gen.max_new_tokens
            > self.actor.vllm.max_seq_len_to_capture
        ):
            raise RuntimeError(
                f"vllm max seq len to capture {self.actor.vllm.max_seq_len_to_capture} is "
                f"smaller than the prompt length + generation length {self.dataset.max_prompt_len + self.ppo.gen.max_new_tokens}"
            )
        if not os.path.exists(os.getenv("REAL_CODE_METADATA_PATH")):
            raise RuntimeError(
                "Dataset json path REAL_CODE_METADATA_PATH does not exist."
            )

        domain = os.getenv("FUNCTIONCALL_SERVICE_DOMAIN", "")
        if not (domain.startswith("http://") and ":" in domain):
            raise RuntimeError(
                "function call address FUNCTIONCALL_SERVICE_DOMAIN is invalid."
            )

        

        # interfaces
        actor_interface = ModelInterfaceAbstraction(
            "ppo_actor",
            args={
                **copy.deepcopy(self.ppo_kwargs),
                # NOTE: to_container converts the object to a dict
                # It is used for unifying the profiling API, which requires to
                # pass external interface configurations in the launch command.
                # Customized dataclass objects will not work in that case.
                "generation_config": (
                    OmegaConf.to_container(self.ppo.gen, resolve=True)
                    if isinstance(self.ppo.gen, (OmegaConf, DictConfig))
                    else dataclasses.asdict(self.ppo.gen)
                ),
                "early_stop_imp_ratio": self.ppo.early_stop_imp_ratio,
                "adv_norm": self.ppo.adv_norm,
                "group_size": self.group_size,
                "generation_size": self.generation_size,
                "group_adv_norm": self.group_adv_norm,
                "mask_too_long": self.mask_too_long,
                "use_dense_reward": self.use_dense_reward,
                "reward_delta": self.reward_delta,
            },
        )
        ref_interface = copy.deepcopy(actor_interface)
        ref_interface.args["enable_save"] = False

        critic_interface = ModelInterfaceAbstraction(
            "ppo_critic",
            args={
                **copy.deepcopy(self.ppo_kwargs),
                "group_size": self.group_size,
                "mask_too_long": self.mask_too_long,
                "use_dense_reward": self.use_dense_reward,
                "reward_delta": self.reward_delta,
            },
        )
        critic_interface.args.pop("eps_clip")
        rw_interface = ModelInterfaceAbstraction(
            "rw_code",
            args=dict(
                rw_type=self.rw_type,
                task=self.task,
                check_xml_format=self.check_xml_format,
                tokenizer_path=self.actor.path,
                enable_save=False,
                output_scaling=self.ppo.reward_output_scaling,
                rm_output_scaling=self.rm_output_scaling,
                output_bias=self.ppo.reward_output_bias,
                group_size=self.group_size,
                check_verifier_status=self.check_verifier_status,
                max_sync_length=self.ppo.gen.max_new_tokens
                + self.dataset.max_prompt_len
                + 128,
            ),
        )
        rollout = MFCDef(
            name="actor_gen",
            model_name="actor",
            mb_spec=self.actor_gen.mb_spec,
            interface_type=ModelInterfaceType.GENERATE,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_interface,
            input_keys=["packed_prompts"],
            output_keys=[
                "seq_no_eos_mask",
                "packed_input_ids",
                "packed_logprobs",
                "prompt_mask",
            ],
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_reward = MFCDef(
            name="rew_inf",
            model_name="reward",
            mb_spec=self.rew_inf.mb_spec,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=rw_interface,
            model_type=self.rew.type,
            model_path=self.rew.path,
            min_n_seqs_per_pass=1 / self.group_size,
            input_keys=["packed_input_ids", "packed_prompts"],
            output_keys=["rewards", "dense_rewards"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_ref_inputs = ["packed_input_ids"]
        inf_ref_logits = MFCDef(
            name="ref_inf",
            model_name="ref",
            mb_spec=self.ref_inf.mb_spec,
            interface_type=ModelInterfaceType.INFERENCE,
            model_type=self.ref.type,
            model_path=self.ref.path,
            interface_impl=ref_interface,
            min_n_seqs_per_pass=1 / self.group_size,
            input_keys=inf_ref_inputs,
            output_keys=["packed_ref_logprobs"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_values = MFCDef(
            name="critic_inf",
            model_name="critic",
            mb_spec=self.critic_inf.mb_spec,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=critic_interface,
            model_type=self.critic.type,
            model_path=self.critic.path,
            min_n_seqs_per_pass=1 / self.group_size,
            input_keys=["packed_input_ids", "seq_no_eos_mask"],
            output_keys=["values"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        train_actor_inputs = [
            "packed_input_ids",
            "packed_logprobs",
            "packed_ref_logprobs",
            "rewards",
            "dense_rewards",
            "values",
            "prompt_mask",
            "seq_no_eos_mask",
        ]
        if self.ppo.disable_value:
            train_actor_inputs.remove("values")
        train_actor = MFCDef(
            name="actor_train",
            model_name="actor",
            mb_spec=self.actor_train.mb_spec,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_interface,
            input_keys=train_actor_inputs,
            log_return_value=True,
            min_n_seqs_per_pass=self.ppo.ppo_n_minibatches / self.group_size,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        train_critic = MFCDef(
            name="critic_train",
            model_name="critic",
            mb_spec=self.critic_train.mb_spec,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=critic_interface,
            model_type=self.critic.type,
            model_path=self.critic.path,
            input_keys=[
                "packed_input_ids",
                "packed_logprobs",
                "packed_ref_logprobs",
                "rewards",
                "dense_rewards",
                "values",
                "prompt_mask",
                "seq_no_eos_mask",
            ],
            log_return_value=True,
            min_n_seqs_per_pass=self.ppo.ppo_n_minibatches / self.group_size,
            n_seqs=self.dataset.train_bs_n_seqs,
        )
        if self.ppo.disable_value:
            return {
                "actor_gen": rollout,
                "actor_train": train_actor,
                # "critic_inf": inf_values,
                # "critic_train": train_critic,
                "ref_inf": inf_ref_logits,
                "rew_inf": inf_reward,
            }
        else:
            return {
                "actor_gen": rollout,
                "actor_train": train_actor,
                "critic_inf": inf_values,
                "critic_train": train_critic,
                "ref_inf": inf_ref_logits,
                "rew_inf": inf_reward,
            }

    @property
    def allocations(self):
        if self.ppo.disable_value:
            return {
                "actor_gen": self.actor_gen,
                "actor_train": self.actor_train,
                # "critic_inf": self.critic_inf,
                # "critic_train": self.critic_train,
                "ref_inf": self.ref_inf,
                "rew_inf": self.rew_inf,
            }
        else:
            return {
                "actor_gen": self.actor_gen,
                "actor_train": self.actor_train,
                "critic_inf": self.critic_inf,
                "critic_train": self.critic_train,
                "ref_inf": self.ref_inf,
                "rew_inf": self.rew_inf,
            }

    @property
    def datasets(self):
        return [
            DatasetAbstraction(
                "code_prompt",
                args=dict(
                    dataset_path=self.dataset.path,
                    max_length=self.dataset.max_prompt_len,
                    fill_to_max_length=self.dataset.fill_to_max_length,
                    filter_threshold=self.dataset_filter_threshold,
                    max_filter_percentage=self.dataset_max_filter_percentage,
                ),
            )
        ]

    @property
    def tokenizer_name_or_path(self) -> str:
        return self.actor.path

    @property
    def search_kwargs(self):
        return {
            "num_gen_tokens": self.ppo.gen.max_new_tokens,
            "n_ppo_minibatches": self.ppo.ppo_n_minibatches,
            "seq_len": self.dataset.max_prompt_len,
        }

    @property
    def max_prompt_len(self):
        return self.dataset.max_prompt_len

    def initial_setup(self) -> ExperimentConfig:
        rpc_allocs = self._get_rpc_allocations()

        resolve_replica_ids(rpc_allocs, self.models)
        resolve_rpc_hooks(
            rpc_allocs, self.models
        )  # inplace modify MFCDefs in rpc allocations

        pprint.pprint(rpc_allocs)

        ######### update ref model using ema, ref_ema_eta = 0 means fixed ref model #########
        def _find_rpc(name):
            return next(alloc.rpc for alloc in rpc_allocs if alloc.rpc.name == name)

        # Remove the offload hook of ref_inf, because
        # we need to receive parameters from peer GPUs and update it immediately.
        if self.ref_ema_eta is not None:

            ref_inf = _find_rpc("ref_inf")
            ref_inf._post_hooks = []

            # Add an unidirectional parameter reallocation hook.
            actor_train = _find_rpc("actor_train")
            actor_train.add_post_hook(
                ParamReallocHook(
                    target=ref_inf.model_name,
                    eta=self.ref_ema_eta,
                )
            )
        ######### The main difference from normal PPO #########

        model_worker = self._get_model_worker_configs(rpc_allocs)

        return ExperimentConfig(
            exp_ctrl=self.exp_ctrl,
            wandb=self.wandb,
            model_rpcs=[rpc_alloc.rpc for rpc_alloc in rpc_allocs],
            model_worker=model_worker,
        )


register_quickstart_exp("ppo-code", PPOCODEConfig)
