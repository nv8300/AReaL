# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import collections
import dataclasses
import functools
import itertools
import time
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.distributed as dist

import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
import realhf.base.logging as logging
import realhf.impl.model.utils.ppo_functional as ppo_functional
from realhf.api.core.data_api import MicroBatchSpec, SequenceSample
from realhf.base.datapack import flat2d
from realhf.impl.model.interface.math_parser import parse_lines_in_parallel
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_generate import concat_prompt_to_generation_output
from realhf.impl.model.utils.functional import (
    gather_packed_shifted_log_probs,
    masked_normalization,
)

logger = logging.getLogger("PackedPPOInterface")


def get_score(prompt_ids, generated, query_ids, tokenizer):
    prompt_strs = tokenizer.batch_decode(
        prompt_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
    )
    seq_strs = tokenizer.batch_decode(
        generated, clean_up_tokenization_spaces=False, skip_special_tokens=True
    )
    query_id_strs = [query_id.split("@")[0] for query_id in query_ids]
    return parse_lines_in_parallel(seq_strs, query_id_strs)


def topk(scores, gen_lengths, k) -> list:
    indexed = list(enumerate(zip(scores, gen_lengths)))

    sorted_indices = sorted(indexed, key=lambda x: (x[1][0], x[1][1]), reverse=True)[:k]

    return [idx for idx, _ in sorted_indices]


def _ppo_actor_loss_from_model_outputs(
    logits: torch.FloatTensor,  # [tot_seqlen, vocab_size]
    input_: SequenceSample,
    kl_adapter: ppo_functional.KLController,  # const
    eps_clip: float,  # const
    early_stop_imp_ratio: Optional[float],  # const
    early_stop_kl: Optional[float],  # const
    temperature: Optional[float] = 1,
) -> Tuple[torch.FloatTensor, Dict]:
    """Loss function for ppo actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    packed_input_ids = input_.data["packed_input_ids"]
    cu_seqlens = (
        torch.nn.functional.pad(
            torch.tensor(flat2d(input_.seqlens["packed_input_ids"])).cumsum(0),
            (1, 0),
        )
        .int()
        .to(logits.device)
    )
    ppo_loss_mask = input_.data["ppo_loss_mask"]
    advantages = input_.data["advantages"]
    old_logp = input_.data["old_logp"]
    kl_rewards = input_.data["kl_rewards"]

    if temperature is not None:
        logits /= temperature
    n_tokens = ppo_loss_mask.count_nonzero()
    logprobs = gather_packed_shifted_log_probs(
        logits, cu_seqlens, packed_input_ids
    ).float()
    loss, ppo_stat = ppo_functional.actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        loss_mask=ppo_loss_mask,
    )

    importance_weight = ppo_stat["importance_weight"].float() * n_tokens
    clip_ratio = ppo_stat["clip_ratio"].float() * n_tokens
    approx_kl = ppo_stat["approx_kl"].float() * n_tokens

    # Logging and early stopping according to KL (logp vs ref) or importance ratio (new logp vs old logp).
    mean_ref_kl = (kl_rewards.detach().float() * ppo_loss_mask).sum()
    logging_loss = torch.where(ppo_loss_mask, loss.detach().float(), 0.0).sum()
    dist.all_reduce(n_tokens, group=constants.data_parallel_group())
    dist.all_reduce(mean_ref_kl, group=constants.data_parallel_group())
    dist.all_reduce(importance_weight, group=constants.data_parallel_group())
    dist.all_reduce(clip_ratio, group=constants.data_parallel_group())
    dist.all_reduce(approx_kl, group=constants.data_parallel_group())
    dist.all_reduce(logging_loss, group=constants.data_parallel_group())

    # Early stopping.
    kl_adapter.update(mean_ref_kl / n_tokens, n_steps=cu_seqlens.shape[0] - 1)
    _imp = importance_weight / n_tokens
    _kl = approx_kl / n_tokens
    if early_stop_imp_ratio is not None and _imp > early_stop_imp_ratio:
        logger.warning(
            f"Current importance ratio {_imp.item():.4f} is larger "
            f"than early stop threshold {early_stop_imp_ratio}. Abandon this minibatch."
        )
        loss = loss * 0.0
    if early_stop_kl is not None and _kl > early_stop_kl:
        logger.warning(
            f"Current approximate KL divergence {_kl.item():.4f} is larger "
            f"than early stop threshold {early_stop_kl}. Abort actor update."
        )
        loss = loss * 0.0

    stats = dict(
        ppo_approx_kl=approx_kl,
        actor_loss=logging_loss,
        actor_clip_ratio=clip_ratio,
        importance_weight=importance_weight,
    )

    return loss, stats


def splited_sum_bool_tensor(t: torch.BoolTensor, chunk_size=256 * 1024 * 1024) -> int:
    """Sum a boolean tensor by splitting them into chunks and sum the chunks
    separately.

    to avoid memory overhead introduced by torch default sum method
    (which will apply for a block of memory of size `8 * t.numel()`
    bytes.)
    """
    flatten = t.flatten()
    splitted = flatten.split(chunk_size // 8, dim=0)
    r = 0
    for chunk in splitted:
        r += chunk.sum()
    return r


@dataclasses.dataclass
class PPOActorInterface(model_api.ModelInterface):
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

    @torch.no_grad()
    def generate(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> SequenceSample:
        module = model.module

        module.eval()

        # Remap the key `packed_prompts` to `packed_input_ids`,
        # because the pipe runner only recognizes `packed_input_ids`.
        # x = SequenceSample.from_default(
        #     ids=input_.ids,
        #     seqlens=input_.seqlens["packed_prompts"],
        #     data=dict(packed_input_ids=input_.data["packed_prompts"]),
        # )

        packed_input_ids = input_.data["packed_prompts"]
        new_input_ids = []
        offset = 0
        for x in input_.seqlens["packed_prompts"]:
            new_input_ids += [
                packed_input_ids[offset : offset + x[0]]
            ] * self.generation_size
            offset += x[0]
        assert offset == sum(x[0] for x in input_.seqlens["packed_prompts"])

        if model.backend_name != "vllm":
            # Replicate prompts
            grouped_input = SequenceSample.from_default(
                ids=list(range(input_.bs * self.generation_size)),
                seqlens=[
                    x[0]
                    for x in input_.seqlens["packed_prompts"]
                    for _ in range(self.generation_size)
                ],
                data=dict(packed_input_ids=torch.cat(new_input_ids)),
            )
        else:
            grouped_input = SequenceSample(
                ids=input_.ids,
                seqlens=dict(packed_input_ids=input_.seqlens["packed_prompts"]),
                keys=["packed_input_ids"],
                dtypes=dict(packed_input_ids=torch.long),
                trailing_shapes=dict(packed_input_ids=()),
                data=dict(packed_input_ids=input_.data["packed_prompts"]),
            )

        res = module.generate(
            input_=grouped_input,
            tokenizer=model.tokenizer,
            gconfig=self.gconfig,
            mb_spec=mb_spec,
        )
        if res is None:
            return None

        gen_tokens, logprobs, _ = res

        pad_token_id = model.tokenizer.pad_token_id
        eos_token_id = model.tokenizer.eos_token_id
        seq_no_eos_mask = (gen_tokens[:, -1] != eos_token_id).logical_and(
            gen_tokens[:, -1] != pad_token_id
        )
        # We also want gen_lengths to include the eos token, where the reward model outputs a score for this sequence.
        gen_lengths = (gen_tokens != pad_token_id).logical_and(
            gen_tokens != eos_token_id
        ).sum(dim=-1) + 1
        gen_lengths = gen_lengths.clip(max=gen_tokens.shape[-1])
        input_seq_lens = [
            x for x in input_.seqlens["packed_prompts"] for _ in range(self.group_size)
        ]
        input_token_ids = torch.cat(new_input_ids)

        if self.generation_size is not None and self.generation_size > self.group_size:

            # best of k
            query_ids = [
                query_id for query_id in input_.ids for _ in range(self.generation_size)
            ]
            scores = get_score(new_input_ids, gen_tokens, query_ids, model.tokenizer)
            input_ids_topk, gen_tokens_topk, logprobs_topk, gen_lengths_topk = (
                [],
                [],
                [],
                [],
            )
            for data_idx in range(0, len(gen_tokens), self.generation_size):
                topk_idx = topk(
                    scores[data_idx : data_idx + self.generation_size],
                    gen_lengths[data_idx : data_idx + self.generation_size],
                    self.group_size,
                )
                topk_idx = [data_idx + x for x in topk_idx]
                gen_tokens_topk += gen_tokens[topk_idx]
                logprobs_topk += logprobs[topk_idx]
                gen_lengths_topk += gen_lengths[topk_idx]
                input_ids_topk += [new_input_ids[x] for x in topk_idx]

            input_token_ids = torch.cat(input_ids_topk)

            gen_tokens = torch.stack(gen_tokens_topk)
            logprobs = torch.stack(logprobs_topk)
            gen_lengths = torch.stack(gen_lengths_topk)
            seq_no_eos_mask = (gen_tokens[:, -1] != eos_token_id).logical_and(
                gen_tokens[:, -1] != pad_token_id
            )

        (
            packed_input_ids,
            packed_logprobs,
            _,
            seq_lengths,
            prompt_mask,
        ) = concat_prompt_to_generation_output(
            packed_prompts=input_token_ids,
            prompt_lengths=torch.tensor(flat2d(input_seq_lens)).to(model.device),
            gen_tokens=gen_tokens,
            logprobs=logprobs,
            logits_mask=None,
            gen_lengths=gen_lengths,
        )

        # Partition generated data into groups.
        seqlens = [
            seq_lengths[i * self.group_size : (i + 1) * self.group_size]
            .cpu()
            .numpy()
            .tolist()
            for i in range(input_.bs)
        ]

        data = dict(
            seq_no_eos_mask=seq_no_eos_mask,
            packed_input_ids=packed_input_ids,
            packed_logprobs=packed_logprobs,
            prompt_mask=prompt_mask,
        )

        res = SequenceSample(
            keys=[
                "packed_input_ids",
                "prompt_mask",
                "packed_logprobs",
                "seq_no_eos_mask",
            ],
            trailing_shapes=dict(
                packed_input_ids=(),
                prompt_mask=(),
                packed_logprobs=(),
                seq_no_eos_mask=(),
            ),
            dtypes=dict(
                packed_input_ids=torch.long,
                prompt_mask=torch.bool,
                packed_logprobs=torch.float,
                seq_no_eos_mask=torch.bool,
            ),
            seqlens=dict(
                packed_input_ids=seqlens,
                packed_logprobs=[[x - 1 for x in slens] for slens in seqlens],
                prompt_mask=seqlens,
                seq_no_eos_mask=[[1] * self.group_size for _ in seqlens],
            ),
            data=data,
            ids=input_.ids,
            prompt_mask=prompt_mask,
        )

        return res

    @torch.no_grad()
    def inference(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> SequenceSample:
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
            dtypes=dict(packed_ref_logprobs=torch.float16),
            trailing_shapes=dict(packed_ref_logprobs=()),
            data=dict(packed_ref_logprobs=logprobs),
            seqlens=dict(
                packed_ref_logprobs=[
                    [x - 1 for x in slen] for slen in input_.seqlens["packed_input_ids"]
                ]
            ),
        )

        return res

    def train_step(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> Dict:
        module = model.module
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()

        old_logp: torch.FloatTensor = input_.data["packed_logprobs"].float()
        ref_logp: torch.FloatTensor = input_.data["packed_ref_logprobs"].float()
        prompt_mask = input_.data["prompt_mask"]
        input_lens = torch.tensor(
            flat2d(input_.seqlens["packed_input_ids"]), device=model.device
        )
        cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
        reward_score = input_.data["rewards"].float()
        if "dense_rewards" in input_.data:
            dense_reward_score = input_.data["dense_rewards"].float()
        if not self.disable_value:
            values = input_.data["values"].float()
        else:
            values = torch.zeros_like(
                input_.data["packed_input_ids"], dtype=torch.float32
            )
        seq_no_eos_mask = input_.data["seq_no_eos_mask"]

        if not self.disable_value:
            if self.value_norm:
                denormalized_values = self.rms.denormalize(values)
            else:
                denormalized_values = values
        else:
            denormalized_values = values

        for i in range(seq_no_eos_mask.shape[0]):
            if not seq_no_eos_mask[i]:
                # Set value at the EOS token to be zero.
                denormalized_values[cu_seqlens[i + 1] - 1] = 0.0
                values[cu_seqlens[i + 1] - 1] = 0.0

        # Shift the loss mask by one token for each packed sequences.
        short1cu_seqlens = cu_seqlens.clone()
        short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
        loss_mask = prompt_mask.logical_not()

        if self.mask_too_long:
            for i in range(seq_no_eos_mask.shape[0]):
                if seq_no_eos_mask[i]:
                    loss_mask[cu_seqlens[i] : cu_seqlens[i + 1]] = False

        shift_one_indices = torch.cat(
            [
                torch.arange(
                    cu_seqlens[i] + 1,
                    cu_seqlens[i + 1],
                    dtype=torch.long,
                    device=cu_seqlens.device,
                )
                for i in range(cu_seqlens.shape[0] - 1)
            ]
        )
        loss_mask = loss_mask[shift_one_indices]

        # Apply the mask to log probabilities.
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute rewards and GAEs.
        if self.use_dense_reward:
            kl_rewards, rewards = ppo_functional.get_packed_reward_dense(
                kl_ctl=self.kl_adapter.value,
                clip_reward_value=self.max_reward_clip,
                log_probs=old_logp,
                ref_log_probs=ref_logp,
                dense_reward_score=dense_reward_score,
                short1cu_seqlens=short1cu_seqlens,
                seq_no_eos_mask=seq_no_eos_mask,
                reward_delta=self.reward_delta,
            )
        else:
            kl_rewards, rewards = ppo_functional.get_packed_rewards(
                kl_ctl=self.kl_adapter.value,
                clip_reward_value=self.max_reward_clip,
                log_probs=old_logp,
                ref_log_probs=ref_logp,
                reward_score=(reward_score),
                short1cu_seqlens=short1cu_seqlens,
                seq_no_eos_mask=seq_no_eos_mask,
                mask_no_eos_with_zero=self.mask_no_eos_with_zero,
            )
        advantages, returns = ppo_functional.get_packed_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=(
                denormalized_values
                if not self.disable_value
                else denormalized_values.new_zeros(denormalized_values.shape)
            ),
            rewards=rewards,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        # Optionally perform normalization.
        if self.value_norm:
            self.rms.update(returns, mask=loss_mask)
        if self.adv_norm:
            if self.group_adv_norm == False:
                advantages = masked_normalization(advantages, loss_mask)
            else:
                logger.info(f"adv_shape: {advantages.shape}")
                logger.info(f"prompt_mask_shape: {prompt_mask.shape}")
                n_samples = len(cu_seqlens) - 1
                assert n_samples % self.group_size == 0
                adv_list = []
                for i in range(0, n_samples, self.group_size):
                    for j in range(1, self.group_size):
                        assert (
                            prompt_mask[cu_seqlens[i] : cu_seqlens[i + 1]].sum()
                            == prompt_mask[
                                cu_seqlens[i + j] : cu_seqlens[i + j + 1]
                            ].sum()
                        )
                    adv_list.append(
                        masked_normalization(
                            advantages[
                                short1cu_seqlens[i] : short1cu_seqlens[
                                    i + self.group_size
                                ]
                            ],
                            loss_mask[
                                short1cu_seqlens[i] : short1cu_seqlens[
                                    i + self.group_size
                                ]
                            ],
                            all_reduce=False,
                        )
                    )

                advantages = torch.cat(adv_list, 0)

        # Prepare data to be splitted into mini-batches.
        input_ = SequenceSample.from_default(
            ids=list(range(input_.bs * self.group_size)),
            data=dict(
                advantages=advantages,
                old_logp=old_logp,
                ppo_loss_mask=loss_mask,
                packed_input_ids=input_.data["packed_input_ids"],
                kl_rewards=kl_rewards,
            ),
            seqlens=[int(x) for x in input_lens.cpu().numpy().tolist()],
        )
        # NOTE: We cannot randomly shuffle data here because
        # data must have the same shape across different pipeline stages.
        datas, *_ = input_.split(MicroBatchSpec(n_mbs=self.n_minibatches))
        logger.info(
            f"PPO minibatch split (size {self.n_minibatches}): "
            f"#seqs: {[s.bs for s in datas]}, "
            f"#tokens: {[sum([sum(lens) for lens in s.seqlens[s._get_split_key()]]) for s in datas]}"
        )

        if self.use_dense_reward:
            dense_reward_score = dense_reward_score[shift_one_indices]
        ### Logging code starts. ###
        _n_seqs = torch.tensor(
            [reward_score.shape[0]], dtype=torch.float32, device=model.device
        )
        no_eos_ratios = seq_no_eos_mask.sum()
        # _n_tokens = loss_mask.count_nonzero()
        _n_tokens = prompt_mask.logical_not().count_nonzero()
        _n_valid_tokens = loss_mask.count_nonzero()
        task_reward = reward_score.sum()
        if self.use_dense_reward:
            dense_reward = (dense_reward_score * loss_mask).sum()
        final_reward = (rewards * loss_mask).sum()
        _advantages = advantages.sum()
        _kl_rewards = (kl_rewards * loss_mask).sum()
        prompt_len = prompt_mask.count_nonzero().float()
        seq_len = input_lens.float().sum()

        dist.all_reduce(_n_seqs, group=constants.data_parallel_group())
        dist.all_reduce(no_eos_ratios, group=constants.data_parallel_group())
        dist.all_reduce(task_reward, group=constants.data_parallel_group())
        if self.use_dense_reward:
            dist.all_reduce(dense_reward, group=constants.data_parallel_group())
        dist.all_reduce(final_reward, group=constants.data_parallel_group())
        dist.all_reduce(_advantages, group=constants.data_parallel_group())
        dist.all_reduce(prompt_len, group=constants.data_parallel_group())
        dist.all_reduce(seq_len, group=constants.data_parallel_group())
        dist.all_reduce(_n_tokens, group=constants.data_parallel_group())
        dist.all_reduce(_n_valid_tokens, group=constants.data_parallel_group())
        dist.all_reduce(_kl_rewards, group=constants.data_parallel_group())

        global_stats = dict(
            task_reward=float(task_reward / _n_seqs),
            kl_reward=float(_kl_rewards / _n_tokens),
            final_reward=float(final_reward / _n_seqs),
            advantage=float(_advantages / _n_tokens),
            avg_seq_len=float(seq_len / _n_seqs),
            avg_prompt_len=float(prompt_len / _n_seqs),
            n_tokens=int(_n_tokens),
            n_valid_tokens=int(_n_valid_tokens),
            n_seqs=int(_n_seqs),
            no_eos_ratio=float(no_eos_ratios / _n_seqs),
            disable_value=int(self.disable_value),
            mask_no_eos_with_zero=int(self.mask_no_eos_with_zero),
        )
        if self.use_dense_reward:
            dense_reward = (float(dense_reward / _n_seqs),)

        ### Logging code ends. ###

        # Run mini-batched PPO training!
        train_stats = collections.defaultdict(lambda: 0)

        for data in datas:
            stats = module.train_batch(
                input_=data,
                mb_spec=mb_spec,
                version_steps=model.version.global_step,
                loss_fn=functools.partial(
                    _ppo_actor_loss_from_model_outputs,
                    kl_adapter=self.kl_adapter,
                    eps_clip=self.eps_clip,
                    early_stop_imp_ratio=self.early_stop_imp_ratio,
                    early_stop_kl=self.early_stop_kl,
                    temperature=self.gconfig.temperature,
                ),
                loss_weight_fn=lambda x: x.data["ppo_loss_mask"].count_nonzero(),
                token_normalize_scope=self.token_normalize_scope,
            )

            if stats:
                for k, v in stats.items():
                    train_stats[k] += v
        cur_epoch = model.version.epoch
        model.inc_version()

        # FIXME: It only logs the MoE aux loss of the final PPO mini-batch.
        global_stats.update(
            constants.log_global_stats_tracker(
                return_dict=True, clear_stats_after_logging=True
            )
        )
        if train_stats:
            train_stats = dict(
                ppo_approx_kl=float(train_stats["ppo_approx_kl"] / _n_tokens),
                actor_loss=float(train_stats["actor_loss"] / _n_tokens),
                actor_clip_ratio=float(train_stats["actor_clip_ratio"] / _n_tokens),
                importance_weight=float(train_stats["importance_weight"] / _n_tokens),
            )
            train_stats = dict(**train_stats, **global_stats)

        return dict(train_stats)

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

    # Mock methods for profiling only.
    def _mock_train_step(
        self,
        model: model_api.Model,
        dataset_input: SequenceSample,
    ) -> Dict:
        prompt_lens = flat2d(dataset_input.seqlens["packed_prompts"])
        bs = len(prompt_lens)
        seqlens = [x + self.gconfig.max_new_tokens for x in prompt_lens]
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        mconfig = module.config
        mdtype = module.dtype
        short1_seqlens = [x - 1 for x in seqlens]

        packed_logprobs = torch.randn(
            (sum(short1_seqlens),), dtype=mdtype, device=model.device
        )
        packed_ref_logprobs = torch.randn_like(packed_logprobs)
        prompt_mask = torch.zeros(
            (sum(seqlens),), dtype=torch.bool, device=model.device
        )
        packed_input_ids = torch.randint(
            0,
            mconfig.vocab_size,
            (sum(seqlens),),
            dtype=torch.long,
            device=model.device,
        )
        rewards = torch.randn(bs, dtype=mdtype, device=model.device)
        seq_no_eos_mask = torch.randint(
            0, 2, (bs,), dtype=torch.bool, device=model.device
        )
        values = torch.randn(
            (sum(seqlens),),
            dtype=mdtype,
            device=model.device,
        )

        return SequenceSample.from_default(
            seqlens=seqlens,
            ids=dataset_input.ids,
            data=dict(
                packed_logprobs=packed_logprobs,
                packed_ref_logprobs=packed_ref_logprobs,
                prompt_mask=prompt_mask,
                packed_input_ids=packed_input_ids,
                rewards=rewards,
                seq_no_eos_mask=seq_no_eos_mask,
                values=values,
            ),
        )


def _ppo_critic_loss_from_model_outputs(
    new_values: torch.FloatTensor,
    input_: SequenceSample,
    value_eps_clip: float,
    kl_adapter: ppo_functional.KLController,
    rms=None,
) -> Tuple[torch.FloatTensor, Dict]:

    cu_seqlens = (
        torch.nn.functional.pad(
            torch.tensor(flat2d(input_.seqlens["packed_input_ids"])).cumsum(0),
            (1, 0),
        )
        .int()
        .to(new_values.device)
    )
    ppo_loss_mask = input_.data["ppo_loss_mask"]
    returns = input_.data["returns"]
    values = input_.data["values"]
    kl_rewards = input_.data["kl_rewards"]

    leave_one_indices = torch.cat(
        [
            torch.arange(
                cu_seqlens[i],
                cu_seqlens[i + 1] - 1,
                dtype=torch.long,
                device=cu_seqlens.device,
            )
            for i in range(cu_seqlens.shape[0] - 1)
        ]
    )
    new_values = new_values[leave_one_indices].view(-1).float()
    values = values[leave_one_indices].view(-1).float()

    loss, loss_stat = ppo_functional.critic_loss_fn(
        value=new_values,
        old_value=values,
        target_value=returns,
        value_eps_clip=value_eps_clip,
        loss_mask=ppo_loss_mask,
    )

    if rms is not None:
        denormalized_values = rms.denormalize(new_values)
    else:
        denormalized_values = new_values

    # Logging.
    n_tokens = ppo_loss_mask.count_nonzero()
    mean_ref_kl = (kl_rewards.detach().float() * ppo_loss_mask).sum()
    logging_loss = loss.detach().float() * n_tokens
    clip_ratio = loss_stat["clip_ratio"].float() * n_tokens
    denormalized_values = (
        torch.where(ppo_loss_mask, denormalized_values, 0.0).sum().detach().float()
    )
    dist.all_reduce(n_tokens, group=constants.data_parallel_group())
    dist.all_reduce(mean_ref_kl, group=constants.data_parallel_group())
    dist.all_reduce(logging_loss, group=constants.data_parallel_group())
    dist.all_reduce(clip_ratio, group=constants.data_parallel_group())
    dist.all_reduce(denormalized_values, group=constants.data_parallel_group())

    # Update KL coefficient to be consistent with actor.
    kl_adapter.update(mean_ref_kl, n_steps=cu_seqlens.shape[0] - 1)

    return loss, dict(
        value_loss=logging_loss,
        value_clip_ratio=clip_ratio,
        denormalized_values=denormalized_values,
    )


@dataclasses.dataclass
class PPOCriticInterface(model_api.ModelInterface):
    n_minibatches: int = 4
    enable_save: bool = True
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 0.95
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0
    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000
    value_norm: bool = False
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp"
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5
    disable_value: bool = False

    group_size: int = 1
    mask_no_eos_with_zero: bool = False
    mask_too_long: bool = False
    use_dense_reward: bool = False
    reward_delta: bool = True
    token_normalize_scope: Literal["global", "dp"] = "global"

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

    @torch.no_grad()
    def inference(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> SequenceSample:
        assert model.module.module.config.is_critic
        module = model.module
        module.eval()

        input_flattend = SequenceSample.from_default(
            ids=list(range(input_.bs * self.group_size)),
            seqlens=flat2d(input_.seqlens["packed_input_ids"]),
            data=dict(packed_input_ids=input_.data["packed_input_ids"]),
        )
        if self.disable_value:
            scores = torch.zeros_like(input_.data["packed_input_ids"]).to(torch.float16)
        else:
            scores = module.forward(input_=input_flattend, mb_spec=mb_spec)

        if scores is None:
            return None
        scores = scores.view(-1)
        # res = SequenceSample.from_default(
        #     ids=input_.ids,
        #     data=dict(values=scores),
        #     seqlens=input_.seqlens["packed_input_ids"],
        # )
        res = SequenceSample(
            keys=["values"],
            ids=input_.ids,
            dtypes=dict(values=torch.float16),
            trailing_shapes=dict(values=()),
            data=dict(values=scores),
            seqlens=dict(values=input_.seqlens["packed_input_ids"]),
        )

        return res

    def train_step(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> Dict:
        assert model.module.module.config.is_critic

        if self.disable_value:
            return dict()

        module = model.module
        tokenizer = model.tokenizer
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()

        old_logp: torch.FloatTensor = input_.data["packed_logprobs"].float()
        ref_logp: torch.FloatTensor = input_.data["packed_ref_logprobs"].float()
        prompt_mask = input_.data["prompt_mask"]
        input_lens = torch.tensor(
            flat2d(input_.seqlens["packed_input_ids"]), device=model.device
        )
        cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
        reward_score = input_.data["rewards"].float()
        if "dense_rewards" in input_.data:
            dense_reward_score = input_.data["dense_rewards"].float()
        values = input_.data["values"].float()
        seq_no_eos_mask = input_.data["seq_no_eos_mask"]

        if self.value_norm:
            denormalized_values = self.rms.denormalize(values)
        else:
            denormalized_values = values

        for i in range(seq_no_eos_mask.shape[0]):
            if not seq_no_eos_mask[i]:
                # Set value at the EOS token to be zero.
                denormalized_values[cu_seqlens[i + 1] - 1] = 0.0
                values[cu_seqlens[i + 1] - 1] = 0.0

        # Shift the loss mask by one token for each packed sequences.
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        short1cu_seqlens = cu_seqlens.clone()
        short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
        loss_mask = prompt_mask.logical_not()

        if self.mask_too_long:
            for i in range(seq_no_eos_mask.shape[0]):
                if seq_no_eos_mask[i]:
                    loss_mask[cu_seqlens[i] : cu_seqlens[i + 1]] = False

        shift_one_indices = torch.cat(
            [
                torch.arange(
                    cu_seqlens[i] + 1,
                    cu_seqlens[i + 1],
                    dtype=torch.long,
                    device=cu_seqlens.device,
                )
                for i in range(cu_seqlens.shape[0] - 1)
            ]
        )
        loss_mask = loss_mask[shift_one_indices]

        # Apply the mask to log probabilities.
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute rewards and GAEs.
        if self.use_dense_reward:
            kl_rewards, rewards = ppo_functional.get_packed_reward_dense(
                kl_ctl=self.kl_adapter.value,
                clip_reward_value=self.max_reward_clip,
                log_probs=old_logp,
                ref_log_probs=ref_logp,
                dense_reward_score=dense_reward_score,
                short1cu_seqlens=short1cu_seqlens,
                seq_no_eos_mask=seq_no_eos_mask,
                reward_delta=self.reward_delta,
            )
        else:
            kl_rewards, rewards = ppo_functional.get_packed_rewards(
                kl_ctl=self.kl_adapter.value,
                clip_reward_value=self.max_reward_clip,
                log_probs=old_logp,
                ref_log_probs=ref_logp,
                reward_score=(reward_score),
                short1cu_seqlens=short1cu_seqlens,
                seq_no_eos_mask=seq_no_eos_mask,
                mask_no_eos_with_zero=self.mask_no_eos_with_zero,
            )
        _, returns = ppo_functional.get_packed_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=denormalized_values,
            rewards=rewards,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        # Optionally perform normalization.
        if self.value_norm:
            self.rms.update(returns, mask=loss_mask)
            normalized_returns = self.rms.normalize(returns)
        else:
            normalized_returns = returns

        # Prepare data to be splitted into mini-batches.
        input_ = SequenceSample.from_default(
            ids=list(range(input_.bs * self.group_size)),
            data=dict(
                returns=normalized_returns,
                values=values,
                ppo_loss_mask=loss_mask,
                packed_input_ids=input_.data["packed_input_ids"],
                kl_rewards=kl_rewards,
            ),
            seqlens=[int(x) for x in input_lens.cpu().numpy().tolist()],
        )
        # NOTE: We cannot randomly shuffle data here because
        # data must have the same shape across different pipeline stages.
        datas, *_ = input_.split(MicroBatchSpec(n_mbs=self.n_minibatches))
        logger.info(
            f"PPO minibatch split (size {self.n_minibatches}): "
            f"#seqs: {[s.bs for s in datas]}, "
            f"#tokens: {[sum([sum(lens) for lens in s.seqlens[s._get_split_key()]]) for s in datas]}"
        )

        # Logging.
        returns = torch.where(loss_mask, returns, 0.0).sum()
        n_tokens = loss_mask.count_nonzero()
        dist.all_reduce(returns, group=constants.data_parallel_group())
        dist.all_reduce(n_tokens, group=constants.data_parallel_group())
        global_stats = dict(returns=float(returns / n_tokens), n_tokens=int(n_tokens))

        # Run mini-batched PPO training!
        train_stats = collections.defaultdict(lambda: 0)
        for data in datas:
            stats = module.train_batch(
                input_=data,
                mb_spec=mb_spec,
                version_steps=model.version.global_step,
                loss_fn=functools.partial(
                    _ppo_critic_loss_from_model_outputs,
                    value_eps_clip=self.value_eps_clip,
                    kl_adapter=self.kl_adapter,
                    rms=None if not self.value_norm else self.rms,
                ),
                loss_weight_fn=lambda x: x.data["ppo_loss_mask"].count_nonzero(),
                token_normalize_scope=self.token_normalize_scope,
            )

            if stats:
                for k, v in stats.items():
                    train_stats[k] += v

        cur_epoch = model.version.epoch
        model.inc_version()

        # FIXME: It only logs the MoE aux loss of the final PPO mini-batch.
        global_stats.update(
            constants.log_global_stats_tracker(
                return_dict=True, clear_stats_after_logging=True
            )
        )
        if train_stats:
            train_stats = dict(
                value_loss=float(train_stats["value_loss"] / n_tokens),
                value_clip_ratio=float(train_stats["value_clip_ratio"] / n_tokens),
                denormalized_values=float(
                    train_stats["denormalized_values"] / n_tokens
                ),
                **global_stats,
            )

        return dict(train_stats)

    # Mock methods for profiling only.
    def _mock_inference(
        self,
        model: model_api.Model,
        dataset_input: SequenceSample,
    ) -> SequenceSample:
        seqlens = flat2d(dataset_input.seqlens["packed_prompts"])
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

    # Mock methods for profiling only.
    def _mock_train_step(
        self,
        model: model_api.Model,
        dataset_input: SequenceSample,
    ) -> Dict:
        seqlens = flat2d(dataset_input.seqlens["packed_prompts"])
        bs = len(seqlens)
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        mconfig = module.config
        mdtype = module.dtype
        short1_seqlens = [x - 1 for x in seqlens]

        packed_logprobs = torch.randn(
            (sum(short1_seqlens),), dtype=mdtype, device=model.device
        )
        packed_ref_logprobs = torch.randn_like(packed_logprobs)
        prompt_mask = torch.zeros(
            (sum(seqlens),), dtype=torch.bool, device=model.device
        )
        packed_input_ids = torch.randint(
            0,
            mconfig.vocab_size,
            (sum(seqlens),),
            dtype=torch.long,
            device=model.device,
        )
        rewards = torch.randn(bs, dtype=mdtype, device=model.device)
        seq_no_eos_mask = torch.randint(
            0, 2, (bs,), dtype=torch.bool, device=model.device
        )
        values = torch.randn(
            (sum(seqlens),),
            dtype=mdtype,
            device=model.device,
        )

        return SequenceSample.from_default(
            seqlens=seqlens,
            ids=dataset_input.ids,
            data=dict(
                packed_logprobs=packed_logprobs,
                packed_ref_logprobs=packed_ref_logprobs,
                prompt_mask=prompt_mask,
                packed_input_ids=packed_input_ids,
                rewards=rewards,
                seq_no_eos_mask=seq_no_eos_mask,
                values=values,
            ),
        )


model_api.register_interface("ppo_actor", PPOActorInterface)
model_api.register_interface("ppo_critic", PPOCriticInterface)
