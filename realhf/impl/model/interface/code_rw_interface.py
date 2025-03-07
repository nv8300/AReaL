# Copyright 2025 Ant Group Inc.
import collections
import dataclasses
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from ast import parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import colorama
import numpy as np
import torch
import torch.distributed as dist

import realhf.api.core.model_api as model_api
import realhf.base.logging as logging
from functioncall.code.verify import code_verify
from realhf.api.core.data_api import SequenceSample, load_hf_tokenizer
from realhf.base import constants

logger = logging.getLogger("Packed Reward Modeling Interface", "benchmark")


class CodeVerifierException(Exception):
    pass


def extract_python_code(text, min_length=20, strict_syntax=True):
    code_pattern = r"(?i)```(?:python|py)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    valid_blocks = []
    for block in code_blocks:
        clean_block = block.strip()
        if len(clean_block) < min_length:
            continue

        # verify code syntax
        if strict_syntax:
            try:
                parse(clean_block, mode="exec")
            except (SyntaxError, IndentationError):
                continue

        valid_blocks.append(clean_block)

    if not valid_blocks:
        return None
    # return the last code block
    return valid_blocks[-1]


def check_with_elementtree(text):
    def escape_between_tags(text, tags=["think", "answer"]):
        """转义标签之间的内容，但保留标签本身."""
        # 构建标签模式
        tag_pattern = "|".join(tags)
        parts = []
        current_pos = 0

        # 匹配开始和结束标签
        pattern = f"</?({tag_pattern})[^>]*>"

        for match in re.finditer(pattern, text):
            # 添加标签之前的内容（需要转义）
            if current_pos < match.start():
                parts.append(html.escape(text[current_pos : match.start()]))

            # 添加标签本身（不转义）
            parts.append(match.group())
            current_pos = match.end()

        # 添加最后剩余的内容
        if current_pos < len(text):
            parts.append(html.escape(text[current_pos:]))

        return "".join(parts)

    text = escape_between_tags(text)
    if not text.strip().startswith("<think>"):
        text = "<think>" + text
    try:
        xml_text = f"<root>{text}</root>"
        x = ET.fromstring(xml_text)
        if x.text is not None and x.text.strip() != "":
            return False, f"Error: extra content before <think>. {x.text}"
        if len(x) != 2:
            return False, f"Error: there are {len(x)} tags."
        if x[0].tag is None or x[0].tag != "think":
            return False, f"Error: <think> tag is missing. {x[0].tag}"
        if x[0].tail is not None and x[0].tail.strip() != "":
            return (
                False,
                f"Error: extra content between <think> and <answer>. {x[0].tail}",
            )
        if x[1].tag is None or x[1].tag != "answer":
            return False, f"Error: <answer> tag is missing. {x[1].tag}"
        if x[1].tail is not None and x[1].tail.strip() != "":
            return False, f"Error: extra content after <answer>, {x[1].tail}"

        return True, x[1].text if x[1].text is not None else ""
    except ET.ParseError as e:
        return False, f"Error: XML格式错误, {str(e)}"


def retokenize(
    task,
    tokenizer,
    packed_input_ids,
    input_cu_seqlens,
    prompts,
    prompt_cu_seqlens,
    query_ids,
    check_xml_format=False,
    do_eval=False,
):
    input_ids = [
        packed_input_ids[start:end]
        for start, end in zip(input_cu_seqlens[:-1], input_cu_seqlens[1:])
    ]
    prompt_ids = [
        prompts[start:end]
        for start, end in zip(prompt_cu_seqlens[:-1], prompt_cu_seqlens[1:])
    ]
    seq_strs = tokenizer.batch_decode(
        input_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
    )
    prompt_strs = tokenizer.batch_decode(
        prompt_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
    )
    # query_id_strs = query_ids
    query_id_strs = [query_id.split("@")[0] for query_id in query_ids]

    format_rewards = []

    queryid_to_results = collections.defaultdict(list)
    # 8 processes on each node, with 10 subprocesses each
    if do_eval == True:
        _answers = [
            seq_str.split(prompt_str)[1]
            for seq_str, prompt_str in zip(seq_strs, prompt_strs)
        ]

        codes = [extract_python_code(_answer) for _answer in _answers]
        logger.info(
            f"code_rw_interface, size: {len(query_id_strs)}, valid code size: {len(codes)}, query_id_0: {query_id_strs[0]}"
        )
        format_rewards = code_verify(codes, query_id_strs)

        if check_xml_format:
            with ThreadPoolExecutor(max_workers=22) as executor:
                futures = [
                    executor.submit(check_with_elementtree, answer_str)
                    for answer_str in _answers
                ]
                # xml_rewards = []
                for idx, future in enumerate(futures):
                    xml_reward, _ = future.result()
                    # xml_rewards.append(xml_reward)
                    if xml_reward == 1 and format_rewards[idx] == 0:
                        format_rewards[idx] = -0.8
                    elif xml_reward == 0 and format_rewards[idx] == 0:
                        format_rewards[idx] = -1

        for query_id_str, format_reward in zip(query_id_strs, format_rewards):
            if query_id_str not in queryid_to_results:
                queryid_to_results[query_id_str] = []
            queryid_to_results[query_id_str].append(format_reward)
    else:
        for query_id_str in query_id_strs:
            if query_id_str not in queryid_to_results:
                queryid_to_results[query_id_str] = []
            queryid_to_results[query_id_str].append(0)
            format_rewards.append(0)

    return format_rewards, prompt_strs, prompt_ids, seq_strs, queryid_to_results


@dataclasses.dataclass
class PackedCodeRewardInterface(model_api.ModelInterface):

    enable_save: bool = False
    tokenizer_path: str = "/storage/models/Qwen__Qwen2.5-1.5B"
    output_scaling: float = 1.0
    rm_output_scaling: float = 1.0
    rm_output_bias: float = 0.0
    output_bias: float = 0.0
    loss_fun = torch.nn.CrossEntropyLoss(reduction="none")
    max_sync_length: int = 2048
    rw_type: str = "sparse"
    task: str = "code"  # math or countdown or code
    check_xml_format: bool = False
    post_process: str = "sigmoid"
    group_size: int = 1
    check_verifier_status: bool = False

    _call_count: int = 0

    def __post_init__(self):
        self.tokenizer = load_hf_tokenizer(self.tokenizer_path)
        logger.info(f"rm_output_scaling: {self.rm_output_scaling}")
        logger.info(f"rm_output_bias: {self.rm_output_bias}")
        logger.info(f"output_scaling: {self.output_scaling}")
        logger.info(f"output_bias: {self.output_bias}")
        logger.info(f"max_sync_length: {self.max_sync_length}")
        logger.info(f"rw_type: {self.rw_type}")
        logger.info(f"post_process: {self.post_process}")

        while True:
            gen_file_path = os.path.join(
                constants.LOG_ROOT,
                constants.experiment_name(),
                constants.trial_name(),
                "generated",
                f"v{self._call_count}r{dist.get_rank()}.txt",
            )
            if os.path.exists(gen_file_path):
                self._call_count += 1
            else:
                break
        logger.info(f"call_count: {self._call_count}")

    def inference(
        self,
        model: model_api.Model,
        data_: SequenceSample,
        mb_spec,
    ) -> SequenceSample:

        packed_input_ids: torch.Tensor = data_.data["packed_input_ids"].squeeze()

        input_seqlens = torch.tensor(data_.seqlens["packed_input_ids"]).view(-1)
        input_cu_seqlens = torch.nn.functional.pad(
            input_seqlens.cumsum(0), (1, 0)
        ).int()

        packed_prompts = data_.data["packed_prompts"]
        prompts = []
        prompt_seqlens = []
        offset = 0
        for x in data_.seqlens["packed_prompts"]:
            prompts += [packed_prompts[offset : offset + x[0]]] * self.group_size
            offset += x[0]
            prompt_seqlens.extend(x * self.group_size)

        assert offset == sum(x[0] for x in data_.seqlens["packed_prompts"])
        # non_packed_prompts = copy.deepcopy(prompts)
        prompts = torch.cat(prompts)
        prompt_seqlens = torch.tensor(prompt_seqlens).view(-1)
        prompt_cu_seqlens = torch.nn.functional.pad(
            prompt_seqlens.cumsum(0), (1, 0)
        ).int()

        query_ids = [data_id for data_id in data_.ids for _ in range(self.group_size)]

        format_rewards, prompt_strs, prompt_ids, seq_strs, queryid_to_results = (
            retokenize(
                self.task,
                self.tokenizer,
                packed_input_ids,
                input_cu_seqlens,
                prompts,
                prompt_cu_seqlens,
                query_ids,
                check_xml_format=self.check_xml_format,
                do_eval=constants.is_last_pipe_stage(),
            )
        )

        assert self.rw_type == "sparse"
        dense_scores = torch.zeros_like(packed_input_ids).float()
        scores = torch.FloatTensor(format_rewards).to(packed_input_ids.device)
        scores[scores == 0] = -1

        if len(scores) == 0:
            return None

        assert dense_scores.shape == packed_input_ids.shape
        scores = (
            scores.to(packed_input_ids.device) - self.output_bias
        ) * self.output_scaling

        logger.info(f"Code reward logging info @v{model.version.global_step}")
        logger.info(
            f"before: Format success rate: {torch.FloatTensor(format_rewards).mean().item()}"
        )
        logger.info(
            f"number of samples: {len(scores)}, {scores.shape}, group_size: {self.group_size}"
        )

        gen_file_path = os.path.join(
            constants.LOG_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
            "generated",
            f"v{self._call_count}r{dist.get_rank()}.txt",
        )
        os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)
        logger.info(f"Generated samples and rewards will be dumped to: {gen_file_path}")
        with open(gen_file_path, "w") as _f:
            for idx, (score, prompt_str, seq_str) in enumerate(
                zip(scores, prompt_strs, seq_strs)
            ):
                info = "\n".join(
                    [
                        f"idx: {idx} / {len(scores)}",
                        f"reward is {score.item()}, prompt is {colorama.Fore.YELLOW + colorama.Style.DIM}{prompt_str}{colorama.Style.RESET_ALL}",
                        f"sequence is: {colorama.Fore.YELLOW + colorama.Style.DIM}{seq_str.split(prompt_str)[1]}{colorama.Style.RESET_ALL}.",
                    ]
                )
                _f.write(info + "\n")

        gen_file_path = os.path.join(
            constants.LOG_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
            "generated_jsonl",
            f"v{self._call_count}r{dist.get_rank()}.jsonl",
        )
        os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)
        logger.info(f"Generated samples and rewards will be dumped to: {gen_file_path}")
        with open(gen_file_path, "w") as _f:
            for idx, (score, prompt_str, seq_str) in enumerate(
                zip(scores, prompt_strs, seq_strs)
            ):
                _f.write(
                    json.dumps(
                        {
                            "prompt": prompt_str,
                            "generated": seq_str.split(prompt_str)[1],
                            "reward": score.item(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        logger.info(
            f"Format success rate: {torch.FloatTensor(format_rewards).mean().item()}"
        )

        pass_at_k = np.mean(
            [sum([xx == 1 for xx in x]) > 0 for x in queryid_to_results.values()]
        )
        avg_num_samples = np.mean([len(x) for x in queryid_to_results.values()])
        logger.info(f"pass@k: {pass_at_k}, num_samples: {avg_num_samples}")

        logger.info(f"number of samples: {len(scores)}, {scores.shape}")

        logger.info(f"reward: {sum(scores) / len(scores)}")

        train_pass_monitor_file_path = os.path.join(
            constants.LOG_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
            "training_monitor",
            f"v{self._call_count}r{dist.get_rank()}.jsonl",
        )
        os.makedirs(os.path.dirname(train_pass_monitor_file_path), exist_ok=True)
        logger.info(
            f"pass monitor result will be dumped to: {train_pass_monitor_file_path}"
        )
        with open(train_pass_monitor_file_path, "w") as monitor_file:
            for key, value in queryid_to_results.items():
                pass1 = sum(value) / len(value)
                pass8 = int(sum(value) > 0)
                monitor_file.write(
                    json.dumps(
                        {"query_id": key, "pass1": pass1, "pass8": pass8},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        self._call_count += 1

        if scores.dtype != torch.float32:
            scores = scores.to(torch.float32)
        if dense_scores.dtype != torch.float32:
            dense_scores = dense_scores.to(torch.float32)

        res = SequenceSample(
            keys=["rewards", "dense_rewards"],
            trailing_shapes=dict(rewards=(), dense_rewards=()),
            dtypes=dict(rewards=torch.float32, dense_rewards=torch.float32),
            ids=data_.ids,
            seqlens=dict(
                rewards=[
                    torch.tensor([1 for _ in range(len(x))], dtype=torch.int32)
                    for x in data_.seqlens["packed_input_ids"]
                ],
                dense_rewards=data_.seqlens["packed_input_ids"],
            ),
            data=dict(rewards=scores, dense_rewards=dense_scores),
        )

        # record rewards for each piece of data
        avg_scores = []
        offset = 0
        for i in range(data_.bs):
            score_lis = scores[
                offset : offset + len(data_.seqlens["packed_input_ids"][i])
            ]
            avg_scores.append(score_lis.mean().item())
            offset += len(data_.seqlens["packed_input_ids"][i])
        assert offset == sum(len(x) for x in data_.seqlens["packed_input_ids"])

        res.metadata["scores"] = avg_scores

        if self.check_verifier_status:
            avg_score = torch.tensor(
                np.mean(avg_scores), device=constants.current_device()
            )
            dist.all_reduce(
                avg_score, op=dist.ReduceOp.SUM, group=constants.parallelism_group()
            )
            avg_score /= constants.parallelism_group_size()
            avg_score = avg_score.item()
            minimal_score = (-1 - self.output_bias) * self.rm_output_scaling

            if avg_score <= minimal_score or np.isclose(avg_score, minimal_score):
                raise CodeVerifierException(
                    "All rewards are at minimal value. Probably there are something wrong with the verifier!"
                )
        return res


model_api.register_interface("rw_code", PackedCodeRewardInterface)
