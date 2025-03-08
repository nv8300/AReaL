# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import uuid
from typing import Callable, Dict, Hashable, List, Optional

import numpy as np
import torch.utils.data

from realhf.api.core import data_api
from realhf.base import logging

logger = logging.getLogger("Prompt Dataset")


class PromptDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
        fill_to_max_length: bool = False,
    ):
        """A dataset with prompts. Usually used for PPO.

        Args:
            util (api.data.DatasetUtility): .
            max_length (Optional[int], optional): The maximum length of each sequence in the batch.
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                a key "prompt". Defaults to None.
            dataset_builder (Optional[Callable[[], List[Dict]]], optional): Alternative to dataset_path.
                A callable that returns a list of dictionary. Defaults to None.
            fill_to_max_length (bool):Whether to fill prompts to the maximum length. If True,
                prompts will be left-filled with non-pad tokens. Only used for testing.
        """
        self._util = util
        self.max_length = max_length

        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        prompts_str = [x["prompt"] for x in data]
        self.ids = [x["id"] for x in data]
        util.tokenizer.padding_side = "left"
        prompt_encodings = util.tokenizer(
            prompts_str,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_length=True,
            return_attention_mask=False,
        )

        if fill_to_max_length:
            for i in range(len(prompt_encodings["length"])):
                x = prompt_encodings["input_ids"][i]
                if max_length > len(x):
                    # Fill with the final non-pad token to the left.
                    prompt_encodings["input_ids"][i] = [x[-1]] * (
                        max_length - len(x)
                    ) + x
                    prompt_encodings["length"][i] = max_length

        self.prompt_lengths = prompt_encodings["length"]
        self.prompts = prompt_encodings["input_ids"]
        assert all(len(x) == l for x, l in zip(self.prompts, self.prompt_lengths))

        logger.info(f"Number of prompts in the dataset: {len(self.prompts)}")

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return data_api.SequenceSample.from_default(
            ids=[self.ids[idx]],
            seqlens=[self.prompt_lengths[idx]],
            data=dict(packed_prompts=torch.tensor(self.prompts[idx], dtype=torch.long)),
        )


class MATHPromptDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
        fill_to_max_length: bool = False,
        filter_threshold: float = 1e4,
        max_filter_percentage: float = 0.0,
    ):
        """A dataset with prompts. Usually used for PPO.

        Args:
            util (api.data.DatasetUtility): .
            max_length (Optional[int], optional): The maximum length of each sequence in the batch.
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                a key "prompt". Defaults to None.
            dataset_builder (Optional[Callable[[], List[Dict]]], optional): Alternative to dataset_path.
                A callable that returns a list of dictionary. Defaults to None.
        """
        self._util = util
        self.max_length = max_length

        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        prompts_str = [x["prompt"] for x in data]
        self.ids = [x["query_id"] for x in data]
        if "scores" in data[0]:
            self.base_scores = [np.mean(x["scores"]) for x in data]
        util.tokenizer.padding_side = "left"
        prompt_encodings = util.tokenizer(
            prompts_str,
            truncation=True,
            # max_length=max_length,
            padding=False,
            return_length=True,
            return_attention_mask=False,
        )

        if fill_to_max_length:
            for i in range(len(prompt_encodings["length"])):
                x = prompt_encodings["input_ids"][i]
                if max_length > len(x):
                    # Fill with the final non-pad token to the left.
                    prompt_encodings["input_ids"][i] = [x[-1]] * (
                        max_length - len(x)
                    ) + x
                    prompt_encodings["length"][i] = max_length

        logger.info(f"{len(data)} samples, checking lengths (max_length={max_length})")
        indices = [
            i for i, x in enumerate(prompt_encodings["length"]) if x <= max_length
        ]
        logger.info(f"{len(indices)} samples remain")

        self.prompt_lengths = [int(prompt_encodings["length"][idx]) for idx in indices]
        self.prompts = [prompt_encodings["input_ids"][idx] for idx in indices]
        self.ids = [self.ids[idx] + f"@idx:{idx}-{util.dp_rank}" for idx in indices]
        if "scores" in data[0]:
            self.base_scores = [self.base_scores[idx] for idx in indices]

        assert all(len(x) == l for x, l in zip(self.prompts, self.prompt_lengths))

        logger.info(f"Number of prompts in the dataset: {len(self.prompts)}")

        self.active_indices = list(range(len(self.prompts)))
        self.filter_threshold = filter_threshold
        self.max_filter_percentage = max_filter_percentage

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        # print(self.base_scores)
        idx = self.active_indices[idx]
        if hasattr(self, "base_scores"):
            return data_api.SequenceSample.from_default(
                ids=[self.ids[idx]],
                seqlens=[self.prompt_lengths[idx]],
                data=dict(
                    packed_prompts=torch.tensor(self.prompts[idx], dtype=torch.long),
                    base_scores=torch.tensor(
                        [self.base_scores[idx]], dtype=torch.float32
                    ),
                ),
            )
        else:
            return data_api.SequenceSample.from_default(
                ids=[self.ids[idx]],
                seqlens=[self.prompt_lengths[idx]],
                data=dict(
                    packed_prompts=torch.tensor(self.prompts[idx], dtype=torch.long)
                ),
            )

    def filter(self, eval_scores: Dict[Hashable, float]):
        # Get all data indices that have a higher score than the threshold.
        idx2scores_to_remove = {}
        for pop_idx, idx in enumerate(self.active_indices):
            data_id = self.ids[idx]
            if data_id not in eval_scores:
                continue
            if eval_scores[data_id] > self.filter_threshold:
                idx2scores_to_remove[pop_idx] = eval_scores[data_id]

        # Control the number of samples to be removed according to max_filter_percentage.
        n = int(len(self.active_indices) * self.max_filter_percentage)
        indices_to_remove = sorted(
            idx2scores_to_remove.keys(),
            key=lambda x: idx2scores_to_remove[x],
            reverse=True,
        )[:n]

        for pop_idx in sorted(indices_to_remove, reverse=True):
            self.active_indices.pop(pop_idx)
        logger.info(
            f"Math prompt dataset DP rank {self.util.dp_rank} filtered"
            f" {len(indices_to_remove)} samples, {len(self.active_indices)} samples remain. "
            f"Original dataset size: {len(self.prompts)}. "
            f"Filter threshold: {self.filter_threshold}. "
            f"Max filter percentage: {self.max_filter_percentage}. "
            f"Current number of eval scores: {len(eval_scores)}."
        )


class CODEPromptDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
        fill_to_max_length: bool = False,
        filter_threshold: float = 1e4,
        max_filter_percentage: float = 0.0,
    ):
        """A dataset with prompts. Usually used for PPO.

        Args:
            util (api.data.DatasetUtility): .
            max_length (Optional[int], optional): The maximum length of each sequence in the batch.
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                a key "prompt". Defaults to None.
            dataset_builder (Optional[Callable[[], List[Dict]]], optional): Alternative to dataset_path.
                A callable that returns a list of dictionary. Defaults to None.
        """
        self._util = util
        self.max_length = max_length

        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        prompts_str = [x["question"] for x in data]
        self.ids = [x["id"] for x in data]
        if "scores" in data[0]:
            self.base_scores = [np.mean(x["scores"]) for x in data]
        util.tokenizer.padding_side = "left"
        prompt_encodings = util.tokenizer(
            prompts_str,
            truncation=True,
            # max_length=max_length,
            padding=False,
            return_length=True,
            return_attention_mask=False,
        )

        if fill_to_max_length:
            for i in range(len(prompt_encodings["length"])):
                x = prompt_encodings["input_ids"][i]
                if max_length > len(x):
                    # Fill with the final non-pad token to the left.
                    prompt_encodings["input_ids"][i] = [x[-1]] * (
                        max_length - len(x)
                    ) + x
                    prompt_encodings["length"][i] = max_length

        logger.info(f"{len(data)} samples, checking lengths (max_length={max_length})")
        indices = [
            i for i, x in enumerate(prompt_encodings["length"]) if x <= max_length
        ]
        logger.info(f"{len(indices)} samples remain")

        self.prompt_lengths = [int(prompt_encodings["length"][idx]) for idx in indices]
        self.prompts = [prompt_encodings["input_ids"][idx] for idx in indices]
        self.ids = [
            str(self.ids[idx]) + f"@idx:{idx}-{util.dp_rank}" for idx in indices
        ]
        if "scores" in data[0]:
            self.base_scores = [self.base_scores[idx] for idx in indices]

        assert all(len(x) == l for x, l in zip(self.prompts, self.prompt_lengths))

        logger.info(f"Number of prompts in the dataset: {len(self.prompts)}")

        self.active_indices = list(range(len(self.prompts)))
        self.filter_threshold = filter_threshold
        self.max_filter_percentage = max_filter_percentage

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        # print(self.base_scores)
        print(
            f"CODEPromptDataset idx{idx}, use idx: {self.active_indices[idx]}, size: {len(self.ids)}"
        )
        idx = self.active_indices[idx]

        if hasattr(self, "base_scores"):
            return data_api.SequenceSample.from_default(
                ids=[self.ids[idx]],
                seqlens=[self.prompt_lengths[idx]],
                data=dict(
                    packed_prompts=torch.tensor(self.prompts[idx], dtype=torch.long),
                    base_scores=torch.tensor(
                        [self.base_scores[idx]], dtype=torch.float32
                    ),
                ),
                metadata=dict(random_id=[uuid.uuid4()]),
            )
        else:
            return data_api.SequenceSample.from_default(
                ids=[self.ids[idx]],
                seqlens=[self.prompt_lengths[idx]],
                data=dict(
                    packed_prompts=torch.tensor(self.prompts[idx], dtype=torch.long)
                ),
                metadata=dict(random_id=[uuid.uuid4()]),
            )

    def filter(self, eval_scores: Dict[Hashable, float]):
        # Get all data indices that have a higher score than the threshold.
        idx2scores_to_remove = {}
        for pop_idx, idx in enumerate(self.active_indices):
            data_id = self.ids[idx]
            if data_id not in eval_scores:
                continue
            if eval_scores[data_id] > self.filter_threshold:
                idx2scores_to_remove[pop_idx] = eval_scores[data_id]

        # Control the number of samples to be removed according to max_filter_percentage.
        n = int(len(self.active_indices) * self.max_filter_percentage)
        indices_to_remove = sorted(
            idx2scores_to_remove.keys(),
            key=lambda x: idx2scores_to_remove[x],
            reverse=True,
        )[:n]

        for pop_idx in sorted(indices_to_remove, reverse=True):
            self.active_indices.pop(pop_idx)
        logger.info(
            f"Code prompt dataset DP rank {self.util.dp_rank} filtered"
            f" {len(indices_to_remove)} samples, {len(self.active_indices)} samples remain. "
            f"Original dataset size: {len(self.prompts)}. "
            f"Filter threshold: {self.filter_threshold}. "
            f"Max filter percentage: {self.max_filter_percentage}. "
            f"Current number of eval scores: {len(eval_scores)}."
        )


if not __name__ == "__main__":
    data_api.register_dataset("prompt", PromptDataset)
    data_api.register_dataset("math_prompt", MATHPromptDataset)
    data_api.register_dataset("code_prompt", CODEPromptDataset)
else:
    from transformers import AutoTokenizer

    dataset = MATHPromptDataset(
        data_api.DatasetUtility(
            seed=0,
            dp_rank=0,
            world_size=1,
            tokenizer=AutoTokenizer.from_pretrained(
                "/storage/openpsi/models/Qwen__Qwen2-1.5B-Instruct/"
            ),
        ),
        max_length=512,
        dataset_path="/storage/openpsi/data/math/Qwen_RL_training_xss/0101/train_rl@0101_with_qwqsft-7b_score.jsonl",
    )

    dataset = CODEPromptDataset(
        data_api.DatasetUtility(
            seed=0,
            dp_rank=0,
            world_size=1,
            tokenizer=AutoTokenizer.from_pretrained(
                "/storage/openpsi/models/Qwen__Qwen2-1.5B-Instruct/"
            ),
        ),
        max_length=512,
        dataset_path="/storage/datasets/codeparrot-apps-test.jsonl",
    )
    data_generator = enumerate(dataset)
    dataset_batch_counter, cur_sample = next(data_generator)
    print(
        f"size: {len(dataset)}, index: {dataset_batch_counter}, cur_sample: {cur_sample}"
    )
