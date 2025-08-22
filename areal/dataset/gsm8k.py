from typing import Optional

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node


def get_gsm8k_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
):
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset


def get_gsm8k_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
):
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["question"]
                + "\nPlease put your final answer within \\boxed{}.",
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["question"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset
