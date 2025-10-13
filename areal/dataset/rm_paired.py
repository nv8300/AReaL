import sys
import os
from typing import Optional

from datasets import Dataset
from datasets.distributed import split_dataset_by_node

# from areal.utils import logging, stats_tracker
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.utils import load_file


# logger = logging.getLogger("Get RM Paired Dataset")

def process(sample, tokenizer):
    rw_paired_data = []
    for i in range(len(sample["pos_answers"])):
        pos_seq_token = tokenizer.encode(
            sample["prompt"] + sample["pos_answers"][i] + tokenizer.eos_token
        )
        neg_seq_token = tokenizer.encode(
            sample["prompt"] + sample["neg_answers"][i] + tokenizer.eos_token
        )
        rw_paired_data.append({"input_pos_ids": pos_seq_token, "input_neg_ids": neg_seq_token})
    return rw_paired_data




def get_rm_paired_dataset(
    path: str,
    split: str,
    tokenizer,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
):
    rw_paired_data = load_file(path)
    dataset_list = []
    for i in range(len(rw_paired_data)):
        dataset_list += process(rw_paired_data[i], tokenizer)
    dataset = Dataset.from_list(dataset_list)

    if max_length is not None:
    # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_pos_ids"]) <= max_length and len(x["input_neg_ids"]) <= max_length)

    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-1.7B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output_dataset = get_rm_paired_dataset(path="areal/dataset/data/rm_paired_valid.jsonl", split="train", tokenizer=tokenizer, rank=0, world_size=1, max_length=None)
    print(output_dataset[0])


