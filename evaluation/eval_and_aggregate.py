import argparse
import json
import os
import subprocess
from glob import glob

import numpy as np
import wandb
from rm_maj_eval import group_pred
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_names", default="math_500,aime24,amc23", type=lambda x: x.split(",")
    )
    parser.add_argument(
        "--model_path",
        default="/storage/openpsi/models/Qwen__Qwen2-1.5B-Instruct/",
        type=str,
    )
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--num_sample_nodes", default=8, type=int)
    parser.add_argument("--samples_per_node", default=4, type=int)
    parser.add_argument("--n_sampling", default=32, type=int)
    parser.add_argument("--prompt_type", default="deepscaler", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate_train", action="store_true")
    parser.add_argument("--max_gen_tokens", default=32768, type=int)

    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = args.model_path
    return args


def eval_maj_k_metrics(data_list, k=8):
    # print(f"evaluating maj@{k}")

    count, right_count = 0, 0
    for sample in data_list:
        assert len(sample["score"]) >= k, sample
        groups, majority_pred = group_pred(
            sample["pred"][:k], strip=False, use_symbol=False
        )
        idx = groups[majority_pred][0]
        right_count += sample["score"][idx]
        count += 1

    task_acc = right_count / count * 100
    # print(f"maj@{k}: {task_acc:.1f}")
    return task_acc


def pass_at_k(data_list, k=8):

    def cur_pass_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    # count, right_count = 0, 0
    pass_at_ks = []
    for sample in data_list:
        assert len(sample["score"]) >= k, sample
        correct = sum(sample["score"])
        pass_at_ks.append(cur_pass_k(len(sample["score"]), correct, k))

    return np.mean(pass_at_ks) * 100


def get_metrics(fname_pattern, tokenizer, is_greedy):

    generated = []
    lengths = []
    results = {}

    for fname in glob(fname_pattern):
        datas = load_jsonl(fname)
        for data in tqdm(datas, desc=fname):

            # tokenize
            generated.extend(data["code"])
            if len(generated) > 2000:
                encodings = tokenizer(generated, return_length=True)
                lengths.extend(encodings["length"])
                generated = []

            # answer score
            cur_idx = data["idx"]
            if cur_idx not in results:
                results[cur_idx] = {"pred": [], "score": []}

            results[cur_idx]["pred"] += data["pred"]
            results[cur_idx]["score"] += data["score"]

    if generated:
        encodings = tokenizer(generated, return_length=True)
        lengths.extend(encodings["length"])

    print(len(lengths))
    assert len(lengths) != 0
    if is_greedy:
        return {
            "greedy_length": np.mean(lengths),
            "greedy_acc": pass_at_k(results.values(), 1),
            "num_questions": len(lengths),
        }
    else:

        return {
            "sample_length": np.mean(lengths),
            "sample_pass@1": pass_at_k(results.values(), 1),
            "pass@8": pass_at_k(results.values(), 8),
            "pass@16": pass_at_k(results.values(), 16),
        }


def process_single_data_name(args, data_name, base_dir, tokenizer):
    cur_dir = os.path.join(base_dir, data_name)
    greedy_prefix = f"test_{args.prompt_type}_-1_seed0_t0.0_s0_e-1_n1"
    sampling_prefix = (
        f"test_{args.prompt_type}_-1_seed*_t0.6_s0_e-1_n{args.samples_per_node}"
    )

    greedy_length_metrics = get_metrics(
        os.path.join(cur_dir, greedy_prefix + ".jsonl"), tokenizer, True
    )
    sampling_metrics = get_metrics(
        os.path.join(cur_dir, sampling_prefix + ".jsonl"), tokenizer, False
    )

    sample_length = sampling_metrics.pop("sample_length")
    output = dict(
        num_questions=greedy_length_metrics["num_questions"],
        greedy_length=greedy_length_metrics["greedy_length"],
        sample_length=sample_length,
        greedy_acc=greedy_length_metrics["greedy_acc"],
        **sampling_metrics,
    )

    return output


if __name__ == "__main__":
    args = parse_args()
    print(f"Evaluation output to {args.output_path}")
    assert args.num_sample_nodes * args.samples_per_node >= args.n_sampling

    eval_dir = (
        "math_eval"
        if args.max_gen_tokens == 4096
        else f"math_eval_{args.max_gen_tokens}"
    )

    base_dir = os.path.join(args.output_path, eval_dir)
    os.makedirs(base_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    result_path = os.path.join(base_dir, f"aggregate_parallel_{args.prompt_type}.json")

    if (
        args.prompt_type == "qwen-boxed"
        and os.path.exists(os.path.join(base_dir, f"aggregate_parallel.json"))
        and not os.path.exists(result_path)
    ):
        os.system(
            f'cp {os.path.join(base_dir, f"aggregate_parallel.json")} {result_path}'
        )

    if not os.path.exists(result_path) or args.overwrite or args.evaluate_train:
        log_path = os.path.join(base_dir, "logs")
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, "greedy.log"), "w") as f:
            subprocess.run(
                [
                    "sh",
                    "sh/eval_greedy.sh",
                    args.model_path,
                    str(args.max_gen_tokens),
                    ",".join(args.data_names),
                    args.prompt_type,
                    args.output_path,
                ],
                text=True,
                stdout=f,
                stderr=f,
            )

        for i in range(args.num_sample_nodes):
            with open(
                os.path.join(
                    log_path, f"seed-{i+1}-sample-{args.samples_per_node}.log"
                ),
                "w",
            ) as f:
                subprocess.run(
                    [
                        "sh",
                        "sh/eval_sample_with_seed.sh",
                        args.model_path,
                        str(i + 1),
                        str(args.samples_per_node),
                        str(args.max_gen_tokens),
                        ",".join(args.data_names),
                        args.prompt_type,
                        args.output_path,
                    ],
                    text=True,
                    stdout=f,
                    stderr=f,
                )

        all_results = dict()
        for data_name in args.data_names:
            all_results[data_name] = process_single_data_name(
                args, data_name, base_dir, tokenizer
            )

        if not args.evaluate_train:
            with open(result_path, "w") as f:
                json.dump(all_results, f, indent=2)

    else:
        with open(result_path) as f:
            all_results = json.load(f)

    try:
        from prettytable import PrettyTable

        table = PrettyTable()
        field_names = ["dataset"] + list(all_results[args.data_names[0]].keys())
        table.field_names = field_names
        for k, v in all_results.items():
            table.add_row([k, *[round(v[x], 1) for x in field_names[1:]]])

        print(table)
    except:
        print(json.dumps(all_results, indent=2))
