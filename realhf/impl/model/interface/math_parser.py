# Copyright 2025 Ant Group Inc.

import json
import os
import signal
import subprocess
import uuid
from typing import *

import requests

from realhf.base import logging
from realhf.base.constants import parallelism_rank

logger = logging.getLogger("math parser")


def get_box(s):
    pos = -1
    cnt = 0
    for i in range(len(s)):
        if s[i] == "{":
            cnt += 1
            if cnt == 1:
                pos = i + 1
        if s[i] == "}":
            cnt -= 1
            if cnt == 0:
                return s[pos:i]

    return None


def get_answer(answer):
    pos = answer.find("\\boxed{")
    if pos == -1:
        return []
    return [get_box(answer[pos:])] + get_answer(answer[pos + 1 :])


def loadJson(dataDir):
    with open(dataDir, "r") as f:
        if dataDir.endswith(".jsonl"):
            samples = [json.loads(line) for line in f.readlines()]
        else:
            samples = json.load(f)

    return samples


headers = {
    "Content-Type": "application/json",
}

id2info = None


def parse_line(prompt_str, generated, query_id):
    global id2info
    if id2info is None:
        try:
            id2info = loadJson(os.environ["REAL_MATH_METADATA_PATH"])
        except KeyError as e:
            raise KeyError("The json file REAL_MATH_METADATA_PATH is not set") from e
    info = id2info[query_id.split("@idx:")[0]]

    tmp_id = str(uuid.uuid4())
    with open(f"/tmp/{tmp_id}-input.jsonl", "w", encoding="utf-8") as f:
        for cur_solution in info["solutions"]:
            f.write(json.dumps({"answer": generated, "solution": cur_solution}) + "\n")

    venv_python = "/sympy/bin/python3"
    logger.info(f"math verify working dir: `{os.getcwd()}`")
    pro = subprocess.Popen(
        " ".join(
            [
                venv_python,
                "math_verify_utils_qwen.py",
                "--tmp_id",
                tmp_id,
            ]
        ),
        shell=True,
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
    )
    pro.wait()
    try:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass

    label = 0
    try:
        with open(f"/tmp/{tmp_id}-output.jsonl", "r") as f:
            for line in f.readlines():
                output_data = json.loads(line)
                label = output_data["retval"] or label
    except FileNotFoundError as e:
        # The subprocess may fail to parse the input (maybe due to reaching the maximum recursion length)
        # We just return 0 for the reward.
        logger.warning(
            f"Failed to parse: query_id `{query_id}`, prompt `{prompt_str}`, seq `{generated}`. Set 0 reward."
        )
        label = 0
    finally:
        if os.path.exists(f"/tmp/{tmp_id}-input.jsonl"):
            os.remove(f"/tmp/{tmp_id}-input.jsonl")
        if os.path.exists(f"/tmp/{tmp_id}-output.jsonl"):
            os.remove(f"/tmp/{tmp_id}-output.jsonl")
        return label


def parse_lines_in_parallel(
    prompt_strs: List,
    generateds: List,
    query_ids: List,
    max_workers: int,
    check_xml_format=False,
) -> List:
    global id2info
    if id2info is None:
        try:
            id2info = loadJson(os.environ["REAL_MATH_METADATA_PATH"])
        except KeyError as e:
            raise KeyError("The json file REAL_MATH_METADATA_PATH is not set") from e
    assert len(prompt_strs) == len(generateds) == len(query_ids), (
        len(prompt_strs),
        len(generateds),
        len(query_ids),
    )
    bs = len(prompt_strs)
    mbs = (bs + max_workers - 1) // max_workers

    tmp_ids = []
    all_query_indices = []
    for i in range(max_workers):
        tmp_id = str(uuid.uuid4())
        query_indices = []
        s = slice(i * mbs, (i + 1) * mbs)
        offset = i * mbs
        with open(f"/tmp/{tmp_id}-input.jsonl", "w", encoding="utf-8") as f:
            for idx, (query_id, generated) in enumerate(
                zip(query_ids[s], generateds[s])
            ):
                info = id2info[query_id.split("@idx:")[0]]
                for cur_solution in info["solutions"]:
                    f.write(
                        json.dumps({"answer": generated, "solution": cur_solution})
                        + "\n"
                    )
                    query_indices.append(idx + offset)
        tmp_ids.append(tmp_id)
        all_query_indices.append(query_indices)

    venv_python = "/sympy/bin/python3"
    logger.info(f"math verify working dir: `{os.getcwd()}`")
    procs = []
    for tmp_id in tmp_ids:
        pro = subprocess.Popen(
            " ".join(
                [
                    venv_python,
                    "math_verify_utils_qwen.py",
                    "--tmp_id",
                    tmp_id,
                    # "--check_xml_format",
                    # "True" if check_xml_format else "False",
                ]
            ),
            shell=True,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
        )
        procs.append(pro)
    for pro in procs:
        try:
            pro.wait()
        except Exception as e:
            pass
        try:
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    labels = [0 for _ in prompt_strs]
    for i, (tmp_id, query_indices) in enumerate(zip(tmp_ids, all_query_indices)):
        try:
            with open(f"/tmp/{tmp_id}-output.jsonl", "r") as f:
                for _ansidx, line in enumerate(f.readlines()):
                    output_data = json.loads(line)
                    labels[query_indices[_ansidx]] = (
                        output_data["retval"] or labels[query_indices[_ansidx]]
                    )
        except FileNotFoundError as e:
            # The subprocess may fail to parse the input (maybe due to reaching the maximum recursion length)
            # We just return 0 for the reward.
            logger.warning(f"Failed to parse generated answers. Set 0 reward.")
        finally:
            if os.path.exists(f"/tmp/{tmp_id}-input.jsonl"):
                os.remove(f"/tmp/{tmp_id}-input.jsonl")
            if os.path.exists(f"/tmp/{tmp_id}-output.jsonl"):
                os.remove(f"/tmp/{tmp_id}-output.jsonl")
    return labels


if __name__ == "__main__":
    sample = {
        "prompt": "",
        "query_id": "35ecd821a9e7e31da9ef0663a25347ce",
        # "answer_in_box": ["\\boxed{\\frac{1}{2}}", "<think></think><answer>\\boxed{\\frac{1}{2}}</answer>"]
        "answer": "<think>\n1. The problem requires us to determine the number of sequences of 144 hand movements such that every position appears exactly once and the hands return to the initial position at the end.\n2. We know that each movement involves one hand moving clockwise to the next number while the other hand stays in place.\n3. Considering the 12-hour clock, we can represent each positioning of the hands as a combination of the positions of both hands. Since both hands can be in any of the 12 positions, there are 12 x 12 = 144 different positionings.\n4. Given that at each position only one hand moves, every single movement is unique, leading to a total of 144 unique movements.\n5. These 144 movements must form a Hamiltonian Cycle, where each edge represents a valid movement between two positions.\n6. The problem thus reduces to finding a Hamiltonian cycle in a directed graph. Since the appearance of each movement is unique, it also determines the direction of the movement.\n7. We consider the edges that are rotations of each other as equivalent. Taking the rotational symmetry into account, we have 144/12 = 12 equivalence classes.\n8. The problem now is to determine the number of ways to arrange these 12 classes of rotations in a circle, which is 11 factorial.\n9. We must find the value of 11! and then compute the result modulo 1000.\n</think>\n<answer>\n320\n</answer>",
    }

    print(
        parse_lines_in_parallel(
            [sample["prompt"] for _ in range(100)],
            # [sample["answer_in_box"][0] for _ in range(50)] + [sample["answer_in_box"][1] for _ in range(50)],
            [sample["answer"]] * 100,
            [sample["query_id"] for _ in range(100)],
            max_workers=8,
            check_xml_format=True,
        )
    )
