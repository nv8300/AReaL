import json
import os
import time
from typing import List

from functioncall.base import logging
from functioncall.base.call import batch_function_call

logger = logging.getLogger("Functioncall")


def loadJson(dataDir):
    with open(dataDir, "r") as f:
        if dataDir.endswith(".jsonl"):
            samples = [json.loads(line) for line in f.readlines()]
        else:
            samples = json.load(f)

    return samples


id2info = None


def math_verify(generateds: List, query_ids: List, batch_size=10, timeout=5) -> List:
    global id2info
    if id2info is None:
        id2info = loadJson(
            os.getenv(
                "REAL_MATH_MEATADATA_PATH",
                "/storage/datasets/id2info.json",
            )
        )
    assert len(generateds) == len(query_ids), (
        len(generateds),
        len(query_ids),
    )

    st = time.monotonic()
    query_indices = []
    parameters = []
    # Collect all (generated, solution) pairs with their original indices
    for idx, (query_id, generated) in enumerate(zip(query_ids, generateds)):
        base_query_id = query_id.split("@idx:")[0]
        info = id2info[base_query_id]
        for cur_solution in info["solutions"]:
            parameters.append((generated, cur_solution, idx))
            query_indices.append(idx)

    # Process in batches
    start_time = time.time()
    batch_args_list = []
    for i in range(0, len(parameters), batch_size):
        answers, solutions, indices = zip(*parameters[i : i + batch_size])
        batch_args = {
            "answers": list(answers),
            "solutions": list(solutions),
            "query_ids": [query_ids[i] for i in indices],
        }

        #print(batch_args)
        batch_args_list.append(batch_args)

    results_batch = batch_function_call(batch_args_list, "python_math", timeout)

    labels = [0] * len(query_ids)
    # Map results back to original indices
    index = 0
    for batch_idx, results in enumerate(results_batch):
        query_index = query_indices[index]
        if not isinstance(results, list) or len(results) == 0:
            index += len(batch_args_list[batch_idx]["answers"])
            logger.warning(
                f"Invalid functioncall math results: {results}, batch index:{batch_idx}, query index: {query_index}, params: {batch_args_list[batch_idx]['answers']}."
            )
            continue

        for result in results:
            if (
                isinstance(result, list)
                and len(result) > 0
                and (isinstance(result[0], int) and result[0] in [0, 1])
            ):
                labels[query_index] = result[0] or labels[query_index]
            else:
                logger.warning(
                    f"Invalid functioncall math result: {result}, index:{index}, qeury_id: {query_ids[query_index]}."
                )

            index += 1

    logger.info(
        f"verify math with query size={len(query_ids)}, takes {time.time() - start_time:.4f} seconds, result: {labels}"
    )
    return labels


if __name__ == "__main__":
    sample = {
        "prompt": "",
        "query_id": "fe11b471-1aa9-4867-958f-a0a811c85f92",
        "answer": "\\boxed{-\\frac{2}{3}}",
    }
    start_time = time.time()
    batch_size = 10
    result = math_verify(
        [sample["answer"]] * batch_size, [sample["query_id"] for _ in range(batch_size)]
    )

    hint = f"batch_size: {batch_size}, total time :   {(time.time() - start_time) * 1000:.0f} ms"
    print(result)
    print(hint)
