import json
import os
from collections import defaultdict

from functioncall.base import logging
from functioncall.base.call import batch_function_call

logger = logging.getLogger("Functioncall")


def try_load_json(data):
    """
    Attempts to load the given data as JSON. If successful, returns the parsed JSON.
    Otherwise, returns None and logs an error.
    """
    try:
        loaded_data = json.loads(data)
        return loaded_data
    except json.JSONDecodeError as e:
        # print(f"Failed to load JSON: {e}")
        return data


def load_problems_with_testcase_batch(path, debug=False, test_case_batch_size=None):
    problem_map = defaultdict(list)
    for idx, line in enumerate(open(path, "rb")):
        if line is None:
            continue

        # parse one problem
        row = json.loads(line.strip().decode("utf-8"))
        query_id = str(row["id"])
        input_output = json.loads(row["input_output"]) if "input_output" in row else {}
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])
        assert len(inputs) == len(
            outputs
        ), f"Inputs({len(inputs)}) and outputs({len(outputs)}) mismatch for {query_id}"

        other_io_fields = {
            k: v for k, v in input_output.items() if k not in ["inputs", "outputs"]
        }
        # create batches for testcases
        if not test_case_batch_size or test_case_batch_size <= 0:
            test_case_batch_size = len(inputs)

        for batch_idx in range(0, len(inputs), test_case_batch_size):
            batch_io = {
                **other_io_fields,
                "inputs": inputs[batch_idx : batch_idx + test_case_batch_size],
                "outputs": outputs[batch_idx : batch_idx + test_case_batch_size],
            }

            sub_problem = {
                "problem_id": query_id,
                "input_output": json.dumps(batch_io),
                "batche_index": batch_idx,
            }
            if debug:
                sub_problem["solutions"] = row.get("solutions", [])
            problem_map[query_id].append(sub_problem)

    return problem_map


global_problems = None


def code_verify(generateds, query_ids, debug=False, timeout=20, timeout_for_testcase=6):
    assert len(generateds) == len(query_ids), (
        len(generateds),
        len(query_ids),
    )
    payload_list = []

    global global_problems
    if global_problems is None:
        global_problems = load_problems_with_testcase_batch(
            os.getenv(
                "REAL_CODE_METADATA_PATH",
                "/storage/datasets/codeparrot-apps-test.jsonl",
            ),
            debug=True,
            test_case_batch_size=20,
        )
    for idx, query_id in enumerate(query_ids):
        if query_id not in global_problems:
            payload_list.append(None)
            logger.warning(
                f"Invalid query id : {query_id}, type: {type(query_id)}, should be in problem dataset!"
            )
            continue

        problems = global_problems[query_id]
        for problem in problems:
            payload_list.append(
                {
                    "problem": problem,
                    "code": generateds[idx],
                    "debug": debug,
                    "timeout": timeout_for_testcase,
                    "query_index": idx,
                }
            )

    logger.debug(
        f"code_verify, payload_list size: {len(payload_list)}, query size: {len(query_ids)}, query_id_0: {query_ids[0]}"
    )
    rsp_list = batch_function_call(payload_list, "python_code", timeout=timeout)

    results = [1] * len(query_ids)
    for idx, rsp in enumerate(rsp_list):
        query_index = payload_list[idx]["query_index"]
        query_id = query_ids[query_index]

        value = 0
        if rsp and "result" in rsp and not any(x != True for x in rsp["result"]):
            value = 1

        else:
            logger.debug(
                f"Functioncall code verify failed, query index: {query_index}, query id: {query_id}, results: {rsp}"
            )

        # logger.debug(f"query index: {idx}, value: {value}, results[query_index]: {results[query_index]}")

        results[query_index] = results[query_index] and value

    return results


if __name__ == "__main__":

    def create_test_params(count=10):
        global global_problems
        codes, query_ids = [], []
        idx = 0
        for query_id, problems in global_problems.items():
            if idx >= count:
                break

            problem = problems[0]
            if "solutions" not in problem or not problem["solutions"]:
                continue

            codes.append(try_load_json(problem["solutions"])[0])
            query_ids.append(query_id)
            idx += 1

        return codes, query_ids

    codes, query_ids = create_test_params(1000)
    result = code_verify(codes, query_ids, True, 100)
    print(result)
