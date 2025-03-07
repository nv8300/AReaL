import json
import multiprocessing
import os
import sys
import time
import traceback
from io import StringIO

from function.testing_util import run_test

from functioncall.base import logging

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
        logger.info(f"Failed to load JSON: {e}")
        return data


def capture_stdout(code):
    original_stdout = sys.stdout
    fake_stdout = StringIO()

    try:
        sys.stdout = fake_stdout  # 重定向输出
        exec(code, {"__builtins__": __builtins__})  # 在隔离环境中执行
    except Exception as e:
        return f"error: {str(e)}, traceback: {traceback.format_exc()}"
    finally:
        sys.stdout = original_stdout  # 恢复原stdout
    return fake_stdout.getvalue()


def _temp_run(problem, generation, debug, result):
    start_time = time.time()

    try:
        if debug:
            logger.debug(f"Running test for problem: {problem}")
        result.append(run_test(sample=problem, test=generation, debug=debug))
        if debug:
            logger.debug(f"Test completed with result: {result}")
    except Exception as e:
        if debug:
            logger.error(f"Error in _temp_run: {e}, problem:{problem}", exe_info=True)

    execution_time = time.time() - start_time
    logger.info(
        f'[_temp_run] query_id: {problem["problem_id"]}, start_time: {str(start_time)}, Time elapsed: {execution_time * 1000:.0f} ms'
    )


def check_correctness(problem, generation, timeout, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    if debug:
        result = capture_stdout(
            "from function.testing_util import run_test\n"
            + "run_test(sample=problem, test=generation, debug=debug)"
        )
        return result[0], result[1]

    start_time = time.time()
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(
        target=_temp_run, args=(problem, generation, debug, result)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        if debug:
            logger.debug(f"Process is still alive. Killing the process.")
        p.kill()
    if not result:
        # Remark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
        if debug:
            logger.debug(f"Global timeout occurred, returning default result.")
    if debug:
        logger.debug(f"Final result: {result}")

    execution_time = time.time() - start_time
    logger.info(
        f'[check_correctness] query_id: {problem["problem_id"]}, start_time: {str(start_time)}, Time elapsed: {execution_time * 1000:.0f} ms'
    )
    return result[0]


def load_problems(path):
    problem_map = {}
    for idx, line in enumerate(open(path, "rb")):
        if line is not None:
            line = line.strip().decode("utf-8")
            row = json.loads(line)

            query_id = str(row["id"])
            problem_map[query_id] = {
                "problem_id": query_id,
                # "question": row["question"],
                "solutions": row["solutions"],
                "input_output": row["input_output"],
                # "difficulty": level,
                # "url": row["url"],
                # "starter_code": row["starter_code"],
            }

    return problem_map


global_problems = load_problems(
    os.getenv(
        "REAL_CODE_METADATA_PATH",
        "/storage/datasets/codeparrot-apps-test.jsonl",
    )
)


def code_verify(generateds, query_ids, debug=False):
    assert len(generateds) == len(query_ids), (
        len(generateds),
        len(query_ids),
    )

    result = []
    global global_problems

    for idx, query_id in enumerate(query_ids):
        if query_id not in global_problems:
            continue

        problem = global_problems[query_id]
        test_code = generateds[idx]
        logger.debug(f"run_batch_code, query_id: {query_id}")

        try:
            curr_res, metadata = check_correctness(
                problem=problem, generation=test_code, timeout=6000, debug=debug
            )

            if any(x != True for x in curr_res):
                logger.debug(
                    f'id:{problem["problem_id"]}, Results were not all True: {metadata}'
                )
                result.append(f"{query_id} failed")
            else:
                # print(f"id:{problem["problem_id"]}, result : {curr_res}")
                result.append(f"{query_id} success")

        except Exception as e:
            logger.error(f"test framework exception = {repr(e)}{e}\n", exe_info=True)
            result.append(f"{query_id} failed")
            break
        finally:
            # print(f"id:{problem["problem_id"]}, result : {curr_res}")
            # assert isinstance(curr_res, list)
            pass

    return result


if __name__ == "__main__":

    def create_test_params(index_list):
        global global_problems
        codes, query_ids = [], []

        for index in index_list:
            if str(index) not in global_problems:
                continue

            problem = global_problems[str(index)]
            if "solutions" not in problem or not problem["solutions"]:
                continue

            codes.append(try_load_json(problem["solutions"])[0])
            query_ids.append(str(index))
        return codes, query_ids

    codes, query_ids = create_test_params(list(range(805, 806)))
    result = code_verify(codes, query_ids, True)
    print(result)
