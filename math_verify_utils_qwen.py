# Copyright 2025 Ant Group Inc.

import argparse
import json
from parser import extract_answer

from grader import call_with_timeout, math_equal


def process_results(answer, solution):

    try:
        extracted_answer = extract_answer(answer, "math", use_last_number=False)
        extracted_solution = extract_answer(solution, "math", use_last_number=True)

        # if extract_answer.strip() == "":
        #     print (answer)
        # raise
        if extracted_answer is None or extracted_answer.strip() in ["None", "none", ""]:
            retval = 0
        elif math_equal(extracted_answer, extracted_solution, timeout=False):
            # elif call_with_timeout(math_equal, extracted_answer, extracted_solution):
            retval = 1
        else:
            retval = 0

        return retval, (extracted_answer, extracted_solution)
    except:
        return 0, ("None", "None")


def process_results_process(a, b, output_queue):
    result = process_results(a, b)
    output_queue.put(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp_id", type=str, required=True)
    args = parser.parse_args()

    all_input_data = []
    with open(f"/tmp/{args.tmp_id}-input.jsonl", "r") as temp_file:
        for line in temp_file.readlines():
            all_input_data.append(json.loads(line))

    with open(f"/tmp/{args.tmp_id}-output.jsonl", "w", encoding="utf-8") as temp_file:
        for input_data in all_input_data:
            # r, (ans, sol) = process_results(
            #     input_data["answer"], input_data["solution"]
            # )
            tmp = call_with_timeout(
                process_results_process, input_data["answer"], input_data["solution"]
            )
            if isinstance(tmp, bool):
                r, (ans, sol) = 0, ("None", "None")
            else:
                r, (ans, sol) = tmp

            res = {"retval": r, "ans": ans, "sol": sol}
            temp_file.write(json.dumps(res) + "\n")

    # print (process_results("answer is: \\boxed{2.0}", "the anser is: \\boxed{200\\%}"))
