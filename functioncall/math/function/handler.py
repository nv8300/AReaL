import json
from parser import extract_answer

from grader import math_equal


def process_results(answer, solution):
    extracted_answer = extract_answer(answer, "math", use_last_number=False)
    extracted_solution = extract_answer(solution, "math", use_last_number=True)

    print(
        f"extracted_answer: {extracted_answer}, extracted_solution: {extracted_solution}, equal: {math_equal(extracted_answer, extracted_solution)}"
    )
    if extracted_answer is None or extracted_answer.strip() in ["None", "none", ""]:
        retval = 0
    elif math_equal(extracted_answer, extracted_solution, timeout=True):
        retval = 1
    else:
        retval = 0

    return retval, (extracted_answer, extracted_solution)


def handle(event, context):
    answers = event.get("answers", "")
    solutions = event.get("solutions", "")

    #print(f"math payload:{event}\n")
    # answers and solutions are json lists, and call process_results then collect result into a list
    if isinstance(answers, str):
        answers = json.loads(answers)
    if isinstance(solutions, str):
        solutions = json.loads(solutions)

    results = []
    for answer, solution in zip(answers, solutions):
        results.append(process_results(answer, solution))

    return results
