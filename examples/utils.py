import importlib
import inspect
import json
import re
import copy
import builtins
import operator
import re
from functools import reduce
from typing import TYPE_CHECKING, Callable, List, Optional, Type, Union

import ast

from areal.bfcl.constants.category_mapping import TEST_COLLECTION_MAPPING, TEST_FILE_MAPPING
from areal.bfcl.constants.enums import ReturnFormat

CLASS_FILE_PATH_MAPPING = {
    "GorillaFileSystem": "areal.bfcl.multi_turn_eval.func_source_code.gorilla_file_system",
    "MathAPI": "areal.bfcl.eval_checker.multi_turn_eval.func_source_code.math_api",
    "MessageAPI": "areal.bfcl.eval_checker.multi_turn_eval.func_source_code.message_api",
    "TwitterAPI": "areal.bfcl.eval_checker.multi_turn_eval.func_source_code.posting_api",
    "TicketAPI": "areal.bfcl.eval_checker.multi_turn_eval.func_source_code.ticket_api",
    "TradingBot": "areal.bfcl.eval_checker.multi_turn_eval.func_source_code.trading_bot",
    "TravelAPI": "areal.bfcl.eval_checker.multi_turn_eval.func_source_code.travel_booking",
    "VehicleControlAPI": "areal.bfcl.eval_checker.multi_turn_eval.func_source_code.vehicle_control",
}

# These classes are stateless and do not require any initial configuration
STATELESS_CLASSES = [
    "MathAPI",
]

def load_file(file_path, sort_by_id=False):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))

    if sort_by_id:
        result.sort(key=sort_key)
    return result


def parse_test_category_argument(test_category: str):
    test_name_total = set()
    test_filename_total = set()

    if test_category in TEST_COLLECTION_MAPPING:
        for test_name in TEST_COLLECTION_MAPPING[test_category]:
            test_name_total.add(test_name)
            test_filename_total.add(TEST_FILE_MAPPING[test_name])
    elif test_category in TEST_FILE_MAPPING:
        test_name_total.add(test_category)
        test_filename_total.add(TEST_FILE_MAPPING[test_category])
    else:
        # Invalid test category name
        raise Exception(f"Invalid test category name provided: {test_category}")

    return sorted(list(test_filename_total)), sorted(list(test_name_total))


def execute_multi_turn_func_call(
    func_call_list: list[str],  # a list of strings of func calls
    initial_config: dict,
    involved_classes: list,
    model_name: str,
    test_entry_id: str,
    long_context: bool = False,
    is_evaL_run: bool = False,
) -> tuple[list[str], dict]:
    """
    Execute the function call, construct the return result into a string, and return it to the list composed of the call result str of the function call.
    """
    if is_evaL_run:
        model_name += "_eval"

    class_method_name_mapping = {}
    involved_instances = {}
    for class_name in involved_classes:
        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        # TODO: Handler the model name issue from handler more elegantly
        instance_name = (
            f"{model_name.replace('-', '_').replace('.', '_').replace('/', '_')}_{test_entry_id}_{class_name.lower()}_instance"
        )
        if instance_name not in globals():
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            class_instance = class_()
            if class_name not in STATELESS_CLASSES:
                class_initial_config = initial_config.get(class_name, {})
                # Deep copy the initial configuration to avoid mutation issues
                class_instance._load_scenario(
                    copy.deepcopy(class_initial_config), long_context=long_context
                )
            globals()[instance_name] = class_instance
        # This happens in subsequent turns
        else:
            class_instance = globals()[instance_name]

        involved_instances[class_name] = class_instance

        # Retrieve all method names and map them to the instance
        for method_name, method in inspect.getmembers(
            class_instance, predicate=inspect.ismethod
        ):
            # Skip private methods
            if method_name.startswith("_"):
                continue
            class_method_name_mapping[method_name] = instance_name

    execution_results = []
    for func_call in func_call_list:
        # Add the instance name to the method calls
        func_call = _process_method_calls(func_call, class_method_name_mapping)

        # Evaluate the function call
        try:
            # We need to make a copy here because otherwise the `eval(func_call)` would error. 
            func_call_copy = func_call
            # Before calling `eval`, we need to make sure that the function call is safe
            # We do so by checking if the function is `kill` or `exit`, etc.
            # Extract the function name first
            if "(" in func_call_copy:
                func_call_copy = func_call_copy.split("(")[0]
            # Situation where the function call is a method call
            if "." in func_call_copy:
                func_call_copy = func_call_copy.split(".")[1]
            if func_call_copy in ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]:
                raise Exception(f"Function call {func_call_copy} is not allowed.")

            func_call_result = eval(func_call)

            if type(func_call_result) == str:
                pass
            elif type(func_call_result) == dict:
                # Some function returns a object instance, which is not serializable
                try:
                    func_call_result = json.dumps(func_call_result)
                except:
                    func_call_result = str(func_call_result)
            else:
                func_call_result = str(func_call_result)

            execution_results.append(func_call_result)
        except Exception as e:
            execution_results.append(f"Error during execution: {str(e)}")

    return execution_results, involved_instances


def is_empty_execute_response(input_list: list):
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False


def ast_parse(
    input_str: str,
    language: ReturnFormat = ReturnFormat.PYTHON,
    has_tool_call_tag: bool = False,
) -> list[dict]:
    if has_tool_call_tag:
        match = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", input_str, re.DOTALL)
        if match:
            input_str = match.group(1).strip()
        else:
            raise ValueError(f"No tool call tag found in input string: {input_str}")

    if language == ReturnFormat.PYTHON:
        # We only want to remove wrapping quotes that could have been added by the model.
        cleaned_input = input_str.strip().strip("'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return extracted

    # elif language == ReturnFormat.JAVA:
    #     # Remove the [ and ] from the string
    #     # Note: This is due to legacy reasons, we should fix this in the future.
    #     return parse_java_function_call(input_str[1:-1])

    # elif language == ReturnFormat.JAVASCRIPT:
    #     # Note: Same as above, we should fix this in the future.
    #     return parse_javascript_function_call(input_str[1:-1])

    # elif language == ReturnFormat.VERBOSE_XML:
    #     # Remove ```xml and anything before/after XML
    #     match = re.search(r"<functions>(.*?)</functions>", input_str, re.DOTALL)
    #     if not match:
    #         raise ValueError(
    #             f"No XML function call found in input string: {input_str}. Missing <functions> tag."
    #         )
    #     return parse_verbose_xml_function_call(match.group(0))

    # elif language == ReturnFormat.CONCISE_XML:
    #     # Remove anything before/after <functions> and </functions>
    #     match = re.search(r"<functions>(.*?)</functions>", input_str, re.DOTALL)
    #     if not match:
    #         raise ValueError(
    #             f"No XML function call found in input string: {input_str}. Missing <functions> tag."
    #         )
    #     return parse_concise_xml_function_call(match.group(0))

    elif language == ReturnFormat.JSON:
        json_match = re.search(r"\[.*\]", input_str, re.DOTALL)
        if json_match:
            input_str = json_match.group(0)
        return parse_json_function_call(input_str)

    else:
        raise NotImplementedError(f"Unsupported language: {language}")


def parse_json_function_call(source_code):
    json_match = re.search(r"\[\s*{.*?}\s*(?:,\s*{.*?}\s*)*\]", source_code, re.DOTALL)
    if json_match:
        source_code = json_match.group(0)

    try:
        json_dict = json.loads(source_code)
    except json.JSONDecodeError as e:
        return []

    function_calls = []
    for function_call in json_dict:
        if isinstance(function_call, dict):
            function_name = function_call["function"]
            arguments = function_call["parameters"]
            function_calls.append({function_name: arguments})
    return function_calls


def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


    def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output



def decoded_output_to_execution_list(decoded_output: list[dict]) -> list[str]:
    """
    Convert decoded output to a list of executable function calls.

    Args:
        decoded_output (list): A list of dictionaries representing function calls.

    Returns:
        list: A list of strings, each representing an executable function call.
    """
    execution_list = []
    for function_call in decoded_output:
        for key, value in function_call.items():
            args_str = ", ".join(f"{k}={parse_nested_value(v)}" for k, v in value.items())
            execution_list.append(f"{key}({args_str})")
    return execution_list


def parse_nested_value(value):
    """
    Parse a potentially nested value from the AST output.

    Args:
        value: The value to parse, which could be a nested dictionary, which includes another function call, or a simple value.

    Returns:
        str: A string representation of the value, handling nested function calls and nested dictionary function arguments.
    """
    if isinstance(value, dict):
        # Check if the dictionary represents a function call (i.e., the value is another dictionary or complex structure)
        if all(isinstance(v, dict) for v in value.values()):
            func_name = list(value.keys())[0]
            args = value[func_name]
            args_str = ", ".join(f"{k}={parse_nested_value(v)}" for k, v in args.items())
            return f"{func_name}({args_str})"
        else:
            # If it's a simple dictionary, treat it as key-value pairs
            return (
                "{"
                + ", ".join(f"'{k}': {parse_nested_value(v)}" for k, v in value.items())
                + "}"
            )
    return repr(value)

    
def is_empty_execute_response(input_list: list):
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False


def process_method_calls(function_call_string: str, instance_mapping: dict) -> str:
    """
    Prepends the instance name to the function name for each of the function name represented in the string, you will
    also be provided with the mapping of method name to instance name.

    Example input:
    ```
    f(x = g((1, 2), h(3)), y = (4), z = (5, 6))
    ```

    Example return:
    ```
    a.f(x=a.g((1, 2), a.h(3)), y=(4), z=(5, 6))
    ```

    Args:
        function_call_string (str): The function call string to parse.
        class_mapping (dict): A dictionary mapping method names to instance names.

    Returns:
        str: The parsed function call string with instance names prepended to method names.
    """

    def replace_function(match):
        func_name = match.group(1)
        if func_name in instance_mapping:
            return f"{instance_mapping[func_name]}.{func_name}"
        return func_name

    # Regular expression to match function names
    pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"

    # Replace function names with their class-prepended versions
    processed_string = re.sub(pattern, replace_function, function_call_string)

    return processed_string


def load_file(file_path, sort_by_id=False, allow_concatenated_json=False):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            try:
                content = json.loads(line)
                result.append(content)
            except Exception as e:
                if not allow_concatenated_json:
                    raise e

                # Although this really shouldn't happen, sometimes a result file might have more than one JSON objects concatenated on a single line instead of one per line (e.g. '{"id": 1, xxx}{"id": 2, xxx}').
                # We can parse them incrementally by using `json.JSONDecoder.raw_decode`, which returns both the parsed object and the index where it stopped parsing.
                line_jsons = []
                decoder = json.JSONDecoder()
                idx = 0
                while idx < len(line):
                    # Skip whitespace between objects (if any)
                    while idx < len(line) and line[idx].isspace():
                        idx += 1

                    if idx >= len(line):
                        break

                    try:
                        json_obj, idx = decoder.raw_decode(line, idx)
                        line_jsons.append(json_obj)
                    except json.JSONDecodeError:
                        # If decoding fails at any point, the entire line is invalid.
                        raise e

                # After parsing, we must ensure the entire line has been consumed.
                # If `idx` is not at the end of the line, it means there's trailing
                # garbage, which is an error.
                if idx < len(line):
                    raise e

                if not line_jsons:
                    # If the line was non-empty but contained no JSON objects (e.g., only whitespace),
                    # it's an error.
                    raise e

                result.extend(line_jsons)

    if sort_by_id:
        result.sort(key=sort_key)
    return result


def sort_key(entry):
    """
    Index comes in two forms: TestCategory_Index or TestCategory_Index-FuncDocSubIndex-PromptSubIndex; both 0-indexed.

    TestCategory_Index: For example, `simple_20` means the 21st entry in the `simple` test category.

    TestCategory_Index-FuncDocSubIndex-PromptSubIndex is used when there are multiple prompts for a single function doc; this only happens in the live dataset.
    FuncDocSubIndex increments for each unique function doc.
    PromptSubIndex is per function doc. It resets to 0 for each function doc.
        For example, `live_simple_19-3-15` means the 20th entry in the `live_simple` test category.
        This entry has the 4th unique function doc and the 16th prompt for that function doc (there are at least 15 other prompts for this same function doc in this category).

    In either case, the universal index is enough to sort the entries.
    """
    entry_id = entry["id"].split(":")[0]
    parts = entry_id.rsplit("_", 1)
    test_category, index = parts[0], parts[1]
    # This handles the case where the index is in the form TestCategory_Index-FuncDocSubIndex-PromptSubIndex
    if "-" in index:
        assert index.count("-") == 2, f"Invalid index format: {index}"
        index = index.split("-")[0]

    # Make sure the memory prereq entries are inferenced first to avoid the memory entries being blocked due to dependencies.

    if is_multi_turn(test_category):
        priority = 3

    return (priority, test_category, int(index))