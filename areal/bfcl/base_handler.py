import re
import json

from areal.bfcl.constants.default_prompts import DEFAULT_SYSTEM_PROMPT_FORMAT, PROMPT_TEMPLATE_MAPPING, PROMPT_STYLE_TEMPLATES, OUTPUT_FORMAT_MAPPING, PARAM_TYPE_MAPPING


def pre_query_processing_prompting(test_entry: dict) -> dict:
    functions: list = test_entry["function"]
    test_entry_id: str = test_entry["id"]

    test_entry["question"][0] = system_prompt_pre_processing_chat_model(
        test_entry["question"][0], functions, test_entry_id
    )

    return {"message": [], "function": functions}


def system_prompt_pre_processing_chat_model(
    prompts: list[dict], function_docs: list[dict], test_entry_id: str
) -> list[dict]:
    """
    Add a system prompt to the chat model to instruct the model on the available functions and the expected response format.
    If the prompts list already contains a system prompt, append the additional system prompt content to the existing system prompt.
    """
    assert type(prompts) == list

    prompt_format = extract_prompt_format_from_id(test_entry_id)

    system_prompt = formulate_system_prompt(
        format_sensitivity_config=prompt_format, functions=function_docs
    )

    # System prompt must be in the first position
    # If the question comes with a system prompt, append its content at the end of the chat template.
    if prompts[0]["role"] == "system":
        prompts[0]["content"] = system_prompt + "\n\n" + prompts[0]["content"]
    # Otherwise, use the system prompt template to create a new system prompt.
    else:
        prompts.insert(
            0,
            {"role": "system", "content": system_prompt},
        )

    return prompts


def extract_prompt_format_from_id(test_entry_id: str) -> str:
    """
    Extract the prompt format from the test entry ID.
    """
    if ":" not in test_entry_id:
        return DEFAULT_SYSTEM_PROMPT_FORMAT
    else:
        assert (
            len(test_entry_id.split(":")) == 3
        ), f"Test entry ID {test_entry_id} should contain exactly two colons, since they are supposed to be the format sensitivity ids."
        return test_entry_id.split(":")[1]

def formulate_system_prompt(format_sensitivity_config: str, functions: list[dict]) -> str:
    """
    Formulate the default system prompt based on the provided parameters.
    """
    (
        return_format,
        has_tool_call_tag,
        function_doc_format,
        prompt_format,
        prompt_style,
    ) = parse_prompt_variation_params(format_sensitivity_config)

    formatted_function_doc = format_function_doc(functions, function_doc_format)

    prompt_template = PROMPT_TEMPLATE_MAPPING[prompt_format]
    style_template = PROMPT_STYLE_TEMPLATES[prompt_style]

    persona = style_template["persona"]
    task = style_template["task"]
    if has_tool_call_tag:
        tool_call_format = style_template["tool_call_with_tag"].format(
            output_format=OUTPUT_FORMAT_MAPPING[return_format],
            param_types=PARAM_TYPE_MAPPING[return_format],
        )
    else:
        tool_call_format = style_template["tool_call_no_tag"].format(
            output_format=OUTPUT_FORMAT_MAPPING[return_format],
            param_types=PARAM_TYPE_MAPPING[return_format],
        )
    multiturn_behavior = style_template["multiturn_behavior"]
    available_tools = style_template["available_tools"].format(
        format=function_doc_format,
        functions=formatted_function_doc,
    )

    system_prompt = prompt_template.format(
        persona=persona,
        task=task,
        tool_call_format=tool_call_format,
        multiturn_behavior=multiturn_behavior,
        available_tools=available_tools,
    )

    return system_prompt


def parse_prompt_variation_params(input_str: str) -> tuple[str, bool, str, str, str]:
    """
    Parse a query string of the form:
      ret_fmt=…&tool_call_tag=…&func_doc_fmt=…&prompt_fmt=…&style=…

    Returns a 5-tuple containing, **in order**:
        1. return_format (str)
        2. has_tool_call_tag (bool)
        3. function_doc_format (str)
        4. prompt_format (str)
        5. prompt_style (str)

    Raises:
        ValueError: If the input string does not conform to the expected format.
    """
    _PATTERN = re.compile(
        r"^"
        r"ret_fmt=(?P<return_format>python|json|verbose_xml|concise_xml)"
        r"&tool_call_tag=(?P<has_tool_call_tag>True|False)"
        r"&func_doc_fmt=(?P<function_doc_format>python|xml|json)"
        r"&prompt_fmt=(?P<prompt_format>plaintext|markdown)"
        r"&style=(?P<prompt_style>classic|experimental)"
        r"$"
    )

    match = _PATTERN.match(input_str)
    if not match:
        raise ValueError(f"Invalid query format: {input_str!r}")

    # Extract named groups
    return_format = match.group("return_format")
    has_tool_call_tag = match.group("has_tool_call_tag") == "True"
    function_doc_format = match.group("function_doc_format")
    prompt_format = match.group("prompt_format")
    prompt_style = match.group("prompt_style")

    return (
        return_format,
        has_tool_call_tag,
        function_doc_format,
        prompt_format,
        prompt_style,
    )


def format_function_doc(functions: list[dict], function_doc_format: str) -> str:
    """
    Format the function documentation based on the specified format.
    """

    if function_doc_format == "json":
        functions = json.dumps(functions, indent=4)

    else:
        raise ValueError(f"Invalid function doc format: {function_doc_format}")

    return functions


