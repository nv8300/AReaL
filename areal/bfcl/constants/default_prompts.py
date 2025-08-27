MAXIMUM_STEP_LIMIT = 20

DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.
"""

DEFAULT_SYSTEM_PROMPT = (
    DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC
    + """
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""
)

# This is the default system prompt format
DEFAULT_SYSTEM_PROMPT_FORMAT = "ret_fmt=python&tool_call_tag=False&func_doc_fmt=json&prompt_fmt=plaintext&style=classic"

PROMPT_TEMPLATE_MAPPING = {
    "plaintext": _PLAINTEXT_SYSTEM_PROMPT_TEMPLATE,
    "markdown": _MARKDOWN_SYSTEM_PROMPT_TEMPLATE,
}

_PLAINTEXT_SYSTEM_PROMPT_TEMPLATE = (
    "{persona}{task}\n\n{tool_call_format}\n\n{multiturn_behavior}\n\n{available_tools}"
)
_MARKDOWN_SYSTEM_PROMPT_TEMPLATE = "{persona}\n\n## Task\n{task}\n\n## Tool Call Format\n{tool_call_format}\n\n## Multi-turn Behavior\n{multiturn_behavior}\n\n## Available Tools\n{available_tools}"



PROMPT_STYLE_TEMPLATES = {
    "classic": {
        "persona": "You are an expert in composing functions.",
        "task": "You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.",
        "tool_call_no_tag": "You should only return the function calls in your response.\n\nIf you decide to invoke any of the function(s), you MUST put it in the format of {output_format}. {param_types} You SHOULD NOT include any other text in the response.",
        "tool_call_with_tag": "You should only return the function calls in the <TOOLCALL> section. If you decide to invoke any of the function(s), you MUST put it in the format of <TOOLCALL>{output_format}</TOOLCALL>. {param_types} You SHOULD NOT include any other text in the response.",
        "multiturn_behavior": "At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.",
        "available_tools": "Here is a list of functions in {format} format that you can invoke.\n{functions}\n",
    },
    "experimental": {
        "persona": "You are an expert in generating structured function calls.",
        "task": "You are given a user query and a set of available functions. Your task is to produce one or more function/tool calls to fulfill the user's request. If no suitable function exists, or required parameters are missing, clearly indicate this.",
        "tool_call_no_tag": "Respond with only the function calls.\n\nYou MUST format it exactly as {output_format}. {param_types} Do NOT include any other text.",
        "tool_call_with_tag": "Return only the function calls enclosed in <TOOLCALL> tags.\n\nYou MUST format it exactly as <TOOLCALL>{output_format}</TOOLCALL>. {param_types} Do NOT include any other text.",
        "multiturn_behavior": "At every turn, aim to complete the user's tasks within that turn. Continue emitting function calls until the request is satisfied to the best of your ability. Once no more calls are needed, the system will proceed to the next turn.",
        "available_tools": "Below is a list of callable functions in the {format} style:\n{functions}\n",
    },
}


OUTPUT_FORMAT_MAPPING = {
    "python": "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]",
    "json": '```json\n[{"function":"func_name1","parameters":{"param1":"value1","param2":"value2"...}},{"function":"func_name2","parameters":{"param":"value"}}]\n```',
    "verbose_xml": '<functions><function name="func_name1"><params><param name="param1" value="value1" type="type1"/><param name="param2" value="value2" type="type2"/>...</params></function><function name="func_name2"><param name="param3" value="value3" type="type3"/></function></functions>',
    "concise_xml": '<functions><function name="func_name1"><param name="param1" type="type1">value1</param><param name="param2" type="type2">value2</param>...</function><function name="func_name2"><param name="param3" type="type3">value</param></function></functions>',
}

PARAM_TYPE_MAPPING = {
    "python": "",
    "json": "",
    "verbose_xml": "The type fields of the parameters in your function calls must be one of: string, integer, float, boolean, array, dict, or tuple.",
    "concise_xml": "The type fields of the parameters in your function calls must be one of: string, integer, float, boolean, array, dict, or tuple.",
}


def format_prompt(messages, function):
    """
    "chat_template":
    {%- if tools %}
        {{- '<|im_start|>system\n' }}
        {%- if messages[0].role == 'system' %}
            {{- messages[0].content + '\n\n' }}
        {%- endif %}
        {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
        {%- for tool in tools %}
            {{- "\n" }}
            {{- tool | tojson }}
        {%- endfor %}
        {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
    {%- else %}
        {%- if messages[0].role == 'system' %}
            {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
    {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
    {%- for message in messages[::-1] %}
        {%- set index = (messages|length - 1) - loop.index0 %}
        {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endfor %}
    {%- for message in messages %}
        {%- if message.content is string %}
            {%- set content = message.content %}
        {%- else %}
            {%- set content = '' %}
        {%- endif %}
        {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
            {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
        {%- elif message.role == "assistant" %}
            {%- set reasoning_content = '' %}
            {%- if message.reasoning_content is string %}
                {%- set reasoning_content = message.reasoning_content %}
            {%- else %}
                {%- if '</think>' in content %}
                    {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                    {%- set content = content.split('</think>')[-1].lstrip('\n') %}
                {%- endif %}
            {%- endif %}
            {%- if loop.index0 > ns.last_query_index %}
                {%- if loop.last or (not loop.last and reasoning_content) %}
                    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
                {%- else %}
                    {{- '<|im_start|>' + message.role + '\n' + content }}
                {%- endif %}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
            {%- if message.tool_calls %}
                {%- for tool_call in message.tool_calls %}
                    {%- if (loop.first and content) or (not loop.first) %}
                        {{- '\n' }}
                    {%- endif %}
                    {%- if tool_call.function %}
                        {%- set tool_call = tool_call.function %}
                    {%- endif %}
                    {{- '<tool_call>\n{"name": "' }}
                    {{- tool_call.name }}
                    {{- '", "arguments": ' }}
                    {%- if tool_call.arguments is string %}
                        {{- tool_call.arguments }}
                    {%- else %}
                        {{- tool_call.arguments | tojson }}
                    {%- endif %}
                    {{- '}\n</tool_call>' }}
                {%- endfor %}
            {%- endif %}
            {{- '<|im_end|>\n' }}
        {%- elif message.role == "tool" %}
            {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
                {{- '<|im_start|>user' }}
            {%- endif %}
            {{- '\n<tool_response>\n' }}
            {{- content }}
            {{- '\n</tool_response>' }}
            {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
                {{- '<|im_end|>\n' }}
            {%- endif %}
        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\n' }}
        {%- if enable_thinking is defined and enable_thinking is false %}
            {{- '<think>\n\n</think>\n\n' }}
        {%- endif %}
    {%- endif %}
    """
    formatted_prompt = ""

    if messages[0]["role"] == "system":
        formatted_prompt += f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"

    last_query_index = len(messages) - 1
    for offset, message in enumerate(reversed(messages)):
        idx = len(messages) - 1 - offset
        if (
            message["role"] == "user"
            and type(message["content"]) == str
            and not (
                message["content"].startswith("<tool_response>")
                and message["content"].endswith("</tool_response>")
            )
        ):
            last_query_index = idx
            break

    for idx, message in enumerate(messages):
        role = message["role"]
        content = message["content"]

        if role == "user" or (role == "system" and idx != 0):
            formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        elif role == "assistant":
            reasoning_content = ""
            if "reasoning_content" in message and message["reasoning_content"]:
                reasoning_content = message["reasoning_content"]

            elif "</think>" in content:
                parts = content.split("</think>")
                reasoning_content = (
                    parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                )
                content = parts[-1].lstrip("\n")

            if idx > last_query_index:
                if idx == len(messages) - 1 or reasoning_content:
                    formatted_prompt += (
                        f"<|im_start|>{role}\n<think>\n"
                        + reasoning_content.strip("\n")
                        + f"\n</think>\n\n"
                        + content.lstrip("\n")
                    )
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"
            else:
                formatted_prompt += f"<|im_start|>{role}\n{content}"

            formatted_prompt += "<|im_end|>\n"

        elif role == "tool":
            prev_role = messages[idx - 1]["role"] if idx > 0 else None
            next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

            if idx == 0 or prev_role != "tool":
                formatted_prompt += "<|im_start|>user"

            formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

            if idx == len(messages) - 1 or next_role != "tool":
                formatted_prompt += "<|im_end|>\n"

    formatted_prompt += "<|im_start|>assistant\n"
    return formatted_prompt
