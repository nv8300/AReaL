import asyncio
import os
import uuid
import time
import re
import importlib
import copy
import inspect
import json

import aiofiles
import aiofiles.os
import colorama
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.bfcl.base_handler import pre_query_processing_prompting
from areal.bfcl.constants.default_prompts import format_prompt
from examples.utils import ast_parse, decoded_output_to_execution_list, is_empty_execute_response, process_method_calls, load_file
from areal.bfcl.constants.enums import ReturnFormat

from areal.bfcl.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES
from areal.bfcl.constants.category_mapping import POSSIBLE_ANSWER_PATH, VERSION_PREFIX


logger = logging.getLogger("Multi-Turn workflow")


class MultiTurnWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_steps: int,
        turn_discount: float,
        model_name: str = "qwen",
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        test_category: str = "multi_turn_base",
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.turn_discount = turn_discount
        self.rollout_stat_scope = rollout_stat_scope
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
        self.model_name_underline_replaced = (
            model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        )
        self.model_name = model_name
        self.possible_answer_dict = load_ground_truth_entry(test_category)
        self.test_category = test_category
        #logger.info("finish MultiTurnWorkflow init")

        # Create tokens that should be amended if the answer is incorrect.
        # This method eliminates the encode-decode inconsistency issue and cancels system prompts.
        # messages = [{"role": "assistant", "content": "some random message."}]
        # s1 = self.tokenizer.apply_chat_template(messages, tokenize=True)
        # messages += [
        #     {
        #         "role": "user",
        #         "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
        #         "Please carefully read the original question, check the preivous errors, and try to answer it again.",
        #     }
        # ]
        # s2 = self.tokenizer.apply_chat_template(
        #     messages, tokenize=True, add_generation_prompt=True
        # )
        # self.multi_turn_prompt_ids = s2[len(s1) :]

    async def _run_one_episode(self, engine: InferenceEngine, test_entry, rid):
        #logger.info("!!! start _run_one_episode")
        # Enforces `n_samples=1`
        # Placeholders for the results
        seq, logprobs, loss_mask, versions = [], [], [], []

        initial_config: dict = test_entry.get("initial_config", {})
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]
        assert test_category == self.test_category, (
                test_category,
                self.test_category,
            )

        #logger.info(f"finish test_entry get: {test_entry_id}")

        total_input_token_count: list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_model_result_list_decoded: list[list[float]] = []
        total_latency: list[list[float]] = []
        # The model response that will be used for later evaluation
        all_model_response: list[list] = []
        # The debugging log for human to understand
        all_inference_log: list[list[dict]] = []
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed.

        #logger.info("finish total list init")


        inference_data: dict = pre_query_processing_prompting(test_entry)
        all_multi_turn_messages: list[list[dict]] = test_entry["question"]
        #logger.info("finish pre_query_processing_prompting")

        # Multi-Turn multi-step call: calculate reward for a multi-turn case, discount accumulate within a turn
        all_count = 0
        discount = 1
        #logger.info(f'before inference_data: {inference_data["message"]}\tturn_name: {test_entry_id}')
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            #logger.info(f"!!! start turn_name: {test_entry_id}\t turn_idx: {turn_idx}")
            current_turn_message: list[dict]
            if turn_idx == 0:
                inference_data = add_first_turn_message_prompting(
                    inference_data, current_turn_message
                )
            else:
                inference_data = add_next_turn_user_message_prompting(
                    inference_data, current_turn_message
                )
            #logger.info(f"!!! turn_name: {test_entry_id}\t turn_idx: {turn_idx}, finish add_turn_user_message_prompting")

            current_turn_response = []
            current_turn_inference_log: list[dict] = {
                "begin_of_turn_query": current_turn_message
            }
            current_turn_input_token_count: list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_model_result_list_decoded: list[float] = []
            current_turn_latency: list[float] = []
            count = 0
            #logger.info(f'middle inference_data: {inference_data["message"]}\tturn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}')

            while True:
                logger.info(f"!!! start step turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}")
                print("-" * 100)
                print(
                    f"ID: {test_entry_id.replace('multi_turn_', '')}, Turn: {turn_idx}, Step: {count}"
                )
                current_step_inference_log: list[dict] = []
                # Add to the current_turn_inference_log at beginning of each step so that we don't need to bother dealing with the break statements
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log

                #logger.info(f'inference_data: {inference_data["message"]}\tturn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}')
                formatted_prompt = query_prompting(inference_data)
                #logger.info(f"!!! formatted_prompt for turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}\n formatted_prompt: {formatted_prompt}\n")
                #logger.info(f"!!! formatted_prompt for turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}\n formatted_prompt: {formatted_prompt}\n")
                input_ids = self.tokenizer.encode(formatted_prompt)

                start_time = time.time()
                req = ModelRequest(
                    rid=rid,
                    input_ids=input_ids,
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                )
                resp = await engine.agenerate(req)
                end_time = time.time()
                query_latency = end_time - start_time
                if len(input_ids) > 15000:
                    logger.info(f"**** get req for turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}\t input_id_len: {len(input_ids)}")


                prompt_str = self.tokenizer.decode(input_ids)
                model_responses = self.tokenizer.decode(resp.output_tokens[:-1])
                #logger.info(f"!!! get model_responses for turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}\n  model_responses: {model_responses}\n")

                model_response_data = {
                    "model_responses": model_responses,
                    "input_token": input_ids,
                    "output_token": resp.output_tokens,
                }


                # Add the assistant message to the chat history
                inference_data = add_assistant_message_prompting(
                    inference_data, model_response_data
                )
                #logger.info(f"add_assistant_message_prompting for turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}")
                # Process the metadata
                current_turn_input_token_count.append(input_ids)
                current_turn_output_token_count.append(resp.output_tokens)
                current_turn_latency.append(query_latency)
                current_turn_response.append(model_responses)
                log_entry = {
                    "role": "assistant",
                    "content": model_responses,
                }
                current_step_inference_log.append(log_entry)

                #logger.info(f"current_step_inference_log for turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}")

                # Amend results
                try:
                    input_len = len(resp.input_tokens) - len(seq)  # 新增加的输入token数量（当前响应相比之前新增的部分）
                    seq_decode = self.tokenizer.decode(seq)
                    input_decode = self.tokenizer.decode(resp.input_tokens[:-input_len])
                    assert len(seq) == 0 or len(resp.input_tokens[:-input_len]) == len(seq), (
                        seq_decode,
                        input_decode,
                        len(seq),
                        len(resp.input_tokens[:-input_len]),
                    )
                    seq += resp.input_tokens[-input_len:] + resp.output_tokens
                    logprobs += [0.0] * input_len + resp.output_logprobs
                    loss_mask += [0] * input_len + [1] * resp.output_len
                    versions += [-1] * input_len + resp.output_versions
                    discount *= self.turn_discount
                    #logger.info(f"!!!seq, logprobs, versions, discount process: {discount}  for turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}")

                # Try decoding the model response
                # try:
                    decoded_model_responses = default_decode_execute_prompting(model_responses, has_tool_call_tag=False)
                    #logger.info(f"decoded_model_responses for turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}")
                    current_turn_model_result_list_decoded.append(decoded_model_responses)
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )
                    model_response_data["model_responses_decoded"] = decoded_model_responses

                    if is_empty_execute_response(decoded_model_responses):
                        #logger.info(f"is_empty_execute_response TRUE for turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}")
                        print("Empty response from the model. Proceed to next turn.")
                        current_step_inference_log.append(
                            {
                                "role": "handler_log",
                                "content": f"Empty response from the model. Proceed to next turn.",
                                "model_response_decoded": decoded_model_responses,
                            }
                        )
                        break
                except Exception as e:
                    print("Failed to decode the model response. Proceed to next turn.")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Error decoding the model response. Proceed to next turn.",
                            "error": str(e),
                        }
                    )
                    break
                
                # Obtain the execution results
                #logger.info(f"!!! start execute_multi_turn_func_call for turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}")
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name_underline_replaced,
                    test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )
                #logger.info(f"!!! finish execute_multi_turn_func_call for turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}")

                # Add the execution results to the chat history for the next turn
                inference_data = add_execution_results_prompting(
                    inference_data, execution_results, model_response_data
                )
                #logger.info(f"add_execution_results_prompting for turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}")

                for execution_result in execution_results:
                    current_step_inference_log.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                        }
                    )
                
                                    
                count += 1
                # Force quit after too many steps
                #logger.info(f"finish for this step turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}")
                if count > self.max_steps:
                    force_quit = True
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {self.max_steps} steps.",
                        }
                    )
                    #logger.info(f"count > self.max_steps: {test_entry_id}\t turn_idx: {turn_idx}\t step: {count}")
                    break
                
            all_count += count
            # Add to the total list
            all_model_response.append(current_turn_response)
            all_inference_log.append(current_turn_inference_log)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_model_result_list_decoded.append(current_turn_model_result_list_decoded)
            total_latency.append(current_turn_latency)
            #logger.info(f"finish for this turn turn_name: {test_entry_id}\t turn_idx: {turn_idx}\t all_step: {all_count}")

            if force_quit:
                break

        #logger.info(f"multi_turn_ground_truth_list, test_entry_id: {test_entry_id}")
        multi_turn_ground_truth_list = self.possible_answer_dict[test_entry_id]["ground_truth"]
        raw_reward = await self.async_reward_fn(
                total_model_result_list_decoded,
                multi_turn_ground_truth_list, 
                test_entry, 
                test_category, 
                self.model_name 
            )

        #reward = float(raw_reward * discount)
        reward = float(raw_reward)
        
        logger.info(f"!!!! finish reward caculate test_entry_id: {test_entry_id}\tturn_idx:{turn_idx}\traw_reward: {raw_reward}\treward: {reward}\ttotal_step: {all_count}")
        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward, num_turns=turn_idx, num_steps=all_count)

        res = dict(
            input_ids=torch.tensor(seq),
            logprobs=torch.tensor(logprobs),
            loss_mask=torch.tensor(loss_mask),
            versions=torch.tensor(versions),
            rewards=torch.tensor(float(reward)),
            attention_mask=torch.ones(len(seq), dtype=torch.bool),
        )
        res = {k: v.unsqueeze(0) for k, v in res.items()}
        #logger.info(f"!!!! before return test_entry_id: {test_entry_id}")
        return (
            TensorDict(res, batch_size=[1]),
            formatted_prompt,
            model_responses,
            reward,
            len(seq),
        )
        

    async def arun_episode(self, engine: InferenceEngine, data):
        #logger.info("*** start arun_episode")
        rid = uuid.uuid4().hex
        tasks = [
            self._run_one_episode(engine, data, rid)
            for _ in range(self.gconfig.n_samples)
        ]
        results = await asyncio.gather(*tasks)
        #logger.info("*** get results in arun_episode")

        if self.dump_dir is not None:
            version = engine.get_version()
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            # Get the unique identifier for this prompt
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"{qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                n_samples = self.gconfig.n_samples
                for i, (_, p, c, r, sl) in enumerate(results):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {sl}, reward is {r}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    await f.write(info + "\n")

        #logger.info("finish Dump rollout to file in arun_episode")
        data = [res[0] for res in results]
        #logger.info("before return in arun_episode")
        return concat_padded_tensors(data)


def add_first_turn_message_prompting(
    inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

def add_next_turn_user_message_prompting(
    inference_data: dict, user_message: list[dict]
) -> dict:
    inference_data["message"].extend(user_message)
    return inference_data


def query_prompting(inference_data: dict):
    # core inference func
    function: list[dict] = inference_data["function"]
    message: list[dict] = inference_data["message"]

    formatted_prompt: str = format_prompt(message, function)
    inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}
    return formatted_prompt

def add_assistant_message_prompting(inference_data: dict, model_response_data: dict) -> dict:
    inference_data["message"].append(
        {"role": "assistant", "content": model_response_data["model_responses"]}
    )
    return inference_data


def default_decode_execute_prompting(
    result: str, has_tool_call_tag: bool = False
) -> list[str]:
    # Note: For execute, there are only Python entries, so we don't need to check the language.
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    decoded_output = ast_parse(
        result, language=ReturnFormat.PYTHON, has_tool_call_tag=has_tool_call_tag
    )
    return decoded_output_to_execution_list(decoded_output)


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
    TODO: Add docstring
    """
    if is_evaL_run:
        model_name += "_eval"

    class_method_name_mapping = {}
    involved_instances = {}
    for class_name in involved_classes:
        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        # TODO: Handler the model name issue from handler more elegantly
        instance_name = (
            f"{model_name}_{test_entry_id}_{class_name}_instance"
        )
        instance_name = re.sub(r'[-./]', '_', instance_name)
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
        func_call = process_method_calls(func_call, class_method_name_mapping)

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


def add_execution_results_prompting(inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
    for execution_result, decoded_model_response in zip(
        execution_results, model_response_data["model_responses_decoded"]
    ):
        inference_data["message"].append(
            {
                "role": "tool",
                "name": decoded_model_response,
                "content": execution_result,
            }
        )

    return inference_data


def load_ground_truth_entry(test_category: str) -> dict:
    """
    This function retrieves the ground truth entry for a given test category.
    The input should not be a test category goup, but a specific test category.
    """
    ground_truth_list = load_file(POSSIBLE_ANSWER_PATH / f"{VERSION_PREFIX}_{test_category}.json")
    ground_truth_dict = {item["id"]: item for item in ground_truth_list}
    return ground_truth_dict
