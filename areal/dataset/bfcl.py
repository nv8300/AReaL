from datasets import Dataset
from datasets.distributed import split_dataset_by_node

import sys
import os
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.utils import parse_test_category_argument, load_file
from areal.bfcl.constants.category_mapping import PROMPT_PATH, MULTI_TURN_FUNC_DOC_PATH, MULTI_TURN_FUNC_DOC_FILE_MAPPING


def get_bfcl_dataset(split: str, tokenizer, rank: int, world_size: int, max_length: Optional[int] = None):
    all_test_file_paths, all_test_categories, all_test_entries_involved, test_cases_total = get_involved_test_entries(test_categories="multi_turn")

    # check data format for Dataset
    issues, inconsistency_idx = check_type_consistency(test_cases_total)
    filtered_test_cases_total = [item for idx, item in enumerate(test_cases_total) if idx not in inconsistency_idx]
    
    dataset = Dataset.from_list(test_cases_total)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset



def get_involved_test_entries(test_categories="multi_turn"):
    all_test_file_paths, all_test_categories, all_test_entries_involved = [], [], []
    all_test_file_paths, all_test_categories = parse_test_category_argument(test_categories)
    # Make a copy here since we are removing list elemenets inside the for loop
    for test_category, file_to_open in zip(
        all_test_categories[:], all_test_file_paths[:]
    ):
        all_test_entries_involved.extend(load_file(PROMPT_PATH / file_to_open))
    
    test_cases_to_generate = process_multi_turn_test_case(all_test_entries_involved)
    test_cases_total = sorted(test_cases_to_generate, key=sort_key)

    return all_test_file_paths, all_test_categories, all_test_entries_involved, test_cases_total


# def collect_test_cases(
#     all_test_entries_involved
# ):
#     test_cases_to_generate = [
#         test_case
#         for test_case in all_test_entries_involved
#     ]
#     test_cases_to_generate = process_multi_turn_test_case(test_cases_to_generate)

#     return sorted(test_cases_to_generate, key=sort_key)


def process_multi_turn_test_case(test_cases):
    """
    Multi-turn test cases don't have the function doc in the prompt. We need to add them here.
    """
    for entry in test_cases:
        involved_classes = entry["involved_classes"]
        entry["function"] = []
        for func_collection in involved_classes:
            # func_doc is a list of dict
            func_doc = load_file(
                MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection]
            )
            entry["function"].extend(func_doc)

        # Handle Miss Func category; we need to remove the holdout function doc
        if "missed_function" in entry:
            for turn_index, missed_func_names in entry["missed_function"].items():
                entry["missed_function"][turn_index] = []
                for missed_func_name in missed_func_names:
                    for i, func_doc in enumerate(entry["function"]):
                        if func_doc["name"] == missed_func_name:
                            # Add the missed function doc to the missed_function list
                            entry["missed_function"][turn_index].append(func_doc)
                            # Remove it from the function list
                            entry["function"].pop(i)
                            break

    return test_cases


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
    parts = entry["id"].rsplit("_", 1)
    test_category, index = parts[0], parts[1]
    # This handles the case where the index is in the form TestCategory_Index-FuncDocSubIndex-PromptSubIndex
    if "-" in index:
        index = index.split("-")[0]
    return (test_category, int(index))



def check_type_consistency(data, path="root"):
    """
    Data type consistency check
    """
    
    issues = []
    inconsistency_idx = []
    
    if isinstance(data, list):
        if not data:
            return issues  # 空列表没有问题
        
        # 检查第一个元素的类型作为基准
        first_type = type(data[0])
        first_structure = get_structure(data[0])
        
        for i, item in enumerate(data):
            current_path = f"{path}[{i}]"
            
            # 检查类型是否一致
            if type(item) != first_type:
                issues.append(f"{current_path}: type inconsistency ({type(item).__name__} vs {first_type.__name__})")
                inconsistency_idx.append(i)
            
            # 如果是复杂类型，递归检查
            if isinstance(item, (list, dict)):
                current_structure = get_structure(item)
                if current_structure != first_structure:
                    issues.append(f"{current_path}: structure inconsistency")
                    inconsistency_idx.append(i)
                
                # 递归检查
                issues.extend(check_type_consistency(item, current_path))
    
    elif isinstance(data, dict):
        # 检查所有字典是否有相同的键
        keys = set(data.keys())
        
        for key, value in data.items():
            current_path = f"{path}.{key}"
            
            # 递归检查值
            if isinstance(value, (list, dict)):
                issues.extend(check_type_consistency(value, current_path))
    
    return issues, inconsistency_idx


def get_structure(obj):
    """
    Get the type structure information of the object
    """
    if isinstance(obj, list):
        if not obj:
            return "empty_list"
        return f"list[{get_structure(obj[0])}]"
    
    elif isinstance(obj, dict):
        if not obj:
            return "empty_dict"
        return f"dict[{sorted(obj.keys())}]"
    
    else:
        return type(obj).__name__


if __name__ == '__main__':
    import os

    # rank = int(os.getenv("RANK"))
    rank = 0
    # world_size = int(os.getenv("WORLD_SIZE"))
    world_size = 1
    # tokenizer = load_hf_tokenizer(config.tokenizer_path)
    tokenizer = None

    output_dataset = get_bfcl_dataset(rank=rank,
        world_size=world_size,
        split="train",
        tokenizer=tokenizer)
    print(output_dataset)
