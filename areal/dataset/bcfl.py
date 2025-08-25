from datasets import Dataset
from datasets.distributed import split_dataset_by_node

from examples.utils import parse_test_category_argument, load_file
from bfcl.constants.category_mapping import PROMPT_PATH, MULTI_TURN_FUNC_DOC_PATH, MULTI_TURN_FUNC_DOC_FILE_MAPPING


def get_bfcl_dataset(split: str, tokenizer, rank: int, world_size: int, max_length: Optional[int] = None):
    all_test_file_paths, all_test_categories, all_test_entries_involved, test_cases_total = get_involved_test_entries(test_categories="multi_turn")
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



# def get_gsm8k_rl_dataset(
#     path: str,
#     split: str,
#     tokenizer,
#     rank: int,
#     world_size: int,
#     max_length: Optional[int] = None,
# ):
#     dataset = load_dataset(path=path, name="main", split=split)

#     def process(sample):
#         messages = [
#             {
#                 "role": "user",
#                 "content": sample["question"]
#                 + "\nPlease put your final answer within \\boxed{}.",
#             }
#         ]
#         return {"messages": messages}

#     dataset = dataset.map(process).remove_columns(["question"])

#     # Filter out sequences longer than max_length if tokenizer and max_length are provided
#     if max_length is not None:

#         def filter_length(sample):
#             # Tokenize the user content to check length
#             content = sample["messages"][0]["content"]
#             tokens = tokenizer.encode(content)
#             return len(tokens) <= max_length

#         dataset = dataset.filter(filter_length)

#     dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
#     return dataset

if __name__ == '__main__':
    import os

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    # tokenizer = load_hf_tokenizer(config.tokenizer_path)
    tokenizer = None

    output_dataset = get_bfcl_dataset(rank=rank,
        world_size=world_size,
        split="train",
        type='rl',
        tokenizer=tokenizer)
    print(output_dataset)