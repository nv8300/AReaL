import json

from areal.bfcl.constants.category_mapping import TEST_COLLECTION_MAPPING, TEST_FILE_MAPPING


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