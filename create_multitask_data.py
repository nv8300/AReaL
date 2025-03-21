import json
import random

data = []
with open("/storage/openpsi/data/code/apps/codeparrot-apps-test.jsonl", "r") as f:
    code_data = [json.loads(l) for l in f.readlines()]

original_keys = list(code_data[0].keys())
print(original_keys)
for d in code_data:
    print(d["starter_code"], type(d["starter_code"]))
    # print(json.loads(d["solutions"])[0])
    exit(0)
    d["query_id"] = d["id"]
    d["prompt"] = d["question"]
    d["task"] = "code"
    for k in original_keys:
        d.pop(k)
    data.append(d)

with open("/storage/openpsi/users/gjx/data/DeepScaleR/prompts.jsonl", "r") as f:
    math_data = [json.loads(l) for l in f.readlines()]

for d in math_data:
    data.append(dict(prompt=d["prompt"], task="math", query_id=d["query_id"]))

random.shuffle(data)
with open("/storage/openpsi/users/bowei.fw/data/code_math.jsonl", "w") as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
