import json

ft_data = {}

with open('tests/test_result/hypothesis.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    for item in data:
        ft_data[item["question"]] = {
            "replace": [],
            "hypothesis": [item["ans"]],
            "ground_truth": ""
        }

with open('data/Hypothesis/questions.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    for item in data:
        ft_data[item["q"]]["replace"].extend(item["replace"])
        ft_data[item["q"]]["ground_truth"] = item["ground_truth"]

with open('sentence-transformer/data/v1.0.json', 'w', encoding='utf-8') as json_file:
    json.dump(ft_data, json_file, ensure_ascii=False, indent=4)