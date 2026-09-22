import json
from metrics_fix import fix_all_metrics_in_json
from dataset_fix import fix_all_dataset_in_json
from json_data_fix import change_data

with open("docs/extracted_info/all_papers_structured_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)
fixed_data = fix_all_metrics_in_json(data)
fixed_data = fix_all_dataset_in_json(fixed_data)
fixed_data = change_data(fixed_data)
with open("docs/extracted_info/all_papers_structured.json", "w", encoding="utf-8") as f:
    json.dump(fixed_data, f, indent=2, ensure_ascii=False)