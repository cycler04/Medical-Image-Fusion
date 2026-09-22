from collections import Counter
import json
from metrics_fix import fix_all_metrics_in_json
from dataset_fix import fix_all_dataset_in_json

with open("docs/extracted_info/all_papers_structured.json", "r", encoding="utf-8") as f:
    data = json.load(f)


field = "evaluation_metrics"
counter = Counter()

for entry in data:
    values = entry["metadata"].get(field, [])
    # if not entry["metadata"]["is_image_fusion"]:
    #     continue
    if isinstance(values, list):
        if values and isinstance(values[0], dict) and "canonical_name" in values[0]:
            for value in values:
                counter[value["canonical_name"]] += 1
        else:
            counter.update(values)
    else:
        counter[values] += 1

for value, count in counter.most_common():
    print(f"{count:>5}  {value}")

# with open("model_counts.txt", "w", encoding="utf-8") as f:
#      f.write(f"{'Count':>5}  Value\n")
#      f.write("-" * 30 + "\n")
#      for value, count in counter.most_common():
#          f.write(f"{count:>5}  {value}\n")

