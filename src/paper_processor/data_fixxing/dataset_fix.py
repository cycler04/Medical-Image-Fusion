import json
import re
from ftfy import fix_text

DATASET_ALIASES = {
    # Generic Modality Pairings often used as dataset names in literature, Assuming from Harvard Whole Brain Atlas
    "Harvard Whole Brain Atlas": [
        "CT-MRI",
        "MRI-CT",
        "Multimodal medical image datasets (CT-MRI)",
        "PET-MRI",
        "MRI-PET",
        "Multimodal medical image datasets (MRI-PET)",
        "PET/MRI database",
        "SPECT-MRI",
        "MRI-SPECT"
    ],
    
    "MSRS": [
        "MSRS",
        "MSRS dataset",
        "MSRS Dataset",
        "MSRS (Multi-Spectral Red-Visible Image Fusion Dataset)"
    ],

    "TNO": [
        "TNO",
        "TNO Dataset"
    ],

    "M3FD": [
        "M3FD",
        "MÂ³FD",
        "M3FD Dataset"
    ],

    "BraTS": [
        "BraTS dataset",
        "BRATS",
        "BraTS 2018",
        "BraTS 2019",
        "BRATS2019",
        "BraTS 2020",
        "BraTS 2021",
        "BraTS-TCGA-LGG"
    ],

    "LLVIP": [
        "LLVIP",
        "LLIVP"  # Catching the typo from your list
    ],

    "AWMM-100K": [
        "AWMM-100K",
        "AWMM-100k"
    ],

    "Lytro": [
        "Lytro",
        "LYTRO"
    ],

    "CIC-IDS": [
        "CIC-IDS-2017",
        "CIC-IDS 2017"
    ],

    "KDD": [
        "NSL-KDD",
        "KDD Cup 1999"
    ],

    "SEED": [
        "SEED",
        "SEED-IV"
    ],

    "FINRISK": [
        "FINRISK1997",
        "FINRISK2002"
    ],

    "GFP-PC": [
        "GFP-PC",
        "GFP Dataset",
        "GFP/PC dataset",
        "publicly available benchmark dataset (GFP-PC)"
    ],

    "Chinese Abdominal Multimodal Medical Image Dataset": [
        "Chinese abdominal multimodal medical image dataset",
        "abdominal multi-modal medical image dataset from China",
        "Ab-MI (Multimodal Abdominal Medical Images)"
    ],

    "Rocket": [
        "Rocket-1",
        "Rocket-2"
    ],


}
# ---------------------------------------------------
# Build reverse lookup table
# ---------------------------------------------------

def normalize_text(text):
    text = fix_text(text)

    text = text.lower()

    # remove (...) acronym parts
    text = re.sub(r"\([^)]*\)", "", text)

    # remove separators
    text = re.sub(r"[_\-/–^]", "", text)

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # remove spaces
    text = re.sub(r"\s+", "", text)

    return text.strip()


LOOKUP = {}

for canonical, aliases in DATASET_ALIASES.items():
    for alias in aliases:
        LOOKUP[normalize_text(alias)] = canonical


# ---------------------------------------------------
# Normalize one dataset
# ---------------------------------------------------

def normalize_dataset(dataset_name):
    normalized = normalize_text(dataset_name)

    return LOOKUP.get(normalized, dataset_name)
# ---------------------------------------------------
# Fix dataset entries inside your JSON
# ---------------------------------------------------

def fix_dataset(data):
    if "metadata" not in data or "datasets" not in data["metadata"]:
        return data
    datasets = data["metadata"]["datasets"]

    fixed_datasets = []

    for dataset in datasets:
        if 'harvard' in dataset.lower() or 'whole brain' in dataset.lower():
            dataset = "Harvard Whole Brain Atlas"
        dataset = normalize_dataset(dataset)
        if dataset not in fixed_datasets:
            fixed_datasets.append(dataset)

    data["metadata"]["datasets"] = fixed_datasets

    return data

def fix_all_dataset_in_json(json_data):
    for entry in json_data:
        fix_dataset(entry)
    return json_data

# ---------------------------------------------------
# Example
# ---------------------------------------------------
if __name__ == "__main__":
    with open("data/research_paper/extracted_info/all_papers_structured.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    fixed_data = fix_all_dataset_in_json(data)
    with open("all_papers_structured.json", "w", encoding="utf-8") as f:
        json.dump(fixed_data, f)
