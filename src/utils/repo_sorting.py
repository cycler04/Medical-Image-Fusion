import json

# Load JSON file
with open("runability_report_ref.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter repos with checkpoints
filtered = [
    repo for repo in data
    if repo.get("has_checkpoint", False)
]

# Prioritize repos with pretrained links
def has_pretrained_link(repo):
    linked = repo.get("linked_url", [])
    
    keywords = [
        "pretrained",
        "pre-trained"
        "checkpoint",
        "weights",
        "model"
    ]
    
    for item in linked:
        desc = item.get("description", "").lower()
        if any(k in desc for k in keywords):
            return True
    
    return False

# Sort:
# 1. repos WITH pretrained links first
# 2. lower overall_ease_score first
filtered_sorted = sorted(
    filtered,
    key=lambda x: (
        not has_pretrained_link(x),   # False first => has pretrained first
        -x.get("overall_ease_score", 0)
    )
)

# Print result
for repo in filtered_sorted:
    print("=" * 80)
    print("Repo:", repo.get("repo_name"))
    print("Ease Score:", repo.get("overall_ease_score"))
    print("Has Pretrained:", has_pretrained_link(repo))
    
    pretrained_links = [
        item["url"]
        for item in repo.get("linked_url", [])
        if any(
            k in item.get("description", "").lower()
            for k in ["pretrained", "checkpoint", "weights", "model"]
        )
    ]
    datasets = [
        repo["dataset_used"]
    ]
    
    if pretrained_links:
        print("Pretrained Links:")
        for link in pretrained_links:
            print(" -", link)
    if datasets:
        print("Dataset Used:")
        for dataset in datasets:
            print(" -", dataset)