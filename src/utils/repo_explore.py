import json
from datetime import datetime

INPUT_JSON = "ref_method_data_with_repo_metadata.json"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# -----------------------------
# Groups
# -----------------------------
pre_2021 = []
from_2021_to_2023 = []
from_2024_plus = []
unknown = []

# -----------------------------
# Process repos
# -----------------------------
for paper in data:

    repo_meta = paper.get("repo_metadata", {})

    created_at = repo_meta.get("created_at")

    repo_name = (
        repo_meta.get("full_name")
        or paper.get("stem")
        or "UNKNOWN"
    )

    if not created_at:
        unknown.append(repo_name)
        continue

    try:
        year = datetime.fromisoformat(
            created_at.replace("Z", "+00:00")
        ).year

        if year < 2021:
            pre_2021.append((repo_name, year))

        elif 2021 <= year < 2024:
            from_2021_to_2023.append((repo_name, year))

        else:
            from_2024_plus.append((repo_name, year))

    except Exception:
        unknown.append(repo_name)

# -----------------------------
# Sort by year
# -----------------------------
pre_2021.sort(key=lambda x: x[1])
from_2021_to_2023.sort(key=lambda x: x[1])
from_2024_plus.sort(key=lambda x: x[1])

# -----------------------------
# Print helper
# -----------------------------
def print_group(title, items):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for repo_name, year in items:
        print(f"{year}  |  {repo_name}")

    print(f"\nTotal: {len(items)}")


# -----------------------------
# Print results
# -----------------------------
print_group("PRE 2021 REPOS", pre_2021)

print_group("2021 - 2023 REPOS", from_2021_to_2023)

print_group("2024+ REPOS", from_2024_plus)

# -----------------------------
# Unknown dates
# -----------------------------
print("\n" + "=" * 60)
print("UNKNOWN / MISSING DATES")
print("=" * 60)

for repo_name in unknown:
    print(repo_name)

print(f"\nTotal unknown: {len(unknown)}")