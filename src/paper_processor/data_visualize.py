from collections import defaultdict, Counter
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "docs/extracted_info/all_papers_structured.json"

# Supports dot-notation for nested fields, including array-of-objects subfields.
# Examples:
#   "proposed_method_detail.model_family"
#   "proposed_method_detail.contributions.improvement"
#   "experiment_setup.datasets.datasets_name"
#   "experiment_setup.compared_methods.name"
#   "experiment_setup.evaluation_metrics.canonical_name"
#   "fusion_modalities"
FIELD       = "experiment_setup.evaluation_metrics.canonical_name"

# Top-level scalar field to group bars by, or None for no grouping.
# Example: "year"
GROUP_BY    = None

TOP_N       = 10    # Limit to top N values, or None for all

OUTPUT_FILE = (
    f"{FIELD.replace('.', '_')}_by_{GROUP_BY}.png"
    if GROUP_BY else
    f"{FIELD.replace('.', '_')}_counts.png"
)
# ───────────────────────────────────────────────────────────────────────────────


def get_nested(obj: dict, keys: list[str]) -> any:
    """
    Traverse a nested dict/list structure by a sequence of keys.
    Automatically fans out over lists of dicts at any level.
    Returns a flat list of all terminal values found.
    """
    if obj is None:
        return []

    # Base case: no more keys to traverse
    if not keys:
        if isinstance(obj, list):
            return obj
        return [obj]

    key, rest = keys[0], keys[1:]

    # obj is a list → recurse into each element
    if isinstance(obj, list):
        results = []
        for item in obj:
            results.extend(get_nested(item, keys))
        return results

    # obj is a dict → descend into the next key
    if isinstance(obj, dict):
        child = obj.get(key)
        if child is None:
            return []
        return get_nested(child, rest)

    return []


def extract_values(entry: dict, field: str) -> list[str]:
    """
    Extract a flat list of non-empty string values for a dot-notation field path.
    The path is split on '.' and resolved from the top-level entry dict.
    Top-level keys (e.g. 'year', 'title') are looked up in entry directly;
    everything else is looked up inside entry (which may itself be nested).
    """
    keys = field.split(".")
    raw = get_nested(entry, keys)

    flat = []
    for v in raw:
        if isinstance(v, list):
            flat.extend(v)
        else:
            flat.append(v)

    return [str(v) for v in flat if v not in (None, "", "None", "null")]


# ── Load data ──────────────────────────────────────────────────────────────────
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ── Aggregate ──────────────────────────────────────────────────────────────────
if GROUP_BY:
    group_value: dict[str, Counter] = defaultdict(Counter)
    for entry in data:
        group = entry.get(GROUP_BY)
        if group is None:
            # Also check one level deep (e.g. entry["metadata"]["year"])
            for sub in entry.values():
                if isinstance(sub, dict):
                    group = sub.get(GROUP_BY)
                    if group is not None:
                        break
        if not group:
            continue
        for v in extract_values(entry, FIELD):
            group_value[str(group)][v] += 1

    groups = sorted(group_value.keys())
    total_counts: Counter = Counter()
    for c in group_value.values():
        total_counts.update(c)
    all_values = [v for v, _ in total_counts.most_common(TOP_N)]

else:
    total_counts: Counter = Counter()
    for entry in data:
        for v in extract_values(entry, FIELD):
            total_counts[v] += 1
    all_values = [v for v, _ in total_counts.most_common(TOP_N)]

if not all_values:
    print(f"No data found for field '{FIELD}'.")
    exit()

# ── Plot ───────────────────────────────────────────────────────────────────────
cmap = plt.get_cmap("tab20")

if GROUP_BY:
    # Grouped bar chart: x-axis = groups, one bar per FIELD value
    matrix = np.array([[group_value[g].get(v, 0) for g in groups] for v in all_values])

    x = np.arange(len(groups))
    n = len(all_values)
    bar_width = 0.8 / n
    colors = [cmap(i / max(n, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.2), 6))

    for i, (val, row) in enumerate(zip(all_values, matrix)):
        offsets = x - 0.4 + (i + 0.5) * bar_width
        ax.bar(offsets, row, width=bar_width, label=val, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_xlabel(GROUP_BY.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Paper count", fontsize=12)
    ax.legend(
        title=FIELD.split(".")[-1].replace("_", " ").title(),
        bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9
    )

else:
    # Single horizontal bar chart
    values_sorted = all_values[::-1]   # most common at top
    counts_sorted = [total_counts[v] for v in values_sorted]
    colors = [cmap(i / max(len(values_sorted), 1)) for i in range(len(values_sorted))]

    fig, ax = plt.subplots(figsize=(8, max(4, len(values_sorted) * 0.4 + 1)))

    y = np.arange(len(values_sorted))
    bars = ax.barh(y, counts_sorted, color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(values_sorted, fontsize=11)
    ax.set_xlabel("Paper count", fontsize=12)
    ax.bar_label(bars, padding=4, fontsize=10)

ax.set_title(
    f"{FIELD.replace('.', ' › ').replace('_', ' ').title()}"
    + (f"  ·  by {GROUP_BY.replace('_', ' ').title()}" if GROUP_BY else "  ·  all years"),
    fontsize=14, fontweight="bold"
)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x" if not GROUP_BY else "y", linestyle="--", alpha=0.4)

plt.tight_layout()
# plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {OUTPUT_FILE}")