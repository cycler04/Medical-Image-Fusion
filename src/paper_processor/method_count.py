import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict, Counter
import json
import math

DATA_PATH = "data/research_paper/usable_info/all_papers_structured.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

family_methods: dict[str, Counter] = defaultdict(Counter)

for entry in data:
    model_family = entry.get("proposed_method_detail", {}).get("model_family")
    if not model_family:
        continue
    for method in entry.get("experiment_setup", {}).get("compared_methods", []):
        name = method.get("name")
        if name:
            family_methods[model_family][name] += 1


def plot_families(
    family_filter: str | list[str] | None = None,
    top_n: int = 10,
    cols: int = 3,
):
    """
    Plot top compared methods per model family.

    Args:
        family_filter: None          ΓåÆ all families
                       "Mamba"       ΓåÆ single family (single chart)
                       ["Mamba","S4"] ΓåÆ specific families (grid)
        top_n:  How many top methods to show per family.
        cols:   Max columns in the grid (ignored for single family).
    """
    # --- Resolve which families to plot ---
    all_families = sorted(family_methods.keys())

    if family_filter is None:
        families = all_families
    elif isinstance(family_filter, str):
        families = [family_filter]
    else:
        families = [f for f in family_filter if f in family_methods]

    missing = (
        {family_filter} if isinstance(family_filter, str) else set(family_filter or [])
    ) - set(family_methods.keys())
    if missing:
        print(f"Warning: families not found in data: {missing}")

    if not families:
        print("No families to plot.")
        return

    # --- Layout ---
    n = len(families)
    if n == 1:
        fig, ax = plt.subplots(figsize=(8, max(3, top_n * 0.4)))
        axes = [ax]
    else:
        cols = min(cols, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        axes = axes.flatten().tolist()

    # --- Plot each family ---
    for i, family in enumerate(families):
        ax = axes[i]
        top = family_methods[family].most_common(top_n)
        methods, counts = zip(*top)
        methods, counts = methods[::-1], counts[::-1]

        ax.barh(methods, counts, color="#3788dd", edgecolor="none")
        ax.set_title(family, fontsize=12, fontweight="bold")
        ax.set_xlabel("Papers", fontsize=10)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    title = (
        f"Top {top_n} compared methods ΓÇö {families[0]}"
        if n == 1
        else f"Top {top_n} compared methods by model family"
    )
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("family_methods.png", dpi=150, bbox_inches="tight")
    plt.show()


# --- Usage ---
# plot_families()                                  # all families
plot_families("Traditional non-DL")                           # single family
# plot_families(["Mamba", "Transformer"])   # specific list
# plot_families(family_filter=None, top_n=5)       # all families, top 5 only
