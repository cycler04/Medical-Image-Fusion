import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================

mod_pair = "SPECT-MRI"

# Metrics where LOWER is better
lower_better_metrics = {
    "MSE",
    "LABF",
    "NABF",
    "NABF1"
}

# =========================
# LOAD CSV
# =========================

df = pd.read_csv(f"outputs/metrics/{mod_pair}.csv")

# First column = model name
model_col = df.columns[0]

# Metric columns
metric_cols = df.columns[1:]

# =========================
# COMPUTE OVERALL Z-SCORE
# =========================

all_zscores = []

for col in metric_cols:

    values = df[col].astype(float)

    # Flip sign if lower is better
    if col.strip().upper() in {m.upper() for m in lower_better_metrics}:
        values = -values

    mean = values.mean()
    std = values.std(ddof=0)

    # Avoid divide-by-zero
    if std == 0:
        z = np.zeros(len(values))
    else:
        z = (values - mean) / std

    all_zscores.append(z)

# Mean z-score across metrics
z_matrix = np.column_stack(all_zscores)

df["z_score_mean"] = z_matrix.mean(axis=1)

# Higher z-score = better rank
df["rank"] = (
    df["z_score_mean"]
    .rank(ascending=False, method="min")
    .astype(int)
)

# Sort by rank
df = df.sort_values("rank")

# =========================
# SAVE
# =========================

output_path = f"outputs/metrics/{mod_pair}_ranked.xlsx"

df.to_excel(output_path, index=False)

print(df[[model_col, "z_score_mean", "rank"]])

print(f"\nSaved to: {output_path}")