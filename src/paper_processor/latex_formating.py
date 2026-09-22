import pandas as pd
import numpy as np

mod_pair = "CT-MRI"

# Read CSV
df = pd.read_csv(f"outputs/metrics/{mod_pair}.csv")

model_col = df.columns[0]
metric_cols = df.columns[1:]

# Convert metrics to numeric
metrics = df[metric_cols].astype(float)

# -----------------------------
# COLUMN-WISE Z-SCORE (STANDARD)
# -----------------------------
col_mean = metrics.mean(axis=0)
col_std = metrics.std(axis=0).replace(0, 1e-8)

z_col = (metrics - col_mean) / col_std

# Rename columns for clarity
z_col = z_col.add_suffix("_zscore")

# -----------------------------
# OPTIONAL: OVERALL MODEL SCORE
# (mean z-score across metrics)
# -----------------------------
df["Z_SCORE_MEAN"] = z_col.mean(axis=1)

# -----------------------------
# MERGE BACK
# -----------------------------
df_out = pd.concat([df, z_col], axis=1)

# -----------------------------
# EXPORT TO EXCEL
# -----------------------------
output_path = f"outputs/metrics/{mod_pair}_zscore.xlsx"
df_out.to_excel(output_path, index=False)

print(f"Saved to: {output_path}")