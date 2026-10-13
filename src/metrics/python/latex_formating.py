import pandas as pd
import numpy as np

# Read CSV
df = pd.read_csv("outputs/metrics/CT-MRI.csv")

# Assume first column is method/model names
metric_cols = df.columns[1:]

# Copy for formatting
styled = df.copy().astype(object)   

latex_tables = []
chunk_size = 8

for i in range(0, len(metric_cols), chunk_size):
    chunk_cols = metric_cols[i : i + chunk_size]
    styled = df.copy().astype(object)

    for col in chunk_cols:
        # Get descending sorted indices
        sorted_idx = df[col].astype(float).sort_values(ascending=False).index

        # Top 1
        idx1 = sorted_idx[0]
        styled.loc[idx1, col] = (
            r"\textbf{" + f"{df.loc[idx1, col]:.2f}" + "}"
        )

        # Top 2
        if len(sorted_idx) > 1:
            idx2 = sorted_idx[1]
            styled.loc[idx2, col] = (
                r"\underline{" + f"{df.loc[idx2, col]:.2f}" + "}"
            )

        # Top 3
        if len(sorted_idx) > 2:
            idx3 = sorted_idx[2]
            styled.loc[idx3, col] = (
                r"\textit{" + f"{df.loc[idx3, col]:.2f}" + "}"
            )

        # Remaining values
        for idx in sorted_idx[3:]:
            styled.loc[idx, col] = f"{df.loc[idx, col]:.2f}"

    latex_table = styled.loc[:, df.columns[:1].tolist() + chunk_cols.tolist()].to_latex(
        index=False,
        escape=False,
        longtable=True
    )
    latex_tables.append(latex_table)


with open("table.tex", "w") as f:
    for latex_table in latex_tables:
        f.write("{\\small\n")
        f.write(latex_table)
        f.write("}\n\n")