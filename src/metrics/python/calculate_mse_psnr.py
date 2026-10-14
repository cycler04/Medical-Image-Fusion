"""
Calculate MSE and PSNR metrics for fused images in results/ directory.
Computes 3 values per pair for MSE (MSE_MRI, MSE_Other, MSE_avg)
and 3 values per pair for PSNR (PSNR_MRI, PSNR_Other, PSNR_avg).

Generates:
1. Per-model, per-modality CSV files in results/<Model>/<Modality>_mse_psnr.csv
2. Per-model consolidated CSV files in results/<Model>/mse_psnr_metrics.csv
3. Global individual-images CSV in results/mse_psnr_all_images.csv
4. Global summary CSV in results/mse_psnr_summary.csv
"""

import os
import sys
import csv
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure src/metrics/python is on path for importing existing quality_metrics
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src" / "metrics" / "python"))

from quality_metrics import mse, psnr

RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_DIR = PROJECT_ROOT / "dataset" / "Harvard-AANLIB"

MODALITY_CONFIG = {
    "MRI-CT": {
        "dataset_subdir": "CT-MRI",
        "source1": "MRI",
        "source2": "CT",
    },
    "MRI-PET": {
        "dataset_subdir": "PET-MRI",
        "source1": "MRI",
        "source2": "PET",
    },
    "MRI-SPECT": {
        "dataset_subdir": "SPECT-MRI",
        "source1": "MRI",
        "source2": "SPECT",
    },
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_gray_normalized(path: Path) -> np.ndarray:
    """Load image as grayscale float32 array normalized to [0, 1]."""
    return np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0


def compute_pair_metrics(s1_img: np.ndarray, s2_img: np.ndarray, fused_img: np.ndarray):
    """
    Compute MSE and PSNR (3 values per pair each):
    - MSE(Source1, Fused)
    - MSE(Source2, Fused)
    - MSE_avg = 0.5 * (MSE_1 + MSE_2)
    - PSNR(Source1, Fused)
    - PSNR(Source2, Fused)
    - PSNR_avg = 0.5 * (PSNR_1 + PSNR_2)
    """
    mse_1 = float(mse(s1_img, fused_img))
    mse_2 = float(mse(s2_img, fused_img))
    mse_avg = 0.5 * (mse_1 + mse_2)

    psnr_1 = float(psnr(s1_img, fused_img, max_value=1.0))
    psnr_2 = float(psnr(s2_img, fused_img, max_value=1.0))
    psnr_avg = 0.5 * (psnr_1 + psnr_2)

    return {
        "mse_1": mse_1,
        "mse_2": mse_2,
        "mse_avg": mse_avg,
        "psnr_1": psnr_1,
        "psnr_2": psnr_2,
        "psnr_avg": psnr_avg,
    }


def main():
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Dataset dir: {DATASET_DIR}")

    models = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"Discovered models: {[m.name for m in models]}")

    all_individual_rows = []
    summary_rows = []

    for model_dir in models:
        model_name = model_dir.name
        model_individual_rows = []

        for mod_name, config in MODALITY_CONFIG.items():
            fused_mod_dir = model_dir / mod_name
            if not fused_mod_dir.is_dir():
                continue

            s1_name = config["source1"]
            s2_name = config["source2"]
            ds_mod = config["dataset_subdir"]

            s1_dir = DATASET_DIR / ds_mod / s1_name
            s2_dir = DATASET_DIR / ds_mod / s2_name

            fused_files = sorted([
                f for f in fused_mod_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            ])

            print(f"\nProcessing [{model_name}] / [{mod_name}] ({len(fused_files)} images)...")

            mod_results = []

            for fpath in fused_files:
                stem = fpath.stem
                # Find source images
                s1_file = s1_dir / fpath.name
                s2_file = s2_dir / fpath.name

                if not s1_file.exists() or not s2_file.exists():
                    print(f"Warning: missing source for {fpath.name}")
                    continue

                F = load_gray_normalized(fpath)
                S1 = load_gray_normalized(s1_file)
                S2 = load_gray_normalized(s2_file)

                m = compute_pair_metrics(S1, S2, F)

                item = {
                    "image": fpath.name,
                    f"MSE_{s1_name}": m["mse_1"],
                    f"MSE_{s2_name}": m["mse_2"],
                    "MSE_avg": m["mse_avg"],
                    f"PSNR_{s1_name}": m["psnr_1"],
                    f"PSNR_{s2_name}": m["psnr_2"],
                    "PSNR_avg": m["psnr_avg"],
                }
                mod_results.append(item)

                # For model consolidated list
                model_individual_rows.append({
                    "model": model_name,
                    "modality": mod_name,
                    "image": fpath.name,
                    "source_1": s1_name,
                    "source_2": s2_name,
                    "MSE_Source1": m["mse_1"],
                    "MSE_Source2": m["mse_2"],
                    "MSE_avg": m["mse_avg"],
                    "PSNR_Source1": m["psnr_1"],
                    "PSNR_Source2": m["psnr_2"],
                    "PSNR_avg": m["psnr_avg"],
                })

                all_individual_rows.append({
                    "model": model_name,
                    "modality": mod_name,
                    "image": fpath.name,
                    "source_1": s1_name,
                    "source_2": s2_name,
                    "MSE_Source1": m["mse_1"],
                    "MSE_Source2": m["mse_2"],
                    "MSE_avg": m["mse_avg"],
                    "PSNR_Source1": m["psnr_1"],
                    "PSNR_Source2": m["psnr_2"],
                    "PSNR_avg": m["psnr_avg"],
                })

            if not mod_results:
                continue

            # Calculate average for this modality
            avg_mse_1 = float(np.mean([x[f"MSE_{s1_name}"] for x in mod_results]))
            avg_mse_2 = float(np.mean([x[f"MSE_{s2_name}"] for x in mod_results]))
            avg_mse_all = float(np.mean([x["MSE_avg"] for x in mod_results]))

            avg_psnr_1 = float(np.mean([x[f"PSNR_{s1_name}"] for x in mod_results]))
            avg_psnr_2 = float(np.mean([x[f"PSNR_{s2_name}"] for x in mod_results]))
            avg_psnr_all = float(np.mean([x["PSNR_avg"] for x in mod_results]))

            # Save per-modality CSV
            mod_csv_path = model_dir / f"{mod_name}_mse_psnr.csv"
            fieldnames = [
                "image",
                f"MSE_{s1_name}",
                f"MSE_{s2_name}",
                "MSE_avg",
                f"PSNR_{s1_name}",
                f"PSNR_{s2_name}",
                "PSNR_avg",
            ]
            with open(mod_csv_path, "w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                for row in mod_results:
                    writer.writerow({
                        "image": row["image"],
                        f"MSE_{s1_name}": f"{row[f'MSE_{s1_name}']:.6f}",
                        f"MSE_{s2_name}": f"{row[f'MSE_{s2_name}']:.6f}",
                        "MSE_avg": f"{row['MSE_avg']:.6f}",
                        f"PSNR_{s1_name}": f"{row[f'PSNR_{s1_name}']:.4f}",
                        f"PSNR_{s2_name}": f"{row[f'PSNR_{s2_name}']:.4f}",
                        "PSNR_avg": f"{row['PSNR_avg']:.4f}",
                    })
                # Add Average summary row
                writer.writerow({
                    "image": "Average",
                    f"MSE_{s1_name}": f"{avg_mse_1:.6f}",
                    f"MSE_{s2_name}": f"{avg_mse_2:.6f}",
                    "MSE_avg": f"{avg_mse_all:.6f}",
                    f"PSNR_{s1_name}": f"{avg_psnr_1:.4f}",
                    f"PSNR_{s2_name}": f"{avg_psnr_2:.4f}",
                    "PSNR_avg": f"{avg_psnr_all:.4f}",
                })
            print(f"Saved -> {mod_csv_path}")

            summary_rows.append({
                "model": model_name,
                "modality": mod_name,
                "num_images": len(mod_results),
                "source_1": s1_name,
                "source_2": s2_name,
                "MSE_Source1": avg_mse_1,
                "MSE_Source2": avg_mse_2,
                "MSE_avg": avg_mse_all,
                "PSNR_Source1": avg_psnr_1,
                "PSNR_Source2": avg_psnr_2,
                "PSNR_avg": avg_psnr_all,
            })

        # Save consolidated CSV for this model
        if model_individual_rows:
            model_csv_path = model_dir / "mse_psnr_metrics.csv"
            fieldnames = [
                "model", "modality", "image", "source_1", "source_2",
                "MSE_Source1", "MSE_Source2", "MSE_avg",
                "PSNR_Source1", "PSNR_Source2", "PSNR_avg",
            ]
            with open(model_csv_path, "w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                for row in model_individual_rows:
                    writer.writerow({
                        "model": row["model"],
                        "modality": row["modality"],
                        "image": row["image"],
                        "source_1": row["source_1"],
                        "source_2": row["source_2"],
                        "MSE_Source1": f"{row['MSE_Source1']:.6f}",
                        "MSE_Source2": f"{row['MSE_Source2']:.6f}",
                        "MSE_avg": f"{row['MSE_avg']:.6f}",
                        "PSNR_Source1": f"{row['PSNR_Source1']:.4f}",
                        "PSNR_Source2": f"{row['PSNR_Source2']:.4f}",
                        "PSNR_avg": f"{row['PSNR_avg']:.4f}",
                    })
            print(f"Saved consolidated -> {model_csv_path}")

    # Save global individual CSV
    all_csv_path = RESULTS_DIR / "mse_psnr_all_images.csv"
    fieldnames = [
        "model", "modality", "image", "source_1", "source_2",
        "MSE_Source1", "MSE_Source2", "MSE_avg",
        "PSNR_Source1", "PSNR_Source2", "PSNR_avg",
    ]
    with open(all_csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_individual_rows:
            writer.writerow({
                "model": row["model"],
                "modality": row["modality"],
                "image": row["image"],
                "source_1": row["source_1"],
                "source_2": row["source_2"],
                "MSE_Source1": f"{row['MSE_Source1']:.6f}",
                "MSE_Source2": f"{row['MSE_Source2']:.6f}",
                "MSE_avg": f"{row['MSE_avg']:.6f}",
                "PSNR_Source1": f"{row['PSNR_Source1']:.4f}",
                "PSNR_Source2": f"{row['PSNR_Source2']:.4f}",
                "PSNR_avg": f"{row['PSNR_avg']:.4f}",
            })
    print(f"\nSaved global individual images -> {all_csv_path}")

    # Save summary CSV
    summary_csv_path = RESULTS_DIR / "mse_psnr_summary.csv"
    summary_fields = [
        "model", "modality", "num_images", "source_1", "source_2",
        "MSE_Source1", "MSE_Source2", "MSE_avg",
        "PSNR_Source1", "PSNR_Source2", "PSNR_avg",
    ]
    with open(summary_csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                "model": row["model"],
                "modality": row["modality"],
                "num_images": row["num_images"],
                "source_1": row["source_1"],
                "source_2": row["source_2"],
                "MSE_Source1": f"{row['MSE_Source1']:.6f}",
                "MSE_Source2": f"{row['MSE_Source2']:.6f}",
                "MSE_avg": f"{row['MSE_avg']:.6f}",
                "PSNR_Source1": f"{row['PSNR_Source1']:.4f}",
                "PSNR_Source2": f"{row['PSNR_Source2']:.4f}",
                "PSNR_avg": f"{row['PSNR_avg']:.4f}",
            })
    print(f"Saved summary -> {summary_csv_path}")


if __name__ == "__main__":
    main()

