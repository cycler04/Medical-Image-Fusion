#!/usr/bin/env python3
"""
Fast & Comprehensive 28-Metric Benchmark Engine for 1500-Epoch Run
===================================================================
Evaluates the 2 newly trained 1500-epoch checkpoints (Best & Last) across
Harvard-AANLIB (72 test pairs) and merges them with existing benchmark data for:
- Ours: Exp01 Unified (2000ep)
- Ours: Exp02 Native Gray (2000ep)
- Official MMIF-INet (IF 2025 Pretrained Checkpoint)
- MFS-Fusion (IF 2024 SOTA Baseline)
- MMIF-CDDFuse (Benchmark Baseline)
- PSLPT (Transformer Baseline)
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "metrics" / "python"))

from structure_metrics import ssim, ms_ssim, piella_metrics
from img_metrics import petrovic_metrics, sf
from info_metrics import en, mi, ncie, scd, fmi
from quality_metrics import mli_error, sd, ag, mse_f, psnr_f
from visual_metrics import vif, viff
from others_metrics import cc, qp

DATASET_ROOT = PROJECT_ROOT / "dataset" / "Harvard-AANLIB" / "MyDatasets"
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "csv" / "experiments" / "01_conditional_color_loss" / "1500ep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_model_dir(name: str) -> Path:
    matches = list((RESULTS_ROOT / "experiments").glob(f"**/{name}"))
    if matches:
        return matches[0]
    matches = list((RESULTS_ROOT / "baseline").glob(f"**/{name}"))
    if matches:
        return matches[0]
    if (RESULTS_ROOT / name).exists():
        return RESULTS_ROOT / name
    return RESULTS_ROOT / "experiments" / name

PREV_RAW_CSV = RESULTS_ROOT / "csv" / "experiments" / "01_conditional_color_loss" / "comprehensive" / "benchmark_raw_per_image.csv"

MODALITIES = ["CT-MRI", "PET-MRI", "SPECT-MRI"]
MODALITY_SOURCES = {
    "CT-MRI": ("CT", "MRI"),
    "PET-MRI": ("PET", "MRI"),
    "SPECT-MRI": ("SPECT", "MRI")
}

NEW_MODELS = [
    {
        "name": "Ours: Exp01 (1500ep b64 - Best)",
        "dir": find_model_dir("MMIF-INet-b64-1500ep-best"),
        "subdirs": {"CT-MRI": "CT-MRI", "PET-MRI": "PET-MRI", "SPECT-MRI": "SPECT-MRI"}
    },
    {
        "name": "Ours: Exp01 (1500ep b64 - Last)",
        "dir": find_model_dir("MMIF-INet-b64-1500ep-last"),
        "subdirs": {"CT-MRI": "CT-MRI", "PET-MRI": "PET-MRI", "SPECT-MRI": "SPECT-MRI"}
    }
]


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L")).astype(np.float64)


def compute_all_metrics_single_pair(args):
    model_name, modality, fname, pa_str, pb_str, pf_str = args
    A = load_gray(Path(pa_str))
    B = load_gray(Path(pb_str))
    F = load_gray(Path(pf_str))

    res = {
        "Model": model_name,
        "Modality": modality,
        "File": fname
    }

    # 1. Quality & Contrast
    res["MLI"] = float(mli_error(F))
    res["SD"] = float(sd(F))
    res["AG"] = float(ag(F))
    res["MSE"] = float(mse_f(A, B, F))
    res["PSNR"] = float(psnr_f(A, B, F))

    # 2. Information-Theoretic
    res["EN"] = float(en(F))
    res["MI"] = float(mi(A, B, F))
    res["NCIE"] = float(ncie(A, B, F))
    res["SCD"] = float(scd(A, B, F))
    res["FMI_pixel"] = float(fmi(A, B, F, feature="none"))
    res["FMI_gradient"] = float(fmi(A, B, F, feature="gradient"))
    res["FMI_dct"] = float(fmi(A, B, F, feature="dct"))
    res["FMI_wavelet"] = float(fmi(A, B, F, feature="wavelet"))
    res["FMI_edge"] = float(fmi(A, B, F, feature="edge"))

    # 3. Image Feature & Edge Preservation
    qabf_val, labf_val, nabf_val, _ = petrovic_metrics(A, B, F)
    res["QABF"] = float(qabf_val)
    res["LABF"] = float(labf_val)
    res["NABF"] = float(nabf_val)
    res["SF"] = float(sf(F))

    # 4. Structural Similarity
    res["SSIM"] = float(0.5 * ssim(A, F) + 0.5 * ssim(B, F))
    res["MS_SSIM"] = float(ms_ssim(np.stack([A, B], axis=2), F))
    res["Q"] = float(piella_metrics(A, B, F, sw=1))
    res["Qw"] = float(piella_metrics(A, B, F, sw=2))
    res["Qe"] = float(piella_metrics(A, B, F, sw=3))

    # 5. Visual Perception
    res["VIF"] = float(vif(A, F) + vif(B, F))
    res["VIFF"] = float(viff(A, B, F))

    # 6. Others
    res["CC"] = float(cc(A, B, F))
    res["Qp"] = float(qp(A, B, F))

    return res


def main():
    print("=" * 80)
    print("STARTING RAPID 28-METRIC BENCHMARK FOR 1500-EPOCH CHECKPOINTS")
    print("=" * 80)

    tasks_to_run = []
    for model_cfg in NEW_MODELS:
        mname = model_cfg["name"]
        mdir = model_cfg["dir"]
        for mod in MODALITIES:
            sub = model_cfg["subdirs"][mod]
            fused_mod_dir = mdir / sub
            src_a_name, src_b_name = MODALITY_SOURCES[mod]
            src_mod_dir = DATASET_ROOT / mod / "test"
            src_a_dir = src_mod_dir / src_a_name
            src_b_dir = src_mod_dir / src_b_name

            if not fused_mod_dir.exists():
                print(f"[Warning] Fused directory missing: {fused_mod_dir}")
                continue

            test_files = sorted([f.name for f in src_a_dir.glob("*.png")])
            for fname in test_files:
                pa = src_a_dir / fname
                pb = src_b_dir / fname
                pf = fused_mod_dir / fname
                if pa.exists() and pb.exists() and pf.exists():
                    tasks_to_run.append((mname, mod, fname, str(pa), str(pb), str(pf)))
                else:
                    print(f"Missing file for {mname} {mod}: {fname}")

    print(f"Evaluating {len(tasks_to_run)} image pairs for new 1500ep checkpoints...")

    num_workers = min(os.cpu_count() or 4, 8)
    print(f"Executing on {num_workers} parallel worker processes...")

    start_time = time.time()
    new_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(compute_all_metrics_single_pair, tasks_to_run), total=len(tasks_to_run), desc="Benchmarking 1500ep"):
            new_results.append(res)

    total_duration = time.time() - start_time
    print(f"\n1500ep metrics computed in {total_duration / 60:.2f} minutes ({total_duration:.1f}s)!")

    df_new = pd.DataFrame(new_results)

    # Merge with pre-computed baseline data
    if PREV_RAW_CSV.exists():
        print(f"Merging with pre-computed baseline evaluations from: {PREV_RAW_CSV}...")
        df_prev = pd.read_csv(PREV_RAW_CSV)
        # Combine
        df = pd.concat([df_new, df_prev], ignore_index=True)
    else:
        df = df_new

    # Save consolidated raw results
    raw_csv = OUTPUT_DIR / "benchmark_raw_per_image.csv"
    df.to_csv(raw_csv, index=False)
    print(f"Saved complete raw per-image results -> {raw_csv}")

    metric_cols = [
        "MLI", "SD", "AG", "MSE", "PSNR",
        "EN", "MI", "NCIE", "SCD", "FMI_pixel", "FMI_gradient", "FMI_dct", "FMI_wavelet", "FMI_edge",
        "QABF", "LABF", "NABF", "SF",
        "SSIM", "MS_SSIM", "Q", "Qw", "Qe",
        "VIF", "VIFF",
        "CC", "Qp"
    ]

    model_order = [
        "Ours: Exp01 (1500ep b64 - Best)",
        "Ours: Exp01 (1500ep b64 - Last)",
        "Ours: Exp01 Unified (2000ep)",
        "Ours: Exp02 Native Gray (2000ep)",
        "Official MMIF-INet",
        "MFS-Fusion",
        "MMIF-CDDFuse",
        "PSLPT"
    ]

    # 1. Per-Modality Summary
    modality_summary = df.groupby(["Modality", "Model"])[metric_cols].mean().reset_index()
    modality_summary["sort_key"] = modality_summary["Model"].apply(lambda x: model_order.index(x) if x in model_order else 99)
    modality_summary = modality_summary.sort_values(["Modality", "sort_key"]).drop(columns=["sort_key"])
    modality_csv = OUTPUT_DIR / "benchmark_by_modality.csv"
    modality_summary.to_csv(modality_csv, index=False)
    print(f"Saved per-modality summary -> {modality_csv}")

    # 2. Overall Combined Summary (72 pairs)
    overall_summary = df.groupby("Model")[metric_cols].mean().reset_index()
    overall_summary["sort_key"] = overall_summary["Model"].apply(lambda x: model_order.index(x) if x in model_order else 99)
    overall_summary = overall_summary.sort_values("sort_key").drop(columns=["sort_key"])
    overall_csv = OUTPUT_DIR / "benchmark_overall_72pairs.csv"
    overall_summary.to_csv(overall_csv, index=False)
    print(f"Saved overall 72-pair summary -> {overall_csv}")

    # 3. Z-Score Calculation
    neg_metrics = {"MSE", "LABF", "NABF"}
    z_records = []
    for mod in MODALITIES:
        mod_df = modality_summary[modality_summary["Modality"] == mod].copy()
        z_df = pd.DataFrame()
        z_df["Model"] = mod_df["Model"]
        z_df["Modality"] = mod

        for col in metric_cols:
            vals = mod_df[col].values
            std_val = np.std(vals)
            mean_val = np.mean(vals)
            if std_val < 1e-12:
                z_vals = np.zeros_like(vals)
            else:
                if col in neg_metrics:
                    z_vals = (mean_val - vals) / std_val
                else:
                    z_vals = (vals - mean_val) / std_val
            z_df[f"Z_{col}"] = z_vals

        z_df["Z_Quality"] = z_df[[f"Z_{c}" for c in ["MLI", "SD", "AG", "MSE", "PSNR"]]].mean(axis=1)
        z_df["Z_Info"] = z_df[[f"Z_{c}" for c in ["EN", "MI", "NCIE", "SCD", "FMI_pixel", "FMI_gradient", "FMI_dct", "FMI_wavelet", "FMI_edge"]]].mean(axis=1)
        z_df["Z_Edge"] = z_df[[f"Z_{c}" for c in ["QABF", "LABF", "NABF", "SF"]]].mean(axis=1)
        z_df["Z_Structure"] = z_df[[f"Z_{c}" for c in ["SSIM", "MS_SSIM", "Q", "Qw", "Qe"]]].mean(axis=1)
        z_df["Z_Visual"] = z_df[[f"Z_{c}" for c in ["VIF", "VIFF"]]].mean(axis=1)
        z_df["Z_Composite"] = z_df[[f"Z_{c}" for c in metric_cols]].mean(axis=1)

        z_records.append(z_df)

    all_z_df = pd.concat(z_records, ignore_index=True)
    zscore_csv = OUTPUT_DIR / "benchmark_zscores_by_modality.csv"
    all_z_df.to_csv(zscore_csv, index=False)
    print(f"Saved Z-Scores -> {zscore_csv}")

    # Global Leaderboard by Average Z-Score
    leaderboard = all_z_df.groupby("Model")[["Z_Quality", "Z_Info", "Z_Edge", "Z_Structure", "Z_Visual", "Z_Composite"]].mean().reset_index()
    leaderboard = leaderboard.sort_values("Z_Composite", ascending=False)
    leaderboard_csv = OUTPUT_DIR / "benchmark_global_leaderboard.csv"
    leaderboard.to_csv(leaderboard_csv, index=False)
    print(f"Saved Global Leaderboard -> {leaderboard_csv}")

    print("\n" + "=" * 90)
    print("GLOBAL LEADERBOARD (Z-SCORE RANKING ACROSS ALL 28 METRICS):")
    print("=" * 90)
    print(leaderboard.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
