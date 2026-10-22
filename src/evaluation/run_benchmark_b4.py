#!/usr/bin/env python3
"""
Comprehensive 28-Metric Benchmark Engine for Batch Size 4 Run
============================================================
Evaluates the 2 newly trained batch-size=4 checkpoints (Best & Last) across
Harvard-AANLIB (72 test pairs) and merges them with existing benchmark data:
- Ours: Exp01 (1500ep b4 no-decay - Best)
- Ours: Exp01 (1500ep b4 no-decay - Last)
- Ours: Exp01 (1500ep b64 - Best)
- Ours: Exp01 (1500ep b64 - Last)
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
import matplotlib.pyplot as plt

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
OUTPUT_DIR = RESULTS_ROOT / "benchmark_b4_1500ep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_OUT_DIR = RESULTS_ROOT / "visualizations"
VIZ_OUT_DIR.mkdir(parents=True, exist_ok=True)

PREV_RAW_CSV = RESULTS_ROOT / "benchmark_1500ep" / "benchmark_raw_per_image.csv"
if not PREV_RAW_CSV.exists():
    PREV_RAW_CSV = RESULTS_ROOT / "benchmark_comprehensive" / "benchmark_raw_per_image.csv"

MODALITIES = ["CT-MRI", "PET-MRI", "SPECT-MRI"]
MODALITY_SOURCES = {
    "CT-MRI": ("CT", "MRI"),
    "PET-MRI": ("PET", "MRI"),
    "SPECT-MRI": ("SPECT", "MRI")
}

NEW_MODELS = [
    {
        "name": "Ours: Exp01 (1500ep b4 no-decay - Best)",
        "dir": RESULTS_ROOT / "MMIF-INet-b4-1500ep-no_decay-best",
        "subdirs": {"CT-MRI": "CT-MRI", "PET-MRI": "PET-MRI", "SPECT-MRI": "SPECT-MRI"}
    },
    {
        "name": "Ours: Exp01 (1500ep b4 no-decay - Last)",
        "dir": RESULTS_ROOT / "MMIF-INet-b4-1500ep-no_decay-last",
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


from viz_utils import is_gray_fusion_task, postprocess_fused_for_visualization


def generate_visual_comparison():
    print("\nGenerating publication-quality visual comparisons...")
    b4_best_dir = RESULTS_ROOT / "MMIF-INet-b4-1500ep-no_decay-best"
    b64_best_dir = RESULTS_ROOT / "MMIF-INet-b64-1500ep-best"
    official_dir = RESULTS_ROOT / "MMIF-INet"
    
    mod_sources = {
        "CT-MRI": ("CT", "MRI", "MRI-CT"),
        "PET-MRI": ("PET", "MRI", "MRI-PET"),
        "SPECT-MRI": ("SPECT", "MRI", "MRI-SPECT")
    }

    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    for r_idx, mod in enumerate(MODALITIES):
        s_a_name, s_b_name, official_sub = mod_sources[mod]
        a_dir = DATASET_ROOT / mod / "test" / s_a_name
        b_dir = DATASET_ROOT / mod / "test" / s_b_name
        is_gray = is_gray_fusion_task(mod)
        
        sample_file = sorted(list(a_dir.glob("*.png")))[0].name
        
        im_a = Image.open(a_dir / sample_file)
        im_b = Image.open(b_dir / sample_file)
        im_a = postprocess_fused_for_visualization(im_a, is_gray_task=is_gray)
        im_b = postprocess_fused_for_visualization(im_b, is_gray_task=is_gray)
        
        f_b4 = Image.open(b4_best_dir / mod / sample_file) if (b4_best_dir / mod / sample_file).exists() else None
        f_b64 = Image.open(b64_best_dir / mod / sample_file) if (b64_best_dir / mod / sample_file).exists() else None
        f_off = Image.open(official_dir / official_sub / sample_file) if (official_dir / official_sub / sample_file).exists() else None

        if f_b4 is not None:
            f_b4 = postprocess_fused_for_visualization(f_b4, is_gray_task=is_gray)
        if f_b64 is not None:
            f_b64 = postprocess_fused_for_visualization(f_b64, is_gray_task=is_gray)
        if f_off is not None:
            f_off = postprocess_fused_for_visualization(f_off, is_gray_task=is_gray)

        axes[r_idx, 0].imshow(im_a); axes[r_idx, 0].set_title(f"Source 1 ({s_a_name})", fontsize=11); axes[r_idx, 0].axis('off')
        axes[r_idx, 1].imshow(im_b); axes[r_idx, 1].set_title(f"Source 2 ({s_b_name})", fontsize=11); axes[r_idx, 1].axis('off')
        if f_off: axes[r_idx, 2].imshow(f_off); axes[r_idx, 2].set_title(f"Official MMIF-INet", fontsize=11); axes[r_idx, 2].axis('off')
        if f_b64: axes[r_idx, 3].imshow(f_b64); axes[r_idx, 3].set_title(f"Ours: 1500ep b64 (Decay)", fontsize=11); axes[r_idx, 3].axis('off')
        if f_b4: axes[r_idx, 4].imshow(f_b4); axes[r_idx, 4].set_title(f"Ours: 1500ep b4 (No Decay)", fontsize=11, fontweight='bold', color='green'); axes[r_idx, 4].axis('off')
        
        axes[r_idx, 0].set_ylabel(f"{mod}\n({sample_file})", fontsize=11, fontweight='bold')

    plt.suptitle("Harvard-AANLIB Fusion: Batch Size 4 (No Decay) vs Batch Size 64 vs Official INet", fontsize=14, fontweight='bold')
    plt.tight_layout()
    viz_path = VIZ_OUT_DIR / "b4_vs_b64_comparison.png"
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved visual comparison figure to: {viz_path}")


def main():
    print("=" * 80)
    print("STARTING RAPID 28-METRIC BENCHMARK FOR BATCH-SIZE=4 CHECKPOINTS")
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

    print(f"Evaluating {len(tasks_to_run)} image pairs for new b4 no-decay checkpoints...")

    num_workers = min(os.cpu_count() or 4, 8)
    print(f"Executing on {num_workers} parallel worker processes...")

    start_time = time.time()
    new_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(compute_all_metrics_single_pair, tasks_to_run), total=len(tasks_to_run), desc="Benchmarking b4"):
            new_results.append(res)

    total_duration = time.time() - start_time
    print(f"\nb4 metrics computed in {total_duration / 60:.2f} minutes ({total_duration:.1f}s)!")

    df_new = pd.DataFrame(new_results)

    # Merge with pre-computed baseline data
    if PREV_RAW_CSV.exists():
        print(f"Merging with pre-computed baseline evaluations from: {PREV_RAW_CSV}...")
        df_prev = pd.read_csv(PREV_RAW_CSV)
        # Avoid duplicate model names
        df_prev = df_prev[~df_prev["Model"].isin([m["name"] for m in NEW_MODELS])]
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
        "Ours: Exp01 (1500ep b4 no-decay - Best)",
        "Ours: Exp01 (1500ep b4 no-decay - Last)",
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

    print("\nKEY METRICS SUMMARY (72 TEST PAIRS):")
    key_cols = ["Model", "SSIM", "MS_SSIM", "QABF", "VIFF", "MI", "SD", "SF", "AG", "SCD"]
    available_cols = [c for c in key_cols if c in overall_summary.columns]
    print(overall_summary[available_cols].to_string(index=False))
    print("=" * 90)

    generate_visual_comparison()


if __name__ == "__main__":
    main()
