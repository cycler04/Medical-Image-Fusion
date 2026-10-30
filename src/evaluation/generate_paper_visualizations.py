"""
Generate paper-style visualization figures comparing fused results across all models.
Highlights MMIF-INet artifacts (chromatic aberration, functional quenching, edge ringing)
alongside source images, MFS-Fusion, and MMIF-CDDFuse, with ROI zoom-in insets and benchmark metrics.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src" / "metrics" / "python"))

from quality_metrics import mse, psnr
from img_metrics import petrovic_metrics
from viz_utils import is_gray_fusion_task, postprocess_fused_for_visualization

OUTPUT_DIR = PROJECT_ROOT / "results" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_FIG_DIR = PROJECT_ROOT / "docs" / "figures"
DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Define representative sample pairs with region of interest (ROI: y1, y2, x1, x2)
# Selected to highlight benchmark comparisons across models:
# - CT: bone & parenchyma detail, skull contrast
# - PET: metabolic hotspot preservation, boundary sharpness
# - SPECT: functional perfusion detail and noise suppression
SAMPLES = [
    {
        "modality": "MRI-CT",
        "dataset_mod": "CT-MRI",
        "source1": "MRI",
        "source2": "CT",
        "image_file": "32022.png",
        "roi": (120, 180, 130, 190),  # Ventricle & bone structure
        "roi_label": "Bone & Parenchyma Boundary",
        "artifact_note": "Bone boundary contrast & soft tissue definition",
    },
    {
        "modality": "MRI-CT",
        "dataset_mod": "CT-MRI",
        "source1": "MRI",
        "source2": "CT",
        "image_file": "2004.png",
        "roi": (100, 160, 50, 110),   # Frontal skull & cortex
        "roi_label": "Skull Cortex",
        "artifact_note": "Grayscale preservation & skull cortex sharpness",
    },
    {
        "modality": "MRI-PET",
        "dataset_mod": "PET-MRI",
        "source1": "MRI",
        "source2": "PET",
        "image_file": "41022.png",
        "roi": (135, 195, 110, 170),  # Metabolic lesion/hotspot
        "roi_label": "Metabolic Hotspot",
        "artifact_note": "Dimmed functional intensity & high NABF1 (0.82) in INet",
    },
    {
        "modality": "MRI-PET",
        "dataset_mod": "PET-MRI",
        "source1": "MRI",
        "source2": "PET",
        "image_file": "25020.png",
        "roi": (95, 155, 60, 120),    # Temporal lobe functional activity
        "roi_label": "Metabolic Core",
        "artifact_note": "Suppressed high-brightness metabolic core in INet",
    },
    {
        "modality": "MRI-SPECT",
        "dataset_mod": "SPECT-MRI",
        "source1": "MRI",
        "source2": "SPECT",
        "image_file": "4013.png",
        "roi": (100, 160, 45, 105),   # Functional perfusion lesion
        "roi_label": "Perfusion Area",
        "artifact_note": "Severe artifact injection: NABF 2x higher, PSNR -10.7dB vs CDD",
    },
    {
        "modality": "MRI-SPECT",
        "dataset_mod": "SPECT-MRI",
        "source1": "MRI",
        "source2": "SPECT",
        "image_file": "9009.png",
        "roi": (130, 190, 105, 165),  # Deep brain perfusion
        "roi_label": "Deep Perfusion",
        "artifact_note": "High NABF1 artificial noise in INet",
    },
]

MODELS = ["MFS-Fusion", "MMIF-CDDFuse", "MMIF-INet", "PSLPT"]


def load_img(path: Path, is_gray_task: bool = False) -> np.ndarray:
    """Load image preserving RGB if available, or postprocessing to pure gray via luminance for gray-gray fusion."""
    im = Image.open(path)
    post_im = postprocess_fused_for_visualization(im, is_gray_task=is_gray_task)
    return np.array(post_im)


def load_gray_norm(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")).astype(np.float64) / 255.0


def compute_metrics(s1_p: Path, s2_p: Path, f_p: Path):
    A = load_gray_norm(s1_p)
    B = load_gray_norm(s2_p)
    F = load_gray_norm(f_p)

    m1 = mse(A, F)
    m2 = mse(B, F)
    m_avg = 0.5 * (m1 + m2)

    p1 = psnr(A, F, 1.0)
    p2 = psnr(B, F, 1.0)
    p_avg = 0.5 * (p1 + p2)

    # Petrovic artifact metrics
    A_raw = np.array(Image.open(s1_p).convert("L")).astype(np.float64)
    B_raw = np.array(Image.open(s2_p).convert("L")).astype(np.float64)
    F_raw = np.array(Image.open(f_p).convert("L")).astype(np.float64)
    qabf, labf, nabf, nabf1 = petrovic_metrics(A_raw, B_raw, F_raw)

    return {
        "MSE_avg": m_avg,
        "PSNR_avg": p_avg,
        "NABF": nabf,
        "NABF1": nabf1,
    }


def create_modality_figure(sample_list, figure_title, save_name):
    """
    Create publication-style figure for a specific modality with 2 sample pairs.
    Layout per sample:
    Row 1: Full images [Source1, Source2, MFS-Fusion, MMIF-CDDFuse, MMIF-INet, PSLPT] with red ROI box
    Row 2: Zoomed-in ROI patches with red border
    """
    num_samples = len(sample_list)
    cols = 6
    # Total rows = num_samples * 2 (1 for full image, 1 for zoom-in crop)
    fig, axes = plt.subplots(
        num_samples * 2,
        cols,
        figsize=(18, 3.8 * num_samples * 2),
        gridspec_kw={"height_ratios": [3, 2] * num_samples},
    )
    plt.subplots_adjust(wspace=0.08, hspace=0.18)

    col_names = [None, None, "MFS-Fusion", "MMIF-CDDFuse", "MMIF-INet", "PSLPT"]

    for s_idx, sample in enumerate(sample_list):
        row_full = s_idx * 2
        row_zoom = s_idx * 2 + 1

        mod = sample["modality"]
        ds_mod = sample["dataset_mod"]
        s1_name = sample["source1"]
        s2_name = sample["source2"]
        im_name = sample["image_file"]
        y1, y2, x1, x2 = sample["roi"]
        is_gray = is_gray_fusion_task(mod)

        s1_path = PROJECT_ROOT / "dataset" / "Harvard-AANLIB" / ds_mod / s1_name / im_name
        s2_path = PROJECT_ROOT / "dataset" / "Harvard-AANLIB" / ds_mod / s2_name / im_name

        s1_img = load_img(s1_path, is_gray_task=is_gray)
        s2_img = load_img(s2_path, is_gray_task=is_gray)

        fused_imgs = {}
        metrics_dict = {}
        for m_name in MODELS:
            f_path = PROJECT_ROOT / "results" / m_name / mod / im_name
            fused_imgs[m_name] = load_img(f_path, is_gray_task=is_gray)
            metrics_dict[m_name] = compute_metrics(s1_path, s2_path, f_path)

        all_imgs = [s1_img, s2_img, fused_imgs["MFS-Fusion"], fused_imgs["MMIF-CDDFuse"], fused_imgs["MMIF-INet"], fused_imgs["PSLPT"]]
        names = [f"Source 1 ({s1_name})", f"Source 2 ({s2_name})", "MFS-Fusion", "MMIF-CDDFuse", "MMIF-INet", "PSLPT"]

        for col in range(cols):
            ax_full = axes[row_full, col]
            ax_zoom = axes[row_zoom, col]
            img = all_imgs[col]

            # 1. Full Image
            ax_full.imshow(img)
            ax_full.set_xticks([])
            ax_full.set_yticks([])

            # Draw ROI box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.0, edgecolor="#ff2222", facecolor="none", linestyle="-"
            )
            ax_full.add_patch(rect)

            # Titles on top row of each sample
            if col < 2:
                title = names[col]
            else:
                m_key = col_names[col]
                met = metrics_dict[m_key]
                title = f"{m_key}\nPSNR: {met['PSNR_avg']:.2f} dB\nNABF: {met['NABF']:.4f}"
            ax_full.set_title(title, fontsize=10.5, fontweight="bold", pad=5)

            # 2. Zoomed-in ROI Patch
            crop = img[y1:y2, x1:x2]
            ax_zoom.imshow(crop)
            ax_zoom.set_xticks([])
            ax_zoom.set_yticks([])

            # Add border around zoom patch
            for spine in ax_zoom.spines.values():
                spine.set_edgecolor("#ff2222")
                spine.set_linewidth(2.0)

            # Label on zoom patch
            if col == 0:
                ax_zoom.set_ylabel(f"Zoom ({sample['roi_label']})", fontsize=10, fontweight="bold", color="#cc0000")

        # Add sample descriptor text across the row
        axes[row_full, 0].text(
            -0.12, 0.5, f"Sample: {im_name}\n({mod})",
            transform=axes[row_full, 0].transAxes,
            fontsize=11, fontweight="bold", va="center", ha="right", rotation=90
        )

    plt.suptitle(figure_title, fontsize=15, fontweight="bold", y=0.995)
    out_path = OUTPUT_DIR / save_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    # Also copy to docs/figures
    plt.savefig(DOCS_FIG_DIR / save_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def create_master_summary_figure():
    """
    Create a 3-row master paper comparison figure covering:
    Row 1: MRI-CT (Sample 32022.png)
    Row 2: MRI-PET (Sample 41022.png)
    Row 3: MRI-SPECT (Sample 4013.png)
    With zoomed insets directly in bottom-right corner and metric annotations.
    """
    master_samples = [SAMPLES[0], SAMPLES[2], SAMPLES[4]]
    cols = 6
    num_rows = len(master_samples)

    fig, axes = plt.subplots(num_rows, cols, figsize=(19, 3.6 * num_rows))
    plt.subplots_adjust(wspace=0.03, hspace=0.18)

    for r_idx, sample in enumerate(master_samples):
        mod = sample["modality"]
        ds_mod = sample["dataset_mod"]
        s1_name = sample["source1"]
        s2_name = sample["source2"]
        im_name = sample["image_file"]
        y1, y2, x1, x2 = sample["roi"]

        s1_path = PROJECT_ROOT / "dataset" / "Harvard-AANLIB" / ds_mod / s1_name / im_name
        s2_path = PROJECT_ROOT / "dataset" / "Harvard-AANLIB" / ds_mod / s2_name / im_name
        is_gray = is_gray_fusion_task(mod)

        s1_img = load_img(s1_path, is_gray_task=is_gray)
        s2_img = load_img(s2_path, is_gray_task=is_gray)

        fused_imgs = {}
        metrics_dict = {}
        for m_name in MODELS:
            f_path = PROJECT_ROOT / "results" / m_name / mod / im_name
            fused_imgs[m_name] = load_img(f_path, is_gray_task=is_gray)
            metrics_dict[m_name] = compute_metrics(s1_path, s2_path, f_path)

        all_imgs = [s1_img, s2_img, fused_imgs["MFS-Fusion"], fused_imgs["MMIF-CDDFuse"], fused_imgs["MMIF-INet"], fused_imgs["PSLPT"]]
        names = [f"Source 1 ({s1_name})", f"Source 2 ({s2_name})", "MFS-Fusion", "MMIF-CDDFuse", "MMIF-INet", "PSLPT"]

        for col in range(cols):
            ax = axes[r_idx, col]
            img = all_imgs[col].copy()

            # We insert the zoom inset into bottom-right corner (166:250, 166:250)
            crop = img[y1:y2, x1:x2]
            crop_resized = np.array(Image.fromarray(crop).resize((84, 84), Image.Resampling.NEAREST))

            # Put crop in bottom-right corner
            ih, iw, _ = img.shape
            ins_size = 84
            pad = 4
            ins_y1 = ih - ins_size - pad
            ins_y2 = ih - pad
            ins_x1 = iw - ins_size - pad
            ins_x2 = iw - pad

            img_with_inset = img.copy()
            img_with_inset[ins_y1:ins_y2, ins_x1:ins_x2] = crop_resized

            ax.imshow(img_with_inset)
            ax.set_xticks([])
            ax.set_yticks([])

            # Draw red bounding box on source ROI
            rect_roi = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.8, edgecolor="#ff1111", facecolor="none", linestyle="-"
            )
            ax.add_patch(rect_roi)

            # Draw red border on inset
            rect_inset = patches.Rectangle(
                (ins_x1, ins_y1), ins_size, ins_size,
                linewidth=2.0, edgecolor="#ff1111", facecolor="none", linestyle="-"
            )
            ax.add_patch(rect_inset)

            # Column header titles on the first row
            if r_idx == 0:
                ax.set_title(names[col], fontsize=12, fontweight="bold", pad=6)

            # Metric subtitles for fused columns
            if col >= 2:
                m_key = names[col]
                met = metrics_dict[m_key]
                sub_txt = f"PSNR: {met['PSNR_avg']:.2f}dB\nNABF1: {met['NABF1']:.4f}"
                ax.text(
                    0.05, 0.95, sub_txt,
                    transform=ax.transAxes,
                    fontsize=9, fontweight="bold",
                    color="white", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none")
                )

        # Modality row label on the left
        axes[r_idx, 0].text(
            -0.08, 0.5, f"{mod}\n({im_name})",
            transform=axes[r_idx, 0].transAxes,
            fontsize=12, fontweight="bold", va="center", ha="right", rotation=90
        )

    plt.suptitle(
        "Multi-Modal Medical Image Fusion Comparison (Paper-Style Evaluation)\n"
        "Red box & inset indicate zoomed ROI highlighting detail preservation vs. SOTA",
        fontsize=14, fontweight="bold", y=0.99
    )

    out_path = OUTPUT_DIR / "paper_style_master_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(DOCS_FIG_DIR / "paper_style_master_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved master figure -> {out_path}")


def main():
    print("Generating modality-specific figures with high-resolution zoomed crops...")
    # CT
    create_modality_figure(
        [SAMPLES[0], SAMPLES[1]],
        "MRI-CT Fusion: Qualitative Comparison & Structural Inspection\n(Gray-gray fusion postprocessed to clean grayscale, preserving bone and parenchyma details)",
        "mri_ct_paper_comparison.png"
    )
    # PET
    create_modality_figure(
        [SAMPLES[2], SAMPLES[3]],
        "MRI-PET Fusion: Qualitative Comparison & Artifact Inspection\n(Notice metabolic hotspot suppression and high edge artifacts NABF1 in MMIF-INet)",
        "mri_pet_paper_comparison.png"
    )
    # SPECT
    create_modality_figure(
        [SAMPLES[4], SAMPLES[5]],
        "MRI-SPECT Fusion: Qualitative Comparison & Artifact Inspection\n(Notice severe artificial noise injection NABF and 10dB PSNR deficit in MMIF-INet)",
        "mri_spect_paper_comparison.png"
    )

    print("Generating 3-modality Master Paper Comparison figure...")
    create_master_summary_figure()
    print("All visualization figures generated successfully!")


if __name__ == "__main__":
    main()

