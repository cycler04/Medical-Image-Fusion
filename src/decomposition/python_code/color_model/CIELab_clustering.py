"""
LAB Cluster Visualizer — Two Image Folders
===========================================
Samples pixels from two image folders, converts to CIELAB, and plots them
side-by-side in a multi-panel scatter to reveal color clustering.

Layout (2 × 3)
--------------
Row 0 : a*–b* plane (main cluster view)  |  L*–a* plane  |  L*–b* plane
Row 1 : Strip previews of folder A images | Strip previews of folder B images | 3-D scatter (L*, a*, b*)

Usage:
    python lab_cluster_viz.py                           # demo: uses two auto-generated folders
    python lab_cluster_viz.py path/to/A  path/to/B
    python lab_cluster_viz.py path/to/A  path/to/B --samples 8000 --title "SPECT vs MRI"
"""

import sys
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3-D projection)
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Core CIELAB conversion  (copied from cielab_color_space.py)
# ─────────────────────────────────────────────────────────────────────────────

_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)


def _srgb_linearise(u):
    return np.where(u <= 0.04045, u / 12.92,
                    ((u + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _f_lab(t):
    d = 6.0 / 29.0
    return np.where(t > d ** 3, np.cbrt(t),
                    t / (3.0 * d ** 2) + 4.0 / 29.0).astype(np.float32)


def rgb_to_lab(img_rgb: np.ndarray):
    rgb    = img_rgb.astype(np.float32) / 255.0
    lin    = _srgb_linearise(rgb)
    M      = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz    = lin @ M.T
    f      = _f_lab(xyz / _D65)
    L      = 116.0 * f[..., 1] - 16.0
    a      = 500.0 * (f[..., 0] - f[..., 1])
    b      = 200.0 * (f[..., 1] - f[..., 2])
    return L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Image loading helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")


def _find_images(folder: str) -> list[str]:
    paths = []
    for ext in _IMG_EXTS:
        paths += glob.glob(os.path.join(folder, ext))
        paths += glob.glob(os.path.join(folder, ext.upper()))
    return sorted(set(paths))


def _load_rgb(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _sample_lab(paths: list[str], n_total: int, rng: np.random.Generator):
    """
    Load all images in *paths*, convert to LAB, and return a flat random
    subsample of *n_total* pixels as (L, a, b) float32 arrays.
    """
    L_list, a_list, b_list = [], [], []
    per_image = max(1, n_total // max(len(paths), 1))

    for p in paths:
        img = _load_rgb(p)
        if img is None:
            print(f"  [warn] skipping unreadable file: {p}")
            continue
        L, a, b = rgb_to_lab(img)
        flat    = np.stack([L.ravel(), a.ravel(), b.ravel()], axis=1)  # (N, 3)
        n       = min(per_image, len(flat))
        idx     = rng.choice(len(flat), size=n, replace=False)
        L_list.append(flat[idx, 0])
        a_list.append(flat[idx, 1])
        b_list.append(flat[idx, 2])

    if not L_list:
        raise RuntimeError("No valid images found in folder.")

    return (np.concatenate(L_list),
            np.concatenate(a_list),
            np.concatenate(b_list))


def _strip_previews(paths: list[str], n: int = 6, h: int = 80) -> list[np.ndarray]:
    """Return up to *n* thumbnails (all same height *h*)."""
    thumbs = []
    for p in paths[:n]:
        img = _load_rgb(p)
        if img is None:
            continue
        scale = h / img.shape[0]
        w     = max(1, int(img.shape[1] * scale))
        thumbs.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
    return thumbs


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Pixel colouring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lab_to_display_color(a_arr, b_arr, L_arr,
                           base_hue_deg: float = 0.0,
                           lightness_val: int   = 210) -> np.ndarray:
    """
    Map each pixel's chroma hue (from a* / b*) to an HSV display colour,
    tinted toward *base_hue_deg* by 30 % for per-folder identity.
    Returns float32 (N, 3) in [0, 1].
    """
    hue_rad   = np.arctan2(b_arr, a_arr)
    hue_norm  = (hue_rad + np.pi) / (2.0 * np.pi)
    hue_shift = (hue_norm + base_hue_deg / 360.0) % 1.0
    chroma    = np.sqrt(a_arr ** 2 + b_arr ** 2)
    sat       = np.clip(chroma / 90.0, 0.05, 1.0)

    h_u8 = (hue_shift * 179).astype(np.uint8).reshape(-1, 1, 1)
    s_u8 = (sat       * 255).astype(np.uint8).reshape(-1, 1, 1)
    v_u8 = np.full_like(h_u8, lightness_val)
    hsv  = np.concatenate([h_u8, s_u8, v_u8], axis=2)
    bgr  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr[:, 0, ::-1].astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Demo data (when no real folders are provided)
# ─────────────────────────────────────────────────────────────────────────────

def _make_demo_images(tmpdir_a: str, tmpdir_b: str, n: int = 8) -> None:
    """
    Generate two sets of synthetic images with distinct colour palettes.
    Folder A → warm reds / oranges.
    Folder B → cool blues / greens.
    """
    import tempfile, os
    os.makedirs(tmpdir_a, exist_ok=True)
    os.makedirs(tmpdir_b, exist_ok=True)

    rng = np.random.default_rng(0)

    def _make(folder, hue_center, hue_range, n_images):
        for i in range(n_images):
            h, w   = 128, 128
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            hue    = int((hue_center + rng.uniform(-hue_range, hue_range)) % 180)
            sat    = int(rng.uniform(140, 230))
            val    = int(rng.uniform(130, 220))
            col    = cv2.cvtColor(np.uint8([[[hue, sat, val]]]),
                                   cv2.COLOR_HSV2BGR)[0][0]
            canvas[:] = col
            # add 2-3 blobs of nearby hue
            for _ in range(rng.integers(2, 5)):
                h2  = int((hue + rng.uniform(-20, 20)) % 180)
                s2  = int(rng.uniform(120, 255))
                v2  = int(rng.uniform(120, 255))
                c2  = cv2.cvtColor(np.uint8([[[h2, s2, v2]]]),
                                    cv2.COLOR_HSV2BGR)[0][0]
                cx, cy, r = rng.integers(10, w-10), rng.integers(10, h-10), rng.integers(10, 40)
                cv2.circle(canvas, (cx, cy), int(r), c2.tolist(), -1)
            # subtle noise
            noise = rng.integers(-15, 15, canvas.shape, dtype=np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(folder, f"img_{i:02d}.png"), canvas)

    _make(tmpdir_a, hue_center=10,  hue_range=15, n_images=n)   # warm
    _make(tmpdir_b, hue_center=110, hue_range=20, n_images=n)   # cool


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_clusters(
    folder_a: str,
    folder_b: str,
    label_a:  str  = "Folder A",
    label_b:  str  = "Folder B",
    n_samples: int = 6000,
    title:    str  = "",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Compare two image folders in CIELAB scatter space.

    Parameters
    ----------
    folder_a, folder_b : paths to image folders
    label_a, label_b   : legend labels
    n_samples          : total pixels sampled *per folder*
    title              : optional figure super-title
    save_path          : if given, save the figure here
    """
    rng = np.random.default_rng(42)

    # ── Load images ──────────────────────────────────────────────────────────
    paths_a = _find_images(folder_a)
    paths_b = _find_images(folder_b)

    if not paths_a:
        raise FileNotFoundError(f"No images found in: {folder_a}")
    if not paths_b:
        raise FileNotFoundError(f"No images found in: {folder_b}")

    print(f"  {label_a}: {len(paths_a)} images  →  sampling {n_samples} pixels")
    print(f"  {label_b}: {len(paths_b)} images  →  sampling {n_samples} pixels")

    La, aa, ba = _sample_lab(paths_a, n_samples, rng)
    Lb, ab, bb = _sample_lab(paths_b, n_samples, rng)

    # ── Per-pixel display colours (hue-shifted by folder for identity) ───────
    ca = _lab_to_display_color(aa, ba, La, base_hue_deg=0,   lightness_val=210)
    cb = _lab_to_display_color(ab, bb, Lb, base_hue_deg=130, lightness_val=200)

    # Flat folder identity colours for legend patches
    COLOR_A = "#e06030"
    COLOR_B = "#3090d0"

    # ── Figure ───────────────────────────────────────────────────────────────
    BG    = "#0f0f0f"
    ALPHA = 0.45
    S     = 3

    fig = plt.figure(figsize=(18, 12), facecolor=BG)

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[3, 1],
        hspace=0.35, wspace=0.22,
        left=0.05, right=0.97, top=0.91, bottom=0.05,
    )

    AX_KW   = dict(facecolor="#181818")
    TTL_KW  = dict(color="white",   fontsize=11, fontweight="bold", pad=6)
    LBL_KW  = dict(color="#aaaaaa", fontsize=9)
    TKS_KW  = dict(colors="#666666", labelsize=8)
    SPR_KW  = dict(edgecolor="#2a2a2a")

    def _style(ax, xlabel, ylabel):
        ax.set_xlabel(xlabel, **LBL_KW)
        ax.set_ylabel(ylabel, **LBL_KW)
        ax.tick_params(**TKS_KW)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a2a")
        ax.axhline(0, color="#444", lw=0.7, ls="--")
        ax.axvline(0, color="#444", lw=0.7, ls="--")

    LIMS = 135

    # ────────────────────────────────────────────────────────────────────────
    # [0,0]  a*–b*  (main chroma plane)
    # ────────────────────────────────────────────────────────────────────────
    ax_ab = fig.add_subplot(gs[0, 0], **AX_KW)
    ax_ab.scatter(aa, ba, c=ca, s=S, alpha=ALPHA, linewidths=0, label=label_a)
    ax_ab.scatter(ab, bb, c=cb, s=S, alpha=ALPHA, linewidths=0, label=label_b)
    ax_ab.set_title("a* – b*  Chroma plane", **TTL_KW)
    ax_ab.set_xlim(-LIMS, LIMS);  ax_ab.set_ylim(-LIMS, LIMS)
    _style(ax_ab, "a*  (green ← · → red)", "b*  (blue ↓ · ↑ yellow)")

    # Quadrant labels
    kw = dict(fontsize=7.5, alpha=0.45, ha="center", color="white")
    ax_ab.text( 85,  85, "Red–Yellow",   **kw)
    ax_ab.text(-85,  85, "Green–Yellow", **kw)
    ax_ab.text( 85, -85, "Red–Blue",     **kw)
    ax_ab.text(-85, -85, "Green–Blue",   **kw)

    # ────────────────────────────────────────────────────────────────────────
    # [0,1]  L*–a*
    # ────────────────────────────────────────────────────────────────────────
    ax_La = fig.add_subplot(gs[0, 1], **AX_KW)
    ax_La.scatter(aa, La, c=ca, s=S, alpha=ALPHA, linewidths=0)
    ax_La.scatter(ab, Lb, c=cb, s=S, alpha=ALPHA, linewidths=0)
    ax_La.set_title("L* – a*  Lightness vs. Red–Green", **TTL_KW)
    ax_La.set_xlim(-LIMS, LIMS);  ax_La.set_ylim(0, 100)
    _style(ax_La, "a*  (green ← · → red)", "L*  (dark → light)")
    ax_La.axhline(0, lw=0)   # override double axhline from _style for L* axis

    # ────────────────────────────────────────────────────────────────────────
    # [0,2]  L*–b*
    # ────────────────────────────────────────────────────────────────────────
    ax_Lb = fig.add_subplot(gs[0, 2], **AX_KW)
    ax_Lb.scatter(ba, La, c=ca, s=S, alpha=ALPHA, linewidths=0)
    ax_Lb.scatter(bb, Lb, c=cb, s=S, alpha=ALPHA, linewidths=0)
    ax_Lb.set_title("L* – b*  Lightness vs. Blue–Yellow", **TTL_KW)
    ax_Lb.set_xlim(-LIMS, LIMS);  ax_Lb.set_ylim(0, 100)
    _style(ax_Lb, "b*  (blue ← · → yellow)", "L*  (dark → light)")

    # ────────────────────────────────────────────────────────────────────────
    # [1,0]  Image strip — folder A
    # [1,1]  Image strip — folder B
    # ────────────────────────────────────────────────────────────────────────
    for col_idx, (paths, label, clr) in enumerate(
        [(paths_a, label_a, COLOR_A),
         (paths_b, label_b, COLOR_B)]
    ):
        ax_strip = fig.add_subplot(gs[1, col_idx], **AX_KW)
        thumbs   = _strip_previews(paths, n=7, h=70)

        if thumbs:
            combined = np.concatenate(thumbs, axis=1)
            ax_strip.imshow(combined, interpolation="bilinear")
        else:
            ax_strip.text(0.5, 0.5, "no preview", color="#888",
                          ha="center", va="center", transform=ax_strip.transAxes)

        ax_strip.set_xticks([]); ax_strip.set_yticks([])
        for sp in ax_strip.spines.values():
            sp.set_edgecolor(clr)
            sp.set_linewidth(2)
        ax_strip.set_title(
            f"{label}  ({len(paths)} images · {len(La if col_idx==0 else Lb):,} sampled px)",
            color=clr, fontsize=10, fontweight="bold", pad=5,
        )

    # ────────────────────────────────────────────────────────────────────────
    # [1,2]  3-D scatter  L*, a*, b*
    # ────────────────────────────────────────────────────────────────────────
    ax3d = fig.add_subplot(gs[1, 2], projection="3d", facecolor="#181818")
    ax3d.scatter(aa, ba, La, c=ca, s=2, alpha=0.35, linewidths=0)
    ax3d.scatter(ab, bb, Lb, c=cb, s=2, alpha=0.35, linewidths=0)
    ax3d.set_xlabel("a*", color="#aaa", fontsize=8, labelpad=3)
    ax3d.set_ylabel("b*", color="#aaa", fontsize=8, labelpad=3)
    ax3d.set_zlabel("L*", color="#aaa", fontsize=8, labelpad=3)
    ax3d.set_xlim(-LIMS, LIMS); ax3d.set_ylim(-LIMS, LIMS); ax3d.set_zlim(0, 100)
    ax3d.tick_params(colors="#555", labelsize=7)
    ax3d.set_title("3-D  L*a*b*", color="white", fontsize=10,
                    fontweight="bold", pad=4)
    ax3d.view_init(elev=25, azim=-55)
    ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#222")
    ax3d.yaxis.pane.set_edgecolor("#222")
    ax3d.zaxis.pane.set_edgecolor("#222")
    ax3d.grid(True, color="#2a2a2a", linewidth=0.5)

    # ── Global legend ─────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=COLOR_A, label=label_a),
        mpatches.Patch(color=COLOR_B, label=label_b),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.97),
        framealpha=0.15, edgecolor="#444",
        labelcolor="white", fontsize=10,
    )

    # ── Super-title ────────────────────────────────────────────────────────────
    suptitle = title or f"{label_a}  vs.  {label_b}  —  CIELAB colour distribution"
    fig.suptitle(suptitle, color="white", fontsize=14, fontweight="bold", y=0.97)

    fig.text(0.05, 0.01,
             "D65 illuminant · sRGB linearisation · BT.709 primaries",
             color="#444", fontsize=7.5)

    # ── Stats printout ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'Channel':<8}  {'Folder':<16}  {'min':>7}  {'max':>7}  {'mean':>7}  {'std':>7}")
    print(f"{'─'*60}")
    for ch, va, vb in [("L*", La, Lb), ("a*", aa, ab), ("b*", ba, bb)]:
        print(f"  {ch:<8}  {label_a:<16}  {va.min():>7.2f}  {va.max():>7.2f}"
              f"  {va.mean():>7.2f}  {va.std():>7.2f}")
        print(f"  {'':8}  {label_b:<16}  {vb.min():>7.2f}  {vb.max():>7.2f}"
              f"  {vb.mean():>7.2f}  {vb.std():>7.2f}")
    print(f"{'─'*60}\n")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✓ Saved → {save_path}")
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    folder_a = "data/raw/Harvard-AANLIB/SPECT-MRI/SPECT"
    folder_b = "data/raw/Harvard-AANLIB/SPECT-MRI/MRI"

    label_a = "SPECT"
    label_b = "MRI"
    
    # ── Demo mode ────────────────────────────────────────────────────────────
    if folder_a is None or folder_b is None:
        import tempfile
        tmp = tempfile.mkdtemp()
        folder_a = os.path.join(tmp, "warm_reds")
        folder_b = os.path.join(tmp, "cool_blues")
        print("No folders supplied → generating demo images …")
        _make_demo_images(folder_a, folder_b, n=10)

    visualize_clusters(
        folder_a  = folder_a,
        folder_b  = folder_b,
        label_a   = label_a,
        label_b   = label_b,
        n_samples = 6000,
        title     = "SPECT vs MRI  —  CIELAB colour distribution",
        save_path = None,
    )
