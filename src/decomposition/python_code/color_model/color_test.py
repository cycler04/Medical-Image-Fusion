﻿"""
Color Space Comparison
======================
Imports HSI, HSV, and YCbCr modules and renders a single
side-by-side comparison figure for one input image.

Usage:
    python color_space_comparison.py                      # built-in sample
    python color_space_comparison.py path/to/image.jpg   # your image

Dependencies (must be in the same directory):
    HSI_test.py   HSV_test.py   YCrCb_test.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

# ── Import the three color-space modules ──────────────────────────────
from decomposition.python_code.color_model.HSI_test   import rgb_to_hsi,   hsi_to_rgb,   load_image as load_hsi
from decomposition.python_code.color_model.HSV_test   import rgb_to_hsv,   hsv_to_rgb
from decomposition.python_code.color_model.YCrCb_test import rgb_to_ycbcr, ycbcr_to_rgb


# ─────────────────────────────────────────────────────────────────────
# Helper: false-color render of a single-channel array
# ─────────────────────────────────────────────────────────────────────
def _false_color(ch, vmin=None, vmax=None, cmap="viridis"):
    norm = Normalize(vmin=ch.min() if vmin is None else vmin,
                     vmax=ch.max() if vmax is None else vmax)
    rgba = plt.get_cmap(cmap)(norm(ch))
    return (rgba[..., :3] * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# Master comparison figure
# ─────────────────────────────────────────────────────────────────────
def compare_all(img_rgb: np.ndarray, save_path: str | None = None):
    """
    Convert img_rgb through all three color spaces and plot everything
    in one figure.

    Layout (5 rows x 4 cols):
        Row 0 : Original | HSI recon  | HSV recon  | YCbCr recon
        Row 1 : label    | H  | S  | I        (HSI channels)
        Row 2 : label    | H  | S  | V        (HSV channels)
        Row 3 : label    | Y  | Cb | Cr       (YCbCr channels)
        Row 4 : stats    | H x S (HSI) | S x V (HSV) | Chroma (YCbCr)
    """

    # ── Conversions ───────────────────────────────────────────────────
    H_hsi, S_hsi, I_hsi = rgb_to_hsi(img_rgb)
    H_hsv, S_hsv, V_hsv = rgb_to_hsv(img_rgb)
    Y_y,   Cb_y,  Cr_y  = rgb_to_ycbcr(img_rgb)

    recon_hsi   = hsi_to_rgb(H_hsi, S_hsi, I_hsi)
    recon_hsv   = hsv_to_rgb(H_hsv, S_hsv, V_hsv)
    recon_ycbcr = ycbcr_to_rgb(Y_y, Cb_y, Cr_y)

    # ── Composite / false-color renders ───────────────────────────────
    H_fc_hsi = _false_color(H_hsi, 0, 2*np.pi, "hsv")
    H_fc_hsv = _false_color(H_hsv, 0, 2*np.pi, "hsv")

    hs_comp  = _false_color(H_hsi * S_hsi, cmap="plasma")   # HSI  H x S
    sv_comp  = _false_color(S_hsv * V_hsv, cmap="inferno")  # HSV  S x V

    chroma_r   = np.clip((Cr_y - 128) / 128 + 0.5, 0, 1)
    chroma_b   = np.clip((Cb_y - 128) / 128 + 0.5, 0, 1)
    chroma_g   = 1 - np.clip(chroma_r * 0.5 + chroma_b * 0.5, 0, 1)
    chroma_vis = np.stack([chroma_r, chroma_g, chroma_b], axis=-1)

    # ── Theme ─────────────────────────────────────────────────────────
    BG     = "#0c0c0f"
    FG     = "white"
    ACCENT = {"hsi": "#4ecdc4", "hsv": "#f4a261", "ycbcr": "#f7b731"}
    MUTED  = "#333344"

    TK = dict(color=FG,       fontsize=10, fontweight="bold", pad=5)
    LK = dict(color="#888899", fontsize=8)

    # ── Figure / grid ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22), facecolor=BG)
    gs  = gridspec.GridSpec(
        5, 4, figure=fig,
        hspace=0.38, wspace=0.18,
        left=0.01, right=0.99, top=0.95, bottom=0.03,
        height_ratios=[1.1, 1, 1, 1, 1],
    )

    def _show(ax, data, title, sub="", cmap=None, vmin=None, vmax=None, border=MUTED):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_title(title, **TK)
        if sub:
            ax.set_xlabel(sub, **LK)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(1.8)

    def _panel(ax):
        ax.set_facecolor("#111118")
        for sp in ax.spines.values():
            sp.set_edgecolor(MUTED); sp.set_linewidth(1)
        ax.set_xticks([]); ax.set_yticks([])

    # ── Row 0 : originals & reconstructions ──────────────────────────
    _show(fig.add_subplot(gs[0, 0]), img_rgb,     "Original (RGB)",    sub="input", border="#556677")
    _show(fig.add_subplot(gs[0, 1]), recon_hsi,   "HSI -> RGB recon",  sub="sanity check", border=ACCENT["hsi"])
    _show(fig.add_subplot(gs[0, 2]), recon_hsv,   "HSV -> RGB recon",  sub="sanity check", border=ACCENT["hsv"])
    _show(fig.add_subplot(gs[0, 3]), recon_ycbcr, "YCbCr -> RGB recon",sub="sanity check", border=ACCENT["ycbcr"])

    # ── Row 1 : HSI individual channels ──────────────────────────────
    _show(fig.add_subplot(gs[1, 1]), H_fc_hsi, "H - Hue",        sub="false-color hsv cmap", border=ACCENT["hsi"])
    _show(fig.add_subplot(gs[1, 2]), S_hsi,    "S - Saturation",  sub="0 -> 1", cmap="magma", vmin=0, vmax=1, border=ACCENT["hsi"])
    _show(fig.add_subplot(gs[1, 3]), I_hsi,    "I - Intensity",   sub="0 -> 1  (mean R,G,B)", cmap="gray", vmin=0, vmax=1, border=ACCENT["hsi"])

    # ── Row 2 : HSV individual channels ──────────────────────────────
    _show(fig.add_subplot(gs[2, 1]), H_fc_hsv, "H - Hue",        sub="false-color hsv cmap", border=ACCENT["hsv"])
    _show(fig.add_subplot(gs[2, 2]), S_hsv,    "S - Saturation",  sub="0 -> 1  (chroma/peak)", cmap="magma", vmin=0, vmax=1, border=ACCENT["hsv"])
    _show(fig.add_subplot(gs[2, 3]), V_hsv,    "V - Value",       sub="0 -> 1  (max R,G,B)", cmap="gray", vmin=0, vmax=1, border=ACCENT["hsv"])

    # ── Row 3 : YCbCr individual channels ────────────────────────────
    _show(fig.add_subplot(gs[3, 1]), Y_y,  "Y - Luma",        sub="0 -> 255  (weighted)", cmap="gray", vmin=0, vmax=255, border=ACCENT["ycbcr"])
    _show(fig.add_subplot(gs[3, 2]), Cb_y, "Cb - Blue diff",   sub="0 -> 255  (128=neutral)", cmap="Blues_r", vmin=0, vmax=255, border=ACCENT["ycbcr"])
    _show(fig.add_subplot(gs[3, 3]), Cr_y, "Cr - Red diff",    sub="0 -> 255  (128=neutral)", cmap="Reds", vmin=0, vmax=255, border=ACCENT["ycbcr"])

    # ── Row 4 : composites ────────────────────────────────────────────
    _show(fig.add_subplot(gs[4, 1]), hs_comp,    "HSI  H x S",       sub="hue weighted by saturation", border=ACCENT["hsi"])
    _show(fig.add_subplot(gs[4, 2]), sv_comp,    "HSV  S x V",       sub="saturation weighted by value", border=ACCENT["hsv"])
    _show(fig.add_subplot(gs[4, 3]), chroma_vis, "YCbCr Chroma map", sub="Cb (blue) <-> Cr (red)", border=ACCENT["ycbcr"])

    # ── Left-column: row labels (col 0, rows 1-3) ─────────────────────
    label_defs = [
        (gs[1, 0], ACCENT["hsi"],   "HSI",   ["H = Hue  [0, 2pi]", "S = 1 - min/I", "I = mean(R,G,B)"]),
        (gs[2, 0], ACCENT["hsv"],   "HSV",   ["H = Hue  [0, 2pi]", "S = (max-min)/max", "V = max(R,G,B)"]),
        (gs[3, 0], ACCENT["ycbcr"], "YCbCr", ["Y  = 0.299R+0.587G+0.114B", "Cb = B-Y  (+128 offset)", "Cr = R-Y  (+128 offset)"]),
    ]
    for spec, color, name, lines in label_defs:
        ax = fig.add_subplot(spec)
        _panel(ax)
        ax.text(0.5, 0.88, name, transform=ax.transAxes, color=color,
                fontsize=13, fontweight="bold", ha="center", va="top")
        for i, line in enumerate(lines):
            ax.text(0.5, 0.60 - i * 0.22, line, transform=ax.transAxes,
                    color="#aaaacc", fontsize=7.8, ha="center", va="top",
                    family="monospace")

    # ── Bottom-left: stats panel ──────────────────────────────────────
    stats_ax = fig.add_subplot(gs[4, 0])
    _panel(stats_ax)
    stats_ax.text(0.5, 0.94, "Channel Stats (mean)", transform=stats_ax.transAxes,
                  color=FG, fontsize=9, fontweight="bold", ha="center", va="top")
    rows = [
        (ACCENT["hsi"],   f"HSI  H={np.degrees(H_hsi.mean()):5.1f}  S={S_hsi.mean():.3f}  I={I_hsi.mean():.3f}"),
        (ACCENT["hsv"],   f"HSV  H={np.degrees(H_hsv.mean()):5.1f}  S={S_hsv.mean():.3f}  V={V_hsv.mean():.3f}"),
        (ACCENT["ycbcr"], f"Y={Y_y.mean():5.1f}  Cb={Cb_y.mean():5.1f}  Cr={Cr_y.mean():5.1f}"),
    ]
    for i, (color, txt) in enumerate(rows):
        stats_ax.text(0.5, 0.65 - i * 0.25, txt, transform=stats_ax.transAxes,
                      color=color, fontsize=7.5, ha="center", va="top",
                      family="monospace")

    # ── Row section labels (left margin) ──────────────────────────────
    for y, color, label in [
        (0.965, "#888899",       "ORIGINALS & RECONSTRUCTIONS"),
        (0.785, ACCENT["hsi"],   "HSI CHANNELS"),
        (0.600, ACCENT["hsv"],   "HSV CHANNELS"),
        (0.415, ACCENT["ycbcr"], "YCbCr CHANNELS"),
        (0.220, "#9b8ecf",       "COMPOSITE CHANNELS"),
    ]:
        fig.text(0.005, y, label, color=color, fontsize=8.5,
                 fontweight="bold", va="center")

    fig.suptitle("Color Space Comparison  -  HSI  vs  HSV  vs  YCbCr",
                 color=FG, fontsize=16, fontweight="bold", y=0.978)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved -> {save_path}")
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    image_path = "data/raw/Harvard-AANLIB/SPECT-MRI/SPECT/3015.png"

    # load_image from HSI_test is identical across all three modules
    img = load_hsi(image_path)
    print(f"Image : {image_path or '(built-in sample)'}")
    print(f"Shape : {img.shape}  |  dtype: {img.dtype}")

    compare_all(img)
