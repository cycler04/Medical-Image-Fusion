"""
HSV Color Space Module
======================
Converts an image to HSV (Hue, Saturation, Value) color space
and visualizes all channels alongside the original.

Usage:
    python hsv_color_space.py                      # uses built-in sample image
    python hsv_color_space.py path/to/image.jpg    # uses your own image

API:
    from hsv_color_space import rgb_to_hsv, hsv_to_rgb, visualize_hsv, load_image

    H, S, V = rgb_to_hsv(img_rgb)           # Convert RGB → HSV
    img_rgb = hsv_to_rgb(H, S, V)           # Convert HSV → RGB
    fig     = visualize_hsv(img_rgb)        # Full visualization

Key formulas
------------
    Value      V = max(R, G, B)              ← brightest channel
    Saturation S = (max − min) / max         ← chroma relative to peak
    Hue        H = arccos-based, radians     ← [0, 2π]  (same as HSI model)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import cv2


# ─────────────────────────────────────────────
# 1. Core conversion functions
# ─────────────────────────────────────────────

def rgb_to_hsv(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an RGB image to HSV color space (pure NumPy – no cv2 shortcut).

    The Hue formula uses the cosine / arccos approach (same as the HSI
    model) so both color spaces share the same H channel and can be
    compared directly.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image, either uint8 [0, 255] or float32 [0.0, 1.0].
        Shape must be (H, W, 3).

    Returns
    -------
    H : float32 ndarray, shape (H, W)  — Hue in radians [0, 2π]
    S : float32 ndarray, shape (H, W)  — Saturation     [0, 1]
    V : float32 ndarray, shape (H, W)  — Value          [0, 1]
    """
    rgb = img_rgb.astype(np.float32) / 255.0 if img_rgb.dtype == np.uint8 else img_rgb.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Value — brightest channel
    V = np.maximum(np.maximum(R, G), B)

    # Saturation — chroma relative to peak; 0 where V == 0 (pure black)
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.where(V > 0, (V - min_rgb) / (V + 1e-10), 0.0)

    # Hue — cosine-based formula, identical to HSI for direct comparison
    num   = 0.5 * ((R - G) + (R - B))
    den   = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    H     = np.where(B <= G, theta, 2 * np.pi - theta)

    return H.astype(np.float32), S.astype(np.float32), V.astype(np.float32)


def hsv_to_rgb(H: np.ndarray, S: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Reconstruct an RGB image from HSV channels.

    Uses the standard 60°-sector HSV → RGB decomposition:
        p = V(1 − S)
        q = V(1 − fS)
        t = V(1 − (1−f)S)
    where f is the fractional part of H / 60°.

    Parameters
    ----------
    H : float32 ndarray — Hue in radians [0, 2π]
    S : float32 ndarray — Saturation     [0, 1]
    V : float32 ndarray — Value          [0, 1]

    Returns
    -------
    np.ndarray : uint8 RGB image, shape (H, W, 3)
    """
    H_deg = np.degrees(H) % 360.0           # radians → [0, 360)
    Hi    = (H_deg / 60.0).astype(int) % 6  # sector index 0-5
    f     = (H_deg / 60.0) - (H_deg / 60.0).astype(int)

    p = V * (1.0 - S)
    q = V * (1.0 - f * S)
    t = V * (1.0 - (1.0 - f) * S)

    R = np.select([Hi == 0, Hi == 1, Hi == 2, Hi == 3, Hi == 4, Hi == 5],
                  [V,       q,       p,       p,       t,       V      ], default=V)
    G = np.select([Hi == 0, Hi == 1, Hi == 2, Hi == 3, Hi == 4, Hi == 5],
                  [t,       V,       V,       q,       p,       p      ], default=V)
    B = np.select([Hi == 0, Hi == 1, Hi == 2, Hi == 3, Hi == 4, Hi == 5],
                  [p,       p,       t,       V,       V,       q      ], default=V)

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 2. Helpers
# ─────────────────────────────────────────────

def _make_false_color(channel: np.ndarray, vmin=None, vmax=None,
                      cmap: str = "viridis") -> np.ndarray:
    """Return an H×W×3 uint8 false-color render of a single-channel array."""
    norm = Normalize(vmin=vmin if vmin is not None else channel.min(),
                     vmax=vmax if vmax is not None else channel.max())
    cm   = plt.get_cmap(cmap)
    rgba = cm(norm(channel))
    return (rgba[..., :3] * 255).astype(np.uint8)


def load_image(path: str | None = None) -> np.ndarray:
    """
    Load an image as an RGB uint8 ndarray.
    If *path* is None or omitted, a built-in colorful test image is returned.
    """
    if path:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Built-in sample: colorful gradient + geometric shapes ──
    h, w   = 300, 400
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for x in range(w):
        hue = int(x / w * 180)
        col = np.array(cv2.cvtColor(
            np.uint8([[[hue, 220, 200]]]), cv2.COLOR_HSV2BGR)[0][0])
        canvas[:, x] = col[::-1]

    for y in range(h):
        factor    = 0.4 + 0.6 * y / h
        canvas[y] = np.clip(canvas[y] * factor, 0, 255).astype(np.uint8)

    for (cx, cy, r, color) in [
        (100, 100, 60, (220,  50,  50)),
        (200, 180, 70, ( 50, 180, 220)),
        (320, 120, 55, ( 50, 220,  80)),
        (150, 230, 45, (230, 200,  30)),
        (310, 240, 50, (180,  50, 200)),
    ]:
        cv2.circle(canvas, (cx, cy), r, color, -1)

    pts = np.array([[0, 260], [140, 260], [200, 300], [0, 300]], np.int32)
    cv2.fillPoly(canvas, [pts], (240, 240, 240))

    return canvas


# ─────────────────────────────────────────────
# 3. Visualisation
# ─────────────────────────────────────────────

def visualize_hsv(img_rgb: np.ndarray,
                  save_path: str | None = None) -> plt.Figure:
    """
    Visualize HSV channels of *img_rgb* in a single figure.

    Layout
    ------
    Row 0 : Original RGB  |  (empty)        |  HSV → RGB reconstruction
    Row 1 : H (Hue)       |  S (Saturation) |  V (Value)
    Row 2 : H×S chroma    |  S×V vivid-V    |  (empty)

    Composite panels
    ----------------
    • H × S  — hue weighted by saturation: highlights richly-colored regions
    • S × V  — saturation weighted by value: bright AND colorful areas pop

    Parameters
    ----------
    img_rgb   : uint8 RGB image (H, W, 3)
    save_path : if provided, save figure to this path instead of showing

    Returns
    -------
    matplotlib.figure.Figure
    """
    H_ch, S_ch, V_ch = rgb_to_hsv(img_rgb)
    hsv_recon         = hsv_to_rgb(H_ch, S_ch, V_ch)

    H_vis  = _make_false_color(H_ch, vmin=0, vmax=2 * np.pi, cmap="hsv")
    hs_vis = _make_false_color(H_ch * S_ch, cmap="plasma")
    sv_vis = _make_false_color(S_ch * V_ch,  cmap="inferno")

    # ── Layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10), facecolor="#0f0f0f")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.40, wspace=0.20,
                            left=0.04, right=0.96, top=0.91, bottom=0.04)

    TITLE_KW = dict(color="white",   fontsize=11, fontweight="bold", pad=6)
    LABEL_KW = dict(color="#aaaaaa", fontsize=9)

    def _show(ax, data, title, cmap=None, vmin=None, vmax=None, label=""):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_title(title, **TITLE_KW)
        ax.set_xlabel(label, **LABEL_KW)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")

    def _blank(ax):
        ax.set_visible(False)

    # Row 0 — original & reconstruction
    _show(fig.add_subplot(gs[0, 0]), img_rgb,   "Original (RGB)",          label="input")

    _show(fig.add_subplot(gs[0, 1]), H_vis, "H — Hue",        label="false-color (hsv cmap)")
    _show(fig.add_subplot(gs[0, 2]), S_ch,  "S — Saturation", cmap="magma",   vmin=0, vmax=1,
          label="0 → 1  (chroma / peak)")
    _show(fig.add_subplot(gs[1, 0]), V_ch,  "V — Value",      cmap="gray",    vmin=0, vmax=1,
          label="0 → 1  (max of channels)")

    # Row 2 — composite channels
    _show(fig.add_subplot(gs[1, 1]), hs_vis, "H × S  (Chroma)",   label="hue weighted by saturation")
    _show(fig.add_subplot(gs[1, 2]), sv_vis, "S × V  (Vivid V)",  label="saturation weighted by value")

    # Section labels  (warm orange palette to distinguish from HSI teal)
    # fig.text(0.04, 0.965, "ORIGINALS & RECONSTRUCTION", color="#888888", fontsize=9, fontweight="bold")
    # fig.text(0.04, 0.645, "HSV CHANNELS",               color="#f4a261", fontsize=9, fontweight="bold")
    # fig.text(0.04, 0.330, "HSV COMPOSITES",              color="#f4a261", fontsize=9, fontweight="bold")

    fig.suptitle("HSV Color Space — Hue · Saturation · Value",
                 color="white", fontsize=15, fontweight="bold", y=0.985)

    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✓ Saved → {save_path}")
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────
# 4. Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    image_path = "data/raw/Harvard-AANLIB/SPECT-MRI/SPECT/3015.png"
    img        = load_image(image_path)

    print(f"Image shape : {img.shape}")
    print(f"Image dtype : {img.dtype}")

    visualize_hsv(img)

    H, S, V = rgb_to_hsv(img)
    print("\n── HSV Channel Statistics ──────────────────────────────")
    print(f"  H : min={np.degrees(H.min()):6.1f}°  max={np.degrees(H.max()):6.1f}°  mean={np.degrees(H.mean()):6.1f}°")
    print(f"  S : min={S.min():.3f}     max={S.max():.3f}     mean={S.mean():.3f}")
    print(f"  V : min={V.min():.3f}     max={V.max():.3f}     mean={V.mean():.3f}")
