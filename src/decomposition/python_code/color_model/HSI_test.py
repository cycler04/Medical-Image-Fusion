"""
HSI Color Space Module
======================
Converts an image to HSI (Hue, Saturation, Intensity) color space
and visualizes all channels alongside the original.

Usage:
    python hsi_color_space.py                      # uses built-in sample image
    python hsi_color_space.py path/to/image.jpg    # uses your own image

API:
    from hsi_color_space import rgb_to_hsi, hsi_to_rgb, visualize_hsi, load_image

    H, S, I = rgb_to_hsi(img_rgb)           # Convert RGB → HSI
    img_rgb = hsi_to_rgb(H, S, I)           # Convert HSI → RGB
    fig     = visualize_hsi(img_rgb)        # Full visualization
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

def rgb_to_hsi(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an RGB image to HSI color space (pure NumPy – no cv2 shortcut).

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image, either uint8 [0, 255] or float32 [0.0, 1.0].
        Shape must be (H, W, 3).

    Returns
    -------
    H : float32 ndarray, shape (H, W)  — Hue in radians [0, 2π]
    S : float32 ndarray, shape (H, W)  — Saturation     [0, 1]
    I : float32 ndarray, shape (H, W)  — Intensity      [0, 1]
    """
    rgb = img_rgb.astype(np.float32) / 255.0 if img_rgb.dtype == np.uint8 else img_rgb.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Intensity
    I = (R + G + B) / 3.0

    # Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.where(I > 0, 1.0 - min_rgb / (I + 1e-10), 0.0)

    # Hue
    num   = 0.5 * ((R - G) + (R - B))
    den   = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))

    H = np.where(B <= G, theta, 2 * np.pi - theta)

    return H.astype(np.float32), S.astype(np.float32), I.astype(np.float32)


def hsi_to_rgb(H: np.ndarray, S: np.ndarray, I: np.ndarray) -> np.ndarray:
    """
    Reconstruct an RGB image from HSI channels.

    Parameters
    ----------
    H : float32 ndarray — Hue in radians [0, 2π]
    S : float32 ndarray — Saturation     [0, 1]
    I : float32 ndarray — Intensity      [0, 1]

    Returns
    -------
    np.ndarray : uint8 RGB image, shape (H, W, 3)
    """
    H = H.copy()
    R = np.zeros_like(I)
    G = np.zeros_like(I)
    B = np.zeros_like(I)

    # Sector 1: 0 ≤ H < 2π/3
    mask1 = (H >= 0) & (H < 2 * np.pi / 3)
    B[mask1] = I[mask1] * (1 - S[mask1])
    R[mask1] = I[mask1] * (1 + S[mask1] * np.cos(H[mask1]) / np.cos(np.pi / 3 - H[mask1]))
    G[mask1] = 3 * I[mask1] - (R[mask1] + B[mask1])

    # Sector 2: 2π/3 ≤ H < 4π/3
    mask2 = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
    H2      = H[mask2] - 2 * np.pi / 3
    R[mask2] = I[mask2] * (1 - S[mask2])
    G[mask2] = I[mask2] * (1 + S[mask2] * np.cos(H2) / np.cos(np.pi / 3 - H2))
    B[mask2] = 3 * I[mask2] - (R[mask2] + G[mask2])

    # Sector 3: 4π/3 ≤ H < 2π
    mask3 = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)
    H3      = H[mask3] - 4 * np.pi / 3
    G[mask3] = I[mask3] * (1 - S[mask3])
    B[mask3] = I[mask3] * (1 + S[mask3] * np.cos(H3) / np.cos(np.pi / 3 - H3))
    R[mask3] = 3 * I[mask3] - (G[mask3] + B[mask3])

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

def visualize_hsi(img_rgb: np.ndarray,
                  save_path: str | None = None) -> plt.Figure:
    """
    Visualize HSI channels of *img_rgb* in a single figure.

    Layout
    ------
    Row 0 : Original RGB  |  (empty)  |  HSI → RGB reconstruction
    Row 1 : H (Hue)       |  S (Sat)  |  I (Intensity)
    Row 2 : H×S chroma    |  S×I vivid intensity  |  (empty)

    Parameters
    ----------
    img_rgb   : uint8 RGB image (H, W, 3)
    save_path : if provided, save figure to this path instead of showing

    Returns
    -------
    matplotlib.figure.Figure
    """
    H_ch, S_ch, I_ch = rgb_to_hsi(img_rgb)
    hsi_recon         = hsi_to_rgb(H_ch, S_ch, I_ch)

    H_vis  = _make_false_color(H_ch, vmin=0, vmax=2 * np.pi, cmap="hsv")
    hs_vis = _make_false_color(H_ch * S_ch, cmap="plasma")
    si_vis = _make_false_color(S_ch * I_ch,  cmap="inferno")

    # ── Layout ────────────────────────────────────────────────────────
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
    _show(fig.add_subplot(gs[0, 0]), img_rgb,   "Original (RGB)",         label="input")

    _show(fig.add_subplot(gs[0, 1]), H_vis, "H — Hue",        label="false-color (hsv cmap)")
    _show(fig.add_subplot(gs[0, 2]), S_ch,  "S — Saturation", cmap="magma", vmin=0, vmax=1, label="0 → 1")
    _show(fig.add_subplot(gs[1, 0]), I_ch,  "I — Intensity",  cmap="gray",  vmin=0, vmax=1, label="0 → 1")

    # Row 2 — composite channels
    _show(fig.add_subplot(gs[1, 1]), hs_vis, "H × S  (Chroma)",    label="hue weighted by saturation")
    _show(fig.add_subplot(gs[1, 2]), si_vis, "S × I  (Vivid I)",   label="saturation weighted by intensity")

    # Section labels
    # fig.text(0.04, 0.965, "ORIGINALS & RECONSTRUCTION", color="#888888", fontsize=9, fontweight="bold")
    # fig.text(0.04, 0.645, "HSI CHANNELS",               color="#4ecdc4", fontsize=9, fontweight="bold")
    # fig.text(0.04, 0.330, "HSI COMPOSITES",              color="#4ecdc4", fontsize=9, fontweight="bold")

    fig.suptitle("HSI Color Space — Hue · Saturation · Intensity",
                 color="white", fontsize=15, fontweight="bold", y=0.985)

    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✓ Saved → {save_path}")
    else:
        plt.show()

    return fig

def hue_plane_visualize():
    # --- HSI conversion ---
    def rgb_to_hsi(rgb):
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        I = (R + G + B) / 3.0

        min_rgb = np.minimum(np.minimum(R, G), B)
        S = np.where(I > 0, 1.0 - min_rgb / (I + 1e-10), 0.0)

        num = 0.5 * ((R - G) + (R - B))
        den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10
        theta = np.arccos(np.clip(num / den, -1.0, 1.0))
        H = np.where(B <= G, theta, 2 * np.pi - theta)

        return H, S, I


    # --- Generate plane R + G + B = 1 ---
    n = 200
    r = np.linspace(0, 1, n)
    g = np.linspace(0, 1, n)

    R, G = np.meshgrid(r, g)
    B = 1 - R - G

    # Keep only valid points (inside triangle)
    mask = (B >= 0)

    R = R[mask]
    G = G[mask]
    B = B[mask]

    rgb = np.stack([R, G, B], axis=-1)

    H, S, I = rgb_to_hsi(rgb)

    # --- Plot ---
    colors = plt.cm.hsv(H / (2 * np.pi))

    plt.figure(figsize=(6,6))
    plt.scatter(R, G, c=colors, s=5)

    plt.xlabel("R")
    plt.ylabel("G")
    plt.title("Hue (2D projection of chromaticity plane)")
    plt.axis('equal')
    plt.show()

# ─────────────────────────────────────────────
# 4. Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    image_path = "data/raw/Harvard-AANLIB/SPECT-MRI/SPECT/3015.png"
    img        = load_image(image_path)

    print(f"Image shape : {img.shape}")
    print(f"Image dtype : {img.dtype}")

    visualize_hsi(img)

    H, S, I = rgb_to_hsi(img)
    print("\n── HSI Channel Statistics ──────────────────────────────")
    print(f"  H : min={np.degrees(H.min()):6.1f}°  max={np.degrees(H.max()):6.1f}°  mean={np.degrees(H.mean()):6.1f}°")
    print(f"  S : min={S.min():.3f}     max={S.max():.3f}     mean={S.mean():.3f}")
    print(f"  I : min={I.min():.3f}     max={I.max():.3f}     mean={I.mean():.3f}")
