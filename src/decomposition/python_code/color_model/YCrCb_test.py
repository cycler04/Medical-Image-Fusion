"""
YCbCr Color Space Module
========================
Converts an image to YCbCr color space (BT.601 coefficients) and
visualizes all channels alongside the original.

Usage:
    python ycbcr_color_space.py                      # uses built-in sample image
    python ycbcr_color_space.py path/to/image.jpg    # uses your own image

API:
    from ycbcr_color_space import rgb_to_ycbcr, ycbcr_to_rgb, visualize_ycbcr, load_image

    Y, Cb, Cr = rgb_to_ycbcr(img_rgb)          # Convert RGB → YCbCr
    img_rgb   = ycbcr_to_rgb(Y, Cb, Cr)        # Convert YCbCr → RGB
    fig       = visualize_ycbcr(img_rgb)        # Full visualization
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

def rgb_to_ycbcr(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an RGB image to YCbCr color space.
    Uses BT.601 full-swing coefficients (same as JPEG).

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image (uint8 or float32). Shape must be (H, W, 3).

    Returns
    -------
    Y  : float32 ndarray, shape (H, W) — Luma          [0, 255]
    Cb : float32 ndarray, shape (H, W) — Blue-diff chroma [0, 255]
    Cr : float32 ndarray, shape (H, W) — Red-diff chroma  [0, 255]
    """
    rgb = img_rgb.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    Y  =  0.299   * R + 0.587   * G + 0.114   * B
    Cb = -0.16874 * R - 0.33126 * G + 0.5     * B + 128.0
    Cr =  0.5     * R - 0.41869 * G - 0.08131 * B + 128.0

    return Y.astype(np.float32), Cb.astype(np.float32), Cr.astype(np.float32)


def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    Reconstruct an RGB image from YCbCr channels.

    Parameters
    ----------
    Y  : float32 ndarray — Luma          [0, 255]
    Cb : float32 ndarray — Blue-diff chroma [0, 255]
    Cr : float32 ndarray — Red-diff chroma  [0, 255]

    Returns
    -------
    np.ndarray : uint8 RGB image, shape (H, W, 3)
    """
    cb = Cb - 128.0
    cr = Cr - 128.0

    R = Y                    + 1.402   * cr
    G = Y - 0.34414 * cb    - 0.71414 * cr
    B = Y + 1.772   * cb

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


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

def visualize_ycbcr(img_rgb: np.ndarray,
                    save_path: str | None = None) -> plt.Figure:
    """
    Visualize YCbCr channels of *img_rgb* in a single figure.

    Layout
    ------
    Row 0 : Original RGB  |  (empty)  |  YCbCr → RGB reconstruction
    Row 1 : Y (Luma)      |  Cb       |  Cr
    Row 2 : Cb+Cr chroma map  |  (empty)  |  (empty)

    Parameters
    ----------
    img_rgb   : uint8 RGB image (H, W, 3)
    save_path : if provided, save figure to this path instead of showing

    Returns
    -------
    matplotlib.figure.Figure
    """
    Y_ch, Cb_ch, Cr_ch = rgb_to_ycbcr(img_rgb)
    ycbcr_recon         = ycbcr_to_rgb(Y_ch, Cb_ch, Cr_ch)

    # Fuse Cb & Cr into a false-color RGB chroma map
    chroma_r = np.clip((Cr_ch - 128) / 128 + 0.5, 0, 1)
    chroma_b = np.clip((Cb_ch - 128) / 128 + 0.5, 0, 1)
    chroma_g = 1 - np.clip(chroma_r * 0.5 + chroma_b * 0.5, 0, 1)
    chroma_vis = np.stack([chroma_r, chroma_g, chroma_b], axis=-1)
    Cb_norm = (Cb_ch - 128) / 255 + 0.5
    Cr_norm = (Cr_ch - 128) / 255 + 0.5

    ycbcr_vis = np.stack([
        Y_ch / 255.0,
        Cb_norm,
        Cr_norm
    ], axis=-1)
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
    _show(fig.add_subplot(gs[0, 0]), img_rgb,      "Original (RGB)",              label="input")
    # _show(fig.add_subplot(gs[0, 1]), ycbcr_vis,    "YCbCr Channels (Y=R, Cb=G, Cr=B)", label="sanity check")
    # _show(fig.add_subplot(gs[0, 2]), ycbcr_recon,  "Reconstructed (YCbCr→RGB)",   label="sanity check")

    # Row 1 — individual channels
    _show(fig.add_subplot(gs[1, 0]), Y_ch,  "Y — Luma",       cmap="gray",    vmin=0, vmax=255, label="0 → 255")
    _show(fig.add_subplot(gs[1, 1]), Cb_ch, "Cb — Blue diff", cmap="Blues_r", vmin=0, vmax=255, label="0 → 255")
    _show(fig.add_subplot(gs[1, 2]), Cr_ch, "Cr — Red diff",  cmap="Reds",    vmin=0, vmax=255, label="0 → 255")

    # Row 2 — chroma composite
    _show(fig.add_subplot(gs[0, 2]), chroma_vis, "Cb + Cr  (Chroma map)", label="blue-shift ↔ red-shift")

    # # Section labels
    # fig.text(0.04, 0.965, "ORIGINALS & RECONSTRUCTION", color="#888888", fontsize=9, fontweight="bold")
    # fig.text(0.04, 0.645, "YCbCr CHANNELS",             color="#f7b731", fontsize=9, fontweight="bold")
    # fig.text(0.04, 0.330, "YCbCr COMPOSITES",            color="#f7b731", fontsize=9, fontweight="bold")

    fig.suptitle("YCbCr Color Space — Luma · Blue-diff Chroma · Red-diff Chroma",
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

    visualize_ycbcr(img)

    Y, Cb, Cr = rgb_to_ycbcr(img)
    print("\n── YCbCr Channel Statistics ─────────────────────────────")
    print(f"  Y  : min={Y.min():6.1f}   max={Y.max():6.1f}   mean={Y.mean():6.1f}")
    print(f"  Cb : min={Cb.min():6.1f}   max={Cb.max():6.1f}   mean={Cb.mean():6.1f}")
    print(f"  Cr : min={Cr.min():6.1f}   max={Cr.max():6.1f}   mean={Cr.mean():6.1f}")
