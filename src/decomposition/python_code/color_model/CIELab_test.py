"""
CIELAB Color Space Module
=========================
Converts an image to CIELAB color space (D65 illuminant, sRGB input) and
visualizes all channels alongside the original.

Channel ranges:
    L*  — Lightness             [0,   100]
    a*  — Green (-) ↔ Red (+)   [-128, 127]
    b*  — Blue (-) ↔ Yellow (+) [-128, 127]

Usage:
    python cielab_color_space.py                      # uses built-in sample image
    python cielab_color_space.py path/to/image.jpg    # uses your own image

API:
    from cielab_color_space import rgb_to_lab, lab_to_rgb, visualize_lab, load_image

    L, a, b = rgb_to_lab(img_rgb)          # Convert RGB  → CIELAB
    img_rgb = lab_to_rgb(L, a, b)          # Convert CIELAB → RGB
    fig     = visualize_lab(img_rgb)        # Full visualization
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, LinearSegmentedColormap
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Custom colormaps  (green→grey→red  for a*;  blue→grey→yellow  for b*)
# ─────────────────────────────────────────────────────────────────────────────

_CMAP_A = LinearSegmentedColormap.from_list(
    "a_star", ["#00c060", "#888888", "#e63030"], N=256
)
_CMAP_B = LinearSegmentedColormap.from_list(
    "b_star", ["#3070e0", "#888888", "#e6c000"], N=256
)
_CMAP_L = "gray"

# XYZ false-colour maps — tinted to their rough perceptual correlates:
#   X ≈ red–orange luminance,  Y ≈ pure luminance (green peak),  Z ≈ blue
_CMAP_X = LinearSegmentedColormap.from_list(
    "xyz_X", ["#000000", "#7b2d00", "#ff6a00", "#ffe5cc"], N=256
)
_CMAP_Y = LinearSegmentedColormap.from_list(
    "xyz_Y", ["#000000", "#1a4a00", "#4caf00", "#e8ffd0"], N=256
)
_CMAP_Z = LinearSegmentedColormap.from_list(
    "xyz_Z", ["#000000", "#00185a", "#1565c0", "#c8e0ff"], N=256
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Core conversion functions
# ─────────────────────────────────────────────────────────────────────────────

# D65 reference white (2° observer)
_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)


def _srgb_linearise(u: np.ndarray) -> np.ndarray:
    """Apply sRGB inverse gamma to convert [0,1] sRGB → linear light."""
    return np.where(u <= 0.04045,
                    u / 12.92,
                    ((u + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _srgb_gamma(u: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma to convert linear light → [0,1] sRGB."""
    return np.where(u <= 0.0031308,
                    12.92 * u,
                    1.055 * u ** (1.0 / 2.4) - 0.055).astype(np.float32)


def _f_lab(t: np.ndarray) -> np.ndarray:
    """CIELAB cube-root transfer function f(t)."""
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3,
                    np.cbrt(t),
                    t / (3.0 * delta ** 2) + 4.0 / 29.0).astype(np.float32)


def _f_lab_inv(t: np.ndarray) -> np.ndarray:
    """Inverse of f(t)."""
    delta = 6.0 / 29.0
    return np.where(t > delta,
                    t ** 3,
                    3.0 * delta ** 2 * (t - 4.0 / 29.0)).astype(np.float32)


def rgb_to_lab(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an sRGB image to CIELAB (D65 illuminant).

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image (uint8 or float32). Shape must be (H, W, 3).
        Values are expected in [0, 255].

    Returns
    -------
    L  : float32 ndarray, shape (H, W) — Lightness         [0,  100]
    a  : float32 ndarray, shape (H, W) — Green–Red axis    [-128, 127]
    b  : float32 ndarray, shape (H, W) — Blue–Yellow axis  [-128, 127]
    """
    # 1. Normalise to [0, 1]
    rgb = img_rgb.astype(np.float32) / 255.0

    # 2. sRGB linearisation (inverse gamma)
    rgb_lin = _srgb_linearise(rgb)

    # 3. Linear sRGB → CIE XYZ  (IEC 61966-2-1, D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)

    xyz = rgb_lin @ M.T          # shape (H, W, 3)

    # 4. Normalise by D65 reference white
    xyz_n = xyz / _D65           # broadcast over H×W

    # 5. Apply f(t) per channel
    f = _f_lab(xyz_n)            # shape (H, W, 3)

    # 6. CIELAB formulae
    L_ch = 116.0 * f[..., 1] - 16.0
    a_ch = 500.0 * (f[..., 0] - f[..., 1])
    b_ch = 200.0 * (f[..., 1] - f[..., 2])

    return (L_ch.astype(np.float32),
            a_ch.astype(np.float32),
            b_ch.astype(np.float32))


def rgb_to_xyz(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an sRGB image to CIE XYZ (D65 illuminant).

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image (uint8 or float32). Shape must be (H, W, 3).
        Values are expected in [0, 255].

    Returns
    -------
    X : float32 ndarray, shape (H, W) — CIE X tristimulus  [0, ~0.95]
    Y : float32 ndarray, shape (H, W) — CIE Y (luminance)  [0,  1.00]
    Z : float32 ndarray, shape (H, W) — CIE Z tristimulus  [0, ~1.09]

    Notes
    -----
    Values are normalised so that the D65 reference white maps to (0.95047,
    1.00000, 1.08883).  Display via false-colour maps since XYZ is not
    directly renderable as RGB.
    """
    rgb = img_rgb.astype(np.float32) / 255.0
    rgb_lin = _srgb_linearise(rgb)

    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)

    xyz = rgb_lin @ M.T          # (H, W, 3)
    return (xyz[..., 0].astype(np.float32),   # X
            xyz[..., 1].astype(np.float32),   # Y
            xyz[..., 2].astype(np.float32))   # Z


def lab_to_rgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Reconstruct an sRGB image from CIELAB channels.

    Parameters
    ----------
    L : float32 ndarray — Lightness         [0,  100]
    a : float32 ndarray — Green–Red axis    [-128, 127]
    b : float32 ndarray — Blue–Yellow axis  [-128, 127]

    Returns
    -------
    np.ndarray : uint8 RGB image, shape (H, W, 3)
    """
    # 1. Invert CIELAB formulae
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    xyz = _f_lab_inv(np.stack([fx, fy, fz], axis=-1)) * _D65  # shape (H,W,3)

    # 2. XYZ → linear sRGB
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float32)

    rgb_lin = xyz @ M_inv.T
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)

    # 3. Apply sRGB gamma
    rgb = _srgb_gamma(rgb_lin)

    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_chroma_map(a_ch: np.ndarray,
                     b_ch: np.ndarray) -> np.ndarray:
    """
    Combine a* and b* into a perceptual false-colour chroma hue map.

    Strategy
    --------
    Hue  = atan2(b*, a*)  mapped to [0, 360°]
    Chroma = sqrt(a*² + b*²)  used as saturation-like weight
    Lightness is fixed at 0.65 for visibility.
    Returns an H×W×3 float32 array in [0, 1].
    """
    hue_rad   = np.arctan2(b_ch, a_ch)              # [-π, π]
    hue_norm  = (hue_rad + np.pi) / (2.0 * np.pi)  # [0, 1]
    chroma    = np.sqrt(a_ch ** 2 + b_ch ** 2)
    sat       = np.clip(chroma / 100.0, 0, 1)       # saturate at C*=100

    h_u8  = (hue_norm  * 179).astype(np.uint8)
    s_u8  = (sat       * 255).astype(np.uint8)
    v_u8  = np.full_like(h_u8, 210)

    hsv   = np.stack([h_u8, s_u8, v_u8], axis=-1)
    bgr   = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr[..., ::-1].astype(np.float32) / 255.0


def _make_ab_scatter(a_ch: np.ndarray,
                     b_ch: np.ndarray,
                     L_ch: np.ndarray,
                     n_samples: int = 6000) -> plt.Figure:
    """(Internal) — not used directly in the grid; kept for standalone use."""
    pass


def load_image(path: str | None = None) -> np.ndarray:
    """
    Load an image as an RGB uint8 ndarray.
    If *path* is None, a built-in colorful test image is returned.
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


# ─────────────────────────────────────────────────────────────────────────────
# 3.  a*–b* chromaticity scatter (inset axes)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_ab_scatter(ax: plt.Axes,
                     a_ch: np.ndarray,
                     b_ch: np.ndarray,
                     L_ch: np.ndarray,
                     n_samples: int = 4000) -> None:
    """
    Plot a pixel scatter in the a*–b* plane coloured by pixel hue.
    Uses a random subsample for speed.
    """
    flat_a = a_ch.ravel()
    flat_b = b_ch.ravel()
    flat_L = L_ch.ravel()

    rng = np.random.default_rng(42)
    idx = rng.choice(len(flat_a), size=min(n_samples, len(flat_a)), replace=False)

    a_s, b_s, L_s = flat_a[idx], flat_b[idx], flat_L[idx]

    # Colour each dot by its own reconstructed hue
    hue_rad  = np.arctan2(b_s, a_s)
    hue_norm = (hue_rad + np.pi) / (2.0 * np.pi)
    chroma   = np.sqrt(a_s ** 2 + b_s ** 2)
    sat      = np.clip(chroma / 100.0, 0, 1)

    h_u8  = (hue_norm * 179).astype(np.uint8).reshape(-1, 1, 1)
    s_u8  = (sat      * 255).astype(np.uint8).reshape(-1, 1, 1)
    v_u8  = np.full_like(h_u8, 220)
    hsv   = np.concatenate([h_u8, s_u8, v_u8], axis=2)
    bgr   = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    colors = bgr[:, 0, ::-1].astype(np.float32) / 255.0

    LIMS = 130
    ax.scatter(a_s, b_s, c=colors, s=4, alpha=0.65, linewidths=0)

    # Axes decoration
    ax.axhline(0, color="#555555", lw=0.8, ls="--")
    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_xlim(-LIMS, LIMS)
    ax.set_ylim(-LIMS, LIMS)
    ax.set_xlabel("a*  (green ← · → red)",   color="#aaaaaa", fontsize=8)
    ax.set_ylabel("b*  (blue ↓ · ↑ yellow)", color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#666666", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.set_facecolor("#181818")

    # Quadrant labels
    kw = dict(fontsize=7, alpha=0.55, ha="center")
    ax.text( 80,  80, "+a/+b\nRed–Yellow",   color="#e0a020", **kw)
    ax.text(-80,  80, "−a/+b\nGreen–Yellow", color="#80c040", **kw)
    ax.text( 80, -80, "+a/−b\nRed–Blue",     color="#c050c0", **kw)
    ax.text(-80, -80, "−a/−b\nGreen–Blue",   color="#4080d0", **kw)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_lab(img_rgb: np.ndarray,
                  save_path: str | None = None) -> plt.Figure:
    """
    Visualize XYZ and CIELAB channels of *img_rgb* in a single figure.

    Layout (3 × 3)
    --------------
    Row 0 : Original RGB    |  Chroma hue map          |  Reconstructed RGB
    Row 1 : X (tristimulus) |  Y (luminance / green)   |  Z (blue tristimulus)
    Row 2 : L* (Lightness)  |  a* (Green–Red)          |  b* (Blue–Yellow)

    A bonus a*–b* chromaticity scatter is inset into the Chroma cell.

    Parameters
    ----------
    img_rgb   : uint8 RGB image (H, W, 3)
    save_path : if provided, save figure to this path

    Returns
    -------
    matplotlib.figure.Figure
    """
    L_ch, a_ch, b_ch = rgb_to_lab(img_rgb)
    X_ch, Y_ch, Z_ch = rgb_to_xyz(img_rgb)
    lab_recon         = lab_to_rgb(L_ch, a_ch, b_ch)
    chroma_map        = _make_chroma_map(a_ch, b_ch)

    # ── Precompute stats strings ──────────────────────────────────────────────
    def _stat(arr):
        return f"min {arr.min():.3f}  ·  max {arr.max():.3f}  ·  μ {arr.mean():.3f}"

    def _stat_lab(arr):
        return f"min {arr.min():.1f}  ·  max {arr.max():.1f}  ·  μ {arr.mean():.1f}"

    # ── Figure layout ─────────────────────────────────────────────────────────
    BG = "#0f0f0f"
    fig = plt.figure(figsize=(16, 13), facecolor=BG)

    gs_outer = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.40, wspace=0.18,
        left=0.04, right=0.96, top=0.93, bottom=0.04
    )

    TITLE_KW = dict(color="white",   fontsize=11, fontweight="bold", pad=6)
    LABEL_KW = dict(color="#aaaaaa", fontsize=8.5)

    # ── Helper: show image ────────────────────────────────────────────────────
    def _show(ax, data, title, cmap=None, vmin=None, vmax=None, xlabel=""):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_title(title, **TITLE_KW)
        ax.set_xlabel(xlabel, **LABEL_KW)
        ax.set_xticks([]);  ax.set_yticks([])
        ax.set_facecolor("#181818")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")

    # ── Helper: attach colorbar ───────────────────────────────────────────────
    def _cbar(ax, cmap, vmin, vmax):
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, orientation="vertical")
        cb.ax.tick_params(colors="#888888", labelsize=7)
        cb.outline.set_edgecolor("#333333")

    # ── Row 0 — originals ─────────────────────────────────────────────────────
    ax_orig   = fig.add_subplot(gs_outer[0, 0])
    ax_chroma = fig.add_subplot(gs_outer[0, 1])
    ax_recon  = fig.add_subplot(gs_outer[0, 2])

    # ── Row 0 — originals ─────────────────────────────────────────────────────────
    _show(ax_orig,   img_rgb,    "Original (sRGB)",                xlabel="input image")
    _show(ax_chroma, chroma_map, "a*–b*  Chroma Hue Map",          xlabel="hue = atan2(b*, a*)  ·  saturation = C*")
    _show(ax_recon,  lab_recon,  "Reconstructed  (CIELAB → RGB)",  xlabel="sanity check — should match original")

    # Add invisible colorbars to match spacing of rows 1 & 2
    for ax in [ax_orig, ax_chroma, ax_recon]:
        sm = plt.cm.ScalarMappable(cmap="gray", norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, orientation="vertical")
        cb.ax.set_visible(False)   # hide ticks/labels but keep the space

    # ── Row 1 — CIE XYZ ──────────────────────────────────────────────────────
    ax_X = fig.add_subplot(gs_outer[1, 0])
    ax_Y = fig.add_subplot(gs_outer[1, 1])
    ax_Z = fig.add_subplot(gs_outer[1, 2])

    X_max = float(_D65[0])   # ~0.9505
    Y_max = float(_D65[1])   # 1.0000
    Z_max = float(_D65[2])   # ~1.0888

    _show(ax_X, X_ch, "X  —  Red–Green–Blue mix",
          cmap=_CMAP_X, vmin=0, vmax=X_max,
          xlabel=_stat(X_ch) + f"    D65 white = {X_max:.4f}")
    _show(ax_Y, Y_ch, "Y  —  Luminance  (green-peak weighted)",
          cmap=_CMAP_Y, vmin=0, vmax=Y_max,
          xlabel=_stat(Y_ch) + f"    D65 white = {Y_max:.4f}")
    _show(ax_Z, Z_ch, "Z  —  Blue–Violet tristimulus",
          cmap=_CMAP_Z, vmin=0, vmax=Z_max,
          xlabel=_stat(Z_ch) + f"    D65 white = {Z_max:.4f}")

    for ax, cmap, vmin, vmax in [
        (ax_X, _CMAP_X, 0, X_max),
        (ax_Y, _CMAP_Y, 0, Y_max),
        (ax_Z, _CMAP_Z, 0, Z_max),
    ]:
        _cbar(ax, cmap, vmin, vmax)

    # ── Row 2 — CIELAB channels ───────────────────────────────────────────────
    ax_L = fig.add_subplot(gs_outer[2, 0])
    ax_a = fig.add_subplot(gs_outer[2, 1])
    ax_b = fig.add_subplot(gs_outer[2, 2])

    _show(ax_L, L_ch, "L*  —  Lightness",
          cmap=_CMAP_L, vmin=0, vmax=100,
          xlabel=_stat_lab(L_ch) + "    range [0, 100]")
    _show(ax_a, a_ch, "a*  —  Green (−) ↔ Red (+)",
          cmap=_CMAP_A, vmin=-128, vmax=127,
          xlabel=_stat_lab(a_ch) + "    range [−128, 127]")
    _show(ax_b, b_ch, "b*  —  Blue (−) ↔ Yellow (+)",
          cmap=_CMAP_B, vmin=-128, vmax=127,
          xlabel=_stat_lab(b_ch) + "    range [−128, 127]")

    for ax, cmap, vmin, vmax in [
        (ax_L, _CMAP_L, 0,    100),
        (ax_a, _CMAP_A, -128, 127),
        (ax_b, _CMAP_B, -128, 127),
    ]:
        _cbar(ax, cmap, vmin, vmax)

    # ── Inset: a*–b* scatter inside the chroma cell ──────────────────────────
    # inset = ax_chroma.inset_axes([0.60, 0.02, 0.38, 0.45])
    # _draw_ab_scatter(inset, a_ch, b_ch, L_ch)
    # inset.set_title("a*–b* scatter", color="#cccccc", fontsize=6.5, pad=3)

    # ── Section banners ───────────────────────────────────────────────────────
    # fig.suptitle(
    #     "sRGB  →  CIE XYZ  →  CIELAB   |   D65 illuminant  ·  2° standard observer",
    #     color="white", fontsize=13, fontweight="bold", y=0.975
    # )

    # Row section labels (left gutter)
    fig.text(0.005, 0.790, "ORIGINALS", color="#888888", fontsize=7.5,
             fontweight="bold", rotation=90, va="center")
    fig.text(0.005, 0.500, "CIE XYZ",   color="#f7a800", fontsize=7.5,
             fontweight="bold", rotation=90, va="center")
    fig.text(0.005, 0.200, "CIELAB",    color="#60c0ff", fontsize=7.5,
             fontweight="bold", rotation=90, va="center")

    # fig.text(0.04, 0.010,
    #          "CIE 1931  ·  BT.709 sRGB primaries  ·  D65 reference white  ·  2° standard observer",
    #          color="#555555", fontsize=7.5)

    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✓ Saved → {save_path}")
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    image_path = "data/raw/Harvard-AANLIB/SPECT-MRI/SPECT/3015.png"
    img        = load_image(image_path)

    print(f"Image shape : {img.shape}")
    print(f"Image dtype : {img.dtype}")

    visualize_lab(img)

    X, Y, Z = rgb_to_xyz(img)
    print("\n── CIE XYZ Channel Statistics ────────────────────────────────────")
    print(f"  X  : min={X.min():.4f}   max={X.max():.4f}   mean={X.mean():.4f}   D65 white ≈ 0.9505")
    print(f"  Y  : min={Y.min():.4f}   max={Y.max():.4f}   mean={Y.mean():.4f}   D65 white = 1.0000")
    print(f"  Z  : min={Z.min():.4f}   max={Z.max():.4f}   mean={Z.mean():.4f}   D65 white ≈ 1.0888")

    L, a, b = rgb_to_lab(img)
    print("\n── CIELAB Channel Statistics ─────────────────────────────────────")
    print(f"  L* : min={L.min():7.2f}   max={L.max():7.2f}   mean={L.mean():7.2f}   range [  0, 100]")
    print(f"  a* : min={a.min():7.2f}   max={a.max():7.2f}   mean={a.mean():7.2f}   range [-128, 127]")
    print(f"  b* : min={b.min():7.2f}   max={b.max():7.2f}   mean={b.mean():7.2f}   range [-128, 127]")
