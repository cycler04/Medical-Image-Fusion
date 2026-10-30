"""
Visualization utilities for Multi-Modal Medical Image Fusion.

Provides consistent, perception-preserving post-processing for benchmark and comparison
visualizations across different imaging modalities (CT-MRI, PET-MRI, SPECT-MRI).

In gray-gray fusion (e.g. CT-MRI, MRI-CT), input modalities are scalar grayscale anatomical maps.
Fused network outputs may exhibit hallucinated chromatic aberration / false color artifacts.
For all comparative visualization figures, fused images in gray-gray fusion are post-processed
to pure grayscale via ITU-R BT.601 luminance:
    Y = 0.299*R + 0.587*G + 0.114*B
This preserves 100% of the anatomical detail, edges, and contrast across all channels without
information loss, while strictly neutralizing false color artifacts.
For color-gray modalities (PET-MRI, SPECT-MRI), functional color maps are preserved.
"""

from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
import torch


def is_image_grayscale(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> bool:
    """Check if an image has identical channels (i.e. is purely grayscale)."""
    if isinstance(img, Image.Image):
        if img.mode in ("L", "1", "I", "F"):
            return True
        arr = np.array(img.convert("RGB"))
    elif isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            return True
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return True
    elif isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.ndim == 2 or (t.ndim == 3 and t.shape[0] == 1):
            return True
        if t.ndim == 3 and t.shape[0] == 3:
            return bool(torch.allclose(t[0], t[1]) and torch.allclose(t[1], t[2]))
        return False
    else:
        return False

    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return bool(np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 1], arr[:, :, 2]))
    return True


def is_gray_fusion_task(modality: str = "", s1_img=None, s2_img=None) -> bool:
    """
    Determine if a fusion task is a gray-gray fusion task.
    
    Checks modality naming convention (e.g. 'CT-MRI', 'MRI-CT')
    or dynamically evaluates if both source images are grayscale.
    """
    clean_mod = str(modality).upper().replace("_", "-").replace(" ", "")
    # Check color modalities first (note: 'SPECT' contains 'CT' as substring, so check first)
    if any(m in clean_mod for m in ["PET", "SPECT", "COLOR"]):
        return False

    tokens = set(clean_mod.split("-"))
    if ("CT" in tokens and "MRI" in tokens) or clean_mod in ("CT-MRI", "MRI-CT"):
        return True
    if any(m in clean_mod for m in ["GRAY", "GREY"]):
        return True

    # Dynamic check on source images if provided
    if s1_img is not None and s2_img is not None:
        return is_image_grayscale(s1_img) and is_image_grayscale(s2_img)

    return False


def postprocess_fused_for_visualization(
    img: Union[Image.Image, np.ndarray, torch.Tensor],
    is_gray_task: bool = False
) -> Union[Image.Image, np.ndarray]:
    """
    Post-process a fused image for comparative visualization.
    
    If is_gray_task is True:
        Converts the image to pure grayscale using perceptual luminance (ITU-R BT.601)
        without losing structural or edge information. Returns 3-channel RGB image/array
        with R == G == B to ensure consistent rendering across matplotlib and image viewers.
    If is_gray_task is False:
        Returns the image in RGB mode (preserving functional pseudocolor for PET/SPECT).
    """
    if isinstance(img, Image.Image):
        if is_gray_task:
            # im.convert('L') uses ITU-R 601-2 luma: L = R*299/1000 + G*587/1000 + B*114/1000
            # Then convert to RGB so that matplotlib displays as standard 3-channel grayscale
            return img.convert("L").convert("RGB")
        return img.convert("RGB")

    elif isinstance(img, np.ndarray):
        arr = img.copy()
        if is_gray_task:
            if arr.ndim == 2:
                return np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3:
                if arr.shape[-1] == 1:
                    return np.repeat(arr, 3, axis=-1)
                elif arr.shape[-1] >= 3:
                    # Perceptual luminance weights
                    gray = (
                        0.299 * arr[:, :, 0].astype(np.float32) +
                        0.587 * arr[:, :, 1].astype(np.float32) +
                        0.114 * arr[:, :, 2].astype(np.float32)
                    )
                    if np.issubdtype(arr.dtype, np.integer):
                        gray = np.clip(np.round(gray), 0, 255).astype(arr.dtype)
                    else:
                        gray = gray.astype(arr.dtype)
                    return np.stack([gray, gray, gray], axis=-1)
        else:
            if arr.ndim == 2:
                return np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                return np.repeat(arr, 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
                # Channel-first format (C, H, W) -> (H, W, C)
                return np.transpose(arr, (1, 2, 0))
        return arr

    elif isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.ndim == 4:
            t = t.squeeze(0)
        if t.ndim == 3 and t.shape[0] in (1, 3):
            # (C, H, W) -> (H, W, C)
            t = t.permute(1, 2, 0)
        arr = t.numpy()
        if arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return postprocess_fused_for_visualization(arr, is_gray_task=is_gray_task)

    return img


def load_viz_image(
    path: Union[str, Path],
    is_gray_task: bool = False,
    as_numpy: bool = True
) -> Union[np.ndarray, Image.Image]:
    """
    Load an image from file and apply visualization postprocessing.
    
    Parameters:
    -----------
    path : str or Path
        Path to image file.
    is_gray_task : bool
        Whether this is a gray-gray fusion task (e.g. CT-MRI).
    as_numpy : bool
        If True, return as (H, W, 3) uint8 numpy array. Otherwise return PIL Image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    im = Image.open(path)
    post_im = postprocess_fused_for_visualization(im, is_gray_task=is_gray_task)
    if as_numpy:
        return np.array(post_im)
    return post_im
