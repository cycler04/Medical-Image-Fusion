"""
GPU-accelerated implementations of medical image fusion metrics using PyTorch CUDA.
Focuses on the primary computational bottlenecks:
  - FMI (Feature Mutual Information): per-patch bivariate copula mutual information
  - MS_SSIM (Multi-Scale Structural Similarity): batched patches, convolutional pooling, and GPU tensors

The original CPU implementations in info_metrics.py and structure_metrics.py remain intact.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.ndimage import uniform_filter

# Relative imports from the same metrics directory
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import structure_metrics
import img_metrics
import info_metrics
import quality_metrics
import visual_metrics
import others_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-12


# ── 1. GPU Accelerated Batch Patch Mutual Information (FMI engine) ─────────────

def batch_patch_mi_gpu(x_np: np.ndarray, y_np: np.ndarray, device=DEVICE) -> np.ndarray:
    """
    GPU-accelerated batch patch normalized mutual information via bivariate copulas.
    Mathematically mirrors info_metrics._batch_patch_mi.
    
    Parameters:
      x_np : (B, L) numpy array of patch vectors from image A/B
      y_np : (B, L) numpy array of patch vectors from image F
      device : torch device ('cuda' or 'cpu')
      
    Returns:
      nmi : (B,) numpy array of normalized mutual information scores
    """
    x = torch.as_tensor(x_np, dtype=torch.float64, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float64, device=device)
    B, L = x.shape

    # 1. Identical patch mask
    identical = torch.all(x == y, dim=1)

    # 2. Marginal probability density functions
    def _to_pdf(p):
        mn = p.min(dim=1, keepdim=True).values
        mx = p.max(dim=1, keepdim=True).values
        flat = torch.where(mx == mn, torch.ones_like(p), (p - mn) / (mx - mn + EPS))
        s = flat.sum(dim=1, keepdim=True)
        return flat / torch.where(s == 0, torch.tensor(1.0, dtype=torch.float64, device=device), s)

    xPdf = _to_pdf(x)
    yPdf = _to_pdf(y)

    # 3. Cumulative distribution functions
    xCdf = torch.cumsum(xPdf, dim=1)
    yCdf = torch.cumsum(yPdf, dim=1)

    # 4. Pearson correlation between marginal PDFs
    xm = xPdf - xPdf.mean(dim=1, keepdim=True)
    ym = yPdf - yPdf.mean(dim=1, keepdim=True)
    dot  = (xm * ym).sum(dim=1)
    norm = torch.sqrt((xm**2).sum(dim=1) * (ym**2).sum(dim=1))
    c    = torch.where(norm == 0, torch.tensor(0.0, dtype=torch.float64, device=device), dot / norm)

    # 5. Index-weighted standard deviations
    idx = torch.arange(1, L + 1, dtype=torch.float64, device=device)
    ex  = (idx       * xPdf).sum(dim=1)
    ex2 = (idx**2    * xPdf).sum(dim=1)
    ey  = (idx       * yPdf).sum(dim=1)
    ey2 = (idx**2    * yPdf).sum(dim=1)
    xSd = torch.sqrt(torch.clamp(ex2 - ex**2, min=0.0))
    ySd = torch.sqrt(torch.clamp(ey2 - ey**2, min=0.0))
    sd_prod = xSd * ySd

    xC2d = xCdf[:, :,  None]
    yC2d = yCdf[:, None, :]

    def _accum_H(jpdf):
        mask = jpdf != 0
        safe = torch.where(mask, torch.abs(jpdf), torch.tensor(1.0, dtype=torch.float64, device=device))
        return -(torch.where(mask, jpdf, torch.tensor(0.0, dtype=torch.float64, device=device)) * torch.log2(safe)).sum(dim=(-2, -1))

    def _H1d(arr):
        mask = arr != 0
        safe = torch.where(mask, torch.abs(arr), torch.tensor(1.0, dtype=torch.float64, device=device))
        return -(torch.where(mask, arr, torch.tensor(0.0, dtype=torch.float64, device=device)) * torch.log2(safe)).sum(dim=-1)

    pos_mask = c >= 0
    neg_mask = ~pos_mask
    jointH = torch.zeros(B, dtype=torch.float64, device=device)

    # Upper copula branch (c >= 0)
    if torch.any(pos_mask):
        sub_c = c[pos_mask]
        sub_sd_prod = sd_prod[pos_mask]
        sub_xC = xC2d[pos_mask]
        sub_yC = yC2d[pos_mask]
        sub_xCm = sub_xC[:, :-1, :]
        sub_yCm = sub_yC[:, :, :-1]
        sub_xPdf = xPdf[pos_mask]
        sub_yPdf = yPdf[pos_mask]

        mFG = torch.minimum(sub_xC, sub_yC)
        covUp = (mFG - sub_xC * sub_yC).sum(dim=(-2, -1))
        corrUp = torch.zeros_like(covUp)
        vc = sub_sd_prod != 0
        corrUp[vc] = covUp[vc] / sub_sd_prod[vc]

        phi = torch.zeros_like(sub_c)
        vp = (sub_c != 0) & (sub_sd_prod != 0) & (corrUp != 0)
        phi[vp] = sub_c[vp] / corrUp[vp]
        ph = phi[:, None, None]

        mFGim = torch.minimum(sub_xCm, sub_yC)
        mFGjm = torch.minimum(sub_xC, sub_yCm)
        mFGij = torch.minimum(sub_xCm, sub_yCm)

        B_sub = len(sub_c)
        H_up = torch.zeros(B_sub, dtype=torch.float64, device=device)
        jp = mFG[:, 0, 0] * phi + (1 - phi) * sub_xPdf[:, 0] * sub_yPdf[:, 0]
        pos = jp > 0
        H_up[pos] += (-jp[pos] * torch.log2(jp[pos]))

        up = mFG[:, 1:, 0] - mFGim[:, :, 0]
        jp_ = ph[:, :, 0] * up + (1 - ph[:, :, 0]) * sub_xPdf[:, 1:] * sub_yPdf[:, :1]
        H_up += _H1d(jp_)

        up = mFG[:, 0, 1:] - mFGjm[:, 0, :]
        jp_ = ph[:, 0, :] * up + (1 - ph[:, 0, :]) * sub_xPdf[:, :1] * sub_yPdf[:, 1:]
        H_up += _H1d(jp_)

        up = mFG[:, 1:, 1:] - mFGim[:, :, 1:] - mFGjm[:, 1:, :] + mFGij
        jp_ = ph * up + (1 - ph) * sub_xPdf[:, 1:, None] * sub_yPdf[:, None, 1:]
        H_up += _accum_H(jp_)

        jointH[pos_mask] = H_up

    # Lower copula branch (c < 0)
    if torch.any(neg_mask):
        sub_c = c[neg_mask]
        sub_sd_prod = sd_prod[neg_mask]
        sub_xC = xC2d[neg_mask]
        sub_yC = yC2d[neg_mask]
        sub_xCm = sub_xC[:, :-1, :]
        sub_yCm = sub_yC[:, :, :-1]
        sub_xPdf = xPdf[neg_mask]
        sub_yPdf = yPdf[neg_mask]

        mFG = torch.clamp(sub_xC + sub_yC - 1.0, min=0.0)
        covLo = (mFG - sub_xC * sub_yC).sum(dim=(-2, -1))
        corrLo = torch.zeros_like(covLo)
        vc = sub_sd_prod != 0
        corrLo[vc] = covLo[vc] / sub_sd_prod[vc]

        theta = torch.zeros_like(sub_c)
        vt = (sub_sd_prod != 0) & (corrLo != 0)
        theta[vt] = sub_c[vt] / corrLo[vt]
        th = theta[:, None, None]

        mFGim = torch.clamp(sub_xCm + sub_yC - 1.0, min=0.0)
        mFGjm = torch.clamp(sub_xC + sub_yCm - 1.0, min=0.0)
        mFGij = torch.clamp(sub_xCm + sub_yCm - 1.0, min=0.0)

        B_sub = len(sub_c)
        H_lo = torch.zeros(B_sub, dtype=torch.float64, device=device)
        jp = mFG[:, 0, 0] * theta + (1 - theta) * sub_xPdf[:, 0] * sub_yPdf[:, 0]
        nz = jp != 0
        H_lo[nz] += -jp[nz] * torch.log2(torch.abs(jp[nz]))

        lo = mFG[:, 0, 1:] - mFGjm[:, 0, :]
        jp_ = th[:, 0, :] * lo + (1 - th[:, 0, :]) * sub_xPdf[:, :1] * sub_yPdf[:, 1:]
        H_lo += _H1d(jp_)

        lo = mFG[:, 1:, 0] - mFGim[:, :, 0]
        jp_ = th[:, :, 0] * lo + (1 - th[:, :, 0]) * sub_xPdf[:, 1:] * sub_yPdf[:, :1]
        H_lo += _H1d(jp_)

        lo = mFG[:, 1:, 1:] - mFGim[:, :, 1:] - mFGjm[:, 1:, :] + mFGij
        jp_ = th * lo + (1 - th) * sub_xPdf[:, 1:, None] * sub_yPdf[:, None, 1:]
        H_lo += _accum_H(jp_)

        jointH[neg_mask] = H_lo

    # 6. Marginal entropies
    def _marginal_H(pdf):
        mask = pdf > 0
        safe = torch.where(mask, pdf, torch.tensor(1.0, dtype=torch.float64, device=device))
        return -(torch.where(mask, pdf, torch.tensor(0.0, dtype=torch.float64, device=device)) * torch.log2(safe)).sum(dim=1)

    xH = _marginal_H(xPdf)
    yH = _marginal_H(yPdf)

    # 7. Normalized mutual information
    mi_val = xH + yH - jointH
    denom  = xH + yH
    nmi    = torch.where(
        (mi_val == 0) | (denom == 0),
        torch.tensor(0.0, dtype=torch.float64, device=device),
        mi_val / torch.where(denom == 0, torch.tensor(1.0, dtype=torch.float64, device=device), denom) * 2.0
    )
    # apply flat and identical-patch override (matching CPU behavior)
    x_flat = torch.all(x == x[:, :1], dim=1)
    y_flat = torch.all(y == y[:, :1], dim=1)
    both_flat = x_flat & y_flat
    nmi = torch.where(both_flat | identical, torch.tensor(1.0, dtype=torch.float64, device=device), nmi)
    return nmi.cpu().numpy()


def fmi_gpu(ima: np.ndarray, imb: np.ndarray, imf: np.ndarray, feature: str = "none", w: int = 3, device=DEVICE) -> float:
    """
    GPU-accelerated Feature Mutual Information (FMI).
    """
    aFeat = info_metrics._extract_feature(ima, feature)
    bFeat = info_metrics._extract_feature(imb, feature)
    fFeat = info_metrics._extract_feature(imf, feature)

    hw = int(np.floor(w / 2))
    wsize = 2 * hw + 1
    aFeat = np.ascontiguousarray(aFeat)
    bFeat = np.ascontiguousarray(bFeat)
    fFeat = np.ascontiguousarray(fFeat)

    aW = info_metrics.view_as_windows(aFeat, (wsize, wsize))
    bW = info_metrics.view_as_windows(bFeat, (wsize, wsize))
    fW = info_metrics.view_as_windows(fFeat, (wsize, wsize))

    M, N = aW.shape[:2]
    B = M * N
    aP = aW.reshape(B, wsize * wsize, order='F')
    bP = bW.reshape(B, wsize * wsize, order='F')
    fP = fW.reshape(B, wsize * wsize, order='F')

    fmi_af = batch_patch_mi_gpu(aP, fP, device=device)
    fmi_bf = batch_patch_mi_gpu(bP, fP, device=device)
    fmi_map = ((fmi_af + fmi_bf) / 2.0).reshape(M, N)
    return float(np.nanmean(fmi_map))


# ── 2. GPU Accelerated Multi-Scale SSIM (MS-SSIM engine) ──────────────────────

def _downsample_mirror(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return uniform_filter(arr, size=(2, 2), mode='mirror')[::2, ::2]
    else:
        return uniform_filter(arr, size=(2, 2, 1), mode='mirror')[::2, ::2, :]


def mef_ssim_torch(seq_t: torch.Tensor, fi_t: torch.Tensor, wsize: int = 11, K: float = 0.03) -> torch.Tensor:
    """
    GPU-accelerated single-scale MEF-SSIM.
    seq_t : (H, W, N) torch.Tensor on device
    fi_t  : (H, W) torch.Tensor on device
    """
    H, W, N = seq_t.shape
    bd = wsize // 2
    C = (K * 255) ** 2
    device = seq_t.device

    # 1. Per-pixel statistics via 2D reflection conv
    seq_4d = seq_t.permute(2, 0, 1).unsqueeze(1)  # (N, 1, H, W)
    kernel = torch.ones(1, 1, wsize, wsize, dtype=torch.float64, device=device) / (wsize * wsize)
    pad = bd

    seq_pad = torch.nn.functional.pad(seq_4d, (pad, pad, pad, pad), mode='reflect')
    mu_4d = torch.nn.functional.conv2d(seq_pad, kernel)

    seq2_pad = torch.nn.functional.pad(seq_4d ** 2, (pad, pad, pad, pad), mode='reflect')
    mu2_4d = torch.nn.functional.conv2d(seq2_pad, kernel)

    mu = mu_4d[:, 0, bd:-bd, bd:-bd].permute(1, 2, 0)
    mu2 = mu2_4d[:, 0, bd:-bd, bd:-bd].permute(1, 2, 0)
    sigma = mu2 - mu ** 2
    ed = torch.sqrt(torch.clamp(wsize ** 2 * sigma, min=0.0)) + 1e-3

    # 2. Gaussian weighting window
    ax = torch.arange(-5, 6, dtype=torch.float64, device=device)
    g = torch.exp(-(ax ** 2) / (2 * 1.5 ** 2))
    gwin = torch.outer(g, g)
    gwin = gwin / gwin.sum()
    gw = gwin.view(1, wsize * wsize)

    # 3. Patch extraction via unfold
    seq_patches = torch.nn.functional.unfold(seq_4d, kernel_size=wsize)
    B_cnt = seq_patches.shape[2]
    vecs = seq_patches.permute(2, 1, 0)  # (B, w^2, N)

    fi_4d = fi_t.unsqueeze(0).unsqueeze(0)
    fi_patches = torch.nn.functional.unfold(fi_4d, kernel_size=wsize)
    fv = fi_patches[0].t()  # (B, w^2)

    mu_b = mu.reshape(B_cnt, N)
    ed_b = ed.reshape(B_cnt, N)

    # 4. Structure consistency
    centered = vecs - mu_b.unsqueeze(1)
    denom = torch.linalg.norm(centered, dim=1)
    sumvec = centered.sum(dim=2)
    sumvec_mean = sumvec.mean(dim=1, keepdim=True)
    numerator = torch.linalg.norm(sumvec - sumvec_mean, dim=1)

    R = (numerator + 1e-10) / (denom.sum(dim=1) + 1e-10)
    R = torch.clamp(R, min=1e-10, max=1.0 - 1e-10)
    p = torch.clamp(torch.tan(np.pi / 2 * R), min=0.0, max=10.0)

    wk = (ed_b / wsize) ** p.unsqueeze(1)
    wk = wk / (wk.sum(dim=1, keepdim=True) + 1e-10)
    maxEd = ed_b.max(dim=1).values

    # 5. Reference block
    rblock = (wk.unsqueeze(1) * centered / (ed_b.unsqueeze(1) + 1e-10)).sum(dim=2)
    nrm = torch.linalg.norm(rblock, dim=1, keepdim=True)
    safe_nrm = torch.where(nrm > 0, nrm, torch.tensor(1.0, dtype=torch.float64, device=device))
    rblock = rblock / safe_nrm * maxEd.unsqueeze(1)

    # 6. Gaussian-weighted SSIM
    mu1 = (gw * rblock).sum(dim=1)
    mu2 = (gw * fv).sum(dim=1)
    rv_c = rblock - mu1.unsqueeze(1)
    fv_c = fv - mu2.unsqueeze(1)

    s1  = (gw * (rv_c ** 2)).sum(dim=1)
    s2  = (gw * (fv_c ** 2)).sum(dim=1)
    s12 = (gw * rv_c * fv_c).sum(dim=1)

    qmap = (2 * s12 + C) / (s1 + s2 + C)
    return qmap.mean()


def ms_ssim_gpu(img_seq: np.ndarray, fI: np.ndarray, K: float = 0.03, level: int = 3, device=DEVICE) -> float:
    """
    GPU-accelerated Multi-Scale SSIM.
    """
    weight = np.array([0.0448, 0.2856, 0.3001])[:level]
    weight = weight / weight.sum()

    seq = img_seq.astype(np.float64)
    fi  = fI.astype(np.float64)

    Q = []
    for l in range(level):
        seq_t = torch.as_tensor(seq, dtype=torch.float64, device=device)
        fi_t  = torch.as_tensor(fi, dtype=torch.float64, device=device)
        Q.append(mef_ssim_torch(seq_t, fi_t, K=K).item())
        if l < level - 1:
            seq = _downsample_mirror(seq)
            fi  = _downsample_mirror(fi)

    return float(np.prod(np.array(Q) ** weight))


# ── 3. High-Level All-Metric Computer (GPU-Accelerated) ─────────────────────────

def compute_metrics_gpu(A: np.ndarray, B: np.ndarray, F: np.ndarray, device=DEVICE) -> Dict[str, float]:
    """
    Computes all 28 metrics using GPU acceleration for bottleneck routines (FMI, MS-SSIM).
    Matches CPU baseline output.
    """
    m = {}

    # Quality
    m["MLI"]   = float(quality_metrics.mli_error(F))
    m["SD"]    = float(quality_metrics.sd(F))
    m["AG"]    = float(quality_metrics.ag(F))
    m["MSE"]   = float(quality_metrics.mse_f(A, B, F))
    m["PSNR"]  = float(quality_metrics.psnr_f(A, B, F))

    # Info
    m["EN"]    = float(info_metrics.en(F))
    m["MI"]    = float(info_metrics.mi(A, B, F))
    m["NCIE"]  = float(info_metrics.ncie(A, B, F))
    m["SCD"]   = float(info_metrics.scd(A, B, F))
    m["FMI_pixel"]   = fmi_gpu(A, B, F, feature="none", device=device)
    m["FMI_dct"]     = fmi_gpu(A, B, F, feature="dct", device=device)
    m["FMI_wavelet"] = fmi_gpu(A, B, F, feature="wavelet", device=device)
    m["FMI_edge"]    = fmi_gpu(A, B, F, feature="edge", device=device)

    # Image
    m["QABF_fast"] = float(img_metrics.qabf(A, B, F))
    qabf_full, labf, nabf, nabf1 = img_metrics.petrovic_metrics(A, B, F)
    m["QABF"]  = float(qabf_full)
    m["LABF"]  = float(labf)
    m["NABF"]  = float(nabf)
    m["NABF1"] = float(nabf1)
    m["SF"]    = float(img_metrics.sf(F))

    # Structure
    m["SSIM"]    = float(0.5 * structure_metrics.ssim(A, F) + 0.5 * structure_metrics.ssim(B, F))
    m["MS_SSIM"] = ms_ssim_gpu(np.stack([A, B], axis=2), F, device=device)
    m["Q"]       = float(structure_metrics.piella_metrics(A, B, F, sw=1))
    m["Qw"]      = float(structure_metrics.piella_metrics(A, B, F, sw=2))
    m["Qe"]      = float(structure_metrics.piella_metrics(A, B, F, sw=3))

    # Visual
    m["VIF"]  = float(visual_metrics.vif(A, F) + visual_metrics.vif(B, F))
    m["VIFF"] = float(visual_metrics.viff(A, B, F))

    # Others
    m["CC"] = float(others_metrics.cc(A, B, F))
    m["Qp"] = float(others_metrics.qp(A, B, F))

    return m
