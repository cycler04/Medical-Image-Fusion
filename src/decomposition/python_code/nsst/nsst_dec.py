import numpy as np
from scipy.signal import convolve2d
from decomposition.python_code.nsst.shearing_filters_myer import shearing_filters_myer
from decomposition.python_code.nsst.processing_ultis import conv2_same_matlab, symext, upsample2df, atrousc
from decomposition.python_code.nsst.atrousfilters import atrousfilters
import time

def atrousdec(x, fname, Nlevels):
    """
    2-D nonsubsampled (undecimated) à trous wavelet decomposition.

    Decomposes image x into Nlevels + 1 subbands — one lowpass and
    Nlevels bandpass (detail) subbands — all at full resolution.

    Parameters
    ----------
    x       : ndarray, shape (m, n)  — input image
    fname   : str                    — filter name (see atrousfilters)
    Nlevels : int                    — number of decomposition levels

    Returns
    -------
    y : list of length (Nlevels + 1)
        y[0]          : coarsest lowpass subband
        y[1], …, y[N] : detail subbands, coarse → fine  (y[N] is finest)
    """
    h0, h1, g0, g1 = atrousfilters(fname)

    y = [None] * (Nlevels + 1)

    # ── Level 1 (finest detail) — plain convolution ─────────────────────────
    shift = np.array([1, 1])
    ext_x_h0 = symext(x, h0, shift)
    ext_x_h1 = symext(x, h1, shift)
    y0 = convolve2d(ext_x_h0, h0, mode='valid')
    y1 = convolve2d(ext_x_h1, h1, mode='valid')

    y[Nlevels] = y1      # finest detail  (y{Nlevels+1} in MATLAB, 1-based)
    x_curr = y0
    I2 = np.eye(2)

    # ── Levels 2 … Nlevels — à trous convolution ────────────────────────────
    for i in range(1, Nlevels):            # i = 1 : Nlevels-1  (MATLAB)
        L     = 2 ** i
        shift = -2 ** (i - 1) * np.array([1, 1]) + 2   # delay compensation

        h0_up = upsample2df(h0, i)
        h1_up = upsample2df(h1, i)

        y0 = atrousc(symext(x_curr, h0_up, shift), h0, I2 * L)
        y1 = atrousc(symext(x_curr, h1_up, shift), h1, I2 * L)

        y[Nlevels - i] = y1    # coarser detail levels (y{Nlevels-i+1} MATLAB)
        x_curr = y0

    y[0] = x_curr              # coarsest lowpass
    return y


def nsst_dec(x, shear_parameters, lpfilt):
    """
    Nonsubsampled Shearlet Transform (NSST) decomposition.

    Combines an à trous Laplacian Pyramid with Meyer directional filters
    to separate the image by both scale and orientation — at full resolution.

    Parameters
    ----------
    x : ndarray, shape (m, n)
        Input image (grayscale, float preferred).

    shear_parameters : dict with keys:
        'dcomp' : list of int, length = number_of_levels
            dcomp[i] means level i+1 has 2^dcomp[i] directional subbands.
        'dsize' : list of int, same length as dcomp
            Spatial support size of the shearing filter at each level.

    lpfilt : str
        Filter name for the lowpass/highpass pyramid (see atrousfilters).


    Returns
    -------
    dst : list of length (number_of_levels + 1)
        dst[0]            : ndarray (m, n)     — coarsest lowpass subband
        dst[i]            : ndarray (m, n, K)  — level i detail subbands
                            where K = 2^dcomp[i-1] is the number of directions

    shear_f : list of length number_of_levels
        shear_f[i] : ndarray (dsize[i], dsize[i], 2^dcomp[i])
                     Precomputed shearing filters for level i.
                     Can be reused on other images of the same size.

    """
    x = np.asarray(x, dtype=float)

    level = len(shear_parameters['dcomp'])

    # ── Step 1: à trous Laplacian Pyramid decomposition ─────────────────────
    y = atrousdec(x, lpfilt, level)   # list of (level+1) full-res subbands

    # ── Step 2: directional filtering of each detail subband ────────────────
    dst     = [None] * (level + 1)
    shear_f = [None] * level
    # Apply each directional filter to the i-th detail subband


    dst[0] = y[0]   # coarsest lowpass passed through directly
    for i in range(level):
        dcomp_i = shear_parameters['dcomp'][i]
        dsize_i = shear_parameters['dsize'][i]
        n_dirs  = 2 ** dcomp_i

        # Build (or retrieve) shearing filters for this level
        # Scaled by sqrt(dsize) to normalise energy across directions
        shear_f[i] = (
            shearing_filters_myer(dsize_i, dcomp_i) * np.sqrt(dsize_i)
        )
        detail = np.zeros((*x.shape, n_dirs))

        for k in range(n_dirs):
            detail[:, :, k] = conv2_same_matlab(
                y[i + 1], shear_f[i][:, :, k]
            )
        #     stats(detail[:, :, k], f"Level {i+1} detail dir {k+1}")
        # print()
        dst[i + 1] = detail

    return dst, shear_f

def atrousdec_with_timing(x, fname, Nlevels, verbose=True):
    h0, h1, g0, g1 = atrousfilters(fname)

    y = [None] * (Nlevels + 1)

    t_total_start = time.time()

    # ── Level 1 (finest detail) ───────────────────────────────────────────
    t_level_start = time.time()

    shift = np.array([1, 1])

    # symext timing
    t0 = time.time()
    ext_x_h0 = symext(x, h0, shift)
    ext_x_h1 = symext(x, h1, shift)
    t_sym = time.time() - t0

    # convolution timing
    t0 = time.time()
    y0 = convolve2d(ext_x_h0, h0, mode='valid')
    y1 = convolve2d(ext_x_h1, h1, mode='valid')
    t_conv = time.time() - t0

    y[Nlevels] = y1
    x_curr = y0

    t_level = time.time() - t_level_start

    if verbose:
        print(f"[Atrous Level 1]")
        print(f"  total time : {t_level:.4f}s")
        print(f"  symext     : {t_sym:.4f}s")
        print(f"  conv       : {t_conv:.4f}s")
        print()

    # Accumulators
    t_sym_total = t_sym
    t_conv_total = t_conv
    t_atrous_total = 0

    I2 = np.eye(2)

    # ── Levels 2 … Nlevels ────────────────────────────────────────────────
    for i in range(1, Nlevels):
        t_level_start = time.time()

        L     = 2 ** i
        shift = -2 ** (i - 1) * np.array([1, 1]) + 2

        h0_up = upsample2df(h0, i)
        h1_up = upsample2df(h1, i)

        # symext timing
        t0 = time.time()
        ext0 = symext(x_curr, h0_up, shift)
        ext1 = symext(x_curr, h1_up, shift)
        t_sym = time.time() - t0

        # atrous convolution timing
        t0 = time.time()
        y0 = atrousc(ext0, h0, I2 * L)
        y1 = atrousc(ext1, h1, I2 * L)
        t_atrous = time.time() - t0

        y[Nlevels - i] = y1
        x_curr = y0

        t_level = time.time() - t_level_start

        # accumulate
        t_sym_total += t_sym
        t_atrous_total += t_atrous

        if verbose:
            print(f"[Atrous Level {i+1}]")
            print(f"  total time : {t_level:.4f}s")
            print(f"  symext     : {t_sym:.4f}s")
            print(f"  atrousc    : {t_atrous:.4f}s")
            print()

    y[0] = x_curr

    t_total = time.time() - t_total_start

    # ── Summary ───────────────────────────────────────────────────────────
    if verbose:
        print("=== ATROUS TIMING SUMMARY ===")
        print(f"Total atrous time : {t_total:.4f}s")
        print(f"  symext total    : {t_sym_total:.4f}s  ({t_sym_total/t_total:.1%})")
        print(f"  conv (lvl1)     : {t_conv_total:.4f}s  ({t_conv_total/t_total:.1%})")
        print(f"  atrousc total   : {t_atrous_total:.4f}s  ({t_atrous_total/t_total:.1%})")

    return y

def nsst_dec_with_timing(x, shear_parameters, lpfilt, verbose=True):
    x = np.asarray(x, dtype=float)
    level = len(shear_parameters['dcomp'])

    t_total_start = time.time()

    # ── Step 1: à trous Laplacian Pyramid ────────────────────────────────
    t0 = time.time()
    y = atrousdec_with_timing(x, lpfilt, level)
    t_atrous = time.time() - t0

    # ── Step 2: directional filtering ────────────────────────────────────
    dst     = [None] * (level + 1)
    shear_f = [None] * level

    dst[0] = y[0]

    t_dir_total = 0
    t_conv_total = 0
    t_filter_total = 0

    for i in range(level):
        t_level_start = time.time()

        dcomp_i = shear_parameters['dcomp'][i]
        dsize_i = shear_parameters['dsize'][i]
        n_dirs  = 2 ** dcomp_i

        # ── Filter generation timing ────────────────────────────────────
        t0 = time.time()
        shear_f[i] = (
            shearing_filters_myer(dsize_i, dcomp_i) * np.sqrt(dsize_i)
        )
        t_filter = time.time() - t0
        t_filter_total += t_filter

        detail = np.zeros((*x.shape, n_dirs))

        # ── Convolution timing ──────────────────────────────────────────
        t_conv_level = 0

        for k in range(n_dirs):
            t1 = time.time()

            detail[:, :, k] = conv2_same_matlab(
                y[i + 1], shear_f[i][:, :, k]
            )

            t_conv = time.time() - t1
            t_conv_level += t_conv

        t_conv_total += t_conv_level

        dst[i + 1] = detail

        t_level = time.time() - t_level_start
        t_dir_total += t_level

        if verbose:
            print(f"[Level {i+1}]")
            print(f"  total level time : {t_level:.4f}s")
            print(f"  filter gen time  : {t_filter:.4f}s")
            print(f"  conv total time  : {t_conv_level:.4f}s")
            print(f"  conv per dir     : {t_conv_level / n_dirs:.4f}s")
            print()

    t_total = time.time() - t_total_start

    # ── Summary ─────────────────────────────────────────────────────────
    if verbose:
        print("=== TIMING SUMMARY ===")
        print(f"Total time           : {t_total:.4f}s")
        print(f"Atrous (pyramid)     : {t_atrous:.4f}s  ({t_atrous/t_total:.1%})")
        print(f"Directional total    : {t_dir_total:.4f}s  ({t_dir_total/t_total:.1%})")
        print(f"  ├─ Convolution     : {t_conv_total:.4f}s  ({t_conv_total/t_total:.1%})")
        print(f"  └─ Filter gen      : {t_filter_total:.4f}s  ({t_filter_total/t_total:.1%})")

    return dst, shear_f

if __name__ == "__main__": 
    import cv2
    import matplotlib.pyplot as plt

    # Custom test image: smooth gradient + a central impulse

    img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
    # N = 63
    # img = np.zeros((N, N))
    # for i in range(N):
    #     for j in range(N):
    #         img[i, j] = i + j  # smooth gradient
    # img[N//2, N//2] += 10   
    params = {'dcomp': [2, 2, 2], 'dsize': [16, 16, 16]}

    lpfilt = 'maxflat'   # or gaussian if error


    # print("Decomposing the image using NSST...")
    dst, shear_f = nsst_dec_with_timing(img, params, lpfilt=lpfilt)