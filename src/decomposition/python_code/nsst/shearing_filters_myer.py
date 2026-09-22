
import numpy as np
from numpy.fft import ifft2, fftshift
from decomposition.python_code.nsst.processing_ultis import stats


def meyer_wind(x):
    """
    Scalar Meyer window function.

    Returns 1 in the passband, 0 outside, and smoothly transitions
    in between using a degree-7 polynomial followed by a cosine taper.

    Parameters
    ----------
    x : float
        Input value (typically in range [-0.5, 1.5] for normal usage).

    Returns
    -------
    y : float
    """
    # Passband: (-1/3 + 1/2, 1/3 + 1/2)  =  (1/6, 5/6)
    if (1 / 6) < x < (5 / 6):
        return 1.0

    # Transition bands:
    #   upper: [5/6, 7/6]
    #   lower: [-1/6, 1/6]  (the elseif catches what the if missed)
    elif ((5 / 6) <= x <= (7 / 6)) or ((-1 / 6) <= x <= (1 / 6)):
        w = 3 * abs(x - 0.5) - 1
        z = w ** 4 * (35 - 84 * w + 70 * w ** 2 - 20 * w ** 3)
        return float(np.cos(np.pi / 2 * z) ** 2)

    else:
        return 0.0


def windowing(x, L):
    """
    Decompose signal x into L Meyer-windowed bandpass channels.

    Parameters
    ----------
    x : 1-D ndarray, length N
        Input signal (typically all-ones when building shearing filters).
    L : int
        Number of bandpass channels (directions).

    Returns
    -------
    y : ndarray, shape (N, L)
        Each column y[:, k] is x multiplied by the k-th Meyer window.
    """
    N = len(x)
    y = np.zeros((N, L))
    T = N // L          # samples per band (assumed integer)

    # Build Meyer window of length 2T
    g = np.zeros(2 * T)
    for j in range(2 * T):
        n_val = -T / 2 + j   # centred sample index
        g[j] = meyer_wind(n_val / T)

    # Apply window to each band
    for j in range(L):
        index = 0
        for k in range(int(-T / 2), int(1.5 * T)):
            # Circular index (0-based)
            in_sig = int(np.floor(np.mod(k + j * T, N)))
            y[in_sig, j] = g[index] * x[in_sig]
            index += 1

    return y



def _avg_pol(L, x1, y1, x2, y2):
    """
    Count how many times each Cartesian grid point is visited by the
    pseudo-polar radial slices.  Used to average overlap in rec_from_pol.

    Parameters
    ----------
    L       : int     — grid size (L × L)
    x1, y1  : ndarray — Cartesian coords for the first set of radial slices
    x2, y2  : ndarray — Cartesian coords for the second set of radial slices

    Returns
    -------
    D : ndarray, shape (L, L)  — hit-count matrix (all entries >= 1)
    """
    D = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            # Coordinates are stored 1-based; subtract 1 for Python indexing
            r = int(y1[i, j]) - 1
            c = int(x1[i, j]) - 1
            D[r, c] += 1
    for i in range(L):
        for j in range(L):
            r = int(y2[i, j]) - 1
            c = int(x2[i, j]) - 1
            D[r, c] += 1
    return D
def matlab_round(x):
    return np.floor(np.abs(x) + 0.5) * np.sign(x)

def gen_x_y_cordinates(n):
    """
    Generate pseudo-polar grid coordinates for an n × n block.

    Produces two families of radial lines that together tile the
    frequency plane: one sweeping across rows (x1/y1) and one across
    columns (x2/y2).

    Parameters
    ----------
    n : int
        Block order (e.g. dsize parameter, typically a power of 2).

    Returns
    -------
    x1n, y1n : ndarray, shape (n, n)  — Cartesian coords, family 1 (1-based)
    x2n, y2n : ndarray, shape (n, n)  — Cartesian coords, family 2 (1-based)
    D        : ndarray, shape (n, n)  — overlap count matrix
    """
    n_orig = n
    n = n + 1  # MATLAB convention: work on (n+1) × (n+1) then trim

    x1 = np.zeros((n, n))
    y1 = np.zeros((n, n))
    x2 = np.zeros((n, n))
    y2 = np.zeros((n, n))
    xt = np.zeros((n, n))
    m1 = np.zeros(n)

    for i in range(1, n + 1):          # i = 1 : n  (1-based)
        y0 = 1;  x0 = i
        x_n = n - i + 1;  y_n = n

        if x_n == x0:
            flag = 1
        else:
            m1[i - 1] = (y_n - y0) / (x_n - x0)
            flag = 0

        xt[i - 1, :] = np.linspace(x0, x_n, n)

        for j in range(1, n + 1):      # j = 1 : n  (1-based)
            if flag == 0:
                yval = m1[i - 1] * (xt[i - 1, j - 1] - x0) + y0
                y1[i - 1, j - 1] = matlab_round(yval)
                x1[i - 1, j - 1] = matlab_round(xt[i - 1, j - 1])
                x2[i - 1, j - 1] = y1[i - 1, j - 1]
                y2[i - 1, j - 1] = x1[i - 1, j - 1]
            else:
                x1[i - 1, j - 1] = (n - 1) / 2 + 1
                y1[i - 1, j - 1] = j
                x2[i - 1, j - 1] = j
                y2[i - 1, j - 1] = (n - 1) / 2 + 1

    # print(xt)
    # print(x1)
    # print(y2)    
    # Trim back to n_orig × n_orig and re-index (matching MATLAB)
    n = n_orig
    x1n = np.zeros((n, n))
    y1n = np.zeros((n, n))
    x2n = np.zeros((n, n))
    y2n = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x1n[i, j] = x1[i, j]
            y1n[i, j] = y1[i, j]
            x2n[i, j] = x2[i + 1, j]   # offset by 1 row (MATLAB: i+1)
            y2n[i, j] = y2[i + 1, j]

    x1n = np.flipud(x1n)
    y2n[n - 1, 0] = n   # boundary correction (MATLAB: y2n(n,1)=n)

    D = _avg_pol(n, x1n, y1n, x2n, y2n)
    return x1n, y1n, x2n, y2n, D


def rec_from_pol(l, n, x1, y1, x2, y2, D):
    """
    Re-assemble radial (pseudo-polar) slices into a Cartesian n × n block.

    Parameters
    ----------
    l       : ndarray, shape (2n, n)
              Windowed radial slice values.  Rows 0:n are family-1 slices
              and rows n:2n are family-2 slices.
    n       : int   — output block size
    x1, y1  : ndarray, shape (n, n)  — family-1 Cartesian coords (1-based)
    x2, y2  : ndarray, shape (n, n)  — family-2 Cartesian coords (1-based)
    D       : ndarray, shape (n, n)  — overlap count from gen_x_y_cordinates

    Returns
    -------
    C : ndarray, shape (n, n)  — reconstructed Cartesian block
    """
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Coordinates are 1-based; convert to 0-based
            r1 = int(y1[i, j]) - 1
            c1 = int(x1[i, j]) - 1
            C[r1, c1] += l[i, j]

            r2 = int(y2[i, j]) - 1
            c2 = int(x2[i, j]) - 1
            C[r2, c2] += l[i + n, j]   # second family starts at row n

    # Average pixels visited by multiple radial lines
    C = C / D
    return C


def shearing_filters_myer(n1, level):
    """
    Build 2^level directional (shearing) filters of spatial size n1 × n1
    using the Meyer window function.

    Parameters
    ----------
    n1    : int   — spatial support size of each filter (n1 × n1)
    level : int   — number of directions = 2^level

    Returns
    -------
    w_s : ndarray, shape (n1, n1, 2^level)
          Real-valued spatial-domain shearing filters.
          w_s[:, :, k] is the k-th directional filter.
    """
    n_dirs = 2 ** level

    # Step 1: generate pseudo-polar grid for n1 × n1 block
    x11, y11, x12, y12, F1 = gen_x_y_cordinates(n1)
    # Step 2: Meyer windowing — 1D signal of length 2*n1, split into n_dirs bands
    wf = windowing(np.ones(2 * n1), n_dirs)   # shape: (2*n1, n_dirs)

    # Step 3: for each direction, build the 2D filter
    w_s = np.zeros((n1, n1, n_dirs))
    for k in range(n_dirs):
        # Outer product: (2*n1,) × (n1,) → (2*n1, n1) radial slice matrix
        temp = np.outer(wf[:, k], np.ones(n1))

        # Convert radial slice → Cartesian frequency-domain filter
        freq_filter = rec_from_pol(temp, n1, x11, y11, x12, y12, F1)

        # Inverse FFT to get spatial-domain filter (real part only)
        w_s[:, :, k] = (
            np.real(fftshift(ifft2(fftshift(freq_filter)))) / np.sqrt(n1)
        )

    return w_s

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    shear = shearing_filters_myer(16, 1)

    n_dirs = shear.shape[2]

    # =============================
    # 1. SPATIAL DOMAIN
    # =============================
    plt.figure(figsize=(12, 3))
    for k in range(n_dirs):
        plt.subplot(1, n_dirs, k + 1)
        plt.imshow(shear[:, :, k], cmap='gray')
        plt.title(f"Dir {k}\n(spatial)")
        plt.axis('off')

    plt.suptitle("Shear Filters — Spatial Domain")
    plt.tight_layout()
    plt.show()

    # =============================
    # 2. FREQUENCY
    # =============================
    plt.figure(figsize=(12, 3))
    for k in range(n_dirs):
        f = shear[:, :, k]

        freq = np.fft.fftshift(np.fft.fft2(f))   # NO padding
        mag = np.log1p(np.abs(freq))

        plt.subplot(1, n_dirs, k + 1)
        plt.imshow(mag, cmap='gray', interpolation='nearest')
        plt.title(f"Dir {k}\n(freq real)")
        plt.axis('off')

    plt.suptitle("Shear Filters — Frequency")
    plt.tight_layout()
    plt.show()