from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from numpy.typing import ArrayLike


def compute_mask_from_magnitude(
    magn_4d: np.ndarray,
    thresh: float = 500.0,
    sigma: float = 5.0,
) -> np.ndarray:
    """Create a binary mask from the first-echo magnitude image.

    Parameters
    ----------
    magn_4d : numpy.ndarray
        Multi-echo magnitude image with shape (X, Y, Z, T).
    thresh : float, optional
        Intensity threshold applied after smoothing. Default is 500.0.
    sigma : float, optional
        Gaussian smoothing sigma (in voxels). Default is 5.0.

    Returns
    -------
    numpy.ndarray
        Binary mask (X, Y, Z) with 1 for in-mask voxels.
    """
    first_echo = magn_4d[..., 0]
    smoothed = gaussian_filter(first_echo, sigma=sigma)
    mask = (smoothed > thresh).astype(np.uint8)
    return mask


def unwrap_phase_1d(phase: np.ndarray) -> np.ndarray:
    """Unwrap a 1D phase vector.

    Parameters
    ----------
    phase : numpy.ndarray
        Phase values along echo times (radians).

    Returns
    -------
    numpy.ndarray
        Unwrapped phase.
    """
    return np.unwrap(phase)


def compute_frequency_map(
    magn_4d: np.ndarray,
    phase_4d: np.ndarray,
    echo_times_ms: ArrayLike,
    rmse_thresh: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frequency map (Hz) from multi-echo phase with masking.

    Per-voxel phase is unwrapped along echo dimension, then a linear model
    phi = a*TE + b is fitted using least squares; frequency is a/(2*pi).
    Voxels with scaled fit RMSE >= ``rmse_thresh`` are discarded.

    Parameters
    ----------
    magn_4d : numpy.ndarray
        Multi-echo magnitude image (X, Y, Z, T), used to derive a mask.
    phase_4d : numpy.ndarray
        Multi-echo phase image (X, Y, Z, T), in radians.
    echo_times_ms : array_like
        Echo times (ms), length T.
    rmse_thresh : float, optional
        Threshold on scaled RMSE to retain frequency estimates. Default 0.8.

    Returns
    -------
    freq_map : numpy.ndarray
        Frequency map in Hz with shape (X, Y, Z).
    mask : numpy.ndarray
        Binary mask (X, Y, Z) derived from magnitude.
    """
    nx, ny, nz, nt = phase_4d.shape
    te_s = np.asarray(echo_times_ms, dtype=float)[:nt] / 1000.0
    mask = compute_mask_from_magnitude(magn_4d)

    freq_map = np.zeros((nx, ny, nz), dtype=float)
    X = np.c_[te_s, np.ones(nt)]

    phase_2d = phase_4d.reshape(nx * ny, nz, nt)
    mask_2d = mask.reshape(nx * ny, nz)
    for z in range(nz):
        ind = np.flatnonzero(mask_2d[:, z])
        if ind.size == 0:
            continue

        for idx in ind:
            phi = unwrap_phase_1d(phase_2d[idx, z, :])
            a = np.linalg.pinv(X) @ phi
            # scaled RMSE as in MATLAB port
            y_scaled = phi - phi.min()
            y_scaled = y_scaled / (y_scaled.max() if y_scaled.max() != 0 else 1.0)
            a_scaled = np.linalg.pinv(X) @ y_scaled
            yhat_scaled = a_scaled[0] * te_s + a_scaled[1]
            rmse = float(np.sqrt(np.sum((y_scaled - yhat_scaled) ** 2)))
            if rmse < rmse_thresh:
                f_hz = a[0] / (2.0 * np.pi)
                x = idx % nx
                y = idx // nx
                freq_map[x, y, z] = f_hz

    return freq_map, mask
