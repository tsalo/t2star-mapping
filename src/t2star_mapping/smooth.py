from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SmoothOptions:
    downsample: tuple[int, int, int] = (2, 2, 2)
    poly_order: int = 3


def _grid_indices(
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = shape
    y, x, z = np.meshgrid(np.arange(ny), np.arange(nx), np.arange(nz), indexing="ij")
    # Return in x,y,z order to match array indexing later
    return x.T, y.T, z.T


def _build_model_terms(order: int) -> np.ndarray:
    terms = []
    for pz in range(order + 1):
        for py in range(order + 1):
            for px in range(order + 1):
                terms.append((px, py, pz))
    return np.asarray(terms, dtype=int)


def _design_matrix(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    terms: np.ndarray,
) -> np.ndarray:
    n = x.size
    nt = terms.shape[0]
    M = np.ones((n, nt), dtype=float)
    for i, (px, py, pz) in enumerate(terms):
        M[:, i] = (x**px) * (y**py) * (z**pz)
    return M


def smooth_and_gradZ_polyfit3d(
    freq_3d: np.ndarray,
    mask_3d: np.ndarray,
    opts: SmoothOptions,
) -> tuple[np.ndarray, np.ndarray]:
    # Downsample using nearest neighbor indexing
    dx, dy, dz = opts.downsample
    nx, ny, nz = freq_3d.shape
    xi = np.arange(0, nx, dx)
    yi = np.arange(0, ny, dy)
    zi = np.arange(0, nz, dz)
    freq_i = freq_3d[np.ix_(xi, yi, zi)]
    mask_i = mask_3d[np.ix_(xi, yi, zi)]

    terms = _build_model_terms(opts.poly_order)
    x, y, z = _grid_indices(freq_i.shape)
    # Flatten masked points
    valid = (mask_i > 0) & np.isfinite(freq_i)
    ind = np.flatnonzero(valid)
    if ind.size == 0:
        return np.zeros_like(freq_3d), np.zeros_like(freq_3d)

    M = _design_matrix(x.ravel()[ind], y.ravel()[ind], z.ravel()[ind], terms)
    d = freq_i.ravel()[ind]
    coeff, *_ = np.linalg.lstsq(M, d, rcond=None)

    # Reconstruct smoothed freq at downsampled grid
    M_full = _design_matrix(x.ravel(), y.ravel(), z.ravel(), terms)
    datafit = (M_full @ coeff).reshape(freq_i.shape)

    # Derivative along z at downsampled grid
    def design_matrix_dz(
        xv: np.ndarray,
        yv: np.ndarray,
        zv: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        nt = t.shape[0]
        Mdz = np.zeros((xv.size, nt), dtype=float)
        for i, (px, py, pz) in enumerate(t):
            if pz == 0:
                Mdz[:, i] = 0.0
            else:
                Mdz[:, i] = pz * (xv**px) * (yv**py) * (zv ** (pz - 1))
        return Mdz

    M_dz_full = design_matrix_dz(x.ravel(), y.ravel(), z.ravel(), terms)
    datafit_dz = (M_dz_full @ coeff).reshape(freq_i.shape)

    # Upsample back using nearest neighbor
    freq_smooth = np.zeros_like(freq_3d)
    gradZ = np.zeros_like(freq_3d)
    for ix, x0 in enumerate(xi):
        for iy, y0 in enumerate(yi):
            for iz, z0 in enumerate(zi):
                freq_smooth[
                    x0 : min(x0 + dx, nx), y0 : min(y0 + dy, ny), z0 : min(z0 + dz, nz)
                ] = datafit[ix, iy, iz]
                gradZ[
                    x0 : min(x0 + dx, nx), y0 : min(y0 + dy, ny), z0 : min(z0 + dz, nz)
                ] = datafit_dz[ix, iy, iz]

    return freq_smooth * mask_3d, gradZ * mask_3d
