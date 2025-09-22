from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from numpy.typing import ArrayLike


def compute_mask_from_magnitude(magn_4d: np.ndarray, thresh: float = 500.0, sigma: float = 5.0) -> np.ndarray:
	first_echo = magn_4d[..., 0]
	smoothed = gaussian_filter(first_echo, sigma=sigma)
	mask = (smoothed > thresh).astype(np.uint8)
	return mask


def unwrap_phase_1d(phase: np.ndarray) -> np.ndarray:
	return np.unwrap(phase)


def compute_frequency_map(magn_4d: np.ndarray, phase_4d: np.ndarray, echo_times_ms: ArrayLike, rmse_thresh: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
	nx, ny, nz, nt = phase_4d.shape
	te_s = np.asarray(echo_times_ms, dtype=float)[:nt] / 1000.0
	mask = compute_mask_from_magnitude(magn_4d)

	freq_map = np.zeros((nx, ny, nz), dtype=float)
	X = np.c_[te_s, np.ones(nt)]

	magn_2d = magn_4d.reshape(nx * ny, nz, nt)
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
