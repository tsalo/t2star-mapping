from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .io import load_nifti, save_nifti_like
from .freqmap import compute_frequency_map
from .smooth import SmoothOptions, smooth_and_gradZ_polyfit3d
from .fitting import fit_t2star, FitResult


@dataclass
class PipelineOptions:
	prefix: str = ""
	fitting_method: Literal["ols", "gls", "nlls", "num"] = "nlls"
	echo_times_ms: list[float] | np.ndarray = None
	rmse_thresh: float = 0.8
	mask_thresh: float = 500.0
	smooth_poly_order: int = 3
	downsample: tuple[int, int, int] = (2, 2, 2)
	dz_mm: float = 1.25
	threshold_t2star_max_ms: float = 1000.0


class T2StarPipeline:
	def __init__(self, magn_path: str, phase_path: str, opts: PipelineOptions):
		self.magn_data, self.magn_img = load_nifti(magn_path)
		self.phase_data, self.phase_img = load_nifti(phase_path)
		self.opts = opts
		if self.opts.echo_times_ms is None:
			raise ValueError("echo_times_ms must be provided")

	def run(self, out_dir: str) -> dict[str, str]:
		nx, ny, nz, nt = self.magn_data.shape
		te = np.asarray(self.opts.echo_times_ms, dtype=float)[:nt]

		freq_map, mask = compute_frequency_map(self.magn_data, self.phase_data, te, rmse_thresh=self.opts.rmse_thresh)
		freq_img_path = f"{out_dir}/{self.opts.prefix}freq.nii.gz"
		save_nifti_like(self.magn_img, freq_map.astype(np.float32), freq_img_path)

		mask_img_path = f"{out_dir}/{self.opts.prefix}mask.nii.gz"
		save_nifti_like(self.magn_img, mask.astype(np.uint8), mask_img_path)

		freq_smooth, gradZ = smooth_and_gradZ_polyfit3d(freq_map, mask, SmoothOptions(downsample=self.opts.downsample, poly_order=self.opts.smooth_poly_order))
		freq_smooth_path = f"{out_dir}/{self.opts.prefix}freq_smooth.nii.gz"
		save_nifti_like(self.magn_img, freq_smooth.astype(np.float32), freq_smooth_path)
		gradZ_path = f"{out_dir}/{self.opts.prefix}freqGradZ.nii.gz"
		save_nifti_like(self.magn_img, gradZ.astype(np.float32), gradZ_path)

		# Corrected T2*: iterate voxelwise within mask
		t2_unc = np.zeros((nx, ny, nz), dtype=np.float32)
		t2_cor = np.zeros((nx, ny, nz), dtype=np.float32)
		r2_unc = np.zeros((nx, ny, nz), dtype=np.float32)
		r2_cor = np.zeros((nx, ny, nz), dtype=np.float32)
		iters = np.zeros((nx, ny, nz), dtype=np.int16)

		magn_flat = self.magn_data.reshape(nx * ny, nz, nt)
		mask_flat = mask.reshape(nx * ny, nz)
		gradZ_flat = gradZ.reshape(nx * ny, nz)
		for z in range(nz):
			indices = np.flatnonzero(mask_flat[:, z])
			for idx in indices:
				S = magn_flat[idx, z, :].astype(float)
				res_unc: FitResult = fit_t2star(S, te, method=self.opts.fitting_method)
				r2_unc.flat[idx + z * nx * ny] = res_unc.r_squared
				t2_unc.flat[idx + z * nx * ny] = np.clip(res_unc.T2star_ms, 0, self.opts.threshold_t2star_max_ms)

				# correction using sinc(|gradZ|*TE/2000), echo times in ms, gradZ in Hz/pixel ~ Hz/mm? Follow MATLAB: /2000
				corr = np.sinc(gradZ_flat[idx, z] * te / 2000.0)
				corr = np.abs(corr)
				corr[corr == 0] = 1.0
				S_corr = S / corr
				res_cor: FitResult = fit_t2star(S_corr, te, method=self.opts.fitting_method)
				r2_cor.flat[idx + z * nx * ny] = res_cor.r_squared
				t2_cor.flat[idx + z * nx * ny] = np.clip(abs(res_cor.T2star_ms), 0, self.opts.threshold_t2star_max_ms)
				iters.flat[idx + z * nx * ny] = res_cor.iterations

		unc_path = f"{out_dir}/{self.opts.prefix}t2star_uncorrected_{self.opts.fitting_method}.nii.gz"
		cor_path = f"{out_dir}/{self.opts.prefix}t2star_corrected_{self.opts.fitting_method}.nii.gz"
		r2u_path = f"{out_dir}/{self.opts.prefix}rsquared_uncorrected_{self.opts.fitting_method}.nii.gz"
		r2c_path = f"{out_dir}/{self.opts.prefix}rsquared_corrected_{self.opts.fitting_method}.nii.gz"
		itr_path = f"{out_dir}/{self.opts.prefix}iterations_{self.opts.fitting_method}.nii.gz"

		save_nifti_like(self.magn_img, t2_unc.astype(np.float32), unc_path)
		save_nifti_like(self.magn_img, t2_cor.astype(np.float32), cor_path)
		save_nifti_like(self.magn_img, r2_unc.astype(np.float32), r2u_path)
		save_nifti_like(self.magn_img, r2_cor.astype(np.float32), r2c_path)
		save_nifti_like(self.magn_img, iters.astype(np.int16), itr_path)

		return dict(
			freq=freq_img_path,
			mask=mask_img_path,
			freq_smooth=freq_smooth_path,
			gradZ=gradZ_path,
			t2_uncorrected=unc_path,
			t2_corrected=cor_path,
			r2_uncorrected=r2u_path,
			r2_corrected=r2c_path,
			iterations=itr_path,
		)
