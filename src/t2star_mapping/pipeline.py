from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .io import load_nifti, save_nifti_like
from .freqmap import compute_frequency_map
from .smooth import smooth_and_grad_z_polyfit3d
from .fitting import fit_t2star


@dataclass
class PipelineOptions:
    """Configuration for the T2* pipeline.

    Parameters
    ----------
    prefix : str, optional
        Prefix for output filenames.
    fitting_method : {"ols", "gls", "nlls", "num"}, optional
        T2* fitting method. Default is "nlls".
    echo_times_ms : list[float] or numpy.ndarray
        Echo times in milliseconds.
    rmse_thresh : float, optional
        RMSE threshold for frequency fit masking. Default 0.8.
    mask_thresh : float, optional
        Unused placeholder for future magnitude-based thresholding. Default 500.0.
    smooth_poly_order : int, optional
        3D polynomial order for smoothing. Default 3.
    downsample : tuple[int, int, int], optional
        Downsampling factors for smoothing. Default (2, 2, 2).
    dz_mm : float, optional
        Slice thickness in mm (for interpretation; not explicitly used). Default 1.25.
    threshold_t2star_max_ms : float, optional
        Upper clamp on T2* outputs (ms). Default 1000.
    """

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
    """High-level pipeline for T2* mapping and correction.

    The pipeline computes a frequency map from phase, smooths it and derives
    grad_z, then performs T2* fitting with and without correction.

    Parameters
    ----------
    magn_path : str
        Path to 4D multi-echo magnitude NIfTI.
    phase_path : str
        Path to 4D multi-echo phase NIfTI (radians).
    opts : PipelineOptions
        Pipeline configuration.
    """

    def __init__(self, magn_path: str, phase_path: str, opts: PipelineOptions):
        self.magn_data, self.magn_img = load_nifti(magn_path)
        self.phase_data, self.phase_img = load_nifti(phase_path)
        self.opts = opts
        if self.opts.echo_times_ms is None:
            raise ValueError("echo_times_ms must be provided")

    def run(self, out_dir: str) -> dict[str, str]:
        """Run the pipeline end-to-end and save outputs.

        Parameters
        ----------
        out_dir : str
            Output directory for generated NIfTI files.

        Returns
        -------
        dict[str, str]
            Mapping from logical output names to file paths.
        """
        nx, ny, nz, nt = self.magn_data.shape
        te = np.asarray(self.opts.echo_times_ms, dtype=float)[:nt]

        freq_map, mask = compute_frequency_map(
            self.magn_data,
            self.phase_data,
            te,
            rmse_thresh=self.opts.rmse_thresh,
        )
        freq_img_path = f"{out_dir}/{self.opts.prefix}freq.nii.gz"
        save_nifti_like(self.magn_img, freq_map.astype(np.float32), freq_img_path)

        mask_img_path = f"{out_dir}/{self.opts.prefix}mask.nii.gz"
        save_nifti_like(self.magn_img, mask.astype(np.uint8), mask_img_path)

        freq_smooth, grad_z = smooth_and_grad_z_polyfit3d(
            freq_map,
            mask,
            downsample=self.opts.downsample,
            poly_order=self.opts.smooth_poly_order,
        )
        freq_smooth_path = f"{out_dir}/{self.opts.prefix}freq_smooth.nii.gz"
        save_nifti_like(self.magn_img, freq_smooth.astype(np.float32), freq_smooth_path)
        grad_z_path = f"{out_dir}/{self.opts.prefix}freqGradZ.nii.gz"
        save_nifti_like(self.magn_img, grad_z.astype(np.float32), grad_z_path)

        # Corrected T2*: iterate voxelwise within mask
        t2_unc = np.zeros((nx, ny, nz), dtype=np.float32)
        t2_cor = np.zeros((nx, ny, nz), dtype=np.float32)
        r2_unc = np.zeros((nx, ny, nz), dtype=np.float32)
        r2_cor = np.zeros((nx, ny, nz), dtype=np.float32)
        iters = np.zeros((nx, ny, nz), dtype=np.int16)

        magn_flat = self.magn_data.reshape(nx * ny, nz, nt)
        mask_flat = mask.reshape(nx * ny, nz)
        gradZ_flat = grad_z.reshape(nx * ny, nz)
        for z in range(nz):
            indices = np.flatnonzero(mask_flat[:, z])
            for idx in indices:
                s = magn_flat[idx, z, :].astype(float)
                t2s_u, s0_u, s_fit_u, r2_u, it_u, _ = fit_t2star(
                    s, te, method=self.opts.fitting_method
                )
                r2_unc.flat[idx + z * nx * ny] = r2_u
                t2_unc.flat[idx + z * nx * ny] = np.clip(
                    t2s_u,
                    0,
                    self.opts.threshold_t2star_max_ms,
                )

                # correction using sinc(|grad_z|*TE/2000),
                # echo times in ms, grad_z in Hz/pixel ~ Hz/mm?
                # Follow MATLAB: /2000
                corr = np.sinc(gradZ_flat[idx, z] * te / 2000.0)
                corr = np.abs(corr)
                corr[corr == 0] = 1.0
                s_corr = s / corr
                t2s_c, s0_c, s_fit_c, r2_c, it_c, _ = fit_t2star(
                    s_corr,
                    te,
                    method=self.opts.fitting_method,
                )
                r2_cor.flat[idx + z * nx * ny] = r2_c
                t2_cor.flat[idx + z * nx * ny] = np.clip(
                    abs(t2s_c),
                    0,
                    self.opts.threshold_t2star_max_ms,
                )
                iters.flat[idx + z * nx * ny] = max(it_u, it_c)

        unc_path = f"{out_dir}/{self.opts.prefix}t2star_uncorrected_{self.opts.fitting_method}.nii.gz"
        cor_path = f"{out_dir}/{self.opts.prefix}t2star_corrected_{self.opts.fitting_method}.nii.gz"
        r2u_path = f"{out_dir}/{self.opts.prefix}rsquared_uncorrected_{self.opts.fitting_method}.nii.gz"
        r2c_path = f"{out_dir}/{self.opts.prefix}rsquared_corrected_{self.opts.fitting_method}.nii.gz"
        itr_path = (
            f"{out_dir}/{self.opts.prefix}iterations_{self.opts.fitting_method}.nii.gz"
        )

        save_nifti_like(self.magn_img, t2_unc.astype(np.float32), unc_path)
        save_nifti_like(self.magn_img, t2_cor.astype(np.float32), cor_path)
        save_nifti_like(self.magn_img, r2_unc.astype(np.float32), r2u_path)
        save_nifti_like(self.magn_img, r2_cor.astype(np.float32), r2c_path)
        save_nifti_like(self.magn_img, iters.astype(np.int16), itr_path)

        return dict(
            freq=freq_img_path,
            mask=mask_img_path,
            freq_smooth=freq_smooth_path,
            grad_z=grad_z_path,
            t2_uncorrected=unc_path,
            t2_corrected=cor_path,
            r2_uncorrected=r2u_path,
            r2_corrected=r2c_path,
            iterations=itr_path,
        )
