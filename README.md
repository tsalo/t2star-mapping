# t2star-mapping
Collection of scripts to fit T2* MRI data

Python package for T2* mapping with through-slice dropout correction (Dahnke & Schaeffter, MRM 2005). Translated from the original MATLAB scripts in `scripts/` and implemented with numpy, scipy, nibabel, and nilearn (no FSL required).

## Features
- Frequency map from multi-echo phase with magnitude-based masking
- 3D polynomial smoothing and slice-direction gradient (gradZ)
- T2* fitting methods: OLS, GLS, NLLS (Levenberg–Marquardt), and numerical approximation
- Correction using sinc(|gradZ|*TE/2000)
- NIfTI I/O via nibabel; outputs match MATLAB naming

## Install
- Create a virtualenv and install requirements:
  pip install -r requirements.txt

## CLI usage
- Example:
  python -m t2star_mapping.cli \
    --magn /path/to/magn.nii.gz \
    --phase /path/to/phase.nii.gz \
    --te 6.34 9.54 12.74 15.94 \
    --out /path/to/outdir \
    --prefix top_ \
    --method nlls

## Inputs
- Multi-echo magnitude 4D NIfTI (X×Y×Z×T)
- Multi-echo phase 4D NIfTI (X×Y×Z×T), unwrapped per-voxel during fitting
- Echo times in ms (order must match the 4th dimension)

## Outputs (written to --out)
- prefixfreq.nii.gz
- prefixmask.nii.gz
- prefixfreq_smooth.nii.gz
- prefixfreqGradZ.nii.gz
- prefixt2star_uncorrected_<method>.nii.gz
- prefixt2star_corrected_<method>.nii.gz
- prefixrsquared_uncorrected_<method>.nii.gz
- prefixrsquared_corrected_<method>.nii.gz
- prefixiterations_<method>.nii.gz

## Python API
- `src/t2star_mapping/io.py`: NIfTI read/write helpers
- `src/t2star_mapping/freqmap.py`: Frequency map + mask from phase/magnitude
- `src/t2star_mapping/smooth.py`: 3D polynomial smoothing and gradZ
- `src/t2star_mapping/fitting.py`: T2* fitting implementations
- `src/t2star_mapping/pipeline.py`: End-to-end pipeline
