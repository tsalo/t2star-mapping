import os
from typing import Tuple

import nibabel as nib
import numpy as np


def load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    return data, img


def save_nifti_like(
    reference_img: nib.Nifti1Image,
    data: np.ndarray,
    out_path: str,
    dtype: np.dtype | None = None,
) -> None:
    if dtype is not None:
        data = data.astype(dtype)
    img = nib.Nifti1Image(
        data, affine=reference_img.affine, header=reference_img.header
    )
    nib.save(img, out_path)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
