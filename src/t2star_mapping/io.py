import os
from typing import Tuple

import nibabel as nb
import numpy as np


def load_nifti(path: str) -> Tuple[np.ndarray, nb.Nifti1Image]:
    img = nb.load(path)
    data = np.asanyarray(img.dataobj)
    return data, img


def save_nifti_like(
    reference_img: nb.Nifti1Image,
    data: np.ndarray,
    out_path: str,
    dtype: np.dtype | None = None,
) -> None:
    if dtype is not None:
        data = data.astype(dtype)
    img = nb.Nifti1Image(
        data,
        affine=reference_img.affine,
        header=reference_img.header,
    )
    nb.save(img, out_path)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
