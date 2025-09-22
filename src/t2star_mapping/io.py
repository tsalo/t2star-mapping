import os
from typing import Tuple

import nibabel as nb
import numpy as np


def load_nifti(path: str) -> Tuple[np.ndarray, nb.Nifti1Image]:
    """Load a NIfTI image.

    Parameters
    ----------
    path : str
        Path to the NIfTI file (.nii or .nii.gz).

    Returns
    -------
    data : numpy.ndarray
        Image data array loaded lazily from the NIfTI file.
    img : nibabel.Nifti1Image
        Loaded nibabel image object containing affine and header.
    """
    img = nb.load(path)
    data = np.asanyarray(img.dataobj)
    return data, img


def save_nifti_like(
    reference_img: nb.Nifti1Image,
    data: np.ndarray,
    out_path: str,
    dtype: np.dtype | None = None,
) -> None:
    """Save an array as a NIfTI image using a reference's geometry.

    Parameters
    ----------
    reference_img : nibabel.Nifti1Image
        Reference image whose affine and header will be reused.
    data : numpy.ndarray
        Array data to write.
    out_path : str
        Output file path (.nii or .nii.gz).
    dtype : numpy.dtype or None, optional
        If provided, cast ``data`` to this dtype before saving.
    """
    if dtype is not None:
        data = data.astype(dtype)
    img = nb.Nifti1Image(
        data,
        affine=reference_img.affine,
        header=reference_img.header,
    )
    nb.save(img, out_path)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)
