"""Defines I/O Operations for NIfTI Input/Output."""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def _validate_nifti_path(path: str | Path) -> Path:
    path = Path(path)
    if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        raise ValueError(f"Unsupported image type: {path}. Expected .nii or .nii.gz.")
    return path


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    """Load a NIfTI image."""
    path = _validate_nifti_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def volume_metadata(image: nib.Nifti1Image) -> dict[str, Any]:
    """Return NIfTI Metadata."""
    data = np.asanyarray(image.dataobj)
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    return {
        "dimensions": tuple(int(v) for v in image.shape),
        "voxel_spacing_mm": tuple(float(v) for v in image.header.get_zooms()[:3]),
        "data_type": str(image.get_data_dtype()),
        "intensity_range": (float(np.nanmin(data)), float(np.nanmax(data))),
        "affine": image.affine.copy(),
        "qform": qform,
        "qform_code": int(qcode),
        "sform": sform,
        "sform_code": int(scode),
        "orientation": tuple(nib.aff2axcodes(image.affine)),
    }


def save_processed_volume(
    data: np.ndarray, source: nib.Nifti1Image, output: str | Path, *, overwrite: bool = False
) -> Path:
    """Write Processed MRI Volume."""
    output = _validate_nifti_path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")
    if data.shape != source.shape:
        raise ValueError(f"Data shape {data.shape} does not match source shape {source.shape}")
    output.parent.mkdir(parents=True, exist_ok=True)
    header = source.header.copy()
    header.set_data_dtype(data.dtype)
    result = nib.Nifti1Image(data, source.affine, header=header)
    result.set_qform(*source.get_qform(coded=True))
    result.set_sform(*source.get_sform(coded=True))
    nib.save(result, str(output))
    return output
