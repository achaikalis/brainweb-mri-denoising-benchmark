"""Convert MINC MRI Volumes to the NIfTI File Format"""

import gzip
import json
import shutil
import subprocess
from pathlib import Path


def mnc2nii_path() -> str:
    """Find the mnc2ii dependency or raise an actionable error."""
    executable = shutil.which("mnc2nii")
    if not executable:
        raise FileNotFoundError(
            "mnc2nii is not on PATH. Activate the minc-convert conda virtual environment first."
        )
    return executable


def convert_minc_to_nifti(
    input_minc: str | Path,
    output_nifti: str | Path,
    *,
    overwrite: bool = False,
    gzip_output: bool = True,
    report_path: str | Path | None = None,
) -> Path:
    """Convert MINC with ``mnc2nii input.mnc output.nii`` without modifying the original source."""
    source, target = Path(input_minc), Path(output_nifti)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".mnc":
        raise ValueError(f"Expected a .mnc file, got {source}")
    if target.name.endswith(".nii.gz"):
        uncompressed = target.with_suffix("")
    elif target.suffix == ".nii":
        uncompressed = target
    else:
        raise ValueError("Output must end in .nii or .nii.gz")
    final = target if gzip_output else uncompressed
    if (final.exists() or uncompressed.exists()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {final}")
    executable = mnc2nii_path()
    result = subprocess.run(
        [executable, str(source), str(uncompressed)], capture_output=True, text=True, check=False
    )
    report = {
        "input_filename": str(source),
        "output_filename": str(final),
        "status": "failed",
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
    if result.returncode != 0:
        if report_path:
            Path(report_path).write_text(json.dumps(report, indent=2))
        raise RuntimeError(f"mnc2nii failed ({result.returncode}): {result.stderr.strip()}")
    if gzip_output:
        with uncompressed.open("rb") as src, gzip.open(final, "wb") as dst:
            shutil.copyfileobj(src, dst)
        uncompressed.unlink()
    report["status"] = "success"
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2))
    return final
