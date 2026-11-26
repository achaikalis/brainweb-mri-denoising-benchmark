"""Download the Full BrainWeb Normal Phantom Grid as MINC MRI Volumes.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from brainweb_dl._brainweb import ALLOWED_NOISE_LEVEL, ALLOWED_RF, BASE_URL

logger = logging.getLogger("brainweb_denoising.download")

_EXT = "mnc.gz"

_MINC_MAGIC = b"CDF\x01"


@dataclass(frozen=True)
class BrainWebVolumeID:
    """BrainWeb MRI Volume Identifier."""

    contrast: str  # "T1" | "T2" | "PD"
    res_mm: int = 1
    noise: int = 0  # percent
    field_value: int = 0  # percent RF

    @property
    def alias(self) -> str:
        return f"{self.contrast}+ICBM+normal+{self.res_mm}mm+pn{self.noise}+rf{self.field_value}"

    @property
    def filename(self) -> str:
        return f"{self.contrast}_ICBM_normal_{self.res_mm}mm_pn{self.noise}_rf{self.field_value}.{_EXT}"


def full_grid(
    contrasts: tuple[str, ...] = ("T1", "T2", "PD"),
    noise_levels: tuple[int, ...] = ALLOWED_NOISE_LEVEL,
    field_values: tuple[int, ...] = ALLOWED_RF,
    res_mm: int = 1,
) -> list[BrainWebVolumeID]:
    """Return Every Volume of the BrainWeb Normal Grid to Download."""
    return [
        BrainWebVolumeID(contrast=c, res_mm=res_mm, noise=n, field_value=f)
        for c in contrasts
        for n in noise_levels
        for f in field_values
    ]


def _minc_url(vol: BrainWebVolumeID) -> str:
    """
    Build the McGill CGI URL.
    """
    params = {
        "do_download_alias": vol.alias,
        "format_value": "minc",
        "zip_value": "gnuzip",
        "who_name": "",
        "who_institution": "",
        "who_email": "",
        "download_for_real": "%5BStart+download%21%5D",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items() if v)
    return f"{BASE_URL}?{query}"


def _download_one(
    vol: BrainWebVolumeID,
    dest: Path,
    force: bool = False,
    timeout_s: float = 120.0,
    retries: int = 3,
) -> Path:
    if dest.exists() and not force and _is_valid_minc_gz(dest):
        logger.info("Exists, skipping: %s", dest)
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                _minc_url(vol),
                stream=True,
                headers={"Accept-Encoding": "identity", "Content-Encoding": "gzip"},
                timeout=timeout_s,
            ) as resp:
                resp.raise_for_status()
                disposition = resp.headers.get("Content-disposition", "")
                if "mnc" not in disposition:
                    raise RuntimeError(
                        f"CGI did not offer a MINC download for {vol.alias} "
                        f"(Content-disposition: {disposition!r})"
                    )
                payload = bytearray()
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        payload.extend(chunk)
        except (requests.RequestException, RuntimeError) as err:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, vol.alias, err)
            time.sleep(2 * attempt)
            continue
        tmp.write_bytes(payload)
        if not _is_valid_minc_gz(tmp):
            logger.warning("Attempt %d/%d: invalid payload for %s", attempt, retries, vol.alias)
            tmp.unlink(missing_ok=True)
            time.sleep(2 * attempt)
            continue
        tmp.replace(dest)
        logger.info("Downloaded: %s (%d B)", dest, dest.stat().st_size)
        return dest
    raise RuntimeError(f"Failed to download {vol.alias} after {retries} attempts")


def _is_valid_minc_gz(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as fh:
            return fh.read(4) in (_MINC_MAGIC, b"CDF\x02")
    except (OSError, EOFError, gzip.BadGzipFile):
        return False


def fetch_all_minc(
    out_dir: str | Path,
    force: bool = False,
    manifest_path: str | Path | None = None,
    grid: list[BrainWebVolumeID] | None = None,
) -> list[dict]:
    """Download the full BrainWeb1 MINC grid into ``out_dir``.

    Parameters
    ----------
    out_dir
        Destination directory, e.g. ``data/raw/minc``.
    force
        Re-download existing files even if they validate.
    manifest_path
        Where to write the download manifest (CSV). Defaults to
        ``out_dir / "manifest.csv"``.
    grid
        Explicit volume identifiers; defaults to :func:`full_grid`.

    Returns
    -------
    list[dict]
        One manifest row per volume: id fields, path, size, sha256.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = grid if grid is not None else full_grid()
    logger.info("Downloading %d MINC volumes to %s", len(grid), out_dir)

    rows: list[dict] = []
    for vol in grid:
        path = _download_one(vol, out_dir / vol.filename, force=force)
        data = gzip.decompress(path.read_bytes())
        rows.append(
            {
                "contrast": vol.contrast,
                "res_mm": vol.res_mm,
                "noise_percent": vol.noise,
                "rf_percent": vol.field_value,
                "alias": vol.alias,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "uncompressed_bytes": len(data),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    manifest = Path(manifest_path) if manifest_path else out_dir / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Manifest written: %s", manifest)
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_all_minc("data/raw/minc")
