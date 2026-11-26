"""Define MSE, PSNR, SSIM and Runtime Metrics."""

import time
from collections.abc import Callable

import numpy as np
from skimage.metrics import structural_similarity


def _arrays(reference, test, mask=None):
    r, t = np.asarray(reference, float), np.asarray(test, float)
    if r.shape != t.shape:
        raise ValueError("reference and test must have matching shapes")
    if mask is not None:
        if np.shape(mask) != r.shape:
            raise ValueError("mask shape must match")
        r, t = r[mask.astype(bool)], t[mask.astype(bool)]
    return r, t


def mse(reference, test, mask=None):
    r, t = _arrays(reference, test, mask)
    return float(np.mean((r - t) ** 2))


def psnr(reference, test, *, data_range: float, mask=None):
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    value = mse(reference, test, mask)
    return float(np.inf if value == 0 else 10 * np.log10(data_range**2 / value))


def ssim(reference, test, *, data_range: float, win_size: int | None = None):
    """2D or Volumetric 3D SSIM."""
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    return float(
        structural_similarity(
            np.asarray(reference), np.asarray(test), data_range=data_range, win_size=win_size
        )
    )


def measure_runtime(function: Callable, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - start
