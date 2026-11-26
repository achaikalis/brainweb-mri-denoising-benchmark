"""Controlled degradations. Inputs are never normalised or rescaled."""

import numpy as np
from scipy.ndimage import gaussian_filter


def _rng(rng: np.random.Generator | None, seed: int | None) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")
    return rng if rng is not None else np.random.default_rng(seed)


def add_awgn(
    volume: np.ndarray,
    sigma: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Add Zero-Mean Additive White Gaussian Noise in the Input Intensity Units."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return np.asarray(volume, dtype=float) + _rng(rng, seed).normal(0, sigma, np.shape(volume))


def add_rician_noise(
    magnitude: np.ndarray,
    sigma: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Rician Magnitude Model for Noise-free S >= 0."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    signal = np.asarray(magnitude, dtype=float)
    if np.any(signal < 0):
        raise ValueError("Rician magnitude signal must be non-negative")
    generator = _rng(rng, seed)
    return np.hypot(
        signal + generator.normal(0, sigma, signal.shape), generator.normal(0, sigma, signal.shape)
    )


def add_salt_and_pepper(
    volume: np.ndarray,
    amount: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Add Salt and Pepper Noise."""
    if not 0 <= amount <= 1:
        raise ValueError("amount must be in [0, 1]")
    result, generator = np.asarray(volume, dtype=float).copy(), _rng(rng, seed)
    mask = generator.random(result.shape) < amount
    result[mask] = np.where(
        generator.random(mask.sum()) < 0.5, np.nanmin(result), np.nanmax(result)
    )
    return result


def add_gaussian_blur(volume: np.ndarray, sigma: float | tuple[float, ...]) -> np.ndarray:
    return gaussian_filter(np.asarray(volume, dtype=float), sigma=sigma)


def add_periodic_interference(
    volume: np.ndarray, amplitude: float, frequency: float, axis: int = 0, phase: float = 0.0
) -> np.ndarray:
    coordinates = np.arange(np.shape(volume)[axis], dtype=float)
    shape = [1] * np.ndim(volume)
    shape[axis] = -1
    wave = amplitude * np.sin(2 * np.pi * frequency * coordinates + phase).reshape(shape)
    return np.asarray(volume, dtype=float) + wave
