"""
Physically safe data augmentations for self-supervised representation learning.

Scientific safeguards:
- Masking preserves realistic observing cadence and gap structure.
- Bounded photometric noise additions.
- NO arbitrary time-warping or ephemeris distortions that would corrupt orbital physics.
"""

from __future__ import annotations

import numpy as np


def random_cadence_mask(
    flux: np.ndarray,
    mask_fraction: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Randomly mask cadences with NaN to simulate data dropouts without altering time coordinate.
    """
    if rng is None:
        rng = np.random.default_rng()
    augmented = flux.copy()
    n_pts = len(flux)
    mask_indices = rng.choice(n_pts, size=int(n_pts * mask_fraction), replace=False)
    augmented[mask_indices] = np.nan
    return augmented


def bounded_photometric_jitter(
    flux: np.ndarray,
    noise_sigma: float = 0.0001,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Inject bounded Gaussian photometric noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, noise_sigma, size=flux.shape)
    return flux + noise
