"""Numba helpers shared by the per-problem sequential improvement step kernels."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(inline="always")
def seed_for(base_seed: int, index: int, count: int) -> int:
    """Distinct 32-bit RNG seed for one (base_seed, index) pair.

    ``count`` is the number of indices (the batch size) so distinct ``base_seed``
    values never overlap. Masked to 32 bits because ``np.random.seed`` takes a
    32-bit seed.
    """
    return (base_seed * count + index) & 0xFFFFFFFF


@njit(inline="always")
def dist(locations: np.ndarray, i: int, j: int) -> float:
    dx = locations[i, 0] - locations[j, 0]
    dy = locations[i, 1] - locations[j, 1]
    return np.sqrt(dx * dx + dy * dy)
