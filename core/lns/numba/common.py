from __future__ import annotations

import numpy as np
from numba import njit

ACCEPT_GREEDY = 0
ACCEPT_SA = 1
ACCEPT_UNCONDITIONAL = 2


@njit(inline="always")
def accept_move(
    candidate_objective: float,
    current_objective: float,
    accept_kind: int,
    temperature: float,
    eps: float,
) -> bool:
    if accept_kind == ACCEPT_GREEDY:
        return candidate_objective < current_objective
    if accept_kind == ACCEPT_UNCONDITIONAL:
        return True
    delta = candidate_objective - current_objective
    if delta < eps:
        return True
    return np.random.random() < np.exp(-max(delta, 0.0) / temperature)


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
