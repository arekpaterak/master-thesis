from __future__ import annotations

from typing import NamedTuple

import numpy as np


class CVRPSolutionArrays(NamedTuple):
    next: np.ndarray  # (G, N+1) / (N+1,) int64
    prev: np.ndarray  # (G, N+1) / (N+1,) int64


class CVRPInstanceArrays(NamedTuple):
    locations: np.ndarray  # (G, N+1, 2) / (N+1, 2) float64
    demand: np.ndarray  # (G, N+1) / (N+1,) float64
    capacity: np.ndarray  # (G,) float64 (batched); per-element the kernel slices a scalar
