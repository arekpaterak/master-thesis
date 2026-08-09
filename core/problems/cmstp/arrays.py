from __future__ import annotations

from typing import NamedTuple

import numpy as np


class CMSTPSolutionArrays(NamedTuple):
    parent: np.ndarray  # (G, N+1) / (N+1,) int64


class CMSTPInstanceArrays(NamedTuple):
    locations: np.ndarray  # (G, N+1, 2) / (N+1, 2) float64
    demand: np.ndarray  # (G, N+1) / (N+1,) float64
    capacity: np.ndarray  # (G,) float64 (batched); per-element the kernel slices a scalar
