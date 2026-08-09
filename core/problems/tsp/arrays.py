from __future__ import annotations

from typing import NamedTuple

import numpy as np


class TSPSolutionArrays(NamedTuple):
    next: np.ndarray  # (G, N) / (N,) int64
    prev: np.ndarray  # (G, N) / (N,) int64


class TSPInstanceArrays(NamedTuple):
    locations: np.ndarray  # (G, N, 2) / (N, 2) float64
