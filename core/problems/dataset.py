from __future__ import annotations

import torch

from core.problems.instance import BaseProblemInstance
from core.problems.sampler import Sampler


class DatasetSampler(Sampler):
    """Sampler that serves fixed instances from a preloaded dataset.

    Returns contiguous, non-shuffled slices of the dataset, advancing an internal
    cursor on every call and wrapping back to the start once exhausted. ``reset``
    rewinds the cursor so each pass over the data (e.g. one validation pass per
    epoch) evaluates the same fixed instances, which is what we want for
    reproducible validation and test evaluation.
    """

    def __init__(self, instance: BaseProblemInstance) -> None:
        if not instance.batch_size:
            raise ValueError("DatasetSampler requires a batched instance.")
        self._instance = instance
        self._length = instance.batch_size[0]
        self._cursor = 0

    def __len__(self) -> int:
        return self._length

    @property
    def num_nodes(self) -> int:
        return self._instance.num_nodes

    @property
    def problem_size(self) -> int:
        return self._instance.problem_size

    def reset(self) -> None:
        self._cursor = 0

    def __call__(self, batch_size: int) -> BaseProblemInstance:
        if batch_size > self._length:
            raise ValueError(
                f"Requested batch_size={batch_size} exceeds dataset size {self._length}."
            )
        start = self._cursor
        end = start + batch_size
        if end <= self._length:
            batch = self._instance[start:end]
        else:
            # Wrap around to the start so repeated passes stay aligned to the data.
            wrap = end - self._length
            batch = torch.cat([self._instance[start:], self._instance[:wrap]], dim=0)
        self._cursor = end % self._length
        return batch
