from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

from core.problems.cvrp.dataset import load_cvrp_dataset
from core.problems.instance import BaseProblemInstance
from core.problems.sampler import Sampler

DatasetLoader = Callable[[str | Path], BaseProblemInstance]

# Maps a problem name to a loader that reads a benchmark file into a single
# batched instance. New problems register their loader here.
_DATASET_LOADERS: dict[str, DatasetLoader] = {
    "cvrp": load_cvrp_dataset,
}


def load_dataset(problem_name: str, path: str | Path) -> BaseProblemInstance:
    """Load a benchmark dataset file into a single batched instance."""
    try:
        loader = _DATASET_LOADERS[problem_name]
    except KeyError:
        raise ValueError(
            f"No dataset loader registered for problem {problem_name!r}. "
            f"Available: {sorted(_DATASET_LOADERS)}."
        ) from None
    return loader(path)


class DatasetSampler(Sampler):
    """Sampler that serves fixed instances from a preloaded dataset.

    Returns contiguous, non-shuffled slices of the dataset, advancing an internal
    cursor on every call and wrapping back to the start once exhausted. ``reset``
    rewinds the cursor so each pass over the data (e.g. one validation pass per
    epoch) evaluates the same fixed instances, which is what we want for
    reproducible validation and test evaluation.

    Not registered with the ``Sampler`` registry: it is problem-agnostic and built
    directly from a loaded instance rather than from a problem name.
    """

    def __init__(self, instance: BaseProblemInstance) -> None:
        if not instance.batch_size:
            raise ValueError("DatasetSampler requires a batched instance.")
        self._instance = instance
        self._length = instance.batch_size[0]
        self._cursor = 0

    @classmethod
    def from_file(cls, problem_name: str, path: str | Path) -> DatasetSampler:
        return cls(load_dataset(problem_name, path))

    def __len__(self) -> int:
        return self._length

    @property
    def num_nodes(self) -> int:
        return self._instance.num_nodes

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
