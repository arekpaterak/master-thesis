from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import torch
from tensordict.tensorclass import TensorClass

from core.problems.instance import BaseProblemInstance


class BaseSolution(TensorClass, ABC):
    @property
    def device(self) -> torch.device:
        return self.instance.locations.device

    @property
    @abstractmethod
    def instance(self) -> BaseProblemInstance: ...

    @abstractmethod
    def where(self, mask: torch.Tensor, other: Self) -> Self:
        """Return ``self`` where ``mask`` is True, else ``other`` (elementwise over the batch)."""
        ...

    @property
    @abstractmethod
    def is_complete(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def is_feasible(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def num_unassigned(self) -> torch.Tensor: ...

    @abstractmethod
    def remove_nodes(self, nodes: torch.Tensor, *, stitch: bool = False) -> Self:
        """``nodes`` is [*B, M] for a single set or [*B, R, M] for R parallel sets."""
        ...

    @abstractmethod
    def objective(self) -> torch.Tensor: ...

    @abstractmethod
    def node_degree(self) -> torch.Tensor: ...

    @abstractmethod
    def visualize(self, **kwargs) -> None: ...
