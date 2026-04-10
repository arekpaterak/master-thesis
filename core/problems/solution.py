from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tensordict.tensorclass import TensorClass

from core.problems.instance import BaseProblemInstance


class BaseSolution(TensorClass, ABC):
    @property
    @abstractmethod
    def instance(self) -> BaseProblemInstance: ...

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
    def objective(self) -> torch.Tensor: ...

    @abstractmethod
    def visualize(self, **kwargs) -> None: ...
