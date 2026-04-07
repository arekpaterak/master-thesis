from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tensordict.tensorclass import TensorClass

from core.problems.instance import BaseProblemInstance


class BaseSolution(TensorClass, ABC):
    instance: BaseProblemInstance

    @abstractmethod
    def objective(self) -> torch.Tensor: ...
