from __future__ import annotations

from abc import ABC
from typing import Any, Protocol


class EpochAccumulator(Protocol):
    def update(self, batch_summary: Any) -> None: ...
    def finalize(self, *, epoch: int) -> Any: ...


class Module(ABC):
    train_accumulator_cls: type[EpochAccumulator]
    validation_accumulator_cls: type[EpochAccumulator]

    def training_step(self, *, optimizer: Any, instance: Any, **kwargs: Any) -> Any:
        pass

    def validation_step(self, instance: Any, epoch: int) -> Any:
        pass
