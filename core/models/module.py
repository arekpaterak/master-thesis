from __future__ import annotations

from abc import ABC
from typing import Any, Protocol, Self


class EpochSummary(Protocol):
    @classmethod
    def from_step_summaries(cls, *, epoch: int, batch_summaries: list, **kwargs: Any) -> Self: ...


class Module(ABC):
    epoch_train_summary_cls: type[EpochSummary]
    epoch_validation_summary_cls: type[EpochSummary]

    def training_step(self, *, optimizer: Any, instance: Any, **kwargs: Any) -> Any:
        pass

    def validation_step(self, instance: Any, epoch: int) -> Any:
        pass
