from __future__ import annotations

from collections.abc import Iterator

import torch.nn as nn
import torch.optim as optim

from core.models.attention_model.model import AttentionModel
from core.models.attention_model.training import BatchSummary, EpochSummary, training_step
from core.models.attention_model.training_helpers import prepare_baseline
from core.models.module import Module
from core.problems.instance import BaseProblemInstance


class AMModule(Module):
    epoch_train_summary_cls = EpochSummary

    def __init__(self, model: AttentionModel) -> None:
        self.model = model
        self.baseline = prepare_baseline(model)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.model.parameters()

    def training_step(
        self,
        *,
        optimizer: optim.Optimizer,
        instance: BaseProblemInstance,
        **_,
    ) -> BatchSummary:
        return training_step(
            model=self.model,
            baseline=self.baseline,
            optimizer=optimizer,
            instance=instance,
        )
