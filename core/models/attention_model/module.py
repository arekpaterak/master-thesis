from __future__ import annotations

from collections.abc import Callable, Iterator

import torch.nn as nn
import torch.optim as optim

from core.models.attention_model.model import AttentionModel
from core.models.attention_model.statistical_test import (
    OneSidedPairedTTestResult,
    one_sided_paired_ttest,
)
from core.models.attention_model.training import (
    BatchSummary,
    EpochSummary,
    EpochTrainAccumulator,
    training_step,
)
from core.models.attention_model.training_helpers import prepare_baseline, sync_baseline_
from core.models.module import Module
from core.problems.instance import BaseProblemInstance
from core.problems.sampler import Sampler
from core.random import SeededGenerator


class AMModule(Module):
    train_accumulator_cls = EpochTrainAccumulator

    def __init__(
        self,
        model: AttentionModel,
        instance_sampler: Sampler,
        eval_n_instances: int = 10_000,
        eval_batch_size: int = 512,
        alpha: float = 0.05,
        on_baseline_sync: Callable[[OneSidedPairedTTestResult], None] | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.baseline = prepare_baseline(model)
        self._instance_sampler = instance_sampler
        self._eval_n_instances = eval_n_instances
        self._eval_batch_size = eval_batch_size
        self._alpha = alpha
        self._on_baseline_sync = on_baseline_sync
        self._rng = SeededGenerator(seed)

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
            rng=self._rng,
        )

    def on_epoch_train_end(self, _: EpochSummary) -> None:
        device = next(self.model.parameters()).device
        test = one_sided_paired_ttest(
            self.model,
            self.baseline,
            self._instance_sampler,
            n_instances=self._eval_n_instances,
            batch_size=self._eval_batch_size,
            alpha=self._alpha,
            device=device,
        )
        if test.improved:
            sync_baseline_(self.baseline, self.model)
        if self._on_baseline_sync is not None:
            self._on_baseline_sync(test)
