from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from core.models.module import Module
from core.problems.sampler import Sampler


class Trainer[BatchSummaryT]:
    def __init__(
        self,
        *,
        max_epochs: int,
        batch_size: int,
        training_instances: int | None = None,
        training_batches_per_epoch: int | None = None,
        on_batch_train_end: Callable[[BatchSummaryT], None] | None = None,
        on_epoch_train_end: Sequence[Callable] = (),
        validation_instances: int | None = None,
        validation_batch_size: int | None = None,
        validate_every_n_epochs: int = 10,
        on_epoch_validation_end: Sequence[Callable] = (),
    ) -> None:
        if (training_instances is None) == (training_batches_per_epoch is None):
            raise ValueError(
                "Exactly one of training_instances or training_batches_per_epoch must be provided."
            )
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.instances_per_epoch = training_instances
        self.batches_per_epoch = training_batches_per_epoch
        self.on_batch_train_end = on_batch_train_end
        self.on_epoch_train_end = on_epoch_train_end
        self.validation_instances = validation_instances
        self.validation_batch_size = validation_batch_size
        self.validate_every_n_epochs = validate_every_n_epochs
        self.on_epoch_validation_end = on_epoch_validation_end

    def _epoch_batch_sizes(self) -> list[int]:
        if self.instances_per_epoch is not None:
            full, remainder = divmod(self.instances_per_epoch, self.batch_size)
            return [self.batch_size] * full + ([remainder] if remainder else [])
        return [self.batch_size] * self.batches_per_epoch  # type: ignore[operator]

    def _validation_batch_sizes(self) -> list[int]:
        full, remainder = divmod(self.validation_instances, self.validation_batch_size)  # type: ignore[arg-type]
        return [self.validation_batch_size] * full + ([remainder] if remainder else [])

    def fit(
        self,
        *,
        model: Module,
        optimizer: torch.optim.Optimizer,
        instance_sampler: Sampler,
        validation_instance_sampler: Sampler | None = None,
        **kwargs: Any,
    ) -> None:
        with_validation = validation_instance_sampler is not None
        if with_validation and (
            self.validation_instances is None or self.validation_batch_size is None
        ):
            raise ValueError(
                "validation_instances and validation_batch_size are required when validation_instance_sampler is provided."
            )

        for epoch in range(self.max_epochs):
            batch_summaries = []
            for size in self._epoch_batch_sizes():
                instance = instance_sampler(size)
                batch_summary = model.training_step(
                    optimizer=optimizer,
                    instance=instance,
                    **kwargs,
                )
                batch_summaries.append(batch_summary)
                if self.on_batch_train_end:
                    self.on_batch_train_end(batch_summary)

            epoch_train_summary = model.epoch_train_summary_cls.from_step_summaries(
                epoch=epoch,
                batch_summaries=batch_summaries,
                model=model,
                optimizer=optimizer,
                **kwargs,
            )
            for callback in self.on_epoch_train_end:
                callback(epoch_train_summary)

            if with_validation and epoch % self.validate_every_n_epochs == 0:
                batch_validation_summaries = []
                for size in self._validation_batch_sizes():
                    validation_instance = validation_instance_sampler(size)  # type: ignore[misc]
                    batch_validation_summary = model.validation_step(validation_instance, epoch)
                    batch_validation_summaries.append(batch_validation_summary)
                epoch_validation_summary = model.epoch_validation_summary_cls.from_step_summaries(
                    epoch=epoch,
                    batch_summaries=batch_validation_summaries,
                )
                for callback in self.on_epoch_validation_end:
                    callback(epoch_validation_summary)
