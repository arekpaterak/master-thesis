from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, Self

import torch


class EpochSummaryProtocol(Protocol):
    @classmethod
    def from_step_summaries(cls, *, epoch: int, batch_summaries: list, **kwargs) -> Self: ...


class Trainer[
    ModelT,
    ProblemInstanceT,
    BatchSummaryT,
    EpochTrainSummaryT: EpochSummaryProtocol,
    EpochValidationSummaryT: EpochSummaryProtocol,
]:
    def __init__(
        self,
        *,
        max_epochs: int,
        batch_size: int,
        training_step: Callable[..., BatchSummaryT],
        epoch_train_summary_cls: type[EpochTrainSummaryT],
        instances_per_epoch: int | None = None,
        batches_per_epoch: int | None = None,
        on_step_end: Callable[[BatchSummaryT], None] | None = None,
        on_epoch_end: Sequence[Callable[[EpochTrainSummaryT], None]] = (),
        validation_step: Callable[[ModelT, ProblemInstanceT, int], object] | None = None,
        epoch_validation_summary_cls: type[EpochValidationSummaryT] | None = None,
        validation_instances: int | None = None,
        validation_batch_size: int | None = None,
        validate_every_n_epochs: int = 1,
        on_validation_end: Sequence[Callable[[EpochValidationSummaryT], None]] = (),
    ) -> None:
        if (instances_per_epoch is None) == (batches_per_epoch is None):
            raise ValueError(
                "Exactly one of instances_per_epoch or batches_per_epoch must be provided."
            )
        if validation_step is not None:
            if epoch_validation_summary_cls is None:
                raise ValueError("val_epoch_summary_class is required when validation_step is set.")
            if validation_instances is None or validation_batch_size is None:
                raise ValueError(
                    "validation_instances and validation_batch_size are required when validation_step is set."
                )
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.instances_per_epoch = instances_per_epoch
        self.batches_per_epoch = batches_per_epoch
        self.train_step = training_step
        self.epoch_train_summary_cls = epoch_train_summary_cls
        self.on_step_end = on_step_end
        self.on_epoch_train_end = on_epoch_end
        self.validation_step = validation_step
        self.epoch_validation_summary_cls = epoch_validation_summary_cls
        self.validation_instances = validation_instances
        self.validation_batch_size = validation_batch_size
        self.validate_every_n_epochs = validate_every_n_epochs
        self.on_epoch_val_end = on_validation_end

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
        model: ModelT,
        optimizer: torch.optim.Optimizer,
        instance_sampler: Callable[[int], ProblemInstanceT],
        validation_instance_sampler: Callable[[int], ProblemInstanceT] | None = None,
        **kwargs,
    ) -> list[EpochTrainSummaryT]:
        if self.validation_step is not None and validation_instance_sampler is None:
            raise ValueError("val_instance_sampler is required when validation_step is set.")

        history = []
        for epoch in range(1, self.max_epochs + 1):
            batch_summaries = []
            for size in self._epoch_batch_sizes():
                instance = instance_sampler(size)
                batch_summary = self.train_step(
                    model=model,
                    optimizer=optimizer,
                    instance=instance,
                    **kwargs,
                )
                batch_summaries.append(batch_summary)
                if self.on_step_end:
                    self.on_step_end(batch_summary)

            epoch_train_summary = self.epoch_train_summary_cls.from_step_summaries(
                epoch=epoch,
                batch_summaries=batch_summaries,
                model=model,
                optimizer=optimizer,
                **kwargs,
            )
            history.append(epoch_train_summary)
            for callback in self.on_epoch_train_end:
                callback(epoch_train_summary)

            if self.validation_step and epoch % self.validate_every_n_epochs == 0:
                batch_validation_summaries = []
                for size in self._validation_batch_sizes():
                    validation_instance = validation_instance_sampler(size)
                    batch_validation_summary = self.validation_step(
                        model, validation_instance, epoch
                    )
                    batch_validation_summaries.append(batch_validation_summary)
                epoch_validation_summary = self.epoch_validation_summary_cls.from_step_summaries(  # type: ignore[union-attr]
                    epoch=epoch,
                    batch_summaries=batch_validation_summaries,
                )
                for callback in self.on_epoch_val_end:
                    callback(epoch_validation_summary)

        return history
