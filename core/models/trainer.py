from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
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
        on_epoch_train_start: Callable[[int], None] | None = None,
        on_batch_train_end: Callable[[BatchSummaryT], None] | None = None,
        on_epoch_train_end: Sequence[Callable] = (),
        initial_epoch: int = 0,
        validation_instances: int | None = None,
        validation_batch_size: int | None = None,
        validate_every_n_epochs: int = 10,
        on_epoch_validation_start: Callable[[int, int], None] | None = None,
        on_batch_validation_end: Callable | None = None,
        on_epoch_validation_end: Sequence[Callable] = (),
        epoch_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
        batch_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
        validation_batch_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
        strict_number_of_instances: bool = False,
        validate_before_training: bool = False,
    ) -> None:
        """Initialise the trainer.

        Args:
            strict_number_of_instances: If True, training and validation process
                exactly ``training_instances`` / ``validation_instances`` per
                epoch (the final batch may be smaller than ``batch_size``). If
                False (default), the trainer rounds the batch count up so every
                batch is the same size, which means up to ``batch_size - 1``
                extra instances may be processed each epoch. Stable batch sizes
                are required for ``torch.compile(mode="reduce-overhead")`` and
                avoid noisy final-batch metrics.
        """
        if (training_instances is None) == (training_batches_per_epoch is None):
            raise ValueError(
                "Exactly one of training_instances or training_batches_per_epoch must be provided."
            )
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.instances_per_epoch = training_instances
        self.batches_per_epoch = training_batches_per_epoch
        self.on_epoch_train_start = on_epoch_train_start
        self.on_batch_train_end = on_batch_train_end
        self.on_epoch_train_end = on_epoch_train_end
        self.initial_epoch = initial_epoch
        self.validation_instances = validation_instances
        self.validation_batch_size = validation_batch_size
        self.validate_every_n_epochs = validate_every_n_epochs
        self.on_epoch_validation_start = on_epoch_validation_start
        self.on_batch_validation_end = on_batch_validation_end
        self.on_epoch_validation_end = on_epoch_validation_end
        self.epoch_progress = epoch_progress
        self.batch_progress = batch_progress
        self.validation_batch_progress = validation_batch_progress
        self.strict_number_of_instances = strict_number_of_instances
        self.validate_before_training = validate_before_training

    def _train_batch_sizes(self) -> list[int]:
        if self.instances_per_epoch is not None:
            return _split_into_batches(
                total=self.instances_per_epoch,
                batch_size=self.batch_size,
                strict=self.strict_number_of_instances,
            )
        if self.batches_per_epoch is not None:
            return [self.batch_size] * self.batches_per_epoch
        return []

    def _validation_batch_sizes(self) -> list[int]:
        if self.validation_instances is None:
            raise ValueError("Expected validation_instances to be provided.")
        if self.validation_batch_size is None:
            raise ValueError("Expected validation_batch_size to be provided.")
        return _split_into_batches(
            total=self.validation_instances,
            batch_size=self.validation_batch_size,
            strict=self.strict_number_of_instances,
        )

    def fit(
        self,
        *,
        module: Module,
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

        if with_validation and self.validate_before_training and self.initial_epoch == 0:
            self._run_validation(module, validation_instance_sampler, epoch=self.initial_epoch - 1)

        for epoch in self.epoch_progress(range(self.initial_epoch, self.max_epochs)):
            train_batch_sizes = self._train_batch_sizes()
            if train_batch_sizes:
                if self.on_epoch_train_start:
                    self.on_epoch_train_start(epoch)
                train_accumulator = module.train_accumulator_cls()
                for size in self.batch_progress(train_batch_sizes):
                    instance = instance_sampler(size)
                    batch_summary = module.training_step(
                        optimizer=optimizer,
                        instance=instance,
                        **kwargs,
                    )
                    train_accumulator.update(batch_summary)
                    if self.on_batch_train_end:
                        self.on_batch_train_end(batch_summary)

                epoch_train_summary = train_accumulator.complete(epoch=epoch)

                module.on_epoch_train_end(epoch_train_summary)
                for callback in self.on_epoch_train_end:
                    callback(epoch_train_summary)

            if with_validation and epoch % self.validate_every_n_epochs == 0:
                self._run_validation(module, validation_instance_sampler, epoch=epoch)

    def _run_validation(
        self,
        module: Module,
        validation_instance_sampler: Sampler,
        *,
        epoch: int,
    ) -> None:
        val_batch_sizes = self._validation_batch_sizes()
        validation_instance_sampler.reset()
        if self.on_epoch_validation_start:
            self.on_epoch_validation_start(epoch, len(val_batch_sizes))
        val_accumulator = module.validation_accumulator_cls()
        for size in self.validation_batch_progress(val_batch_sizes):
            validation_instance = validation_instance_sampler(size)
            batch_validation_summary = module.validation_step(validation_instance, epoch)
            val_accumulator.update(batch_validation_summary)
            if self.on_batch_validation_end:
                self.on_batch_validation_end(batch_validation_summary)
        epoch_validation_summary = val_accumulator.complete(epoch=epoch)

        module.on_epoch_validation_end(epoch_validation_summary)
        for callback in self.on_epoch_validation_end:
            callback(epoch_validation_summary)


def _split_into_batches(*, total: int, batch_size: int, strict: bool) -> list[int]:
    """Split ``total`` instances into batches of size ``batch_size``.

    With ``strict=True``, processes exactly ``total`` instances; the final batch
    may be smaller. With ``strict=False`` (default), rounds up so every batch
    has size ``batch_size`` and up to ``batch_size - 1`` extra instances are
    processed.
    """
    if strict:
        full, remainder = divmod(total, batch_size)
        return [batch_size] * full + ([remainder] if remainder else [])
    n_batches = math.ceil(total / batch_size)
    return [batch_size] * n_batches
