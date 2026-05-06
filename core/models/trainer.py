from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


class Trainer[ModelT, InstanceT, BatchSummaryT, EpochSummaryT]:
    def __init__(
        self,
        *,
        max_epochs: int,
        batch_size: int,
        training_step: Callable[..., BatchSummaryT],
        aggregate_epoch: Callable[..., EpochSummaryT],
        instances_per_epoch: int | None = None,
        batches_per_epoch: int | None = None,
        on_step_end: Callable[[BatchSummaryT], None] | None = None,
        on_epoch_end: Sequence[Callable[[EpochSummaryT], None]] = (),
    ) -> None:
        if (instances_per_epoch is None) == (batches_per_epoch is None):
            raise ValueError(
                "Exactly one of instances_per_epoch or batches_per_epoch must be provided."
            )
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.instances_per_epoch = instances_per_epoch
        self.batches_per_epoch = batches_per_epoch
        self.train_step = training_step
        self.aggregate_epoch = aggregate_epoch
        self.on_step_end = on_step_end
        self.on_epoch_end = on_epoch_end

    def _epoch_batch_sizes(self) -> list[int]:
        if self.instances_per_epoch is not None:
            full, remainder = divmod(self.instances_per_epoch, self.batch_size)
            return [self.batch_size] * full + ([remainder] if remainder else [])
        return [self.batch_size] * self.batches_per_epoch  # type: ignore[operator]

    def fit(
        self,
        *,
        model: ModelT,
        optimizer: torch.optim.Optimizer,
        instance_sampler: Callable[[int], InstanceT],
        **kwargs,
    ) -> list[EpochSummaryT]:
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
                if self.on_step_end is not None:
                    self.on_step_end(batch_summary)

            epoch_summary = self.aggregate_epoch(
                epoch=epoch,
                batch_summaries=batch_summaries,
                model=model,
                optimizer=optimizer,
                **kwargs,
            )
            history.append(epoch_summary)
            for callback in self.on_epoch_end:
                callback(epoch_summary)
        return history
