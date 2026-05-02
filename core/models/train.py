from collections.abc import Callable

import torch


class Trainer[ModelT, EpochSummaryT]:
    def __init__(
        self,
        *,
        max_epochs: int,
        train_epoch: Callable[..., EpochSummaryT],
        on_epoch_end: Callable[[EpochSummaryT], None] | None = None,
    ) -> None:
        self.max_epochs = max_epochs
        self.train_epoch = train_epoch
        self.on_epoch_end = on_epoch_end

    def train(
        self,
        *,
        model: ModelT,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ) -> list[EpochSummaryT]:
        history = []
        for epoch in range(1, self.max_epochs + 1):
            epoch_summary = self.train_epoch(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                **kwargs,
            )
            history.append(epoch_summary)
            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch_summary)
        return history
