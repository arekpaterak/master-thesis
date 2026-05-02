from collections.abc import Callable


class Trainer[SummaryT]:
    def __init__(
        self,
        *,
        max_epochs: int,
        train_epoch: Callable[[int], SummaryT],
        on_epoch_end: Callable[[SummaryT], None] | None = None,
    ) -> None:
        self.max_epochs = max_epochs
        self.train_epoch = train_epoch
        self.on_epoch_end = on_epoch_end

    def train(self) -> list[SummaryT]:
        history = []
        for epoch in range(1, self.max_epochs + 1):
            summary = self.train_epoch(epoch)
            history.append(summary)
            if self.on_epoch_end is not None:
                self.on_epoch_end(summary)
        return history
