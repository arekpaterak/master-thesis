from __future__ import annotations

from core.models.trainer import Trainer


def test_trainer_runs_epochs_and_callbacks_in_order():
    seen_step_summaries: list[str] = []
    seen_epoch_summaries: list[str] = []

    def instance_sampler(batch_size: int) -> str:
        return f"instance-{batch_size}"

    def train_step(*, model: object, optimizer: object, instance: object, **_) -> str:
        assert model == "model"
        assert optimizer == "optimizer"
        return f"batch:{instance}"

    class EpochSummary:
        def __init__(self, value: str) -> None:
            self.value = value

        def __eq__(self, other: object) -> bool:
            return isinstance(other, EpochSummary) and self.value == other.value

        @classmethod
        def from_step_summaries(
            cls, *, epoch: int, batch_summaries: list[str], model: object, optimizer: object, **_
        ) -> EpochSummary:
            assert batch_summaries == ["batch:instance-4", "batch:instance-4"]
            return cls(f"epoch-{epoch}")

    history = Trainer(
        max_epochs=2,
        batches_per_epoch=2,
        batch_size=4,
        training_step=train_step,
        epoch_train_summary_cls=EpochSummary,
        on_step_end=seen_step_summaries.append,
        on_epoch_end=[seen_epoch_summaries.append],
    ).fit(model="model", optimizer="optimizer", instance_sampler=instance_sampler)

    assert [s.value for s in history] == ["epoch-1", "epoch-2"]
    assert seen_epoch_summaries == history
    assert seen_step_summaries == [
        "batch:instance-4",
        "batch:instance-4",
        "batch:instance-4",
        "batch:instance-4",
    ]
