from __future__ import annotations

from core.models.module import Module
from core.models.trainer import Trainer


def test_trainer_runs_epochs_and_callbacks_in_order():
    seen_step_summaries: list[str] = []
    seen_epoch_summaries: list[str] = []

    def instance_sampler(batch_size: int) -> str:
        return f"instance-{batch_size}"

    class EpochSummary:
        def __init__(self, value: str) -> None:
            self.value = value

        def __eq__(self, other: object) -> bool:
            return isinstance(other, EpochSummary) and self.value == other.value

        @classmethod
        def from_step_summaries(
            cls, *, epoch: int, batch_summaries: list[str], **_
        ) -> EpochSummary:
            assert batch_summaries == ["batch:instance-4", "batch:instance-4"]
            return cls(f"epoch-{epoch}")

    class FakeModule(Module):
        epoch_train_summary_cls = EpochSummary

        def training_step(self, *, optimizer: object, instance: object, **_) -> str:
            assert optimizer == "optimizer"
            return f"batch:{instance}"

    fake_module = FakeModule()
    Trainer(
        max_epochs=2,
        training_batches_per_epoch=2,
        batch_size=4,
        on_batch_train_end=seen_step_summaries.append,
        on_epoch_train_end=[seen_epoch_summaries.append],
    ).fit(model=fake_module, optimizer="optimizer", instance_sampler=instance_sampler)

    assert [s.value for s in seen_epoch_summaries] == ["epoch-0", "epoch-1"]
    assert seen_step_summaries == [
        "batch:instance-4",
        "batch:instance-4",
        "batch:instance-4",
        "batch:instance-4",
    ]
