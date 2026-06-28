from __future__ import annotations

import pytest

from core.models.module import Module
from core.models.trainer import Trainer, _split_into_batches


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

    class EpochAccumulator:
        def __init__(self) -> None:
            self._batch_summaries: list[str] = []

        def update(self, batch_summary: str) -> None:
            self._batch_summaries.append(batch_summary)

        def complete(self, *, epoch: int) -> EpochSummary:
            assert self._batch_summaries == ["batch:instance-4", "batch:instance-4"]
            return EpochSummary(f"epoch-{epoch}")

    class FakeModule(Module):
        train_accumulator_cls = EpochAccumulator

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
    ).fit(module=fake_module, optimizer="optimizer", instance_sampler=instance_sampler)

    assert [s.value for s in seen_epoch_summaries] == ["epoch-0", "epoch-1"]
    assert seen_step_summaries == [
        "batch:instance-4",
        "batch:instance-4",
        "batch:instance-4",
        "batch:instance-4",
    ]


@pytest.mark.parametrize(
    "total, batch_size, expected",
    [
        (100, 10, [10] * 10),  # divides evenly
        (100, 32, [32, 32, 32, 32]),  # rounds up: 128 total instances
        (1500, 64, [64] * 24),  # ceil(1500/64) = 24 batches of 64 (1536 instances)
        (5, 10, [10]),  # smaller than batch size still gets a full batch
    ],
)
def test_split_into_batches_non_strict_rounds_up(total, batch_size, expected):
    assert _split_into_batches(total=total, batch_size=batch_size, strict=False) == expected


@pytest.mark.parametrize(
    "total, batch_size, expected",
    [
        (100, 10, [10] * 10),  # divides evenly: no remainder
        (100, 32, [32, 32, 32, 4]),  # 3 full + remainder
        (1500, 64, [64] * 23 + [28]),  # 23 full + remainder
        (5, 10, [5]),  # only a partial batch
    ],
)
def test_split_into_batches_strict_keeps_exact_count(total, batch_size, expected):
    assert _split_into_batches(total=total, batch_size=batch_size, strict=True) == expected


def test_trainer_default_uses_stable_batch_sizes():
    trainer = Trainer(
        max_epochs=1,
        training_instances=1500,
        batch_size=64,
    )
    assert trainer._train_batch_sizes() == [64] * 24


def test_trainer_strict_keeps_exact_instance_count():
    trainer = Trainer(
        max_epochs=1,
        training_instances=1500,
        batch_size=64,
        strict_number_of_instances=True,
    )
    assert trainer._train_batch_sizes() == [64] * 23 + [28]
