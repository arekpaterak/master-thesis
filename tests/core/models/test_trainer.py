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

    def aggregate_epoch(
        *, epoch: int, batch_summaries: list[str], model: object, optimizer: object, **_
    ) -> str:
        assert batch_summaries == ["batch:instance-4", "batch:instance-4"]
        return f"epoch-{epoch}"

    history = Trainer(
        max_epochs=2,
        batches_per_epoch=2,
        batch_size=4,
        training_step=train_step,
        aggregate_epoch=aggregate_epoch,
        on_step_end=seen_step_summaries.append,
        on_epoch_end=[seen_epoch_summaries.append],
    ).fit(model="model", optimizer="optimizer", instance_sampler=instance_sampler)

    assert history == ["epoch-1", "epoch-2"]
    assert seen_epoch_summaries == history
    assert seen_step_summaries == [
        "batch:instance-4",
        "batch:instance-4",
        "batch:instance-4",
        "batch:instance-4",
    ]
