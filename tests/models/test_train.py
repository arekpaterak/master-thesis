from core.models.train import Trainer


def test_trainer_runs_epochs_and_callbacks_in_order():
    seen_epochs = []
    seen_summaries = []

    def train_epoch(*, epoch: int, model: object, optimizer: object) -> str:
        assert model == "model"
        assert optimizer == "optimizer"
        seen_epochs.append(epoch)
        return f"epoch-{epoch}"

    history = Trainer(
        max_epochs=3,
        train_epoch=train_epoch,
        on_epoch_end=seen_summaries.append,
    ).train(model="model", optimizer="optimizer")

    assert seen_epochs == [1, 2, 3]
    assert history == ["epoch-1", "epoch-2", "epoch-3"]
    assert seen_summaries == history
