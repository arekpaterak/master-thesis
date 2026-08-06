from pathlib import Path

import pytest

from core.models.utils import checkpoint_epoch, find_checkpoint, list_checkpoints


def _make_checkpoints(run_dir: Path, epochs: list[int]) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    for epoch in epochs:
        (checkpoints_dir / f"epoch_{epoch:04d}.pt").touch()


class TestFindCheckpoint:
    def test_returns_latest_by_default(self, tmp_path: Path):
        _make_checkpoints(tmp_path, [0, 1, 24])
        assert find_checkpoint(tmp_path).name == "epoch_0024.pt"

    def test_returns_pinned_epoch(self, tmp_path: Path):
        _make_checkpoints(tmp_path, [0, 24, 99])
        assert find_checkpoint(tmp_path, epoch=24).name == "epoch_0024.pt"

    def test_raises_for_missing_epoch(self, tmp_path: Path):
        _make_checkpoints(tmp_path, [0, 24])
        with pytest.raises(ValueError, match="epoch 50"):
            find_checkpoint(tmp_path, epoch=50)

    def test_raises_for_empty_run_dir(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No checkpoint"):
            find_checkpoint(tmp_path)


class TestListCheckpoints:
    def test_ascending_by_epoch(self, tmp_path: Path):
        _make_checkpoints(tmp_path, [24, 0, 1])
        assert [checkpoint_epoch(p) for p in list_checkpoints(tmp_path)] == [0, 1, 24]
