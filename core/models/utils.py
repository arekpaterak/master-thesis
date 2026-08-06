from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel
from torch import nn as nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_checkpoints(run_dir: Path) -> list[Path]:
    """Epoch checkpoints under ``run_dir/checkpoints``, ascending by epoch."""
    return sorted((run_dir / "checkpoints").glob("epoch_*.pt"))


def checkpoint_epoch(path: Path) -> int:
    """Epoch number encoded in an ``epoch_XXXX.pt`` filename."""
    return int(path.stem.split("_")[1])


def find_checkpoint(run_dir: Path, epoch: int | None = None) -> Path:
    """Resolve a checkpoint: the pinned ``epoch`` if given, else the latest.

    Raises ``ValueError`` if no checkpoints exist, or if the pinned epoch is missing.
    """
    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        raise ValueError(f"No checkpoint found under {(run_dir / 'checkpoints').resolve()}.")
    if epoch is None:
        return checkpoints[-1]
    path = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
    if not path.exists():
        available = ", ".join(str(checkpoint_epoch(p)) for p in checkpoints)
        raise ValueError(
            f"Checkpoint for epoch {epoch} not found under "
            f"{(run_dir / 'checkpoints').resolve()}. Available epochs: {available}."
        )
    return path


def save_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    loss: float | None = None,
    config: BaseModel | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "loss": loss,
        "config": config.model_dump() if config is not None else None,
        **(extra or {}),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    device = next(model.parameters()).device
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])


def get_model_device(
    model: torch.nn.Module,
) -> torch.device:
    return next(model.parameters()).device


def freeze_parameters_(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
