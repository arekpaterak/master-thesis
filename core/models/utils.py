import os
import shutil
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

    Without a pinned epoch, the stable ``last.pt`` pointer is preferred when present --
    this is the only checkpoint committed for a featured run, so a fresh clone (which has
    no ``epoch_*.pt`` files) still resolves. Raises ``ValueError`` if nothing is found, or
    if the pinned epoch is missing.
    """
    checkpoints_dir = run_dir / "checkpoints"
    if epoch is not None:
        path = checkpoints_dir / f"epoch_{epoch:04d}.pt"
        if not path.exists():
            available = ", ".join(str(checkpoint_epoch(p)) for p in list_checkpoints(run_dir))
            raise ValueError(
                f"Checkpoint for epoch {epoch} not found under "
                f"{checkpoints_dir.resolve()}. Available epochs: {available}."
            )
        return path

    last = checkpoints_dir / "last.pt"
    if last.exists():
        return last
    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        raise ValueError(f"No checkpoint found under {checkpoints_dir.resolve()}.")
    return checkpoints[-1]


def update_last_checkpoint(checkpoint_path: Path) -> None:
    """Point ``checkpoints/last.pt`` at the just-saved checkpoint via a hard link.

    A hard link shares storage with the ``epoch_*.pt`` file (no duplication on disk) while
    giving a stable, epoch-agnostic name to commit and to load from. ``list_checkpoints``
    globs ``epoch_*.pt`` only, so ``last.pt`` never interferes with epoch enumeration.
    Falls back to a copy on filesystems that reject hard links.
    """
    last = checkpoint_path.parent / "last.pt"
    last.unlink(missing_ok=True)
    try:
        os.link(checkpoint_path, last)
    except OSError:
        shutil.copyfile(checkpoint_path, last)


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
