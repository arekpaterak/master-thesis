from pathlib import Path

import torch
from torch import nn as nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    loss: float | None = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "loss": loss,
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
