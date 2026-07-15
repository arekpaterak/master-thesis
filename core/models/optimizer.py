from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    name: Literal["adamw"] = Field(default="adamw", title="Name", description="Optimizer name.")
    learning_rate: float = Field(default=1e-4, title="Learning rate", description="Learning rate.")
    weight_decay: float = Field(default=1e-6, title="Weight decay", description="Weight decay.")


def build_optimizer(config: OptimizerConfig, parameters: Iterable[nn.Parameter]) -> optim.Optimizer:
    match config.name:
        case "adamw":
            return optim.AdamW(
                parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                fused=True,
            )
        case _:
            raise ValueError(f"Unknown optimizer {config.name}")
