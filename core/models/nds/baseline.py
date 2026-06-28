from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field


class Baseline(ABC):
    """Advantage baseline for the NDS REINFORCE loss.

    Calling the baseline returns a ``(B, R)``-broadcastable reward that the policy loss
    subtracts from the per-rollout reward. ``min_rollouts`` is the smallest
    ``num_rollouts`` for which the baseline produces a useful (non-degenerate) advantage.
    """

    def min_rollouts(self) -> int:
        return 1

    @abstractmethod
    def __call__(self, reward: torch.Tensor) -> torch.Tensor: ...


class NoBaseline(Baseline):
    """No variance reduction; advantage equals the raw reward."""

    def __call__(self, reward: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(reward)


class SharedBaseline(Baseline):
    """Shared mean over the rollouts (POMO); the original NDS/all-rollouts baseline."""

    def min_rollouts(self) -> int:
        return 2

    def __call__(self, reward: torch.Tensor) -> torch.Tensor:
        return reward.mean(dim=1, keepdim=True)


class LeaveOneOutBaseline(Baseline):
    """Per-rollout baseline is the mean of the other rollouts."""

    def min_rollouts(self) -> int:
        return 2

    def __call__(self, reward: torch.Tensor) -> torch.Tensor:
        num_rollouts = reward.shape[1]
        if num_rollouts == 1:
            return reward
        return (reward.sum(dim=1, keepdim=True) - reward) / (num_rollouts - 1)


class NoBaselineConfig(BaseModel):
    type: Literal["none"] = "none"


class SharedBaselineConfig(BaseModel):
    type: Literal["mean"] = "mean"


class LeaveOneOutBaselineConfig(BaseModel):
    type: Literal["leave_one_out"] = "leave_one_out"


BaselineConfig = Annotated[
    NoBaselineConfig | SharedBaselineConfig | LeaveOneOutBaselineConfig,
    Field(discriminator="type"),
]


def build_baseline(config: BaselineConfig) -> Baseline:
    match config:
        case NoBaselineConfig():
            return NoBaseline()
        case SharedBaselineConfig():
            return SharedBaseline()
        case LeaveOneOutBaselineConfig():
            return LeaveOneOutBaseline()
