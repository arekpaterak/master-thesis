from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, ClassVar, Literal

import torch
from pydantic import BaseModel, Field


class Baseline(ABC):
    """Advantage baseline for the NDS REINFORCE loss.

    Calling the baseline returns a ``(B, R)``-broadcastable reward that the policy loss
    subtracts from the per-rollout reward.
    """

    @abstractmethod
    def __call__(self, reward: torch.Tensor) -> torch.Tensor: ...


class NoBaseline(Baseline):
    """No variance reduction; advantage equals the raw reward."""

    @classmethod
    def from_config(cls, config: NoBaselineConfig) -> NoBaseline:
        return cls()

    def __call__(self, reward: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(reward)


class SharedBaseline(Baseline):
    """Shared mean over the rollouts (POMO); the original NDS/all-rollouts baseline."""

    @classmethod
    def from_config(cls, config: SharedBaselineConfig) -> SharedBaseline:
        return cls()

    def __call__(self, reward: torch.Tensor) -> torch.Tensor:
        return reward.mean(dim=1, keepdim=True)


class LeaveOneOutBaseline(Baseline):
    """Per-rollout baseline is the mean of the other rollouts."""

    @classmethod
    def from_config(cls, config: LeaveOneOutBaselineConfig) -> LeaveOneOutBaseline:
        return cls()

    def __call__(self, reward: torch.Tensor) -> torch.Tensor:
        num_rollouts = reward.shape[1]
        if num_rollouts == 1:
            return reward
        return (reward.sum(dim=1, keepdim=True) - reward) / (num_rollouts - 1)


class _BaselineConfigBase(BaseModel):
    min_rollouts: ClassVar[int] = 1
    """Smallest ``num_rollouts`` for which the baseline gives a non-degenerate advantage."""

    @abstractmethod
    def build(self) -> Baseline: ...


class NoBaselineConfig(_BaselineConfigBase):
    type: Literal["none"] = "none"

    def build(self) -> NoBaseline:
        return NoBaseline.from_config(self)


class SharedBaselineConfig(_BaselineConfigBase):
    type: Literal["mean"] = "mean"
    min_rollouts = 2

    def build(self) -> SharedBaseline:
        return SharedBaseline.from_config(self)


class LeaveOneOutBaselineConfig(_BaselineConfigBase):
    type: Literal["leave_one_out"] = "leave_one_out"
    min_rollouts = 2

    def build(self) -> LeaveOneOutBaseline:
        return LeaveOneOutBaseline.from_config(self)


BaselineConfig = Annotated[
    NoBaselineConfig | SharedBaselineConfig | LeaveOneOutBaselineConfig,
    Field(discriminator="type"),
]
