from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field


class Reward(ABC):
    """Maps the objective before/after a reconstruction step to a per-rollout reward.

    ``objective_before`` is the ``(B,)`` objective of the current solution and
    ``objective_after`` is the ``(B, R)`` objective of each rollout's repaired
    solution. The returned ``(B, R)`` reward is non-negative (no credit for a
    worse solution).
    """

    @abstractmethod
    def __call__(
        self,
        *,
        objective_before: torch.Tensor,
        objective_after: torch.Tensor,
    ) -> torch.Tensor: ...


class AbsoluteReward(Reward):
    """Raw objective reduction in problem units (original NDS behaviour)."""

    @classmethod
    def from_config(cls, config: AbsoluteRewardConfig) -> AbsoluteReward:
        return cls()

    def __call__(
        self,
        *,
        objective_before: torch.Tensor,
        objective_after: torch.Tensor,
    ) -> torch.Tensor:
        improvement = objective_before.unsqueeze(-1) - objective_after
        return improvement.clamp(min=0.0)


class RelativeReward(Reward):
    """Objective reduction as a fraction of the current objective.

    Scale-invariant across instance sizes and across the search trajectory, which
    keeps the gradient magnitude stable as the solution improves.
    """

    @classmethod
    def from_config(cls, config: RelativeRewardConfig) -> RelativeReward:
        return cls()

    def __call__(
        self,
        *,
        objective_before: torch.Tensor,
        objective_after: torch.Tensor,
    ) -> torch.Tensor:
        before = objective_before.unsqueeze(-1)
        improvement = (before - objective_after) / before
        return improvement.clamp(min=0.0)


class _RewardConfigBase(BaseModel):
    @abstractmethod
    def build(self) -> Reward: ...


class AbsoluteRewardConfig(_RewardConfigBase):
    type: Literal["absolute"] = "absolute"

    def build(self) -> AbsoluteReward:
        return AbsoluteReward.from_config(self)


class RelativeRewardConfig(_RewardConfigBase):
    type: Literal["relative"] = "relative"

    def build(self) -> RelativeReward:
        return RelativeReward.from_config(self)


RewardConfig = Annotated[
    AbsoluteRewardConfig | RelativeRewardConfig,
    Field(discriminator="type"),
]
