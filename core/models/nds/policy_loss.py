from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field


class PolicyLoss(ABC):
    """REINFORCE-style loss over the rollouts of a single improvement step.

    Operates on already-computed tensors; the training step keeps ownership of the
    solution/objective logic and passes in the per-rollout reward, the per-rollout
    baseline and the log probabilities, all shaped ``(B, R)`` (the baseline may be
    ``(B, 1)`` and broadcasts). The advantage is ``reward - baseline``; the baseline is
    produced separately by a :class:`~core.models.nds.baseline.Baseline`.
    """

    @abstractmethod
    def __call__(
        self,
        *,
        reward: torch.Tensor,
        baseline: torch.Tensor,
        log_p: torch.Tensor,
    ) -> torch.Tensor: ...


class BestRolloutLoss(PolicyLoss):
    """Learns from the single best rollout per instance (original NDS behaviour).

    Only the best rollout's advantage and log probability enter the gradient.
    """

    def __call__(
        self,
        *,
        reward: torch.Tensor,
        baseline: torch.Tensor,
        log_p: torch.Tensor,
    ) -> torch.Tensor:
        advantage = reward - baseline
        best_idx = reward.argmax(dim=1, keepdim=True)
        best_advantage = advantage.gather(1, best_idx)
        best_log_p = log_p.gather(1, best_idx)
        return -(best_advantage * best_log_p).mean()


class AllRolloutsLoss(PolicyLoss):
    """Learns from every rollout (POMO-style) for more gradient per forward pass."""

    def __call__(
        self,
        *,
        reward: torch.Tensor,
        baseline: torch.Tensor,
        log_p: torch.Tensor,
    ) -> torch.Tensor:
        advantage = reward - baseline
        return -(advantage * log_p).mean()


class BestRolloutLossConfig(BaseModel):
    type: Literal["best"] = "best"


class AllRolloutsLossConfig(BaseModel):
    type: Literal["all"] = "all"


PolicyLossConfig = Annotated[
    BestRolloutLossConfig | AllRolloutsLossConfig,
    Field(discriminator="type"),
]


def build_policy_loss(config: PolicyLossConfig) -> PolicyLoss:
    match config:
        case BestRolloutLossConfig():
            return BestRolloutLoss()
        case AllRolloutsLossConfig():
            return AllRolloutsLoss()
