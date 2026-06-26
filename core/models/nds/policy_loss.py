from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field


class PolicyLoss(ABC):
    """REINFORCE-style loss over the rollouts of a single improvement step.

    Operates on already-computed tensors; the training step keeps ownership of the
    solution/objective logic and passes in the per-rollout reward and log
    probabilities, both shaped ``(B, R)``.
    """

    @abstractmethod
    def __call__(
        self,
        *,
        reward: torch.Tensor,
        log_p: torch.Tensor,
    ) -> torch.Tensor: ...


class BestRolloutLoss(PolicyLoss):
    """Learns from the single best rollout per instance (original NDS behaviour).

    Uses the shared mean reward as the baseline and only the best rollout's log
    probability for the gradient.
    """

    def __call__(
        self,
        *,
        reward: torch.Tensor,
        log_p: torch.Tensor,
    ) -> torch.Tensor:
        baseline = reward.mean(dim=1, keepdim=True)
        best_idx = reward.argmax(dim=1, keepdim=True)
        best_reward = reward.gather(1, best_idx)
        best_log_p = log_p.gather(1, best_idx)
        return -((best_reward - baseline) * best_log_p).mean()


class AllRolloutsLoss(PolicyLoss):
    """Learns from every rollout (POMO-style) for more gradient per forward pass."""

    def __init__(self, *, baseline: Literal["mean", "leave_one_out"] = "mean") -> None:
        self._baseline = baseline

    def __call__(
        self,
        *,
        reward: torch.Tensor,
        log_p: torch.Tensor,
    ) -> torch.Tensor:
        advantage = reward - self._compute_baseline(reward)
        return -(advantage * log_p).mean()

    def _compute_baseline(self, reward: torch.Tensor) -> torch.Tensor:
        num_rollouts = reward.shape[1]
        if self._baseline == "mean" or num_rollouts == 1:
            return reward.mean(dim=1, keepdim=True)
        # Leave-one-out: baseline for each rollout is the mean of the other rollouts.
        return (reward.sum(dim=1, keepdim=True) - reward) / (num_rollouts - 1)


class BestRolloutLossConfig(BaseModel):
    type: Literal["best"] = "best"


class AllRolloutsLossConfig(BaseModel):
    type: Literal["all"] = "all"
    baseline: Literal["mean", "leave_one_out"] = Field(
        default="mean",
        title="Baseline",
        description=(
            "Advantage baseline. 'mean' = shared mean over rollouts (POMO); "
            "'leave_one_out' = mean of the other rollouts per sample."
        ),
    )


PolicyLossConfig = Annotated[
    BestRolloutLossConfig | AllRolloutsLossConfig,
    Field(discriminator="type"),
]


def build_policy_loss(config: PolicyLossConfig) -> PolicyLoss:
    match config:
        case BestRolloutLossConfig():
            return BestRolloutLoss()
        case AllRolloutsLossConfig():
            return AllRolloutsLoss(baseline=config.baseline)
