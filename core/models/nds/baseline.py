from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from core.lns.operators.repair import RepairOperator
from core.lns.utils import best_repair
from core.models.nds.reward import Reward
from core.models.rollout import Rollout
from core.models.statistical_test import (
    OneSidedPairedTTestResult,
    one_sided_paired_ttest,
)
from core.models.utils import freeze_parameters_
from core.problems.solution import BaseSolution

if TYPE_CHECKING:
    from core.models.nds.models import NDSModel


@dataclass(frozen=True)
class BaselineContext:
    """Per-step context a baseline may use to compute its reference reward.

    Only the greedy-rollout baseline needs this; the sample-relative baselines
    (none/mean/leave-one-out) ignore it and operate on the rewards alone.
    """

    solution: BaseSolution
    """Current solution before deconstruction, batch shape ``(B,)``."""
    objective_before: torch.Tensor
    """``(B,)`` objective of ``solution`` (precomputed by the training step)."""
    deconstruction_size: int
    apply_rollout: Callable[[BaseSolution, Rollout], BaseSolution]
    repair: RepairOperator
    n_repairs: int
    compute_reward: Reward


class Baseline(ABC):
    """Advantage baseline for the NDS REINFORCE loss.

    Calling the baseline returns a ``(B, R)``-broadcastable reward that the policy loss
    subtracts from the per-rollout reward. The sample-relative variants are pure functions
    of the rewards; the greedy-rollout variant runs a separate frozen model and needs the
    per-step ``BaselineContext``. ``min_rollouts`` is the smallest ``num_rollouts`` for
    which the baseline produces a useful (non-degenerate) advantage.
    """

    def min_rollouts(self) -> int:
        return 1

    @abstractmethod
    def __call__(
        self,
        *,
        reward: torch.Tensor,
        context: BaselineContext,
    ) -> torch.Tensor: ...

    def setup(self, policy: NDSModel) -> None:
        """Initialise any state that depends on the policy model (e.g. a frozen copy)."""

    def epoch_callback(
        self,
        *,
        policy: NDSModel,
        metric: Callable[[NDSModel], torch.Tensor],
        epoch: int,
    ) -> OneSidedPairedTTestResult | None:
        """Hook run after a training epoch; used by the greedy baseline to sync."""
        return None

    def state_dict(self) -> dict | None:
        return None

    def load_state_dict(self, state: dict | None) -> None:
        pass

    def to(self, device: torch.device | str) -> None:
        pass


class NoBaseline(Baseline):
    """No variance reduction; advantage equals the raw reward."""

    def __call__(
        self,
        *,
        reward: torch.Tensor,
        context: BaselineContext,
    ) -> torch.Tensor:
        return torch.zeros_like(reward)


class SharedBaseline(Baseline):
    """Shared mean over the rollouts (POMO); the original NDS/all-rollouts baseline."""

    def min_rollouts(self) -> int:
        return 2

    def __call__(
        self,
        *,
        reward: torch.Tensor,
        context: BaselineContext,
    ) -> torch.Tensor:
        return reward.mean(dim=1, keepdim=True)


class LeaveOneOutBaseline(Baseline):
    """Per-rollout baseline is the mean of the other rollouts."""

    def min_rollouts(self) -> int:
        return 2

    def __call__(
        self,
        *,
        reward: torch.Tensor,
        context: BaselineContext,
    ) -> torch.Tensor:
        num_rollouts = reward.shape[1]
        if num_rollouts == 1:
            return reward
        return (reward.sum(dim=1, keepdim=True) - reward) / (num_rollouts - 1)


class GreedyRolloutBaseline(Baseline):
    """Greedy rollout baseline (Kool et al.).

    Holds a frozen copy of the policy that produces a greedy (argmax) deconstruction of
    the current solution; its repaired reward is the advantage reference. The baseline is
    periodically synced to the policy via a one-sided paired t-test on a fixed eval set.
    """

    def __init__(self, *, alpha: float = 0.05, sync_every_n_epochs: int = 1) -> None:
        self._alpha = alpha
        self._sync_every_n_epochs = sync_every_n_epochs
        self._baseline_model: NDSModel | None = None

    def setup(self, policy: NDSModel) -> None:
        self._baseline_model = prepare_baseline(policy)

    def __call__(
        self,
        *,
        reward: torch.Tensor,
        context: BaselineContext,
    ) -> torch.Tensor:
        assert self._baseline_model is not None, "GreedyRolloutBaseline.setup must run first."
        with torch.no_grad():
            # The model is built without PolyNet (with_poly=False) for the greedy rollout
            # baseline, so this single greedy rollout is the canonical, seed-independent one.
            greedy_rollout = self._baseline_model.rollout(
                context.solution,
                rollouts_per_solution=1,
                deconstruction_size=context.deconstruction_size,
                greedy=True,
            )
            deconstructed = context.apply_rollout(context.solution, greedy_rollout)
            repaired = best_repair(context.repair, deconstructed, context.n_repairs)
            return context.compute_reward(
                objective_before=context.objective_before,
                objective_after=repaired.objective(),
            )

    def epoch_callback(
        self,
        *,
        policy: NDSModel,
        metric: Callable[[NDSModel], torch.Tensor],
        epoch: int,
    ) -> OneSidedPairedTTestResult | None:
        assert self._baseline_model is not None, "GreedyRolloutBaseline.setup must run first."
        if (epoch + 1) % self._sync_every_n_epochs != 0:
            return None
        result = one_sided_paired_ttest(
            model=policy,
            baseline=self._baseline_model,
            metric=metric,
            alpha=self._alpha,
        )
        if result.improved:
            sync_baseline_(self._baseline_model, policy)
        return result

    def state_dict(self) -> dict | None:
        if self._baseline_model is None:
            return None
        return self._baseline_model.state_dict()

    def load_state_dict(self, state: dict | None) -> None:
        if state is None or self._baseline_model is None:
            return
        self._baseline_model.load_state_dict(state)
        self._baseline_model.eval()
        freeze_parameters_(self._baseline_model)

    def to(self, device: torch.device | str) -> None:
        if self._baseline_model is not None:
            self._baseline_model.to(device)


def sync_baseline_(baseline: nn.Module, model: nn.Module) -> None:
    baseline.load_state_dict(copy.deepcopy(model.state_dict()))
    baseline.eval()
    freeze_parameters_(baseline)


def prepare_baseline(model: NDSModel) -> NDSModel:
    baseline = copy.deepcopy(model)
    sync_baseline_(baseline, model)
    return baseline


class NoBaselineConfig(BaseModel):
    type: Literal["none"] = "none"


class SharedBaselineConfig(BaseModel):
    type: Literal["mean"] = "mean"


class LeaveOneOutBaselineConfig(BaseModel):
    type: Literal["leave_one_out"] = "leave_one_out"


class GreedyRolloutBaselineConfig(BaseModel):
    type: Literal["greedy_rollout"] = "greedy_rollout"
    alpha: float = Field(
        default=0.05,
        title="t-test alpha",
        description="Significance threshold for syncing the baseline to the policy.",
    )
    sync_every_n_epochs: int = Field(
        default=1,
        title="Sync every N epochs",
        description="Run the paired t-test (and maybe sync) every N training epochs.",
    )
    num_instances: int = Field(
        default=256,
        title="t-test instances",
        description="Instances used for the policy-vs-baseline paired t-test.",
    )
    batch_size: int = Field(
        default=64,
        title="t-test batch size",
        description="Batch size for the paired t-test evaluation search.",
    )


BaselineConfig = Annotated[
    NoBaselineConfig
    | SharedBaselineConfig
    | LeaveOneOutBaselineConfig
    | GreedyRolloutBaselineConfig,
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
        case GreedyRolloutBaselineConfig():
            return GreedyRolloutBaseline(
                alpha=config.alpha,
                sync_every_n_epochs=config.sync_every_n_epochs,
            )
