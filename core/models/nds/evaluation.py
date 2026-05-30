from __future__ import annotations

import math
from collections.abc import Callable

import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from models.nds.models import NDSModel
from pydantic import BaseModel, computed_field

from core.models.rollout import Rollout
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


class BatchValidationSummary(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    batch_size: int
    objective_initial: float
    objective_final: float
    objective_curve: list[float]
    step_labels: list[int]
    snapshots: list[BaseSolution]


class EpochValidationSummary(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    epoch: int
    objective_initial: float
    objective_final: float
    objective_curve: list[float]
    step_labels: list[int]
    snapshots: list[BaseSolution]

    @computed_field
    @property
    def improvement(self) -> float:
        return self.objective_initial - self.objective_final

    @computed_field
    @property
    def relative_improvement(self) -> float:
        if self.objective_initial == 0.0:
            return 0.0
        return self.improvement / self.objective_initial


class EpochValidationAccumulator:
    def __init__(self) -> None:
        self._total_instances = 0
        self._total_objective_initial = 0.0
        self._total_objective_final = 0.0
        self._weighted_curve: list[float] = []
        self._step_labels: list[int] = []
        self._snapshots: list[BaseSolution] = []

    def update(self, summary: BatchValidationSummary) -> None:
        n = summary.batch_size
        self._total_instances += n
        self._total_objective_initial += summary.objective_initial * n
        self._total_objective_final += summary.objective_final * n
        if not self._weighted_curve:
            self._weighted_curve = [v * n for v in summary.objective_curve]
            self._step_labels = summary.step_labels
            self._snapshots = summary.snapshots
        else:
            for i, v in enumerate(summary.objective_curve):
                self._weighted_curve[i] += v * n

    def finalize(self, *, epoch: int) -> EpochValidationSummary:
        n = self._total_instances
        return EpochValidationSummary(
            epoch=epoch,
            objective_initial=self._total_objective_initial / n,
            objective_final=self._total_objective_final / n,
            objective_curve=[v / n for v in self._weighted_curve],
            step_labels=self._step_labels,
            snapshots=self._snapshots,
        )


def configure_validation_step(
    *,
    generate_start_solution: Callable[[BaseProblemInstance], BaseSolution],
    apply_rollout_fn: Callable[[BaseSolution, Rollout], BaseSolution],
    repair: Callable[[BaseSolution, Rollout], BaseSolution],
    deconstruction_size: int,
    num_improvement_steps: int,
    every_n_lns_steps: int,
) -> Callable[[NDSModel, BaseProblemInstance, int], BatchValidationSummary]:
    snapshot_steps = [0] + list(
        range(every_n_lns_steps, num_improvement_steps + 1, every_n_lns_steps)
    )

    def validation_step(
        model: NDSModel,
        instance: BaseProblemInstance,
        epoch: int,
    ) -> BatchValidationSummary:
        model.eval()

        instance = instance.to(next(model.parameters()).device)
        batch_size = instance.batch_size[0]

        with torch.no_grad():
            solution = generate_start_solution(instance)
            objective_initial = solution.objective().mean().item()

            snapshots: list[BaseSolution] = [_unbatch_first(solution)]
            objective_curve: list[float] = [objective_initial]

            for step in range(1, num_improvement_steps + 1):
                # TODO: Apply all rollouts one after another, match Augmented Simulated Annealing
                rollout = model.rollout(
                    solution,
                    rollouts_per_solution=1,
                    deconstruction_size=deconstruction_size,
                    greedy=False,
                )
                partial = apply_rollout_fn(solution, rollout)
                solution_prime = repair(partial, rollout)
                solution = solution_prime.squeeze(1)

                if step in snapshot_steps:
                    snapshots.append(_unbatch_first(solution))
                    objective_curve.append(solution.objective().mean().item())

            objective_final = solution.objective().mean().item()

        return BatchValidationSummary(
            batch_size=batch_size,
            objective_initial=objective_initial,
            objective_final=objective_final,
            objective_curve=objective_curve,
            step_labels=snapshot_steps,
            snapshots=snapshots,
        )

    return validation_step


def _unbatch_first(solution: BaseSolution) -> BaseSolution:
    return solution[0] if solution.batch_size else solution


def plot_lns_progress(
    snapshots: list[BaseSolution],
    step_labels: list[int],
    *,
    epoch: int,
) -> Figure:
    n_snapshots = len(snapshots)
    n_cols = min(n_snapshots, 5)
    n_rows = math.ceil(n_snapshots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for i, (solution, step) in enumerate(zip(snapshots, step_labels)):
        solution.visualize(ax=axes_flat[i])
        axes_flat[i].set_title(
            axes_flat[i].get_title() + f"\nStep {step}",
            fontsize=9,
        )

    for i in range(n_snapshots, n_rows * n_cols):
        axes_flat[i].set_visible(False)

    fig.suptitle(f"Epoch {epoch}", fontsize=12)
    fig.tight_layout()
    return fig
