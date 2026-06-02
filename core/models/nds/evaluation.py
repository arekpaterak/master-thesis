from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable

import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel, computed_field

from core.eval_config import EvaluationConfig
from core.lns.acceptance_criterion import AcceptanceCriterion
from core.lns.improvement_step import ImprovementStep
from core.lns.search import search
from core.methods.base import Method
from core.models.module import EpochAccumulator
from core.models.nds.models import NDSModel
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


class EpochValidationAccumulator(EpochAccumulator):
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

    def complete(self, *, epoch: int) -> EpochValidationSummary:
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
    improvement_step: ImprovementStep,
    criterion: AcceptanceCriterion,
    num_improvement_steps: int,
    capture_snapshots_every_n_improvement_steps: int | None = 1,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
) -> Callable[[NDSModel, BaseProblemInstance, int], BatchValidationSummary]:
    snapshot_step_set = {0}
    if capture_snapshots_every_n_improvement_steps is not None:
        assert capture_snapshots_every_n_improvement_steps > 0, (
            "Expected capture_snapshots_every_n_improvement_steps > 0"
        )
        snapshot_step_set |= set(
            range(
                capture_snapshots_every_n_improvement_steps,
                num_improvement_steps + 1,
                capture_snapshots_every_n_improvement_steps,
            )
        )

    def validation_step(
        model: NDSModel,
        instance: BaseProblemInstance,
        epoch: int,
    ) -> BatchValidationSummary:
        model.eval()

        device = next(model.parameters()).device
        instance = instance.to(device)
        batch_size = instance.batch_size[0]

        with (
            torch.no_grad(),
            torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled),
        ):
            solution = generate_start_solution(instance)
            objective_initial = solution.objective().mean().item()

            snapshots: list[BaseSolution] = [_unbatch_first(solution)]
            objective_curve: list[float] = [objective_initial]

            def on_step(step: int, solution: BaseSolution) -> None:
                if step in snapshot_step_set:
                    snapshots.append(_unbatch_first(solution))
                    objective_curve.append(solution.objective().mean().item())

            result = search(
                solution,
                improvement_step=improvement_step,
                criterion=criterion,
                n_iterations=num_improvement_steps,
                on_step=on_step,
                step_progress=step_progress,
            )

            objective_final = result.solution.objective().mean().item()

        return BatchValidationSummary(
            batch_size=batch_size,
            objective_initial=objective_initial,
            objective_final=objective_final,
            objective_curve=objective_curve,
            step_labels=sorted(snapshot_step_set),
            snapshots=snapshots,
        )

    return validation_step


def _unbatch_first(solution: BaseSolution) -> BaseSolution:
    return solution[0] if solution.batch_size else solution  # type: ignore


class EvaluationResult(BaseModel):
    """Per-instance per-LNS-step result of a single eval run."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    step_labels: list[int]
    # Shape: (len(step_labels), num_test_instances)
    per_instance_objective: list[list[float]]
    # Cumulative wall-clock seconds at each captured step (length = len(step_labels))
    cumulative_time_s: list[float]
    # Snapshots for the first K instances; outer length = K, inner length = len(step_labels)
    plot_snapshots: list[list[BaseSolution]]


def configure_evaluation_step(
    *,
    method: Method,
    eval_config: EvaluationConfig,
    capture_every_n_improvement_steps: int = 1,
) -> Callable[[BaseProblemInstance], EvaluationResult]:
    num_improvement_steps = method.search.num_improvement_steps

    assert capture_every_n_improvement_steps > 0, "Expected capture_every_n_improvement_steps > 0"
    snapshot_step_set = {0} | set(
        range(
            capture_every_n_improvement_steps,
            num_improvement_steps + 1,
            capture_every_n_improvement_steps,
        )
    )

    def evaluation_step(instance: BaseProblemInstance) -> EvaluationResult:
        per_instance_objective: list[list[float]] = []
        cumulative_time_s: list[float] = []
        plot_snapshots: list[list[BaseSolution]] = []
        k_plots = min(eval_config.num_plot_instances, instance.batch_size[0])

        with torch.no_grad():
            solution = method.generate_start_solution(instance)

            t0 = time.perf_counter()
            per_instance_objective.append(solution.objective().detach().cpu().tolist())
            cumulative_time_s.append(0.0)
            for plot_idx in range(k_plots):
                plot_snapshots.append([_unbatch_at(solution, plot_idx)])

            def on_step(step: int, solution: BaseSolution) -> None:
                if step in snapshot_step_set:
                    per_instance_objective.append(solution.objective().detach().cpu().tolist())
                    cumulative_time_s.append(time.perf_counter() - t0)
                    for plot_idx in range(k_plots):
                        plot_snapshots[plot_idx].append(_unbatch_at(solution, plot_idx))

            search(
                solution,
                improvement_step=method.improvement_step,
                criterion=method.criterion,
                n_iterations=num_improvement_steps,
                on_step=on_step,
            )

        return EvaluationResult(
            step_labels=sorted(snapshot_step_set),
            per_instance_objective=per_instance_objective,
            cumulative_time_s=cumulative_time_s,
            plot_snapshots=plot_snapshots,
        )

    return evaluation_step


def _unbatch_at(solution: BaseSolution, idx: int) -> BaseSolution:
    return solution[idx] if solution.batch_size else solution  # type: ignore


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
