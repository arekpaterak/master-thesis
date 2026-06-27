from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from pydantic import BaseModel, computed_field

from core.lns.acceptance_criterion import AcceptanceCriterion, SimulatedAnnealingAcceptance
from core.lns.improvement_step import ImprovementStep
from core.lns.search import augmented_search, search
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
    n_augmentations: int = 1,
    augmented_threshold_factor: float = 15.0,
    capture_snapshots_every_n_improvement_steps: int | None = 1,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
) -> Callable[[NDSModel, BaseProblemInstance, int], BatchValidationSummary]:
    if n_augmentations > 1 and not isinstance(criterion, SimulatedAnnealingAcceptance):
        raise ValueError(
            "Augmented search (n_augmentations > 1) requires a simulated_annealing "
            "acceptance criterion -- the cross-augmentation threshold depends on "
            "the SA temperature. Set validation.search.acceptance.type to 'sa'."
        )
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
            if n_augmentations > 1:
                instance = instance.augment(n_augmentations)
            solution = generate_start_solution(instance)
            objective_initial = _aggregate_objective(solution).mean().item()

            snapshots: list[BaseSolution] = [_unbatch_first(_best_per_instance(solution))]
            objective_curve: list[float] = [objective_initial]

            def on_step(step: int, solution: BaseSolution) -> None:
                if step in snapshot_step_set:
                    snapshots.append(_unbatch_first(_best_per_instance(solution)))
                    objective_curve.append(_aggregate_objective(solution).mean().item())

            if n_augmentations > 1:
                assert isinstance(criterion, SimulatedAnnealingAcceptance)
                result = augmented_search(
                    solution,
                    improvement_step=improvement_step,
                    criterion=criterion,
                    n_iterations=num_improvement_steps,
                    threshold_factor=augmented_threshold_factor,
                    on_step=on_step,
                    step_progress=step_progress,
                )
            else:
                result = search(
                    solution,
                    improvement_step=improvement_step,
                    criterion=criterion,
                    n_iterations=num_improvement_steps,
                    on_step=on_step,
                    step_progress=step_progress,
                )

            objective_final = _aggregate_objective(result.solution).mean().item()

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


def _aggregate_objective(solution: BaseSolution) -> torch.Tensor:
    """For augmented solutions (B, A) return min over A; otherwise the objective as-is.

    Treats a 2D batch_size as ``(B, A)`` -- the augmented case. A 1D batch_size
    is the plain (B,) case and returns the objective unchanged.
    """
    objective = solution.objective()
    if len(solution.batch_size) == 2:
        return objective.min(dim=-1).values
    return objective


def _best_per_instance(solution: BaseSolution) -> BaseSolution:
    """For augmented solutions, gather the best across the A dim; otherwise unchanged."""
    if len(solution.batch_size) != 2:
        return solution
    best_idx = solution.objective().argmin(dim=-1, keepdim=True)  # (B, 1)
    return solution.gather(-1, best_idx).squeeze(-1)
