from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import torch
from pydantic import BaseModel, computed_field

from core.lns.config import SearchParams
from core.lns.improvement_step import ImprovementStep
from core.methods.base import Method
from core.models.module import EpochAccumulator
from core.models.nds.models import NDSModel
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution
from core.solve import solve


class BatchValidationSummary(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    batch_size: int
    objective_initial: float
    objective_final: float
    batch_time_sec: float
    objective_curve: list[float]
    step_labels: list[int]
    snapshots: list[BaseSolution]


class EpochValidationSummary(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    epoch: int
    objective_initial: float
    objective_final: float
    batch_time_sec: float
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

    @computed_field
    @property
    def steps_to_best(self) -> int:
        best_idx = min(range(len(self.objective_curve)), key=lambda i: self.objective_curve[i])
        return self.step_labels[best_idx]

    @computed_field
    @property
    def steps_with_improvement(self) -> float:
        improvements = sum(b < a for a, b in zip(self.objective_curve, self.objective_curve[1:]))
        return (
            improvements / (len(self.objective_curve) - 1) if len(self.objective_curve) > 1 else 0.0
        )


class EpochValidationAccumulator(EpochAccumulator):
    def __init__(self) -> None:
        self._total_instances = 0
        self._total_batches = 0
        self._total_objective_initial = 0.0
        self._total_objective_final = 0.0
        self._total_batch_time = 0.0
        self._weighted_curve: list[float] = []
        self._step_labels: list[int] = []
        self._snapshots: list[BaseSolution] = []

    def update(self, summary: BatchValidationSummary) -> None:
        n = summary.batch_size
        self._total_instances += n
        self._total_batches += 1
        self._total_objective_initial += summary.objective_initial * n
        self._total_objective_final += summary.objective_final * n
        self._total_batch_time += summary.batch_time_sec
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
            batch_time_sec=self._total_batch_time / self._total_batches,
            objective_curve=[v / n for v in self._weighted_curve],
            step_labels=self._step_labels,
            snapshots=self._snapshots,
        )


def configure_validation_step(
    *,
    generate_start_solution: Callable[[BaseProblemInstance], BaseSolution],
    improvement_step: ImprovementStep,
    search_params: SearchParams,
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
                search_params.num_improvement_steps + 1,
                capture_snapshots_every_n_improvement_steps,
            )
        )

    method = Method(
        search=search_params,
        generate_start_solution=generate_start_solution,
        improvement_step=improvement_step,
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

        snapshots: list[BaseSolution] = []
        objective_curve: list[float] = []

        def on_step(step: int, solution: BaseSolution) -> None:
            if step in snapshot_step_set:
                snapshots.append(_unbatch_first(solution))
                objective_curve.append(solution.objective().mean().item())

        start = time.perf_counter()
        with (
            torch.no_grad(),
            torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled),
        ):
            final = solve(method, instance, on_step=on_step, step_progress=step_progress)
        elapsed = time.perf_counter() - start

        return BatchValidationSummary(
            batch_size=batch_size,
            objective_initial=objective_curve[0],
            objective_final=final.objective().mean().item(),
            batch_time_sec=elapsed,
            objective_curve=objective_curve,
            step_labels=sorted(snapshot_step_set),
            snapshots=snapshots,
        )

    return validation_step


def _unbatch_first(solution: BaseSolution) -> BaseSolution:
    return solution[0] if solution.batch_size else solution  # type: ignore
