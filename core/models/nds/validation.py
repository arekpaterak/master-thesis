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
from core.models.nds.utils import RolloutMetricsRecorder
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution
from core.solve import base_augmentation, best_augmentation, solve


class BatchValidationSummary(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    batch_size: int
    objective_initial: float
    objective_final: float
    batch_time_sec: float
    instances_per_sec: float
    objective_curve: list[float]
    step_labels: list[int]
    snapshots: list[BaseSolution]
    diversity_overlap: float
    unique_rollouts: float
    objective_final_augmented: float | None = None


class EpochValidationSummary(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    epoch: int
    objective_initial: float
    objective_final: float
    batch_time_sec: float
    epoch_time_sec: float
    instances_per_sec: float
    objective_curve: list[float]
    step_labels: list[int]
    snapshots: list[BaseSolution]
    diversity_overlap: float
    unique_rollouts: float
    objective_final_augmented: float | None = None

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


class EpochValidationAccumulator(EpochAccumulator):
    def __init__(self) -> None:
        self._total_instances = 0
        self._total_batches = 0
        self._total_objective_initial = 0.0
        self._total_objective_final = 0.0
        self._total_batch_time = 0.0
        self._total_diversity_overlap = 0.0
        self._total_unique_rollouts = 0.0
        self._weighted_curve: list[float] = []
        self._step_labels: list[int] = []
        self._snapshots: list[BaseSolution] = []
        self._augmented = False
        self._total_objective_final_augmented = 0.0

    def update(self, summary: BatchValidationSummary) -> None:
        n = summary.batch_size
        self._total_instances += n
        self._total_batches += 1
        self._total_objective_initial += summary.objective_initial * n
        self._total_objective_final += summary.objective_final * n
        self._total_batch_time += summary.batch_time_sec
        self._total_diversity_overlap += summary.diversity_overlap * n
        self._total_unique_rollouts += summary.unique_rollouts * n
        if not self._weighted_curve:
            self._weighted_curve = [v * n for v in summary.objective_curve]
            self._step_labels = summary.step_labels
            self._snapshots = summary.snapshots
        else:
            for i, v in enumerate(summary.objective_curve):
                self._weighted_curve[i] += v * n
        if summary.objective_final_augmented is not None:
            self._augmented = True
            self._total_objective_final_augmented += summary.objective_final_augmented * n

    def complete(self, *, epoch: int) -> EpochValidationSummary:
        n = self._total_instances
        return EpochValidationSummary(
            epoch=epoch,
            objective_initial=self._total_objective_initial / n,
            objective_final=self._total_objective_final / n,
            batch_time_sec=self._total_batch_time / self._total_batches,
            epoch_time_sec=self._total_batch_time,
            instances_per_sec=self._total_instances / self._total_batch_time,
            objective_curve=[v / n for v in self._weighted_curve],
            step_labels=self._step_labels,
            snapshots=self._snapshots,
            diversity_overlap=self._total_diversity_overlap / n,
            unique_rollouts=self._total_unique_rollouts / n,
            objective_final_augmented=(
                self._total_objective_final_augmented / n if self._augmented else None
            ),
        )


class NDSValidationStep:
    def __init__(
        self,
        *,
        generate_start_solution: Callable[[BaseProblemInstance], BaseSolution],
        improvement_step: ImprovementStep,
        search_params: SearchParams,
        capture_snapshots_every_n_improvement_steps: int | None = 1,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
        rollout_recorder: RolloutMetricsRecorder | None = None,
    ) -> None:
        self._amp_enabled = amp_enabled
        self._amp_dtype = amp_dtype
        self._step_progress = step_progress
        self._rollout_recorder = rollout_recorder

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
        self._snapshot_step_set = snapshot_step_set

        self._method = Method(
            search=search_params,
            generate_start_solution=generate_start_solution,
            improvement_step=improvement_step,
        )

    def __call__(
        self,
        model: NDSModel,
        instance: BaseProblemInstance,
        epoch: int,
    ) -> BatchValidationSummary:
        model.eval()

        device = next(model.parameters()).device
        instance = instance.to(device)
        batch_size = instance.batch_size[0]

        augmented = self._method.search.n_augmentations > 1
        snapshots: list[BaseSolution] = []
        objective_curve: list[float] = []

        def on_step(step: int, solution: BaseSolution) -> None:
            if step not in self._snapshot_step_set:
                return
            base = base_augmentation(solution) if augmented else solution
            snapshots.append(_unbatch_first(base))
            objective_curve.append(base.objective().mean().item())

        if self._rollout_recorder is not None:
            self._rollout_recorder.reset()

        start = time.perf_counter()
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type, dtype=self._amp_dtype, enabled=self._amp_enabled
            ),
        ):
            solution_final = solve(
                self._method,
                instance,
                reduce_augmentations=False,
                on_step=on_step,
                step_progress=self._step_progress,
            )
        elapsed = time.perf_counter() - start

        base_final = base_augmentation(solution_final) if augmented else solution_final
        recorder = self._rollout_recorder
        return BatchValidationSummary(
            batch_size=batch_size,
            objective_initial=objective_curve[0],
            objective_final=base_final.objective().mean().item(),
            batch_time_sec=elapsed,
            instances_per_sec=batch_size / elapsed,
            objective_curve=objective_curve,
            step_labels=sorted(self._snapshot_step_set),
            snapshots=snapshots,
            diversity_overlap=recorder.diversity_overlap if recorder else 0.0,
            unique_rollouts=recorder.unique_rollouts if recorder else 0.0,
            objective_final_augmented=(
                best_augmentation(solution_final).objective().mean().item() if augmented else None
            ),
        )


def _unbatch_first(solution: BaseSolution) -> BaseSolution:
    return solution[0] if solution.batch_size else solution  # type: ignore
