from __future__ import annotations

from typing import Self

from core.eval.problem_setting import SyntheticProblemSetting
from core.evaluation import EvaluationResult
from core.results.row import Row, SummaryRowBase


class SyntheticSummaryRow(SummaryRowBase):
    """One row of summary_synthetic.csv: an aggregated synthetic EvaluationResult."""

    problem: str
    size: int
    source: str
    destruction_size: int
    n_steps: int
    time_per_instance_s: float
    obj_initial_mean: float
    obj_initial_std: float
    obj_final_mean: float
    obj_final_std: float
    obj_final_median: float
    obj_final_min: float
    obj_final_max: float
    improvement_mean: float
    improvement_std: float
    improvement_pct_mean: float

    @classmethod
    def build(
        cls,
        *,
        config: str,
        method_type: str,
        group: str,
        checkpoint_epoch: int | None,
        setting: SyntheticProblemSetting,
        result: EvaluationResult,
        destruction_size: int,
    ) -> Self:
        objective = result.objective_per_step
        initial, final = objective[0], objective[-1]
        improvement = initial - final
        n_instances = objective.shape[1]
        return cls(
            config=config,
            method_type=method_type,
            setting=setting.label,
            group=group,
            problem_kind="synthetic",
            checkpoint_epoch=checkpoint_epoch,
            n_instances=n_instances,
            total_time_s=round(result.total_time_s, 4),
            problem=setting.problem.name,
            size=setting.problem.size,
            source=setting.source,
            destruction_size=destruction_size,
            n_steps=objective.shape[0] - 1,
            time_per_instance_s=round(result.total_time_s / max(n_instances, 1), 6),
            obj_initial_mean=initial.mean().item(),
            obj_initial_std=initial.std().item(),
            obj_final_mean=final.mean().item(),
            obj_final_std=final.std().item(),
            obj_final_median=final.median().item(),
            obj_final_min=final.min().item(),
            obj_final_max=final.max().item(),
            improvement_mean=improvement.mean().item(),
            improvement_std=improvement.std().item(),
            improvement_pct_mean=(improvement / initial).mean().item() * 100,
        )


class SyntheticInstanceRow(Row):
    """One row of instances.csv: a single synthetic instance's outcome."""

    config: str
    setting: str
    group: str
    instance_id: str
    source: str
    initial_objective: float
    final_objective: float
    improvement: float

    @classmethod
    def build(
        cls, *, config: str, group: str, setting: SyntheticProblemSetting, result: EvaluationResult
    ) -> list[Self]:
        objective = result.objective_per_step
        initial, final = objective[0], objective[-1]
        source = setting.source
        return [
            cls(
                config=config,
                setting=setting.label,
                group=group,
                instance_id=f"{setting.label}-{source}-{i:04d}",
                source=source,
                initial_objective=initial[i].item(),
                final_objective=final[i].item(),
                improvement=(initial[i] - final[i]).item(),
            )
            for i in range(objective.shape[1])
        ]


class SyntheticStepRow(Row):
    """One row of steps.csv: mean objective and elapsed time at one improvement step."""

    step: int
    obj_mean: float
    obj_std: float
    time_total_s: float
    time_per_instance_s: float

    @classmethod
    def build(cls, result: EvaluationResult) -> list[Self]:
        objective = result.objective_per_step
        mean = objective.mean(dim=1)
        std = objective.std(dim=1)
        n_instances = max(objective.shape[1], 1)
        return [
            cls(
                step=step,
                obj_mean=mean[idx].item(),
                obj_std=std[idx].item(),
                time_total_s=result.time_per_step[idx],
                time_per_instance_s=result.time_per_step[idx] / n_instances,
            )
            for idx, step in enumerate(result.step_labels)
        ]
