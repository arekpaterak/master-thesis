from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from core.eval.problem_setting import CVRPLibProblemSetting, SyntheticProblemSetting
from core.evaluation import EvaluationResult


def synthetic_summary_row(
    *,
    config: str,
    method_type: str,
    setting: SyntheticProblemSetting,
    result: EvaluationResult,
) -> dict[str, object]:
    """Aggregate a synthetic EvaluationResult into a single summary row."""
    objective = result.objective_per_step
    initial, final = objective[0], objective[-1]
    improvement = initial - final
    n_instances = objective.shape[1]
    n_steps = objective.shape[0] - 1
    return {
        "config": config,
        "method_type": method_type,
        "problem_label": setting.label,
        "problem_kind": "synthetic",
        "n_instances": n_instances,
        "total_time_s": round(result.total_time_s, 4),
        "problem": setting.problem.name,
        "size": setting.problem.size,
        "n_steps": n_steps,
        "time_per_instance_s": round(result.total_time_s / max(n_instances, 1), 6),
        "obj_initial_mean": initial.mean().item(),
        "obj_initial_std": initial.std().item(),
        "obj_final_mean": final.mean().item(),
        "obj_final_std": final.std().item(),
        "obj_final_median": final.median().item(),
        "obj_final_min": final.min().item(),
        "obj_final_max": final.max().item(),
        "improvement_mean": improvement.mean().item(),
        "improvement_std": improvement.std().item(),
        "improvement_pct_mean": (improvement / initial).mean().item() * 100,
    }


def synthetic_instance_rows(
    *, config: str, setting: SyntheticProblemSetting, result: EvaluationResult
) -> list[dict[str, object]]:
    objective = result.objective_per_step
    initial, final = objective[0], objective[-1]
    return [
        {
            "config": config,
            "problem_label": setting.label,
            "instance_id": f"{setting.label}-seed{setting.test_seed}-{i:04d}",
            "seed": setting.test_seed,
            "initial_objective": initial[i].item(),
            "final_objective": final[i].item(),
            "improvement": (initial[i] - final[i]).item(),
        }
        for i in range(objective.shape[1])
    ]


def synthetic_curve_rows(result: EvaluationResult) -> list[dict[str, object]]:
    objective = result.objective_per_step
    mean = objective.mean(dim=1)
    std = objective.std(dim=1)
    return [
        {"lns_step": step, "obj_mean": mean[idx].item(), "obj_std": std[idx].item()}
        for idx, step in enumerate(result.step_labels)
    ]


def cvrplib_summary_row(
    *,
    config: str,
    method_type: str,
    setting: CVRPLibProblemSetting,
    instance_rows: Sequence[Mapping[str, object]],
    total_time_s: float,
) -> dict[str, object]:
    gaps = pl.Series("gap_pct", [row["gap_pct"] for row in instance_rows], dtype=pl.Float64)
    n_at_or_below = sum(1 for row in instance_rows if row["our_cost"] <= row["bks"])
    return {
        "config": config,
        "method_type": method_type,
        "problem_label": setting.label,
        "problem_kind": "cvrplib",
        "n_instances": len(instance_rows),
        "total_time_s": round(total_time_s, 4),
        "set_name": setting.set_name,
        "gap_pct_mean": gaps.mean(),
        "gap_pct_median": gaps.median(),
        "gap_pct_std": gaps.std(),
        "gap_pct_min": gaps.min(),
        "gap_pct_max": gaps.max(),
        "n_at_or_below_bks": n_at_or_below,
    }


def cvrplib_bucket_rows(
    instance_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not instance_rows:
        return []
    df = pl.DataFrame(list(instance_rows))
    bucketed = (
        df.with_columns(((pl.col("n_customers") // 100) * 100).alias("size_bucket"))
        .group_by("size_bucket")
        .agg(
            pl.col("gap_pct").mean().round(3).alias("mean_gap_pct"),
            pl.col("gap_pct").median().round(3).alias("median_gap_pct"),
            pl.len().alias("count"),
        )
        .sort("size_bucket")
    )
    return bucketed.to_dicts()
