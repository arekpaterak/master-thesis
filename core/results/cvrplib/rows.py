from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import polars as pl

from core.eval.problem_setting import CVRPLibProblemSetting
from core.results.row import Row, SummaryRowBase


class CvrplibInstanceRow(Row):
    """One row of {label}_instances.csv: a single CVRPLIB instance vs its BKS."""

    instance: str
    n_customers: int
    bks: int
    our_cost: int
    gap_pct: float


class CvrplibSummaryRow(SummaryRowBase):
    """One row of summary_cvrplib.csv: aggregated gap over a CVRPLIB set."""

    set_name: str
    destruction_size: int
    gap_pct_mean: float
    gap_pct_median: float
    gap_pct_std: float | None
    gap_pct_min: float
    gap_pct_max: float
    n_at_or_below_bks: int

    @classmethod
    def build(
        cls,
        *,
        config: str,
        method_type: str,
        checkpoint_epoch: int | None,
        setting: CVRPLibProblemSetting,
        instance_rows: Sequence[CvrplibInstanceRow],
        total_time_s: float,
        destruction_size: int,
    ) -> Self:
        gaps = pl.Series("gap_pct", [row.gap_pct for row in instance_rows], dtype=pl.Float64)
        n_at_or_below = sum(1 for row in instance_rows if row.our_cost <= row.bks)
        return cls(
            config=config,
            method_type=method_type,
            problem_label=setting.label,
            problem_kind="cvrplib",
            checkpoint_epoch=checkpoint_epoch,
            n_instances=len(instance_rows),
            total_time_s=round(total_time_s, 4),
            set_name=setting.set_name,
            destruction_size=destruction_size,
            gap_pct_mean=gaps.mean(),
            gap_pct_median=gaps.median(),
            gap_pct_std=gaps.std(),
            gap_pct_min=gaps.min(),
            gap_pct_max=gaps.max(),
            n_at_or_below_bks=n_at_or_below,
        )
