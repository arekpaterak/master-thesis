from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import polars as pl

from core.eval.problem_setting import CVRPLibProblemSetting
from core.results.row import Row, SummaryRowBase


class CVRPLibInstanceRow(Row):
    """One row of {label}_instances.csv: a single CVRPLIB instance vs its BKS."""

    instance: str
    n_customers: int
    bks: int
    obj: int
    gap_pct: float
    time_s: float


# (label, min customers inclusive, max customers inclusive or None for no upper bound)
CUSTOMER_RANGES: list[tuple[str, int, int | None]] = [
    ("[100, 150)", 100, 149),
    ("[150, 200]", 150, 200),
    ("all", 0, None),
]


class CVRPLibSummaryRow(SummaryRowBase):
    """One row of summary_cvrplib.csv: aggregated gap over a CVRPLIB size range."""

    set_name: str
    destruction_size: int
    customer_range: str
    mean_time_s: float
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
        group: str,
        checkpoint_epoch: int | None,
        setting: CVRPLibProblemSetting,
        instance_rows: Sequence[CVRPLibInstanceRow],
        destruction_size: int,
    ) -> list[Self]:
        """One summary row per size range in ``CUSTOMER_RANGES`` (empty ranges skipped)."""
        rows: list[Self] = []
        for range_label, lo, hi in CUSTOMER_RANGES:
            subset = [
                row
                for row in instance_rows
                if lo <= row.n_customers and (hi is None or row.n_customers <= hi)
            ]
            if not subset:
                continue
            gaps = pl.Series("gap_pct", [row.gap_pct for row in subset], dtype=pl.Float64)
            times = pl.Series("time_s", [row.time_s for row in subset], dtype=pl.Float64)
            n_at_or_below = sum(1 for row in subset if row.obj <= row.bks)
            rows.append(
                cls(
                    config=config,
                    method_type=method_type,
                    setting=setting.label,
                    group=group,
                    problem_kind="cvrplib",
                    checkpoint_epoch=checkpoint_epoch,
                    n_instances=len(subset),
                    total_time_s=round(times.sum(), 4),
                    set_name=setting.set_name,
                    destruction_size=destruction_size,
                    customer_range=range_label,
                    mean_time_s=round(times.mean(), 4),
                    gap_pct_mean=gaps.mean(),
                    gap_pct_median=gaps.median(),
                    gap_pct_std=gaps.std(),
                    gap_pct_min=gaps.min(),
                    gap_pct_max=gaps.max(),
                    n_at_or_below_bks=n_at_or_below,
                )
            )
        return rows
