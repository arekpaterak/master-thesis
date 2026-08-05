from __future__ import annotations

from pathlib import Path

import polars as pl

from core.results.cvrplib import CvrplibSummaryRow
from core.results.synthetic import SyntheticInstanceRow


def test_row_write_roundtrip(tmp_path: Path) -> None:
    rows = [
        SyntheticInstanceRow(
            config="random_destroy",
            setting="cvrp20",
            group="main",
            instance_id="cvrp20-seed7-0000",
            source="seed7",
            initial_objective=5.7,
            final_objective=4.1,
            improvement=1.6,
        ),
    ]
    path = SyntheticInstanceRow.write(rows, path=tmp_path / "insts.csv")

    df = pl.read_csv(path)
    assert df.columns == list(SyntheticInstanceRow.model_fields)
    assert df.height == 1
    assert df["improvement"].to_list() == [1.6]


def test_row_write_empty_writes_header(tmp_path: Path) -> None:
    # Schema comes from the class, so an empty list still yields a header-only CSV.
    path = SyntheticInstanceRow.write([], path=tmp_path / "empty.csv")
    df = pl.read_csv(path)
    assert df.columns == list(SyntheticInstanceRow.model_fields)
    assert df.height == 0


def test_row_write_derives_nullable_from_optional_field(tmp_path: Path) -> None:
    # gap_pct_std is Optional; a None must round-trip as a null column, and the base
    # columns must come before the subclass columns.
    row = CvrplibSummaryRow(
        config="random_destroy",
        method_type="random-destroy",
        setting="cvrplib-X",
        group="generalisation",
        problem_kind="cvrplib",
        checkpoint_epoch=None,
        n_instances=1,
        total_time_s=1.0,
        set_name="X",
        destruction_size=15,
        gap_pct_mean=2.0,
        gap_pct_median=2.0,
        gap_pct_std=None,
        gap_pct_min=2.0,
        gap_pct_max=2.0,
        n_at_or_below_bks=0,
    )
    path = CvrplibSummaryRow.write([row], path=tmp_path / "summary_cvrplib.csv")

    df = pl.read_csv(path)
    assert df.columns == list(CvrplibSummaryRow.model_fields)
    assert df["gap_pct_std"].to_list() == [None]
