from __future__ import annotations

from pathlib import Path

import polars as pl

from core.results.schema import EVAL_SCHEMA
from core.results.writer import eval_results_dir, write_eval_results


def test_write_eval_results_roundtrip(tmp_path: Path) -> None:
    rows = [
        {
            "method": "nds-ar",
            "instance_id": "tsp20_seed42_00",
            "seed": 42,
            "lns_step": 0,
            "objective": 5.7,
            "time_s": 0.0,
            "initial_objective": 5.7,
        },
        {
            "method": "nds-ar",
            "instance_id": "tsp20_seed42_00",
            "seed": 42,
            "lns_step": 50,
            "objective": 4.1,
            "time_s": 12.3,
            "initial_objective": 5.7,
        },
    ]

    path = write_eval_results(tmp_path, eval_run_id="eval-001", rows=rows)
    assert path == eval_results_dir(tmp_path) / "eval-001.parquet"

    df = pl.read_parquet(path)
    assert df.schema == EVAL_SCHEMA
    assert df.height == 2
    assert df["objective"].to_list() == [5.7, 4.1]


def test_write_eval_results_glob_across_runs(tmp_path: Path) -> None:
    for run_id in ("eval-001", "eval-002"):
        write_eval_results(
            tmp_path,
            eval_run_id=run_id,
            rows=[
                {
                    "method": "random",
                    "instance_id": "tsp20_00",
                    "seed": 0,
                    "lns_step": 0,
                    "objective": 1.0,
                    "time_s": 0.0,
                    "initial_objective": 1.0,
                }
            ],
        )

    df = pl.read_parquet(eval_results_dir(tmp_path) / "*.parquet")
    assert df.height == 2
