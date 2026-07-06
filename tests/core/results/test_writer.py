from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from core.results.schema import SYNTHETIC_INSTANCE_SCHEMA
from core.results.writer import eval_output_dir, write_config, write_csv_rows


def test_eval_output_dir_composes_suite_and_id(tmp_path: Path) -> None:
    assert eval_output_dir(tmp_path, "random_destroy", "20260706-120000") == (
        tmp_path / "random_destroy" / "20260706-120000"
    )


def test_write_csv_rows_roundtrip(tmp_path: Path) -> None:
    rows = [
        {
            "config": "random_destroy",
            "problem_label": "cvrp20",
            "instance_id": "cvrp20-seed7-0000",
            "seed": 7,
            "initial_objective": 5.7,
            "final_objective": 4.1,
            "improvement": 1.6,
        },
    ]
    path = write_csv_rows(rows, schema=SYNTHETIC_INSTANCE_SCHEMA, path=tmp_path / "insts.csv")

    df = pl.read_csv(path)
    assert df.columns == list(SYNTHETIC_INSTANCE_SCHEMA)
    assert df.height == 1
    assert df["improvement"].to_list() == [1.6]


def test_write_csv_rows_empty_uses_schema(tmp_path: Path) -> None:
    path = write_csv_rows([], schema=SYNTHETIC_INSTANCE_SCHEMA, path=tmp_path / "empty.csv")
    assert pl.read_csv(path).columns == list(SYNTHETIC_INSTANCE_SCHEMA)


def test_write_config_roundtrip(tmp_path: Path) -> None:
    config = {"name": "random_destroy", "method": {"type": "random-destroy", "seed": 0}}
    path = write_config(config, path=tmp_path / "config.yml")
    assert yaml.safe_load(path.read_text()) == config
