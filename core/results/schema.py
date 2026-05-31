from __future__ import annotations

import polars as pl

EVAL_SCHEMA: dict[str, pl.DataType] = {
    "method": pl.Utf8,
    "instance_id": pl.Utf8,
    "seed": pl.Int32,
    "lns_step": pl.Int32,
    "objective": pl.Float64,
    "time_s": pl.Float64,
    "initial_objective": pl.Float64,
}
