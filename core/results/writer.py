from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl

from core.results.schema import EVAL_SCHEMA


def eval_results_dir(run_dir: Path) -> Path:
    return run_dir / "results" / "eval"


def write_eval_results(
    run_dir: Path,
    *,
    eval_run_id: str,
    rows: Iterable[Mapping[str, object]],
) -> Path:
    """Write eval rows (one parquet per eval run) under the shared EVAL_SCHEMA."""
    df = pl.DataFrame(list(rows), schema=EVAL_SCHEMA)
    out_dir = eval_results_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{eval_run_id}.parquet"
    df.write_parquet(path)
    return path
