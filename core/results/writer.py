from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl
import yaml


def eval_output_dir(base_dir: Path, suite_name: str, eval_id: str) -> Path:
    return base_dir / suite_name / eval_id


def write_csv_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    schema: dict[str, pl.DataType],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(list(rows), schema=schema).write_csv(path)
    return path


def write_config(config: Mapping[str, object], *, path: Path) -> Path:
    """Write the effective evaluation config alongside its results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False))
    return path
