from __future__ import annotations

import types
from collections.abc import Sequence
from pathlib import Path
from typing import Self, Union, get_args, get_origin

import polars as pl
from pydantic import BaseModel

_PL_BY_TYPE: dict[type, pl.DataType] = {
    int: pl.Int64,
    float: pl.Float64,
    str: pl.Utf8,
    bool: pl.Boolean,
}


def _pl_dtype(annotation: object) -> pl.DataType:
    """Map a (possibly Optional) row-field annotation to a polars dtype."""
    if get_origin(annotation) in (Union, types.UnionType):
        annotation = next(arg for arg in get_args(annotation) if arg is not type(None))
    return _PL_BY_TYPE[annotation]


class Row(BaseModel):
    model_config = {"frozen": True}

    @classmethod
    def polars_schema(cls) -> dict[str, pl.DataType]:
        return {name: _pl_dtype(field.annotation) for name, field in cls.model_fields.items()}

    @classmethod
    def write(cls, rows: Sequence[Self], *, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame([row.model_dump() for row in rows], schema=cls.polars_schema())
        frame.write_csv(path)
        return path


class SummaryRowBase(Row):
    config: str
    method_type: str
    setting: str
    group: str
    problem_kind: str
    checkpoint_epoch: int | None
    n_instances: int
    total_time_s: float
