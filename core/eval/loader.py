from __future__ import annotations

from pathlib import Path

import yaml

from core.eval.config import EvaluationConfig


def load_evaluation_config(path: str | Path, *, root: Path | None = None) -> EvaluationConfig:
    """Load an evaluation config, expanding path-or-inline references.

    ``method.search`` and each ``problems`` entry may be either an inline mapping or a
    string path. String paths are resolved relative to ``root`` (defaults to the config
    file's directory), loaded, and spliced in before validation.
    """
    path = Path(path)
    root = root or path.parent
    raw = yaml.safe_load(path.read_text()) or {}
    raw.setdefault("name", path.stem)

    method = raw.get("method")
    if isinstance(method, dict) and "search" in method:
        method["search"] = _resolve(method["search"], root)

    if "problems" in raw:
        raw["problems"] = [_resolve(problem, root) for problem in raw["problems"]]

    return EvaluationConfig.model_validate(raw)


def _resolve(value: object, root: Path) -> object:
    """A string is a path to load; anything else is already inline."""
    if isinstance(value, str):
        return yaml.safe_load((root / value).read_text()) or {}
    return value
