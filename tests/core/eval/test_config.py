from __future__ import annotations

from pathlib import Path

import yaml

from core.eval.config import EvaluationConfig
from core.eval.problem_setting import SyntheticProblemSetting
from core.methods.random_destroy import RandomDestroyMethodConfig
from core.problems import CVRPConfig


def test_write_yaml_roundtrips_effective_config(tmp_path: Path) -> None:
    config = EvaluationConfig(
        name="smoke",
        method=RandomDestroyMethodConfig(seed=0),
        settings={
            "main": [
                SyntheticProblemSetting(
                    problem=CVRPConfig(size=20), num_test_instances=4, batch_size=2
                )
            ]
        },
    )
    path = config.write_yaml(path=tmp_path / "config.yml")

    reloaded = EvaluationConfig.model_validate(yaml.safe_load(path.read_text()))
    assert reloaded == config


def test_runs_flatten_groups_in_order() -> None:
    config = EvaluationConfig(
        name="grouped",
        method=RandomDestroyMethodConfig(seed=0),
        settings={
            "main": [SyntheticProblemSetting(problem=CVRPConfig(size=100))],
            "generalisation": [SyntheticProblemSetting(problem=CVRPConfig(size=200))],
        },
    )
    runs = config.runs()

    assert [(run.group, run.setting.label) for run in runs] == [
        ("main", "cvrp100"),
        ("generalisation", "cvrp200"),
    ]
    assert all(run.name == "grouped" and run.method == config.method for run in runs)
