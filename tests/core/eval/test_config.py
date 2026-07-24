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
        problems=[
            SyntheticProblemSetting(problem=CVRPConfig(size=20), num_test_instances=4, batch_size=2)
        ],
    )
    path = config.write_yaml(path=tmp_path / "config.yml")

    reloaded = EvaluationConfig.model_validate(yaml.safe_load(path.read_text()))
    assert reloaded == config
