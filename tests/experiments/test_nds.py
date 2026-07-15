import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from core.lns.config import SearchParams
from core.models.nds.config import (
    JointModelConfig,
    LoggingConfig,
    ModelConfig,
    SplitModelConfig,
    TrainConfig,
    TrainerParams,
)
from core.problems import CVRPConfig

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_NDS_SCRIPT = _PROJECT_ROOT / "experiments" / "train-nds.py"


def _minimal_config(model: ModelConfig) -> TrainConfig:
    return TrainConfig(
        problem=CVRPConfig(size=10),
        model=model,
        trainer=TrainerParams(
            epochs=1,
            instances_per_epoch=5,
            batch_size=2,
            num_main_improvement_steps=1,
        ),
        search=SearchParams(
            repair_type="greedy_insert",
            num_improvement_steps=1,
            num_rollouts=2,
            deconstruction_size=3,
        ),
        logging=LoggingConfig(wandb_project=""),
    )


_CASES = [
    ("cvrp", JointModelConfig()),
    ("tsp", JointModelConfig()),
    ("cvrp", SplitModelConfig()),
    ("tsp", SplitModelConfig()),
]


@pytest.mark.experiment
@pytest.mark.parametrize(
    ("problem", "model"),
    _CASES,
    ids=[f"{problem}-{model.encoder_mode}" for problem, model in _CASES],
)
def test_nds_runnable_as_script(tmp_path, problem, model):
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump(_minimal_config(model).model_dump()))

    result = subprocess.run(
        [sys.executable, str(_NDS_SCRIPT), "--config", str(config_file), "--problem.name", problem],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
