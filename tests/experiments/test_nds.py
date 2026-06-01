import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from core.lns.config import SearchParams
from core.models.nds.config import (
    LoggingConfig,
    ProblemParams,
    TrainConfig,
    TrainerParams,
)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_NDS_SCRIPT = _PROJECT_ROOT / "experiments" / "nds.py"

_MINIMAL_CONFIG = TrainConfig(
    problem=ProblemParams(size=10),
    trainer=TrainerParams(
        epochs=1,
        instances_per_epoch=5,
        batch_size=2,
        num_main_improvement_steps=1,
    ),
    search=SearchParams(
        num_improvement_steps=1,
        num_rollouts=2,
        deconstruction_size=3,
    ),
    logging=LoggingConfig(wandb_project=""),
)


@pytest.mark.experiment
@pytest.mark.parametrize(
    "problem",
    [
        "cvrp",
        "tsp",
    ],
)
def test_nds_runnable_as_script(tmp_path, problem):
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump(_MINIMAL_CONFIG.model_dump()))

    result = subprocess.run(
        [sys.executable, str(_NDS_SCRIPT), "--config", str(config_file), "--problem.name", problem],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
