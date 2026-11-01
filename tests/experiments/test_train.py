import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from core.lns.config import SearchParams, StartSolutionParams
from core.models.nds.config import (
    JointModelConfig,
    LoggingConfig,
    ModelConfig,
    SplitModelConfig,
    TrainConfig,
    TrainerParams,
    ValidationConfig,
)
from core.problems import make_problem_config

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_TRAIN_SCRIPT = _PROJECT_ROOT / "experiments" / "train.py"
_RUNS_DIR = _PROJECT_ROOT / "experiments" / "runs"


def _minimal_config(model: ModelConfig) -> TrainConfig:
    return TrainConfig(
        problem=make_problem_config(name=model.problem_name, size=10),
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
            start_solution_params=StartSolutionParams(
                num_improvement_steps=1, deconstruction_size=3
            ),
        ),
        validation=ValidationConfig(
            num_instances=2,
            batch_size=2,
            plot_enabled=False,
            search=SearchParams(
                repair_type="greedy_insert",
                num_improvement_steps=1,
                num_rollouts=2,
                deconstruction_size=3,
                start_solution_params=StartSolutionParams(
                    num_improvement_steps=1, deconstruction_size=3
                ),
            ),
        ),
        logging=LoggingConfig(wandb_project=""),
    )


_CASES = [
    ("cvrp", JointModelConfig(problem_name="cvrp")),
    ("tsp", JointModelConfig(problem_name="tsp")),
    ("cvrp", SplitModelConfig(problem_name="cvrp")),
    ("tsp", SplitModelConfig(problem_name="tsp")),
]


@pytest.mark.experiment
@pytest.mark.parametrize(
    ("problem", "model"),
    _CASES,
    ids=[f"{problem}-{model.encoder_mode}" for problem, model in _CASES],
)
def test_nds_runnable_as_script(tmp_path, problem, model):
    config_file = tmp_path / f"test-train-{model.encoder_mode}.yml"
    config_file.write_text(yaml.dump(_minimal_config(model).model_dump()))
    run_dir = _RUNS_DIR / problem / config_file.stem

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(_TRAIN_SCRIPT),
                str(config_file),
                "--problem.name",
                problem,
                "--tmp",
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    assert result.returncode == 0, result.stderr
