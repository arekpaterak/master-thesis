import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_NDS_SCRIPT = _PROJECT_ROOT / "experiments" / "nds.py"

_MINIMAL_CONFIG = {
    "problem": {"name": "cvrp", "size": 10},
    "model": {"hidden_dim": 32, "num_heads": 4, "ff_dim": 64, "poly_dim": 32, "seed_dim": 4},
    "trainer": {
        "epochs": 1,
        "batches_per_epoch": 1,
        "batch_size": 2,
        "rollouts_per_instance": 2,
        "num_initial_improvement_steps": 0,
        "num_main_improvement_steps": 1,
        "deconstruction_size": 3,
    },
    "logging": {"wandb_project": ""},
}


@pytest.mark.experiment
def test_nds_runnable(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump(_MINIMAL_CONFIG))

    result = subprocess.run(
        [sys.executable, str(_NDS_SCRIPT), "--config", str(config_file)],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
