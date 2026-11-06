import pytest
import torch

from core.methods import NDSTrainedConfig
from core.models.nds.config import JointModelConfig, TrainConfig
from core.problems import make_problem_config


def test_rejects_a_model_trained_on_another_problem(tmp_path) -> None:
    train_config = TrainConfig(
        problem=make_problem_config(name="cvrp", size=10),
        model=JointModelConfig(problem_name="cvrp"),
    )
    (tmp_path / "checkpoints").mkdir()
    torch.save(
        {"config": train_config.model_dump(mode="json"), "model": {}},
        tmp_path / "checkpoints" / "epoch_0001.pt",
    )

    with pytest.raises(ValueError, match=r"model\.problem_name 'cvrp' does not match .* 'tsp'"):
        NDSTrainedConfig(run_dir=tmp_path).build(
            problem=make_problem_config(name="tsp", size=10), device=torch.device("cpu")
        )
