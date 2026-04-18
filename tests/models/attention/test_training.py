import copy

import pytest
import torch

from core.models.attention_model.model import AttentionModel
from core.models.attention_model.training import train
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.tsp.config import TSPConfig
from core.problems.tsp.sampler import TSPSampler


@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
def test_train_one_epoch(problem: str):
    if problem == "tsp":
        sampler = TSPSampler(TSPConfig(num_locations=10), seed=0)
    else:
        sampler = CVRPSampler(CVRPConfig.for_n(20), seed=0)

    model = AttentionModel(problem_name=problem, d_h=8, num_layers=1)
    params_before = copy.deepcopy([p.detach().clone() for p in model.parameters()])

    train(
        model=model,
        instance_sampler=sampler,
        n_epochs=1,
        steps_per_epoch=1,
        batch_size=8,
        lr=1e-4,
    )

    params_after = list(model.parameters())
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(params_before, params_after)
    ), "Expected at least one parameter to change after training."
