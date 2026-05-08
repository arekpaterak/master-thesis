import copy

import pytest
import torch

from core.models.attention_model.model import AttentionModel
from core.models.attention_model.module import AMModule
from core.models.attention_model.training_helpers import prepare_baseline, sync_baseline_
from core.models.trainer import Trainer
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.tsp.config import TSPConfig
from core.problems.tsp.sampler import TSPSampler


@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
def test_train_one_epoch(problem: str):
    if problem == "tsp":
        sampler = TSPSampler(TSPConfig(num_locations=10), seed=0)
    else:
        sampler = CVRPSampler(CVRPConfig.for_n(20), seed=0)

    model = AttentionModel(problem_name=problem, hidden_dim=8, num_layers=1)
    attention_module = AMModule(model)
    optimizer = torch.optim.AdamW(attention_module.parameters(), lr=1e-4, weight_decay=0.0)
    params_before = copy.deepcopy([p.detach().clone() for p in model.parameters()])

    Trainer(
        max_epochs=1,
        training_batches_per_epoch=1,
        batch_size=8,
    ).fit(model=attention_module, optimizer=optimizer, instance_sampler=sampler)

    params_after = list(model.parameters())
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(params_before, params_after)
    ), "Expected at least one parameter to change after training."


def test_sync_baseline_updates_in_place():
    model = AttentionModel(problem_name="tsp", hidden_dim=8, num_layers=1)
    baseline = prepare_baseline(model)
    baseline_id = id(baseline)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    sync_baseline_(baseline, model)

    assert id(baseline) == baseline_id
    for baseline_param, model_param in zip(baseline.parameters(), model.parameters()):
        assert torch.equal(baseline_param, model_param)
        assert not baseline_param.requires_grad
