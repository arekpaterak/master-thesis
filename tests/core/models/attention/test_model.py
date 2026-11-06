import pytest

from core.models.attention_model.model import AttentionModel
from core.problems.cvrp import CVRPDistribution, CVRPSampler
from core.problems.tsp import TSPDistribution, TSPSampler
from core.random import SeededGenerator


@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
def test_attention_model_rollout(problem: str):
    batch_size = 2
    hidden_dim = 16

    model = AttentionModel(problem_name=problem, hidden_dim=hidden_dim, num_heads=2, num_layers=1)
    if problem == "tsp":
        instance = TSPSampler(TSPDistribution(num_locations=5))(batch_size)
    else:
        instance = CVRPSampler(CVRPDistribution.for_n(20))(batch_size)

    result = model.rollout(instance, greedy=False, rng=SeededGenerator(0))
    assert result.actions.shape[0] == batch_size
    assert result.log_p.shape == (batch_size,)
    assert isinstance(result.mean_entropy, float)
    assert isinstance(result.mean_entropy_norm, float)

    solution = model.env.to_solution(instance, result.actions)
    objective = solution.objective()
    assert objective.shape == (batch_size,)
