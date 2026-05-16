import pytest

from core.models.attention_model.model import AttentionModel, rollout_to_solution
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.tsp import TSPConfig, TSPSampler


@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
def test_attention_model_rollout(problem: str):
    batch_size = 2
    hidden_dim = 16

    model = AttentionModel(problem_name=problem, hidden_dim=hidden_dim, num_heads=2, num_layers=1)
    if problem == "tsp":
        instance = TSPSampler(TSPConfig(num_locations=5))(batch_size)
    else:
        instance = CVRPSampler(CVRPConfig.for_n(20))(batch_size)

    result = model.rollout(instance, greedy=False)
    assert result.actions.shape[0] == batch_size
    assert result.log_p.shape == (batch_size,)
    assert isinstance(result.mean_entropy, float)
    assert isinstance(result.mean_entropy_norm, float)

    solution = rollout_to_solution(result, instance)
    objective = solution.objective()
    assert objective.shape == (batch_size,)
