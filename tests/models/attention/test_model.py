import pytest

from core.models.attention_model.model import AttentionModel
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.tsp import TSPConfig, TSPSampler


@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
def test_attention_model_rollout(problem: str):
    batch_size = 2
    d_h = 16

    model = AttentionModel(problem_name=problem, hidden_dim=d_h, num_heads=2, num_layers=1)
    if problem == "tsp":
        instance = TSPSampler(TSPConfig(num_locations=5))(batch_size)
    else:
        instance = CVRPSampler(CVRPConfig.for_n(20))(batch_size)

    result = model.rollout(instance, greedy=False)
    assert result.actions.shape[0] == batch_size
    assert result.log_p_of_actions.shape[0] == batch_size
    assert result.overall_log_p.shape == (batch_size,)
    assert isinstance(result.entropy, float)
    assert isinstance(result.entropy_norm, float)

    solution = result.to_solution(instance)
    objective = solution.objective()
    assert objective.shape == (batch_size,)
