import torch

from core.models.nds.config import SplitModelParams
from core.models.nds.decoders.config import (
    SimpleOnePassDecoderConfig,
)
from core.models.nds.factory import build_model
from core.models.nds.models.split import SplitNDSModel
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.tsp import TSP, TSPSampler
from core.problems.tsp.graph_solution import TSPGraphSolution


def _cvrp_model() -> SplitNDSModel:
    return build_model(SplitModelParams(decoder=SimpleOnePassDecoderConfig()), "cvrp")


def _cvrp_solution(B: int = 4, N: int = 20) -> CVRPGraphSolution:
    instance = CVRPSampler(CVRP.for_n(N))(B)
    return CVRPGraphSolution.one_customer_per_route(instance)


def _tsp_solution(B: int = 3, N: int = 20) -> TSPGraphSolution:
    instance = TSPSampler(TSP(num_locations=N))(B)
    tour = torch.stack([torch.randperm(N) for _ in range(B)])
    return TSPGraphSolution.create_from_sequence(instance, tour)


def test_factory_builds_split_model() -> None:
    model = _cvrp_model()
    assert isinstance(model, SplitNDSModel)

    # Cache state must not leak into the persisted weights.
    assert all("instance_cache" not in k and "cached" not in k for k in model.state_dict())


def test_cvrp_rollout_shape_and_distinctness() -> None:
    R, k = 10, 15
    model = _cvrp_model().eval()
    solution = _cvrp_solution(B=4, N=20)

    rollout = model.rollout(solution, rollouts_per_solution=R, deconstruction_size=k)

    assert rollout.actions.shape == (4, R, k)
    assert rollout.log_p.shape == (4, R)
    for action_sequence in rollout.actions.reshape(-1, k):
        assert len(set(action_sequence.tolist())) == k


def test_tsp_rollout_shape() -> None:
    R, k = 4, 5
    model = build_model(SplitModelParams(decoder=SimpleOnePassDecoderConfig()), "tsp").eval()
    solution = _tsp_solution(B=3, N=20)

    rollout = model.rollout(solution, rollouts_per_solution=R, deconstruction_size=k)

    assert rollout.actions.shape == (3, R, k)


def test_cache_reused_in_eval_and_bypassed_in_train() -> None:
    model = _cvrp_model()
    solution = _cvrp_solution(B=2, N=20)

    model.eval()
    model.rollout(solution, rollouts_per_solution=2, deconstruction_size=5)
    cached = model._instance_cache
    assert cached is not None
    assert model._cached_instance is solution.instance

    # Same instance object -> cache object is reused, not recomputed.
    model.rollout(solution, rollouts_per_solution=2, deconstruction_size=5)
    assert model._instance_cache is cached

    # A different instance invalidates the cache.
    other = _cvrp_solution(B=2, N=20)
    model.rollout(other, rollouts_per_solution=2, deconstruction_size=5)
    assert model._instance_cache is not cached

    # Training mode never reads or writes the cache.
    model.train()
    sentinel = model._instance_cache
    model.rollout(solution, rollouts_per_solution=2, deconstruction_size=5)
    assert model._instance_cache is sentinel
