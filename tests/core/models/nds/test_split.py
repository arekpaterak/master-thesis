import torch

from core.models.nds.config import SplitModelConfig
from core.models.nds.decoders.config import (
    SimpleOnePassDecoderConfig,
)
from core.models.nds.encoders.config import PositionalEncodingConfig
from core.models.nds.factory import build_model
from core.models.nds.models.split import SplitNDSModel
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.tsp import TSP, TSPSampler
from core.problems.tsp.graph_solution import TSPGraphSolution


def _cvrp_model() -> SplitNDSModel:
    return build_model(SplitModelConfig(decoder=SimpleOnePassDecoderConfig()), "cvrp")


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
    R, M = 10, 15
    model = _cvrp_model().eval()
    solution = _cvrp_solution(B=4, N=20)

    rollout = model.rollout(solution, rollouts_per_solution=R, deconstruction_size=M)

    assert rollout.actions.shape == (4, R, M)
    assert rollout.log_p.shape == (4, R)
    for action_sequence in rollout.actions.reshape(-1, M):
        assert len(set(action_sequence.tolist())) == M


def test_tsp_rollout_shape() -> None:
    R, M = 4, 5
    model = build_model(SplitModelConfig(decoder=SimpleOnePassDecoderConfig()), "tsp").eval()
    solution = _tsp_solution(B=3, N=20)

    rollout = model.rollout(solution, rollouts_per_solution=R, deconstruction_size=M)

    assert rollout.actions.shape == (3, R, M)


def test_fusion_route_encoding_defaults_to_problem() -> None:
    # None resolves to a problem-specific default: on for CVRP, off for TSP.
    cvrp = build_model(SplitModelConfig(decoder=SimpleOnePassDecoderConfig()), "cvrp")
    tsp = build_model(SplitModelConfig(decoder=SimpleOnePassDecoderConfig()), "tsp")

    assert cvrp.encoder.fusion.route is not None
    assert tsp.encoder.fusion.route is None


def test_positional_encoding_can_be_disabled_with_structural_layers_on() -> None:
    # No positional encoding, but neighbours/route encoding still active: the solution
    # must reach the model purely through the structural fusion layers.
    model = build_model(
        SplitModelConfig(
            decoder=SimpleOnePassDecoderConfig(),
            positional_encoding=None,
            with_fusion_neighbours_encoding=True,
            with_fusion_route_encoding=True,
        ),
        "cvrp",
    ).eval()

    assert (
        model.encoder.fusion.projection.in_features == model.encoder.fusion.projection.out_features
    )
    assert model.encoder.fusion.neighbours is not None
    assert model.encoder.fusion.route is not None

    solution = _cvrp_solution(B=2, N=20)
    rollout = model.rollout(solution, rollouts_per_solution=3, deconstruction_size=5)
    assert rollout.actions.shape == (2, 3, 5)


def test_positional_encoding_enabled_widens_fusion_projection() -> None:
    model = build_model(
        SplitModelConfig(
            decoder=SimpleOnePassDecoderConfig(),
            positional_encoding=PositionalEncodingConfig(dim=16),
        ),
        "cvrp",
    )

    # CVRP fuses IPE + XPE, so the projection input is hidden_dim + 2 * pe_dim.
    default_hidden_dim = SplitModelConfig().hidden_dim
    assert model.encoder.fusion.projection.in_features == default_hidden_dim + 2 * 16


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
