from collections.abc import Callable

import torch

from core.models.nds.config import SplitModelConfig
from core.models.nds.decoders.config import (
    SimpleOnePassDecoderConfig,
)
from core.models.nds.destroy import ModelNodeDestroy
from core.models.nds.encoders.config import PositionalEncodingConfig, SplitEncoderConfig
from core.models.nds.factory import build_model
from core.models.nds.models.split import SplitNDSModel
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.tsp import TSP, TSPSampler
from core.problems.tsp.graph_solution import TSPGraphSolution


def _cvrp_model() -> SplitNDSModel:
    return build_model(SplitModelConfig(decoder=SimpleOnePassDecoderConfig()), "cvrp")


def _count_encode(model: SplitNDSModel) -> Callable[[], int]:
    """Patch the encoder to count how many times the instance is encoded."""
    original = model.encoder.encode_instance
    count = 0

    def counting(instance):
        nonlocal count
        count += 1
        return original(instance)

    model.encoder.encode_instance = counting  # type: ignore[method-assign]
    return lambda: count


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

    # Session cache state must not leak into the persisted weights.
    assert all("session_cache" not in k and "cache" not in k for k in model.state_dict())


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


def test_route_encoding_defaults_to_problem() -> None:
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
            encoder=SplitEncoderConfig(with_neighbours_encoding=True, with_route_encoding=True),
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


def test_search_session_binds_cache_and_encodes_once() -> None:
    model = _cvrp_model().eval()
    solution = _cvrp_solution(B=2, N=20)
    encode_count = _count_encode(model)

    with model.search_session(solution.instance) as step:
        step(solution, rollouts_per_solution=2, deconstruction_size=5)
        step(solution, rollouts_per_solution=2, deconstruction_size=5)

    # Encoded once on session enter and reused for both rollouts via the bound callable.
    assert encode_count() == 1

    # The model holds no ambient cache: a bare rollout after the session recomputes.
    model.rollout(solution, rollouts_per_solution=2, deconstruction_size=5)
    assert encode_count() == 2


def test_rollout_recomputes_without_session() -> None:
    model = _cvrp_model().eval()
    solution = _cvrp_solution(B=2, N=20)
    encode_count = _count_encode(model)

    # No active session and no explicit cache: pure rollout recomputes each call.
    model.rollout(solution, rollouts_per_solution=2, deconstruction_size=5)
    model.rollout(solution, rollouts_per_solution=2, deconstruction_size=5)
    assert encode_count() == 2


def test_model_node_destroy_session_encodes_once() -> None:
    # The search-time path: ModelNodeDestroy.search_session binds the model's rollout
    # callable, so repeated select_nodes over one instance encode only once.
    model = _cvrp_model()
    destroy = ModelNodeDestroy(model=model, deconstruction_size=5)
    solution = _cvrp_solution(B=2, N=20)
    encode_count = _count_encode(model)

    with destroy.search_session(solution):
        assert destroy._rollout is not None
        destroy.select_nodes(solution, num_rollouts=2)
        destroy.select_nodes(solution, num_rollouts=2)

    assert encode_count() == 1
    assert destroy._rollout is None
