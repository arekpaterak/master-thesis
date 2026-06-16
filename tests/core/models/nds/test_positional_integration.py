import torch

from core.models.embeddings.cvrp import CVRPInitialEmbedding
from core.models.embeddings.tsp import TSPInitialEmbedding
from core.models.nds.config import PositionalEncodingConfig
from core.models.nds.models.ar import AutoregressiveNDSModel
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.features import (
    CVRPInstanceFeatureExtractor,
    CVRPSolutionFeatureExtractor,
)
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.tsp import TSP, TSPSampler
from core.problems.tsp.features import (
    TSPInstanceFeatureExtractor,
    TSPSolutionFeatureExtractor,
)
from core.problems.tsp.graph_solution import TSPGraphSolution


def test_cvrp_embedding_concat_grows_projection_and_shape():
    B, N, H, dim = 3, 20, 128, 8
    instance = CVRPSampler(CVRP.for_n(N))(B)
    solution = CVRPGraphSolution.one_customer_per_route(instance)

    config = PositionalEncodingConfig(mode="concat", dim=dim)
    embedding = CVRPInitialEmbedding(hidden_dim=H, positional=config)
    # 3 content features (x, y, demand) + IPE + XPE
    assert embedding.customer_projection.in_features == 3 + 2 * dim
    assert embedding.depot_projection.in_features == 2

    instance_features = CVRPInstanceFeatureExtractor(with_positional=True)(instance)
    solution_features = CVRPSolutionFeatureExtractor(with_positional=True)(solution)
    out = embedding(instance_features, solution_features)
    assert out.shape == (B, N + 1, H)


def test_tsp_embedding_concat_grows_projection_and_shape():
    B, N, H, dim = 3, 20, 128, 8
    instance = TSPSampler(TSP(num_locations=N))(B)
    tour = torch.stack([torch.randperm(N) for _ in range(B)])
    solution = TSPGraphSolution.create_from_sequence(instance, tour)

    config = PositionalEncodingConfig(mode="concat", dim=dim)
    embedding = TSPInitialEmbedding(hidden_dim=H, positional=config)
    assert embedding.node_projection.in_features == 2 + dim  # IPE only, no XPE

    instance_features = TSPInstanceFeatureExtractor()(instance)
    solution_features = TSPSolutionFeatureExtractor(with_positional=True)(solution)
    out = embedding(instance_features, solution_features)
    assert out.shape == (B, N, H)


def test_cvrp_model_rollout_with_positional_encoding():
    B, R, N, k = 2, 4, 20, 5
    instance = CVRPSampler(CVRP.for_n(N))(B)
    solution = CVRPGraphSolution.one_customer_per_route(instance)

    model = AutoregressiveNDSModel(
        problem_name="cvrp",
        positional=PositionalEncodingConfig(mode="concat", dim=8),
    )
    model.eval()
    rollout = model.rollout(solution, rollouts_per_solution=R, deconstruction_size=k)
    assert rollout.actions.shape == (B, R, k)
