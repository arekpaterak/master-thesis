from core.models.embeddings import CVRPInitialEmbedding
from core.models.nds.encoders import NDSEncoder
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.features import (
    CVRPInstanceFeatureExtractor,
    CVRPSolutionFeatureExtractor,
)
from core.problems.cvrp.solve import one_customer_per_route


def test_encoder() -> None:
    B, N = 4, 20
    sampler = CVRPSampler(CVRP.for_n(N))
    instance = sampler(B)
    solution = one_customer_per_route(instance)

    instance_features = CVRPInstanceFeatureExtractor()(instance)
    solution_features = CVRPSolutionFeatureExtractor()(solution)

    encoder = NDSEncoder(problem_name="cvrp")
    embedded_nodes = encoder.forward(instance_features, solution_features)

    assert embedded_nodes.shape == (B, N + 1, 128)


def test_initial_node_embedding() -> None:
    N = 20
    sampler = CVRPSampler(CVRP.for_n(N))
    instance = sampler(4)

    instance_features = CVRPInstanceFeatureExtractor()(instance)
    initial_embedding = CVRPInitialEmbedding(hidden_dim=128)
    initial_node_embeddings = initial_embedding.forward(instance_features)

    assert initial_node_embeddings.shape == (4, N + 1, 128)
