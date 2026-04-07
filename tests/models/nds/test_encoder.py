from core.models.nds.encoder import CVRPInitialNodeEmbedding, NDSEncoder
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.cvrp.solve import one_customer_per_route


def test_encoder() -> None:
    B, N = 4, 20
    sampler = CVRPSampler(CVRPConfig.for_n(N))
    instance = sampler(B)
    solution = one_customer_per_route(instance)

    neighbours_index, tour_index = solution.neighbours_per_customer(), solution.route_per_customer()

    encoder = NDSEncoder()
    embedded_nodes = encoder.forward(
        depot_location=instance.depot_location,
        customer_features=instance.customer_features,
        neighbours_index=neighbours_index,
        tour_index=tour_index,
    )

    assert embedded_nodes.shape == (B, N + 1, 128)


def test_initial_node_embedding() -> None:
    N = 20
    sampler = CVRPSampler(CVRPConfig.for_n(N))
    instance = sampler(4)

    initial_embedding = CVRPInitialNodeEmbedding(d_h=128)
    initial_node_embeddings = initial_embedding(instance.depot_location, instance.customer_features)

    assert initial_node_embeddings.shape == (4, N + 1, 128)
