import pytest
import torch

from core.models.attention_model.decoder import Decoder
from core.states.constructive import CVRPConstructiveState, TSPConstructiveState


@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
def test_decoder_output_shape(problem: str):
    batch_size = 2
    n = 5 if problem == "tsp" else 6
    d_h = 16
    num_heads = 2

    decoder = Decoder(hidden_dim=d_h, num_heads=num_heads)

    h_nodes = torch.rand((batch_size, n, d_h))
    h_graph = torch.rand((batch_size, d_h))

    if problem == "tsp":
        state = TSPConstructiveState.initial(
            batch_size=batch_size,
            num_nodes=n,
            device=h_nodes.device,
        )
    else:
        demand = torch.tensor(
            [[0.0, 0.2, 0.2, 0.2, 0.2, 0.2], [0.0, 0.1, 0.3, 0.2, 0.2, 0.2]],
            dtype=torch.float,
        )
        capacity = torch.ones(batch_size, 1)
        state = CVRPConstructiveState.initial(demand=demand, capacity=capacity)

    log_p = decoder.forward(node_embeddings=h_nodes, graph_embedding=h_graph, state=state)
    assert log_p.shape == (batch_size, n), f"Expected log_p shape {(batch_size, n)}"

    if problem == "tsp":
        partial_tour = torch.tensor([[0, 2], [0, 2]], dtype=torch.long)
        state = TSPConstructiveState.from_partial_tour(partial_tour=partial_tour, num_nodes=n)
        log_p = decoder.forward(node_embeddings=h_nodes, graph_embedding=h_graph, state=state)
        assert log_p.shape == (batch_size, n), f"Expected log_p shape {(batch_size, n)}"

        for t in [0, 2]:
            assert torch.all(log_p[:, t] < -1e4), (
                f"Expected visited node {t} to be masked with low logprob"
            )
