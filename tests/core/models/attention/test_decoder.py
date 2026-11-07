import pytest
import torch

from core.models.attention_model.decoder import Decoder
from core.problems.cvrp import CVRPInstance
from core.problems.cvrp.constructive.env import CVRPConstructiveEnv
from core.problems.tsp import TSPInstance
from core.problems.tsp.constructive.env import TSPConstructiveEnv


@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
def test_decoder_output_shape(problem: str):
    batch_size = 2
    n = 5 if problem == "tsp" else 6
    hidden_dim = 16
    num_heads = 2

    decoder = Decoder(problem_name=problem, hidden_dim=hidden_dim, num_heads=num_heads)

    h_nodes = torch.rand((batch_size, n, hidden_dim))
    h_graph = torch.rand((batch_size, hidden_dim))

    if problem == "tsp":
        env = TSPConstructiveEnv()
        instance = TSPInstance.create(locations=torch.rand(batch_size, n, 2))
    else:
        env = CVRPConstructiveEnv()
        instance = CVRPInstance.create(
            locations=torch.rand(batch_size, n, 2),
            demand=torch.tensor([[0.0, 2, 2, 2, 2, 2], [0.0, 1, 3, 2, 2, 2]]),
            capacity=torch.full((batch_size, 1), 10.0),
        )
    state = env.reset(instance)

    log_p = decoder.forward(
        node_embeddings=h_nodes,
        graph_embedding=h_graph,
        state=state,
        action_mask=env.action_mask(state),
    )
    assert log_p.shape == (batch_size, n), f"Expected log_p shape {(batch_size, n)}"

    if problem == "tsp":
        partial_tour = torch.tensor([[0, 2], [0, 2]], dtype=torch.long)
        state = env.replay(instance, partial_tour)
        log_p = decoder.forward(
            node_embeddings=h_nodes,
            graph_embedding=h_graph,
            state=state,
            action_mask=env.action_mask(state),
        )
        assert log_p.shape == (batch_size, n), f"Expected log_p shape {(batch_size, n)}"

        for t in [0, 2]:
            assert torch.all(log_p[:, t] < -1e4), (
                f"Expected visited node {t} to be masked with low logprob"
            )
