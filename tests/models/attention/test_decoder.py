import torch

from core.models.attention.decoder import Decoder


def test_decoder_output_shape():
    batch_size = 2
    n = 5
    d_h = 16
    num_heads = 2

    decoder = Decoder(d_h=d_h, num_heads=num_heads)

    h_nodes = torch.rand((batch_size, n, d_h))
    h_graph = torch.rand((batch_size, d_h))

    # Empty tour
    tour = []
    log_p = decoder(h_nodes=h_nodes, h_graph=h_graph, tour_idx=tour)
    assert log_p.shape == (batch_size, n), f"Expected log_p shape {(batch_size, n)}"

    # Partial tour
    tour = [0, 2]
    log_p = decoder(h_nodes=h_nodes, h_graph=h_graph, tour_idx=tour)
    assert log_p.shape == (batch_size, n), f"Expected log_p shape {(batch_size, n)}"

    # Masked nodes should have probability close to 0 (logprob close to -inf)
    for t in tour:
        assert torch.all(log_p[:, t] < -1e4), (
            f"Expected visited node {t} to be masked with low logprob"
        )
