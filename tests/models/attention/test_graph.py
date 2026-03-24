import torch

from core.models.attention_model.graph import make_complete_graph


def test_make_complete_graph():
    x = torch.rand((5, 2))
    data = make_complete_graph(x)

    # 5 nodes with self-connections -> 25 edges
    assert data.edge_index.shape == (2, 25), "Expected edge_index shape to be (2, 25)"
    assert torch.allclose(data.x, x), "Expected node features to be the input coordinates"


def test_make_complete_graph_no_self_connections():
    x = torch.rand((5, 2))
    data = make_complete_graph(x, self_connections=False)

    # 5 nodes without self-connections -> 20 edges
    assert data.edge_index.shape == (2, 20), "Expected edge_index shape to be (2, 20)"
