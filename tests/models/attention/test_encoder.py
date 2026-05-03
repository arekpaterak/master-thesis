import torch
from torch_geometric.data import Batch

from core.models.attention_model.encoder import Encoder
from core.models.attention_model.preprocessing import make_complete_graph


def test_encoder_output_shape():
    batch_size = 2
    n = 5
    d_x = 2
    hidden_dim = 16
    num_heads = 2
    num_layers = 1
    ff_dim = 32

    encoder = Encoder(
        hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim
    )

    node_features = torch.rand((batch_size, n, d_x))
    data = Batch.from_data_list([make_complete_graph(node_features[b]) for b in range(batch_size)])

    output = encoder(
        edge_index=data["edge_index"],
        batch=data["batch"],
        node_features=node_features,
    )

    assert output.h_nodes.shape == (batch_size * n, hidden_dim), (
        f"Expected h_nodes shape {(batch_size * n, hidden_dim)}"
    )
    assert output.h_graph.shape == (batch_size, hidden_dim), (
        f"Expected h_graph shape {(batch_size, hidden_dim)}"
    )
