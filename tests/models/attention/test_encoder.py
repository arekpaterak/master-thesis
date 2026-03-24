import torch
from torch_geometric.data import Batch

from core.models.attention_model.encoder import Encoder
from core.models.attention_model.graph import make_complete_graph


def test_encoder_output_shape():
    batch_size = 2
    n = 5
    d_x = 2
    d_h = 16
    num_heads = 2
    num_layers = 1
    d_ff = 32

    encoder = Encoder(d_x=d_x, d_h=d_h, num_heads=num_heads, num_layers=num_layers, d_ff=d_ff)

    instances = torch.rand((batch_size, n, d_x))
    data = Batch.from_data_list([make_complete_graph(instances[b]) for b in range(batch_size)])

    output = encoder(x=data["x"], edge_index=data["edge_index"], batch=data["batch"])

    assert output.h_nodes.shape == (batch_size * n, d_h), (
        f"Expected h_nodes shape {(batch_size * n, d_h)}"
    )
    assert output.h_graph.shape == (batch_size, d_h), f"Expected h_graph shape {(batch_size, d_h)}"
