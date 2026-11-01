import torch

from core.models.attention_model.encoder import Encoder
from core.problems.tsp.features import TSPInstanceFeatures


def _encoder(hidden_dim: int = 16) -> Encoder:
    return Encoder(hidden_dim=hidden_dim, num_heads=2, num_layers=2, ff_dim=32)


def _features(locations: torch.Tensor) -> TSPInstanceFeatures:
    return TSPInstanceFeatures(node_locations=locations, batch_size=(locations.shape[0],))


def test_encoder_output_shape():
    batch_size, n, hidden_dim = 2, 5, 16
    encoder = _encoder(hidden_dim)

    output = encoder(_features(torch.rand((batch_size, n, 2))))

    assert output.h_nodes.shape == (batch_size, n, hidden_dim)
    assert output.h_graph.shape == (batch_size, hidden_dim)
    torch.testing.assert_close(output.h_graph, output.h_nodes.mean(dim=1))


def test_encoder_attends_within_instance_only():
    torch.manual_seed(0)
    encoder = _encoder().eval()  # running BatchNorm statistics: no coupling through the batch
    locations = torch.rand((3, 7, 2))

    with torch.no_grad():
        batched = encoder(_features(locations)).h_nodes
        single = torch.cat([encoder(_features(locations[b : b + 1])).h_nodes for b in range(3)])

    torch.testing.assert_close(batched, single)
