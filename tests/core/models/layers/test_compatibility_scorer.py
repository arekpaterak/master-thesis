import pytest
import torch

from core.models.layers.compatibility_scorer import CompatibilityScorer

B, R, num_nodes, D = 2, 4, 5, 16


def _scorer() -> CompatibilityScorer:
    return CompatibilityScorer(hidden_dim=D, C=10.0, use_projections=False).eval()


@pytest.mark.parametrize(
    ("query_dims", "mask_shape"),
    [
        ((), (B, num_nodes)),  # attention model: exact
        ((R,), (B, R, num_nodes)),  # AR: exact
        ((R,), (B, 1, num_nodes)),  # NAR simple: broadcast over R
        ((R, 3), (B, 1, 1, num_nodes)),  # slot: broadcast over (R, M)
    ],
)
def test_forward_accepts_broadcastable_masks(query_dims, mask_shape) -> None:
    scorer = _scorer()
    h_context = torch.randn(B, *query_dims, D)
    h_nodes = torch.randn(B, num_nodes, D)
    mask = torch.zeros(mask_shape, dtype=torch.bool)
    out = scorer(context=h_context, node_embeddings=h_nodes, mask=mask)
    assert out.shape == (B, *query_dims, num_nodes)


def test_forward_rejects_wrong_key_axis() -> None:
    scorer = _scorer()
    h_context = torch.randn(B, R, D)
    h_nodes = torch.randn(B, num_nodes, D)
    # last dim 1 would otherwise silently broadcast a single mask value over all nodes.
    bad_mask = torch.zeros(B, R, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="mask's last dim must be num_nodes"):
        scorer(context=h_context, node_embeddings=h_nodes, mask=bad_mask)


def test_forward_rejects_missing_query_axis() -> None:
    scorer = _scorer()
    h_context = torch.randn(B, R, D)  # score rank 3
    h_nodes = torch.randn(B, num_nodes, D)
    # (B, num_nodes) omits the rollout axis; right-aligned broadcasting could silently
    # misread batch as the rollout axis when B == R.
    bad_mask = torch.zeros(B, num_nodes, dtype=torch.bool)
    with pytest.raises(ValueError, match="mask must have dim"):
        scorer(context=h_context, node_embeddings=h_nodes, mask=bad_mask)
