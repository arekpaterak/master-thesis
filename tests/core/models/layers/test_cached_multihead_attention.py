import pytest
import torch

from core.models.layers.cached_multihead_attention import CachedMultiheadAttention

B, R, L_k, D, num_heads = 2, 4, 5, 16, 8


def _build() -> tuple[CachedMultiheadAttention, torch.Tensor, torch.Tensor, torch.Tensor]:
    mha = CachedMultiheadAttention(embed_dim=D, num_heads=num_heads).eval()
    q = torch.randn(B, R, D)
    h_nodes = torch.randn(B, L_k, D)
    k, v = mha.project_kv(h_nodes)
    return mha, q, k, v


def test_forward_with_valid_mask_returns_query_shape() -> None:
    mha, q, k, v = _build()
    mask = torch.ones(B, R, L_k, dtype=torch.bool)
    out = mha(q, k, v, attention_mask=mask)
    assert out.shape == (B, R, D)


def test_forward_rejects_wrong_mask_shape() -> None:
    mha, q, k, v = _build()
    # (B, 1, L_k) would otherwise silently broadcast one mask across all rollouts.
    bad_mask = torch.ones(B, 1, L_k, dtype=torch.bool)
    with pytest.raises(ValueError, match="attention_mask must have shape"):
        mha(q, k, v, attention_mask=bad_mask)
