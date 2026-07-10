import torch

from core.models.layers.sinkhorn import log_sinkhorn

B, R, M, N = 2, 3, 4, 12


def _node_mask(masked_nodes=()):
    mask = torch.zeros(B, 1, 1, N, dtype=torch.bool)
    for node in masked_nodes:
        mask[..., node] = True
    return mask


def test_shape_and_rows_sum_to_one():
    scores = torch.randn(B, R, M, N)
    log_p = log_sinkhorn(scores, iterations=20, tau=0.1, node_mask=_node_mask())

    assert log_p.shape == (B, R, M, N)
    row_sums = log_p.logsumexp(dim=-1)  # (B, R, M)
    assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-5)


def test_column_marginals_approach_uniform():
    scores = torch.randn(B, R, M, N)
    log_p = log_sinkhorn(scores, iterations=200, tau=0.1, node_mask=_node_mask())

    # Column marginals (mass per node summed over slots) should approach M / N.
    col_mass = log_p.exp().sum(dim=-2)  # (B, R, N)
    expected = torch.full_like(col_mass, M / N)
    assert torch.allclose(col_mass, expected, atol=1e-2)


def test_masked_nodes_get_no_mass():
    scores = torch.randn(B, R, M, N)
    node_mask = _node_mask(masked_nodes=(0, 5))
    scores = scores.masked_fill(node_mask, float("-inf"))

    log_p = log_sinkhorn(scores, iterations=50, tau=0.1, node_mask=node_mask)

    assert torch.isneginf(log_p[..., 0]).all()
    assert torch.isneginf(log_p[..., 5]).all()
    unmasked = log_p[..., [i for i in range(N) if i not in (0, 5)]]
    assert torch.isfinite(unmasked).all()
    # Masked columns are excluded, marginals approach M / N_valid over the rest.
    col_mass = log_p.exp().sum(dim=-2)  # (B, R, N)
    assert torch.allclose(col_mass[..., 0], torch.zeros(B, R), atol=1e-6)


def test_backward_is_finite_with_masked_nodes():
    # A masked node's column is all -inf; the column logsumexp must not produce NaN
    # gradients in the backward pass (0 * exp(-inf - -inf)).
    node_mask = _node_mask(masked_nodes=(0, 5))
    scores = torch.randn(B, R, M, N).masked_fill(node_mask, float("-inf"))
    scores.requires_grad_(True)

    log_p = log_sinkhorn(scores, iterations=20, tau=0.05, node_mask=node_mask)
    log_p.gather(-1, log_p.argmax(dim=-1, keepdim=True)).sum().backward()

    assert torch.isfinite(scores.grad[~node_mask.expand_as(scores)]).all()


def test_reduces_collisions_vs_independent_softmax():
    # Every slot most prefers node 3, so independent softmax collides there. Each slot
    # has a distinct second preference (node 4 + i), so a distinct assignment exists and
    # Sinkhorn should recover it once node 3 can only absorb its M / N share.
    scores = torch.full((1, 1, M, N), -5.0)
    scores[..., 3] = 5.0  # shared top preference
    for i in range(M):
        scores[0, 0, i, 4 + i] = 4.0  # distinct second preference

    independent = torch.log_softmax(scores, dim=-1).argmax(dim=-1)  # (1, 1, M)
    assert independent.unique().numel() == 1  # all slots collide on node 3

    coupled = log_sinkhorn(scores, iterations=50, tau=0.05, node_mask=_node_mask())
    coupled_actions = coupled.argmax(dim=-1)  # (1, 1, M)
    assert coupled_actions.unique().numel() == M  # all distinct
