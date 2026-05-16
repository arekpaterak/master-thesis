from __future__ import annotations

import torch


def plackett_luce_log_p(log_p: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Compute per-step conditional log-probabilities for a sequence sampled without replacement.

    For each step t, computes log P(selected[t] | selected[:t]) under the model distribution
    defined by log_p. This is the exact joint log-probability decomposition for
    Gumbel-top-k sampling (Plackett-Luce model).

    Args:
        log_p: Log-probabilities over all nodes from the decoder. Shape: (B, R, N).
        selected: Indices of selected nodes. Shape: (B, R, k).

    Returns:
        Per-step conditional log-probabilities. Shape: (B, R, k).
    """
    B, R, k = selected.shape
    N = log_p.shape[-1]

    mask = torch.zeros(B, R, N, dtype=torch.bool, device=log_p.device)
    log_p_steps = []

    for t in range(k):
        remaining_log_p = log_p.masked_fill(mask, float("-inf"))
        log_normalizer = torch.logsumexp(remaining_log_p, dim=-1, keepdim=True)
        conditional_log_p = remaining_log_p - log_normalizer
        log_p_steps.append(conditional_log_p.gather(-1, selected[..., t : t + 1]).squeeze(-1))
        mask = mask.scatter(-1, selected[..., t : t + 1], True)

    return torch.stack(log_p_steps, dim=-1)  # (B, R, k)
