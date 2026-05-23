from __future__ import annotations

import torch
import torch.nn.functional as F


def log_prob_of_sequence_without_replacement(
    logits: torch.Tensor, selected: torch.Tensor
) -> torch.Tensor:
    """Compute per-step conditional log-probabilities for a sequence sampled without replacement.

    For each step t, computes log P(selected[t] | selected[:t]) under the model distribution
    defined by log_p.

    Args:
        logits: Logits for all nodes from the decoder. Shape: (..., N).
        selected: Indices of selected nodes. Shape: (..., k).

    Returns:
        Per-step conditional log-probabilities. Shape: (..., k).
    """
    k = selected.shape[-1]

    log_p = torch.zeros(*logits.shape[:-1], device=logits.device)
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)

    for k_idx in range(k):
        masked_logits = logits.masked_fill(mask, float("-inf"))
        log_p += (
            F.log_softmax(masked_logits, dim=-1)
            .gather(-1, selected[..., k_idx : k_idx + 1])
            .squeeze(-1)
        )
        mask = mask.scatter(-1, selected[..., k_idx : k_idx + 1], True)

    return log_p
