from __future__ import annotations

import torch
import torch.nn.functional as F


def log_prob_of_sequence_without_replacement(
    logits: torch.Tensor, selected: torch.Tensor
) -> torch.Tensor:
    """Compute the joint log-probability of a sequence sampled without replacement.

    For each step t, computes log P(selected[t] | selected[:t]) and accumulates the sum,
    giving the joint log-probability of the full sequence.

    Args:
        logits: Logits for all nodes from the decoder. Shape: (..., N).
        selected: Indices of selected nodes. Shape: (..., M).

    Returns:
        Joint log-probability of the selected sequence. Shape: (...).
    """
    M = selected.shape[-1]

    log_p = torch.zeros(*logits.shape[:-1], device=logits.device)
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)

    for m_idx in range(M):
        masked_logits = logits.masked_fill(mask, float("-inf"))
        log_p += (
            F.log_softmax(masked_logits, dim=-1)
            .gather(-1, selected[..., m_idx : m_idx + 1])
            .squeeze(-1)
        )
        mask = mask.scatter(-1, selected[..., m_idx : m_idx + 1], True)

    return log_p
