from __future__ import annotations

import torch


def log_sinkhorn(
    log_scores: torch.Tensor,
    *,
    iterations: int,
    tau: float,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Couple M slots to N nodes via Sinkhorn normalization in log-space.

    Alternates row and column normalization of the ``M x N`` score matrix so slots
    compete for distinct nodes. Rows (slots) are normalized to sum to 1; columns
    (nodes) are pushed toward a uniform marginal ``M / N_valid``, where ``N_valid``
    is the number of unmasked nodes. The final step is a row normalization, so each
    returned row is a valid categorical log-distribution over nodes.

    Working in log-space (``logsumexp`` instead of ``exp`` + ``sum``) keeps the
    iteration stable for small ``tau``.

    Args:
        log_scores: Raw (unnormalized) logits, (B, R, M, N). Masked nodes must be -inf.
        iterations: Number of Sinkhorn row/column iterations.
        tau: Temperature; smaller values sharpen toward a hard assignment.
        node_mask: Boolean mask broadcastable to (B, 1, 1, N); True = masked out.

    Returns:
        Coupled per-slot log-probabilities, (B, R, M, N), with masked nodes at -inf.
    """
    num_slots = log_scores.shape[-2]
    num_valid = (~node_mask).sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1, 1, 1)
    log_col_target = torch.log(num_slots / num_valid.to(log_scores.dtype))  # (B, 1, 1, 1)

    log_K = log_scores / tau
    for _ in range(iterations):
        # Row step: each slot sums to 1 over nodes.
        log_K = log_K - log_K.logsumexp(dim=-1, keepdim=True)
        # Column step: push node marginals toward M / N_valid. Masked columns are all
        # -inf; feeding that into logsumexp gives -inf whose backward is NaN (0 * exp of
        # -inf - -inf). Replace masked entries with a finite value before the reduction,
        # then discard those columns' result so masked nodes stay at -inf downstream.
        safe_log_K = log_K.masked_fill(node_mask, 0.0)
        col_lse = safe_log_K.logsumexp(dim=-2, keepdim=True)  # (B, R, 1, N)
        col_lse = col_lse.masked_fill(node_mask, 0.0)
        log_K = log_K - col_lse + log_col_target

    # Final row normalization so each row is a valid categorical log-distribution.
    return log_K - log_K.logsumexp(dim=-1, keepdim=True)
