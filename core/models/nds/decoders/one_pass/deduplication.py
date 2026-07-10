from __future__ import annotations

import torch
import torch.nn as nn

from core.models.layers.sinkhorn import log_sinkhorn
from core.models.nds.decoders.config import DeduplicationConfig


class SinkhornDeduplicator(nn.Module):
    """Sinkhorn-normalizes the slot-node score matrix so slots compete for distinct nodes.

    Maps raw masked logits (B, R, M, N) to coupled per-slot log-probabilities of the
    same shape (rows sum to 1). ``node_mask`` is broadcastable to (B, 1, 1, N); True =
    masked out.
    """

    def __init__(self, *, iterations: int, tau: float):
        super().__init__()
        self.iterations = iterations
        self.tau = tau

    def forward(self, raw_logits: torch.Tensor, *, node_mask: torch.Tensor) -> torch.Tensor:
        return log_sinkhorn(
            raw_logits, iterations=self.iterations, tau=self.tau, node_mask=node_mask
        )


def build_deduplicator(config: DeduplicationConfig) -> SinkhornDeduplicator | None:
    """Instantiate the slot deduplicator for ``config``, or None (no coupling) if it is None."""
    if config is None:
        return None
    return SinkhornDeduplicator(iterations=config.iterations, tau=config.tau)
