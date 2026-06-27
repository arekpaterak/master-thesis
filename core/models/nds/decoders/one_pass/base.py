from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from core.models.rollout import Rollout


class OnePassDestroyDecoder(nn.Module, ABC):
    """Abstract base class for one-pass destroy decoders.

    Each concrete subclass implements a different strategy for selecting k nodes
    to destroy in a single forward pass, and returns a Rollout with the selected
    nodes and their log-probabilities under the model.
    """

    @property
    @abstractmethod
    def seed_dim(self) -> int:
        """Dimensionality of the seed vector expected by this decoder."""
        ...

    @abstractmethod
    def rollout(
        self,
        *,
        h_nodes: torch.Tensor,
        seed: torch.Tensor,
        action_mask: torch.Tensor,
        deconstruction_size: int,
        greedy: bool,
        **kwargs,
    ) -> Rollout:
        """Select deconstruction_size nodes and return the resulting Rollout.

        Args:
            h_nodes: Encoded node embeddings. Shape: (B, N, D).
            seed: Random seed vector for stochastic conditioning. Shape: (B, R, seed_dim).
            action_mask: Boolean mask where True marks forbidden nodes. Shape: (B, N).
            deconstruction_size: Number of nodes to select (k).
            greedy: If True, select deterministically; otherwise sample stochastically.
            **kwargs: Decoder-specific additional inputs.

        Returns:
            Rollout with actions (B, R, k) and log_p (B, R).
        """
        ...
