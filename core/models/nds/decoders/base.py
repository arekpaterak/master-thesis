from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from core.models.rollout import Rollout


class DestroyDecoder(nn.Module, ABC):
    """Abstract base for destroy decoders.

    A decoder turns encoded node embeddings into a :class:`Rollout` selecting
    ``deconstruction_size`` nodes to remove. Concrete subclasses differ in how they
    select nodes (one-pass over all nodes, or autoregressive step-by-step), but they
    share the ``rollout`` contract so any model can drive any decoder.
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
        action_mask: torch.Tensor | None,
        deconstruction_size: int,
        greedy: bool,
        **kwargs,
    ) -> Rollout:
        """Select ``deconstruction_size`` nodes and return the resulting Rollout.

        Args:
            h_nodes: Encoded node embeddings. Shape: (B, N, D).
            seed: Random seed vector for stochastic conditioning. Shape: (B, R, seed_dim).
            action_mask: Boolean mask where True marks forbidden nodes, shape (B, N).
                Autoregressive decoders rebuild the (problem-specific, transitionable)
                initial state internally and may ignore this argument.
            deconstruction_size: Number of nodes to select (k).
            greedy: If True, select deterministically; otherwise sample stochastically.
            **kwargs: Decoder-specific additional inputs.

        Returns:
            Rollout with actions (B, R, k) and log_p (B, R).
        """
        ...
