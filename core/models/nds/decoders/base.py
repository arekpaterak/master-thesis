from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from core.envs.destroy import DestroyEnv, DestroyState
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
        env: DestroyEnv,
        state: DestroyState,
        greedy: bool,
        **kwargs,
    ) -> Rollout:
        """Select ``env.horizon`` nodes starting from ``state`` and return the Rollout.

        Args:
            h_nodes: Encoded node embeddings. Shape: (B, N, D).
            seed: Random seed vector for stochastic conditioning. Shape: (B, R, seed_dim).
                Ignored by decoders built without PolyNet (``with_poly=False``).
            env: The destroy MDP; its ``horizon`` is the number of nodes to select (M).
            state: The initial state from ``env.reset``.
            greedy: If True, select deterministically; otherwise sample stochastically.
            **kwargs: Decoder-specific additional inputs.

        Returns:
            Rollout with actions (B, R, M) and log_p (B, R).
        """
        ...
