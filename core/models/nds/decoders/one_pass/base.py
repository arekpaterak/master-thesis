from __future__ import annotations

from abc import ABC

from core.models.nds.decoders.base import DestroyDecoder


class OnePassDestroyDecoder(DestroyDecoder, ABC):
    """Abstract base class for one-pass destroy decoders.

    Each concrete subclass implements a different strategy for selecting M nodes
    to destroy in a single forward pass, and returns a Rollout with the selected
    nodes and their log-probabilities under the model. The ``seed_dim`` and
    ``rollout`` contract is inherited from :class:`DestroyDecoder`.
    """
