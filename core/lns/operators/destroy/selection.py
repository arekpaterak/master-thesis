from __future__ import annotations

from abc import ABC

import numpy as np
import torch


class DestroySelection(ABC):
    """What a destroy operator selects for removal.

    The typed hand-off between a destroy operator and an apply-and-repair: the operator
    emits a selection and an ``ApplyAndRepair`` consumes the matching type.
    """


class NodeSelection(DestroySelection):
    """Nodes to remove, shaped ``[*B, R, M]`` (R parallel rollouts of M nodes each)."""

    def __init__(self, nodes: torch.Tensor) -> None:
        self.nodes = nodes

    def to_numpy(self) -> np.ndarray:
        """``(G, R, M)`` int64 -- batch dims collapsed to a single leading dim G."""
        num_rollouts, deconstruction_size = self.nodes.shape[-2:]
        return (
            self.nodes.reshape(-1, num_rollouts, deconstruction_size)
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(np.int64, copy=False)
        )


class EdgeSelection(DestroySelection):
    """Edges to remove."""
