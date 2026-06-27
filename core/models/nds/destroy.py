from __future__ import annotations

import torch

from core.lns.operators.destroy.base import NodeDestroyOperator
from core.models.nds.factory import NDSModel
from core.problems.solution import BaseSolution


class ModelNodeDestroy[S: BaseSolution](NodeDestroyOperator[S]):
    """Destroy operator that uses an NDS model to select nodes.

    ``select_nodes`` runs the model forward pass and returns all rollouts at once
    as ``[*batch_size, R, M]``. ``__call__`` is not supported; improvement steps
    must use ``select_nodes`` directly.
    """

    def __init__(
        self,
        model: NDSModel,
        deconstruction_size: int,
        stitch: bool = False,
    ) -> None:
        self._model = model
        self._deconstruction_size = deconstruction_size
        self.stitch = stitch

    @torch.no_grad()
    def select_nodes(self, solution: S, num_rollouts: int) -> torch.Tensor:
        """Return ``[*batch_size, R, M]`` node indices from a single model forward pass."""
        self._model.eval()
        rollout = self._model.rollout(
            solution,
            rollouts_per_solution=num_rollouts,
            deconstruction_size=self._deconstruction_size,
            greedy=False,
        )
        return rollout.actions

    def __call__(self, solution: S) -> S:
        raise NotImplementedError(
            "ModelNodeDestroy does not support __call__; "
            "improvement steps must use select_nodes directly."
        )
