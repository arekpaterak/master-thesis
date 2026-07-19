from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from core.lns.operators.destroy.base import NodeDestroyOperator
from core.models.nds.factory import NDSModel
from core.models.nds.metrics import RolloutMetricsRecorder
from core.models.rollout import Rollout
from core.problems.solution import BaseSolution


class ModelNodeDestroy[S: BaseSolution](NodeDestroyOperator[S]):
    """Destroy operator that uses an NDS model to select nodes.

    ``select_nodes`` runs the model forward pass and returns all rollouts at once
    as ``[*batch_size, R, M]``. ``__call__`` is not supported; improvement steps
    must use ``select_nodes`` directly.
    An optional ``recorder`` captures the rollout actions.
    """

    def __init__(
        self,
        model: NDSModel,
        deconstruction_size: int,
        stitch: bool = False,
        recorder: RolloutMetricsRecorder | None = None,
    ) -> None:
        self._model = model
        self._deconstruction_size = deconstruction_size
        self.stitch = stitch
        self._recorder = recorder
        self._rollout: Callable[..., Rollout] | None = None

    @contextmanager
    def search_session(self, solution: S) -> Iterator[None]:
        self._model.eval()
        with self._model.search_session(solution.instance) as rollout:
            self._rollout = rollout
            try:
                yield
            finally:
                self._rollout = None

    @torch.no_grad()
    def select_nodes(self, solution: S, num_rollouts: int) -> torch.Tensor:
        """Return ``[*batch_size, R, M]`` node indices from a single model forward pass."""
        self._model.eval()
        rollout_fn = self._rollout if self._rollout is not None else self._model.rollout
        rollout = rollout_fn(
            solution,
            rollouts_per_solution=num_rollouts,
            deconstruction_size=self._deconstruction_size,
            greedy=False,
        )
        if self._recorder is not None:
            self._recorder.record(rollout.actions)
        return rollout.actions

    def __call__(self, solution: S) -> S:
        raise NotImplementedError(
            "ModelNodeDestroy does not support __call__; "
            "improvement steps must use select_nodes directly."
        )
