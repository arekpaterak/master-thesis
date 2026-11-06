from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from core.lns.operators.destroy.base import DestroyOperator
from core.lns.operators.destroy.selection import NodeSelection
from core.models.nds.metrics import RolloutMetricsRecorder
from core.models.nds.models import NDSModel
from core.models.rollout import Rollout
from core.problems.solution import BaseSolution


class ModelNodeDestroy[S: BaseSolution](DestroyOperator[S, NodeSelection]):
    """Destroy operator that uses an NDS model to select nodes.

    ``select`` runs the model forward pass and returns all rollouts at once as a
    :class:`NodeSelection` wrapping ``[*batch_size, R, M]`` actions. An optional
    ``recorder`` captures the rollout actions.
    """

    selection_type = NodeSelection

    def __init__(
        self,
        model: NDSModel,
        deconstruction_size: int,
        recorder: RolloutMetricsRecorder | None = None,
    ) -> None:
        self._model = model
        self._deconstruction_size = deconstruction_size
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
    def select(self, solution: S, num_rollouts: int) -> NodeSelection:
        """Return a :class:`NodeSelection` of ``[*batch_size, R, M]`` from one model forward pass."""
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
        return NodeSelection(rollout.actions)
