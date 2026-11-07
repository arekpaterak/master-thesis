from __future__ import annotations

import torch

from core.models.nds.models import NDSModel
from core.models.nds.models.split import SplitNDSModel
from core.models.rollout import Rollout
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


class ModelPass:
    """Strategy for producing per-improvement-step rollouts within a training step.

    Subclasses decide whether the instance encoder runs once or every step; the loop
    drives them uniformly: ``rollout`` each step, ``after_backward`` after every
    ``loss.backward``, and ``finish`` once after the loop.
    """

    def rollout(self, solution: BaseSolution, **kwargs) -> Rollout:
        raise NotImplementedError

    def after_backward(self) -> None:
        pass

    def finish(self) -> None:
        pass


class RecomputePass(ModelPass):
    """Re-encode the instance every improvement step (rebuilds the graph each step)."""

    def __init__(self, model: NDSModel) -> None:
        self._model = model

    def rollout(self, solution: BaseSolution, **kwargs) -> Rollout:
        return self._model.rollout(solution, **kwargs)


class EncodeOncePass(ModelPass):
    """Encode the instance once and reuse it across improvement steps.

    The encoder output is detached into a leaf so each step's backward stays bounded;
    its gradient is accumulated and pushed through the encoder once in ``finish``. This
    is exact: the split encoder is a deterministic function of the instance.
    """

    def __init__(self, model: SplitNDSModel, instance: BaseProblemInstance) -> None:
        self._model = model
        base_cache = model.encode_instance(instance)
        self._embeddings = base_cache.instance_node_embeddings
        self._embeddings_detached = self._embeddings.detach().requires_grad_(True)
        self._cache = base_cache._replace(instance_node_embeddings=self._embeddings_detached)
        self._accumulated_gradient = torch.zeros_like(self._embeddings)

    def rollout(self, solution: BaseSolution, **kwargs) -> Rollout:
        return self._model.rollout(solution, cache=self._cache, **kwargs)

    def after_backward(self) -> None:
        self._accumulated_gradient += self._embeddings_detached.grad
        self._embeddings_detached.grad = None

    def finish(self) -> None:
        """One encoder backward with the summed boundary gradient."""
        self._embeddings.backward(gradient=self._accumulated_gradient)
