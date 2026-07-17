from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


def diversity_overlap(actions: torch.Tensor) -> float:
    """Mean node overlap across the ``R`` rollouts of each instance, in ``[0, 1]``."""
    flat = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    num_rollouts = flat.shape[-2]
    if num_rollouts < 2:
        return 0.0

    num_nodes = int(flat.max()) + 1
    presence = torch.zeros(
        flat.shape[0], num_rollouts, num_nodes, dtype=torch.bool, device=flat.device
    )
    presence.scatter_(-1, flat, True)

    rollouts_per_node = presence.sum(dim=-2, keepdim=True)
    others = (presence * (rollouts_per_node - 1)).sum(dim=(-2, -1)).float()
    memberships = presence.sum(dim=(-2, -1)).float()
    overlap = torch.where(
        memberships > 0,
        others / (memberships * (num_rollouts - 1)),
        torch.zeros_like(memberships),
    )
    return overlap.mean().item()


def unique_rollouts(actions: torch.Tensor) -> float:
    """Fraction of the ``R`` rollouts with a distinct node-set, in ``[0, 1]``."""
    flat = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    num_rollouts = flat.shape[-2]
    if num_rollouts < 2:
        return 1.0

    sorted_actions = flat.sort(dim=-1).values
    equal = (sorted_actions.unsqueeze(-2) == sorted_actions.unsqueeze(-3)).all(dim=-1)
    earlier = torch.tril(
        torch.ones(num_rollouts, num_rollouts, dtype=torch.bool, device=flat.device),
        diagonal=-1,
    )
    is_duplicate = (equal & earlier).any(dim=-1)
    num_distinct = (~is_duplicate).sum(dim=-1).float()
    return (num_distinct / num_rollouts).mean().item()


def mean_pairwise_cosine_similarity(embeddings: torch.Tensor) -> float:
    """Mean cosine similarity over distinct node pairs of each instance, in ``[-1, 1]``."""
    flat = embeddings.reshape(-1, embeddings.shape[-2], embeddings.shape[-1])
    num_nodes = flat.shape[-2]
    if num_nodes < 2:
        return 0.0

    # For unit vectors sum_ij <u_i, u_j> = ||sum_i u_i||^2, so the off-diagonal sum needs no
    # (N, N) gram matrix. The subtracted term is sum_i ||u_i||^2 (the count of non-zero rows),
    # which keeps a zero embedding -- normalized to zero -- contributing nothing.
    normalized = F.normalize(flat.float(), dim=-1)
    all_pairs = normalized.sum(dim=-2).square().sum(dim=-1)
    diagonal = normalized.square().sum(dim=(-2, -1))
    off_diagonal = all_pairs - diagonal
    return (off_diagonal / (num_nodes * (num_nodes - 1))).mean().item()


class EncoderMetricsRecorder:
    def __init__(self, offset: int = 0) -> None:
        self._offset = offset
        self.reset()

    @classmethod
    @contextmanager
    def attach(cls, module: nn.Module, *, offset: int = 0) -> Iterator[EncoderMetricsRecorder]:
        """Record ``module``'s forward output for the duration of the block via a forward hook.

        The hook fires on every forward of ``module``, so scope the block to the phase that
        should be measured; outside it the module is untouched.
        """
        recorder = cls(offset)
        handle = module.register_forward_hook(
            lambda _module, _args, output: recorder.record(output)
        )
        try:
            yield recorder
        finally:
            handle.remove()

    def reset(self) -> None:
        self._embeddings_cos_sim_sum = 0.0
        self._count = 0

    def record(self, embeddings: torch.Tensor) -> None:
        # no_grad keeps the diagnostic off the autograd graph; the hook fires mid-forward with
        # grad enabled. No autocast guard is needed -- the metric has no matmul, so its .float()
        # is not downcast (autocast only lowers matmul/linear/conv).
        with torch.no_grad():
            self._embeddings_cos_sim_sum += mean_pairwise_cosine_similarity(
                embeddings[:, self._offset :]
            )
        self._count += 1

    @property
    def embeddings_cos_sim(self) -> float:
        return self._embeddings_cos_sim_sum / self._count if self._count else 0.0


class RolloutMetricsRecorder:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._diversity_overlap_sum = 0.0
        self._unique_rollouts_sum = 0.0
        self._count = 0

    def record(self, actions: torch.Tensor) -> None:
        self._diversity_overlap_sum += diversity_overlap(actions)
        self._unique_rollouts_sum += unique_rollouts(actions)
        self._count += 1

    @property
    def diversity_overlap(self) -> float:
        return self._diversity_overlap_sum / self._count if self._count else 0.0

    @property
    def unique_rollouts(self) -> float:
        return self._unique_rollouts_sum / self._count if self._count else 0.0
