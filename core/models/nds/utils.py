from __future__ import annotations

import torch

from core.models.rollout import Rollout
from core.problems.solution import BaseSolution


def flatten_instance_batch[S: BaseSolution](solution: S) -> tuple[S, tuple[int, ...]]:
    """Collapse all leading instance batch dims of ``solution`` into a single one.

    The NDS encoder/decoder operate on one batch dim. Augmentation adds an extra
    ``A`` batch dim to instances and solutions, giving batch ``(B, A)``. Flattening
    ``(B, A, ...)`` -> ``(B * A, ...)`` lets the model treat every instance
    independently. Pair with :func:`unflatten_rollout` to restore the original
    dims on the produced rollout.

    Returns:
        The flattened solution and the original batch dims, to be passed back to
        :func:`unflatten_rollout`.
    """
    batch_dims = tuple(solution.batch_size)
    return solution.reshape(-1), batch_dims


def unflatten_rollout(rollout: Rollout, batch_dims: tuple[int, ...]) -> Rollout:
    """Restore the original instance batch dims on a flat ``(G, R, ...)`` rollout.

    The rollout dim ``R`` is the innermost batch dim and is kept, so the result
    has batch ``(*batch_dims, R)``.
    """
    rollout_size = rollout.batch_size[-1]
    return rollout.reshape(*batch_dims, rollout_size)


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


class RolloutMetricsRecorder:
    """Accumulates diversity metrics over the destroy rollouts of a search.

    Each :meth:`record` call adds the diversity metrics of one rollout tensor to running
    sums; the properties return the means over all recorded steps. Used to capture rollout
    diversity during validation, where rollouts are produced inside ``solve``.
    """

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
