from __future__ import annotations

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
