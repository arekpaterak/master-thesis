from __future__ import annotations

import torch

NUM_D4_AUGMENTATIONS = 8


def apply_d4_transforms(locations: torch.Tensor, n_augmentations: int) -> torch.Tensor:
    """Apply the first ``n_augmentations`` symmetries of the unit square to coordinates.

    The 8 D4 symmetries (4 rotations + 4 reflections) of ``[0, 1]^2`` are distance-preserving.
    At the index 0 is the identity, so ``n_augmentations == 1``returns the input unchanged (with an extra leading dim).

    Args:
        locations: (..., N, 2) tensor of coordinates in ``[0, 1]^2``.
        n_augmentations: How many augmentations to apply, in 1..8.

    Returns:
        (..., n_augmentations, N, 2) tensor with the requested augmentations stacked on a new dim immediately before the node dim.
    """
    assert 1 <= n_augmentations <= NUM_D4_AUGMENTATIONS, (
        f"Expected 1 <= n_augmentations <= {NUM_D4_AUGMENTATIONS}, got {n_augmentations}"
    )

    x = locations[..., 0]
    y = locations[..., 1]
    transforms = [
        (x, y),
        (1 - x, y),
        (x, 1 - y),
        (1 - x, 1 - y),
        (y, x),
        (1 - y, x),
        (y, 1 - x),
        (1 - y, 1 - x),
    ][:n_augmentations]

    augmented = [torch.stack([new_x, new_y], dim=-1) for new_x, new_y in transforms]
    return torch.stack(augmented, dim=-3)
