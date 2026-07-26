from __future__ import annotations

import torch
from matplotlib.axes import Axes

from core.problems.augmentations import apply_d4_transforms
from core.problems.instance import BaseProblemInstance


class TSPInstance(BaseProblemInstance):
    """Canonical, generator-agnostic representation of a single or batched TSP instance."""

    locations: torch.Tensor
    """(B, N, 2) or (N, 2)"""

    @classmethod
    def create(cls, locations: torch.Tensor) -> TSPInstance:
        batch_size = locations.shape[0] if locations.dim() == 3 else []
        return cls(locations=locations, batch_size=batch_size, device=locations.device)

    def __post_init__(self):
        assert self.locations.shape[-1] == 2, (
            "Expected locations to have shape (batch_size, num_locations, 2) or (num_locations, 2)."
        )

    @property
    def num_locations(self) -> int:
        return self.locations.shape[-2]

    @property
    def num_nodes(self) -> int:
        return self.num_locations

    @property
    def problem_size(self) -> int:
        return self.num_locations

    def normalize_(self) -> None:
        pass  # TSP instances are sampled in [0, 1] and require no normalization.

    def distances(self) -> torch.Tensor:
        return torch.cdist(self.locations, self.locations)

    def augment(self, n_augmentations: int) -> TSPInstance:
        augmented_locations = apply_d4_transforms(self.locations, n_augmentations)
        return TSPInstance(
            locations=augmented_locations,
            batch_size=(*self.batch_size, n_augmentations),
            device=augmented_locations.device,
        )

    def visualize(
        self,
        *,
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
    ) -> None:
        from .render import render_instance

        render_instance(
            self,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            ax=ax,
        )
