from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from core.problems.augmentations import apply_d4_transforms
from core.problems.instance import BaseProblemInstance

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class CMSTPInstance(BaseProblemInstance):
    locations: torch.Tensor
    """(B, N+1, 2) or (N+1, 2) - root first, then N terminals."""
    demand: torch.Tensor
    """(B, N+1) or (N+1,) - raw (unnormalised) demand, root entry is 0."""
    capacity: torch.Tensor
    """(B, 1) or (1,) - raw (unnormalised) per-subtree capacity."""
    coordinate_scale: torch.Tensor
    """(B, 1) or (1,) - cumulative scale factor applied by normalize_(). 1.0 for sampler instances."""

    ROOT = 0

    @classmethod
    def create(
        cls,
        locations: torch.Tensor,
        demand: torch.Tensor,
        capacity: torch.Tensor,
        coordinate_scale: torch.Tensor | None = None,
    ) -> CMSTPInstance:
        batch_size = locations.shape[0] if locations.dim() == 3 else []
        if coordinate_scale is None:
            shape = (locations.shape[0], 1) if locations.dim() == 3 else (1,)
            coordinate_scale = torch.ones(shape, dtype=locations.dtype, device=locations.device)
        return cls(
            locations=locations,
            demand=demand,
            capacity=capacity,
            coordinate_scale=coordinate_scale,
            batch_size=batch_size,
            device=locations.device,
        )

    def __post_init__(self) -> None:
        b_ndim = len(self.batch_size)

        assert self.locations.ndim == b_ndim + 2, (
            f"Expected nodes to have {b_ndim + 2} dims (batch + N+1 + 2), got {self.locations.ndim}"
        )
        assert self.demand.ndim == b_ndim + 1, (
            f"Expected demand to have {b_ndim + 1} dims (batch + N+1), got {self.demand.ndim}"
        )
        assert self.capacity.ndim == b_ndim + 1, (
            f"Expected capacity to have {b_ndim + 1} dims (batch + 1), got {self.capacity.ndim}"
        )
        assert self.coordinate_scale.ndim == b_ndim + 1, (
            f"Expected coordinate_scale to have {b_ndim + 1} dims (batch + 1), "
            f"got {self.coordinate_scale.ndim}"
        )

        assert self.locations.shape[-1] == 2, "Expected nodes last dim to be 2 (x, y)."
        assert self.capacity.shape[-1] == 1, "Expected capacity last dim to be 1."
        assert self.coordinate_scale.shape[-1] == 1, "Expected coordinate_scale last dim to be 1."
        assert self.locations.shape[-2] == self.demand.shape[-1], (
            "Number of nodes mismatch between nodes and demands."
        )

    @property
    def num_nodes(self) -> int:
        return self.locations.shape[-2]

    @property
    def num_terminals(self) -> int:
        """N, the non-root nodes."""
        return self.num_nodes - 1

    @property
    def problem_size(self) -> int:
        return self.num_terminals

    @property
    def root_location(self) -> torch.Tensor:
        return self.locations[..., :1, :]  # (..., 1, 2)

    @property
    def terminal_locations(self) -> torch.Tensor:
        return self.locations[..., 1:, :]

    @property
    def terminal_demands(self) -> torch.Tensor:
        """(..., N) raw terminal demands (excludes the root)."""
        return self.demand[..., 1:]

    @property
    def normalized_demand(self) -> torch.Tensor:
        """(..., N+1) demand normalised by capacity, in [0, 1] (root is 0)."""
        return self.demand / self.capacity

    @property
    def normalized_terminal_demands(self) -> torch.Tensor:
        """(..., N) capacity-normalised terminal demands in (0, 1] (excludes the root)."""
        return self.terminal_demands / self.capacity

    def distances(self) -> torch.Tensor:
        """(..., N+1, N+1) pairwise Euclidean distances between all nodes (root + terminals)."""
        return torch.cdist(self.locations, self.locations)

    def normalize_(self) -> None:
        """Normalize coordinates in-place by the maximum coordinate value.

        Records the scale factor in coordinate_scale so original-scale costs can be
        recovered later. Do not call on sampler instances, already in [0, 1].
        """
        scale = self.locations.amax(dim=(-2, -1), keepdim=True)  # (*batch, 1, 1)
        self.locations.div_(scale)
        self.coordinate_scale.mul_(scale.squeeze(-1))  # (*batch, 1)

    @property
    def is_normalized(self) -> bool:
        return bool((self.coordinate_scale == 1.0).all())

    def augment(self, n_augmentations: int) -> CMSTPInstance:
        augmented_locations = apply_d4_transforms(self.locations, n_augmentations)
        expanded_demand = self.demand.unsqueeze(-2).expand(
            *self.batch_size, n_augmentations, self.demand.shape[-1]
        )
        expanded_capacity = self.capacity.unsqueeze(-2).expand(
            *self.batch_size, n_augmentations, self.capacity.shape[-1]
        )
        expanded_coordinate_scale = self.coordinate_scale.unsqueeze(-2).expand(
            *self.batch_size, n_augmentations, self.coordinate_scale.shape[-1]
        )
        return CMSTPInstance(
            locations=augmented_locations,
            demand=expanded_demand.clone(),
            capacity=expanded_capacity.clone(),
            coordinate_scale=expanded_coordinate_scale.clone(),
            batch_size=(*self.batch_size, n_augmentations),
            device=augmented_locations.device,
        )

    @property
    def terminal_features(self) -> torch.Tensor:
        """(..., N, 3) content features [x, y, normalized_demand] for the encoder."""
        terminal_locations = self.locations[..., 1:, :]
        terminal_demands = self.normalized_terminal_demands.unsqueeze(-1)
        return torch.cat([terminal_locations, terminal_demands], dim=-1)

    def root_anchored_polar_angle(self) -> torch.Tensor:
        """(..., N) float - root-anchored polar angle of each terminal in [-pi, pi]."""
        relative = self.terminal_locations - self.root_location  # (..., N, 2)
        return torch.atan2(relative[..., 1], relative[..., 0])  # (..., N) in [-pi, pi]

    def visualize(
        self,
        *,
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
        with_normalized_demand: bool = True,
    ) -> None:
        from .render import render_instance

        render_instance(
            self,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            ax=ax,
            with_normalized_demand=with_normalized_demand,
        )
