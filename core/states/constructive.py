from dataclasses import dataclass
from typing import Self, cast

import torch


@dataclass(frozen=True)
class TSPConstructiveState:
    """Problem state for TSP constructive decoding."""

    action_mask: torch.Tensor
    """(B, N) - True means node is already visited and cannot be selected."""
    first_node_idx: torch.Tensor | None
    """(B, 1) - First selected node; None before first action."""
    last_node_idx: torch.Tensor | None
    """(B, 1) - Most recently selected node; None before first action."""
    selected_count: int

    def __post_init__(self) -> None:
        if self.action_mask.dim() != 2:
            raise ValueError("Expected action_mask shape (B, N).")

        if self.selected_count < 0:
            raise ValueError("selected_count must be non-negative.")

        has_first = self.first_node_idx is not None
        has_last = self.last_node_idx is not None
        if has_first != has_last:
            raise ValueError("first_node_idx and last_node_idx must both be None or both be set.")

        if self.selected_count == 0:
            if has_first or has_last:
                raise ValueError(
                    "first_node_idx/last_node_idx must be None when selected_count is 0."
                )
            return

        if not has_first or not has_last:
            raise ValueError("first_node_idx/last_node_idx are required when selected_count > 0.")

        batch_size = self.action_mask.size(0)
        assert self.first_node_idx is not None
        assert self.last_node_idx is not None
        if self.first_node_idx.shape != (batch_size, 1):
            raise ValueError("Expected first_node_idx shape (B, 1).")
        if self.last_node_idx.shape != (batch_size, 1):
            raise ValueError("Expected last_node_idx shape (B, 1).")

    @classmethod
    def build_initial(
        cls,
        *,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Self:
        return cls(
            action_mask=torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device),
            first_node_idx=None,
            last_node_idx=None,
            selected_count=0,
        )

    @classmethod
    def from_partial_tour(
        cls,
        *,
        partial_tour: torch.Tensor,
        num_nodes: int,
    ) -> Self:
        batch_size = partial_tour.size(0)
        selected_count = partial_tour.size(1)
        device = partial_tour.device

        if selected_count == 0:
            return cls.build_initial(batch_size=batch_size, num_nodes=num_nodes, device=device)

        action_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        action_mask.scatter_(-1, partial_tour, True)
        return cls(
            action_mask=action_mask,
            first_node_idx=partial_tour[:, :1],
            last_node_idx=partial_tour[:, -1:],
            selected_count=selected_count,
        )

    def transition(self, action: torch.Tensor) -> Self:
        action_idx = action.unsqueeze(-1) if action.dim() == 1 else action
        new_action_mask = self.action_mask.clone()
        new_action_mask.scatter_(-1, action_idx, True)
        first_node_idx = self.first_node_idx if self.first_node_idx is not None else action_idx
        return cast(
            Self,
            type(self)(
                action_mask=new_action_mask,
                first_node_idx=first_node_idx,
                last_node_idx=action_idx,
                selected_count=self.selected_count + 1,
            ),
        )

    @property
    def is_initial(self) -> bool:
        return self.selected_count == 0
