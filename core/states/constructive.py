from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, cast

import torch

from core.problems.cvrp import CVRPSolution


class ConstructiveState(ABC):
    @property
    @abstractmethod
    def action_mask(self) -> torch.Tensor: ...

    @abstractmethod
    def transition(self, action: torch.Tensor) -> Self: ...

    @property
    @abstractmethod
    def is_initial(self) -> bool: ...

    def all_done(self) -> bool:
        return False


@dataclass(frozen=True)
class TSPConstructiveState(ConstructiveState):
    """Problem state for TSP constructive decoding."""

    _action_mask: torch.Tensor
    """(B, N) - True means node is already visited and cannot be selected."""
    first_node_idx: torch.Tensor | None
    """(B, 1) - First selected node; None before first action."""
    last_node_idx: torch.Tensor | None
    """(B, 1) - Most recently selected node; None before first action."""
    _selected_count: int

    def __post_init__(self) -> None:
        if self._action_mask.dim() != 2:
            raise ValueError("Expected action_mask shape (B, N).")

        if self._selected_count < 0:
            raise ValueError("selected_count must be non-negative.")

        has_first = self.first_node_idx is not None
        has_last = self.last_node_idx is not None
        if has_first != has_last:
            raise ValueError("first_node_idx and last_node_idx must both be None or both be set.")

        if self._selected_count == 0:
            if has_first or has_last:
                raise ValueError(
                    "first_node_idx/last_node_idx must be None when selected_count is 0."
                )
            return

        if not has_first or not has_last:
            raise ValueError("first_node_idx/last_node_idx are required when selected_count > 0.")

        batch_size = self._action_mask.size(0)
        assert self.first_node_idx is not None
        assert self.last_node_idx is not None
        if self.first_node_idx.shape != (batch_size, 1):
            raise ValueError("Expected first_node_idx shape (B, 1).")
        if self.last_node_idx.shape != (batch_size, 1):
            raise ValueError("Expected last_node_idx shape (B, 1).")

    @classmethod
    def initial(
        cls,
        *,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Self:
        return cls(
            _action_mask=torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device),
            first_node_idx=None,
            last_node_idx=None,
            _selected_count=0,
        )

    @classmethod
    def from_partial(
        cls,
        *,
        partial_tour: torch.Tensor,
        num_nodes: int,
    ) -> Self:
        batch_size = partial_tour.size(0)
        selected_count = partial_tour.size(1)
        device = partial_tour.device

        if selected_count == 0:
            return cls.initial(batch_size=batch_size, num_nodes=num_nodes, device=device)

        action_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        action_mask.scatter_(-1, partial_tour, True)
        return cls(
            _action_mask=action_mask,
            first_node_idx=partial_tour[:, :1],
            last_node_idx=partial_tour[:, -1:],
            _selected_count=selected_count,
        )

    def transition(self, action: torch.Tensor) -> Self:
        action_idx = action.unsqueeze(-1) if action.dim() == 1 else action
        new_action_mask = self._action_mask.clone()
        new_action_mask.scatter_(-1, action_idx, True)
        first_node_idx = self.first_node_idx if self.first_node_idx is not None else action_idx
        return cast(
            Self,
            type(self)(
                _action_mask=new_action_mask,
                first_node_idx=first_node_idx,
                last_node_idx=action_idx,
                _selected_count=self._selected_count + 1,
            ),
        )

    @property
    def action_mask(self) -> torch.Tensor:
        return self._action_mask

    @property
    def is_initial(self) -> bool:
        return self._selected_count == 0

    def all_done(self) -> bool:
        return bool(self._action_mask.all())


@dataclass(frozen=True)
class CVRPConstructiveState(ConstructiveState):
    """Problem state for CVRP constructive decoding (no split deliveries)."""

    remaining_demand: torch.Tensor
    """(B, N+1) - Remaining demand, depot at index 0 is always 0."""
    remaining_capacity: torch.Tensor
    """(B, 1) - Remaining vehicle capacity in [0, 1]."""
    last_node_idx: torch.Tensor | None
    """(B, 1) - Last selected node; None before the first decode action."""
    _selected_count: int

    def __post_init__(self) -> None:
        if self.remaining_demand.dim() != 2:
            raise ValueError("Expected remaining_demand shape (B, N+1).")
        if self.remaining_capacity.dim() != 2 or self.remaining_capacity.size(-1) != 1:
            raise ValueError("Expected remaining_capacity shape (B, 1).")
        if self.remaining_demand.size(0) != self.remaining_capacity.size(0):
            raise ValueError("Batch dimension mismatch between demand and capacity.")
        if self._selected_count < 0:
            raise ValueError("selected_count must be non-negative.")
        if self.last_node_idx is not None and self.last_node_idx.shape != (
            self.remaining_demand.size(0),
            1,
        ):
            raise ValueError("Expected last_node_idx shape (B, 1).")

    @classmethod
    def initial(
        cls,
        *,
        demand: torch.Tensor,
        capacity: torch.Tensor,
    ) -> Self:
        if demand.dim() != 2:
            raise ValueError("Expected demand shape (B, N+1).")
        if capacity.dim() != 2 or capacity.size(-1) != 1:
            raise ValueError("Expected capacity shape (B, 1).")

        remaining_demand = demand.clone()
        remaining_demand[:, 0] = 0.0
        return cls(
            remaining_demand=remaining_demand,
            remaining_capacity=capacity.clone(),
            last_node_idx=torch.full(
                (demand.size(0), 1),
                fill_value=CVRPSolution.DEPOT,
                dtype=torch.long,
                device=demand.device,
            ),
            _selected_count=1,
        )

    @classmethod
    def from_partial(
        cls,
        *,
        demand: torch.Tensor,
        capacity: torch.Tensor,
        partial_actions: torch.Tensor,
    ) -> Self:
        remaining_demand = demand.clone()
        remaining_demand[:, 0] = 0.0
        state = cls(
            remaining_demand=remaining_demand,
            remaining_capacity=capacity.clone(),
            last_node_idx=None,
            _selected_count=0,
        )
        for step in range(partial_actions.size(1)):
            state = state.transition(partial_actions[:, step])

        return cast(Self, state)

    @property
    def action_mask(self) -> torch.Tensor:
        """True marks forbidden actions."""
        demand = self.remaining_demand
        cap = self.remaining_capacity
        all_served = (demand[:, 1:] <= 0).all(dim=-1, keepdim=True)
        customer_done = demand[:, 1:] <= 0
        customer_over_capacity = demand[:, 1:] > cap
        mask_customers = customer_done | customer_over_capacity

        if self.last_node_idx is None:
            mask_depot = torch.ones((demand.size(0), 1), dtype=torch.bool, device=demand.device)
        else:
            mask_depot = self.last_node_idx.eq(CVRPSolution.DEPOT)

        # Once all customers are served, keep depot selectable to avoid an all-masked row.
        mask_depot = mask_depot & ~all_served

        return torch.cat([mask_depot, mask_customers], dim=-1)

    @property
    def all_served(self) -> torch.Tensor:
        return (self.remaining_demand[:, 1:] <= 0).all(dim=-1, keepdim=True)

    def all_done(self) -> bool:
        if self.last_node_idx is None:
            return False
        all_served = bool(self.all_served.all())
        back_at_depot = bool(self.last_node_idx.eq(CVRPSolution.DEPOT).all())
        return all_served and back_at_depot

    @property
    def is_initial(self) -> bool:
        return self._selected_count <= 1

    def transition(self, action: torch.Tensor) -> Self:
        action_idx = action.unsqueeze(-1) if action.dim() == 1 else action
        if action_idx.shape != (self.remaining_demand.size(0), 1):
            raise ValueError("Expected action shape (B, 1) for CVRP constructive transition.")

        new_demand = self.remaining_demand.clone()
        new_capacity = self.remaining_capacity.clone()
        is_depot = action_idx.eq(0)

        # Visiting depot resets capacity to full vehicle capacity (normalized to 1).
        new_capacity = torch.where(is_depot, torch.ones_like(new_capacity), new_capacity)

        customer_idx = action_idx.clamp_min(0)
        picked_demand = new_demand.gather(-1, customer_idx)

        # No split deliveries: serve customer fully in one visit.
        customer_visit = ~is_depot
        new_capacity = torch.where(
            customer_visit, (new_capacity - picked_demand).clamp_min(0.0), new_capacity
        )
        new_demand.scatter_(
            -1,
            customer_idx,
            torch.where(customer_visit, torch.zeros_like(picked_demand), picked_demand),
        )

        return cast(
            Self,
            type(self)(
                remaining_demand=new_demand,
                remaining_capacity=new_capacity,
                last_node_idx=action_idx,
                _selected_count=self._selected_count + 1,
            ),
        )
