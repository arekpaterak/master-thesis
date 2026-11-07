from __future__ import annotations

from dataclasses import dataclass

import torch

from core.envs.constructive import ConstructiveEnv
from core.problems.cvrp.instance import CVRPInstance
from core.problems.cvrp.solution import CVRPSolution


@dataclass(frozen=True)
class CVRPConstructiveState:
    """Constructive decoding without split deliveries; demand and capacity are normalized."""

    remaining_demand: torch.Tensor
    """(B, N+1) - Remaining demand, depot at index 0 is always 0."""
    remaining_capacity: torch.Tensor
    """(B, 1) - Remaining vehicle capacity in [0, 1]."""
    last_node_idx: torch.Tensor
    """(B, 1) - Last selected node; the depot before the first decode action."""

    def __post_init__(self) -> None:
        if self.remaining_demand.dim() != 2:
            raise ValueError("Expected remaining_demand shape (B, N+1).")
        if self.remaining_capacity.dim() != 2 or self.remaining_capacity.size(-1) != 1:
            raise ValueError("Expected remaining_capacity shape (B, 1).")
        if self.last_node_idx.shape != (self.remaining_demand.size(0), 1):
            raise ValueError("Expected last_node_idx shape (B, 1).")

    @property
    def all_served(self) -> torch.Tensor:
        """(B, 1)"""
        return (self.remaining_demand[:, 1:] <= 0).all(dim=-1, keepdim=True)


class CVRPConstructiveEnv(ConstructiveEnv[CVRPInstance, CVRPSolution, CVRPConstructiveState]):
    def reset(self, instance: CVRPInstance) -> CVRPConstructiveState:
        remaining_demand = instance.normalized_demand.clone()
        remaining_demand[:, CVRPInstance.DEPOT] = 0.0
        return CVRPConstructiveState(
            remaining_demand=remaining_demand,
            remaining_capacity=torch.ones_like(instance.capacity),
            last_node_idx=torch.full(
                (remaining_demand.size(0), 1),
                fill_value=CVRPInstance.DEPOT,
                dtype=torch.long,
                device=remaining_demand.device,
            ),
        )

    def step(self, state: CVRPConstructiveState, action: torch.Tensor) -> CVRPConstructiveState:
        action_idx = action.unsqueeze(-1) if action.dim() == 1 else action
        if action_idx.shape != (state.remaining_demand.size(0), 1):
            raise ValueError("Expected action shape (B, 1) for CVRP constructive transition.")

        is_depot = action_idx.eq(CVRPInstance.DEPOT)
        picked_demand = state.remaining_demand.gather(-1, action_idx)

        # Visiting the depot refills the vehicle; a customer is served fully in one visit.
        capacity = torch.where(
            is_depot, torch.ones_like(state.remaining_capacity), state.remaining_capacity
        )
        capacity = torch.where(is_depot, capacity, (capacity - picked_demand).clamp_min(0.0))
        remaining_demand = state.remaining_demand.clone().scatter_(
            -1, action_idx, torch.where(is_depot, picked_demand, torch.zeros_like(picked_demand))
        )
        return CVRPConstructiveState(
            remaining_demand=remaining_demand,
            remaining_capacity=capacity,
            last_node_idx=action_idx,
        )

    def action_mask(self, state: CVRPConstructiveState) -> torch.Tensor:
        customers = state.remaining_demand[:, 1:]
        mask_customers = (customers <= 0) | (customers > state.remaining_capacity)
        # Once all customers are served, keep the depot selectable to avoid an all-masked row.
        mask_depot = state.last_node_idx.eq(CVRPInstance.DEPOT) & ~state.all_served
        return torch.cat([mask_depot, mask_customers], dim=-1)

    def is_done(self, state: CVRPConstructiveState) -> torch.Tensor:
        back_at_depot = state.last_node_idx.eq(CVRPInstance.DEPOT)
        return (state.all_served & back_at_depot).squeeze(-1)

    def to_solution(self, instance: CVRPInstance, actions: torch.Tensor) -> CVRPSolution:
        return CVRPSolution.create(instance=instance, sequence=actions)
