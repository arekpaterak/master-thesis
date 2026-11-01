from __future__ import annotations

from dataclasses import dataclass

import torch

from core.envs.constructive import ConstructiveEnv
from core.problems.tsp.instance import TSPInstance
from core.problems.tsp.solution import TSPSolution


@dataclass(frozen=True)
class TSPConstructiveState:
    visited: torch.Tensor
    """(B, N) - True means node is already visited and cannot be selected."""
    first_node_idx: torch.Tensor | None
    """(B, 1) - First selected node; None before first action."""
    last_node_idx: torch.Tensor | None
    """(B, 1) - Most recently selected node; None before first action."""

    def __post_init__(self) -> None:
        if self.visited.dim() != 2:
            raise ValueError("Expected visited shape (B, N).")
        if (self.first_node_idx is None) != (self.last_node_idx is None):
            raise ValueError("first_node_idx and last_node_idx must both be None or both be set.")

    @property
    def is_initial(self) -> bool:
        return self.first_node_idx is None


class TSPConstructiveEnv(ConstructiveEnv[TSPInstance, TSPSolution, TSPConstructiveState]):
    def reset(self, instance: TSPInstance) -> TSPConstructiveState:
        (batch_size,) = instance.batch_size
        visited = torch.zeros(
            batch_size, instance.num_nodes, dtype=torch.bool, device=instance.locations.device
        )
        return TSPConstructiveState(visited=visited, first_node_idx=None, last_node_idx=None)

    def step(self, state: TSPConstructiveState, action: torch.Tensor) -> TSPConstructiveState:
        action_idx = action.unsqueeze(-1) if action.dim() == 1 else action
        visited = state.visited.clone().scatter_(-1, action_idx, True)
        first_node_idx = action_idx if state.first_node_idx is None else state.first_node_idx
        return TSPConstructiveState(
            visited=visited, first_node_idx=first_node_idx, last_node_idx=action_idx
        )

    def action_mask(self, state: TSPConstructiveState) -> torch.Tensor:
        return state.visited

    def is_done(self, state: TSPConstructiveState) -> torch.Tensor:
        return state.visited.all(dim=-1)

    def to_solution(self, instance: TSPInstance, actions: torch.Tensor) -> TSPSolution:
        return TSPSolution.create(instance=instance, tour=actions)
