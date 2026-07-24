from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DestroyState:
    """Problem-level state for destroy decoding.

    The ``transition`` is problem-agnostic (it just forbids the chosen node); the only
    problem-specific bit -- which nodes are forbidden initially -- lives in
    ``build_initial`` (per-problem subclass) or in the mask handed to ``from_mask``.
    """

    action_mask: torch.Tensor
    """(B, R, N) - True marks forbidden actions."""
    last_node_idx: torch.Tensor | None
    """(B, R, 1) - Last selected node; None before first action."""

    @classmethod
    def from_mask(cls, action_mask: torch.Tensor) -> DestroyState:
        """Wrap an existing initial mask (B, R, N) as a fresh, transitionable state."""
        return cls(action_mask=action_mask, last_node_idx=None)

    @classmethod
    def build_initial(
        cls,
        *,
        batch_size: int,
        rollout_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> DestroyState:
        action_mask = torch.zeros(
            batch_size, rollout_size, num_nodes, dtype=torch.bool, device=device
        )
        return cls(action_mask=action_mask, last_node_idx=None)

    def transition(self, action: torch.Tensor) -> DestroyState:
        new_action_mask = self.action_mask.clone()
        new_action_mask.scatter_(-1, action, True)
        return type(self)(action_mask=new_action_mask, last_node_idx=action)


@dataclass(frozen=True)
class TSPDestroyState(DestroyState):
    """No nodes are forbidden initially; the base ``build_initial`` already suffices."""


@dataclass(frozen=True)
class CVRPDestroyState(DestroyState):
    @classmethod
    def build_initial(
        cls,
        *,
        batch_size: int,
        rollout_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> CVRPDestroyState:
        action_mask = torch.zeros(
            batch_size, rollout_size, num_nodes, dtype=torch.bool, device=device
        )
        # Depot cannot be selected for removal.
        action_mask[:, :, 0] = True
        return cls(action_mask=action_mask, last_node_idx=None)


@dataclass(frozen=True)
class CMSTDestroyState(DestroyState):
    @classmethod
    def build_initial(
        cls,
        *,
        batch_size: int,
        rollout_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> CMSTDestroyState:
        action_mask = torch.zeros(
            batch_size, rollout_size, num_nodes, dtype=torch.bool, device=device
        )
        # Root cannot be selected for removal.
        action_mask[:, :, 0] = True
        return cls(action_mask=action_mask, last_node_idx=None)


def initial_action_mask(
    problem_name: str, *, batch_size: int, num_nodes: int, device: torch.device
) -> torch.Tensor:
    """Initial forbidden-node mask (B, N) for a fresh rollout of ``problem_name``."""
    match problem_name:
        case "cvrp":
            state_cls: type[DestroyState] = CVRPDestroyState
        case "tsp":
            state_cls = TSPDestroyState
        case "cmst":
            state_cls = CMSTDestroyState
        case _:
            raise ValueError(f"Unsupported problem_name: {problem_name}")
    return state_cls.build_initial(
        batch_size=batch_size, rollout_size=1, num_nodes=num_nodes, device=device
    ).action_mask.squeeze(1)
