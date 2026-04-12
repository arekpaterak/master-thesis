from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, cast

import torch


class DestroyState(ABC):
    """Abstract problem-level state for destroy decoding."""

    action_mask: torch.Tensor
    """(B, R, N) - True marks forbidden actions."""
    last_node_idx: torch.Tensor | None
    """(B, R, 1) - Last selected node; None before first action."""

    @classmethod
    @abstractmethod
    def build_initial(
        cls,
        *,
        batch_size: int,
        rollout_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Self: ...

    @abstractmethod
    def transition(self, action: torch.Tensor) -> Self: ...


@dataclass(frozen=True)
class TSPDestroyState(DestroyState):
    action_mask: torch.Tensor
    last_node_idx: torch.Tensor | None

    @classmethod
    def build_initial(
        cls,
        *,
        batch_size: int,
        rollout_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Self:
        action_mask = torch.zeros(
            batch_size,
            rollout_size,
            num_nodes,
            dtype=torch.bool,
            device=device,
        )
        return cls(action_mask=action_mask, last_node_idx=None)

    def transition(self, action: torch.Tensor) -> Self:
        new_action_mask = self.action_mask.clone()
        new_action_mask.scatter_(-1, action, True)
        return cast(Self, type(self)(action_mask=new_action_mask, last_node_idx=action))


@dataclass(frozen=True)
class CVRPDestroyState(DestroyState):
    action_mask: torch.Tensor
    last_node_idx: torch.Tensor | None

    @classmethod
    def build_initial(
        cls,
        *,
        batch_size: int,
        rollout_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Self:
        action_mask = torch.zeros(
            batch_size,
            rollout_size,
            num_nodes,
            dtype=torch.bool,
            device=device,
        )
        # Depot cannot be selected for removal.
        action_mask[:, :, 0] = True
        return cls(action_mask=action_mask, last_node_idx=None)

    def transition(self, action: torch.Tensor) -> Self:
        new_action_mask = self.action_mask.clone()
        new_action_mask.scatter_(-1, action, True)
        return cast(Self, type(self)(action_mask=new_action_mask, last_node_idx=action))
