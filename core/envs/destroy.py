from __future__ import annotations

from dataclasses import dataclass

import torch

from core.envs.base import Env
from core.problems.instance import BaseProblemInstance


@dataclass(frozen=True)
class DestroyState:
    action_mask: torch.Tensor
    """(B, R, N), or (B, 1, N) shared by all rollouts until the first step; True = forbidden."""
    last_node_idx: torch.Tensor | None
    """(B, R, 1); None before the first step."""
    num_selected: int


class DestroyEnv(Env[BaseProblemInstance, DestroyState]):
    """Select ``horizon`` distinct nodes to remove; the leading fixed nodes are never selectable."""

    def __init__(self, *, horizon: int) -> None:
        self.horizon = horizon

    def reset(
        self, instance: BaseProblemInstance, *, device: torch.device | None = None
    ) -> DestroyState:
        (batch_size,) = instance.batch_size
        action_mask = torch.zeros(
            batch_size,
            1,
            instance.num_nodes,
            dtype=torch.bool,
            device=device or instance.locations.device,
        )
        action_mask[..., : instance.num_fixed_nodes] = True
        return DestroyState(action_mask=action_mask, last_node_idx=None, num_selected=0)

    def step(self, state: DestroyState, action: torch.Tensor) -> DestroyState:
        action = action.unsqueeze(-1) if action.dim() == 2 else action
        action_mask = (
            state.action_mask.expand(*action.shape[:-1], state.action_mask.size(-1))
            .clone()
            .scatter_(-1, action, True)
        )
        return DestroyState(
            action_mask=action_mask, last_node_idx=action, num_selected=state.num_selected + 1
        )

    def action_mask(self, state: DestroyState) -> torch.Tensor:
        return state.action_mask

    def is_done(self, state: DestroyState) -> torch.Tensor:
        return torch.tensor(state.num_selected >= self.horizon)
