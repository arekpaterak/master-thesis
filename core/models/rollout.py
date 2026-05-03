from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Rollout:
    actions: torch.Tensor
    log_p_of_actions: torch.Tensor
    entropy: float | None
    entropy_norm: float | None

    def __post_init__(self) -> None:
        assert self.actions.size() == self.log_p_of_actions.size()

    @property
    def overall_log_p(self) -> torch.Tensor:
        return self.log_p_of_actions.sum(dim=-1)
