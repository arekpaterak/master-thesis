from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class DecodingState(Protocol):
    @property
    def action_mask(self) -> torch.Tensor:
        """Boolean tensor where True means the action is not allowed."""
        ...

    def transition(self, action: torch.Tensor, **kwargs: Any) -> DecodingState:
        """Return a new immutable state after applying action."""
        ...
