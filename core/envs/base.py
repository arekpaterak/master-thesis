from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Env[X, St](ABC):
    """A batched MDP with immutable states; every method is pure."""

    @abstractmethod
    def reset(self, x: X) -> St: ...

    @abstractmethod
    def step(self, state: St, action: torch.Tensor) -> St: ...

    @abstractmethod
    def action_mask(self, state: St) -> torch.Tensor:
        """True marks forbidden actions."""
        ...

    @abstractmethod
    def is_done(self, state: St) -> torch.Tensor:
        """Per-instance termination, broadcastable to the batch shape."""
        ...

    def all_done(self, state: St) -> bool:
        return bool(self.is_done(state).all())

    def replay(self, x: X, actions: torch.Tensor) -> St:
        """The state reached from ``reset(x)`` by taking ``actions`` [..., T] in order."""
        state = self.reset(x)
        for t in range(actions.size(-1)):
            state = self.step(state, actions[..., t])
        return state
