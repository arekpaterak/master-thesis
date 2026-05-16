from __future__ import annotations

import torch
from tensordict.tensorclass import TensorClass


class Rollout(TensorClass):
    actions: torch.Tensor
    """(*batch_size, n) - sequence of action indices (e.g. nodes to destroy or construct)."""
    log_p: torch.Tensor
    """(*batch_size,) - joint log probability of the action sequence under the model."""
    entropy: torch.Tensor
    """(*batch_size,) - entropy of the action distribution."""
    entropy_norm: torch.Tensor
    """(*batch_size,) - entropy normalized by the maximum possible entropy."""

    @classmethod
    def create(
        cls,
        *,
        actions: torch.Tensor,
        log_p: torch.Tensor,
        entropy: torch.Tensor,
        entropy_norm: torch.Tensor,
    ) -> Rollout:
        """Create a Rollout, inferring batch_size from log_p shape."""
        return cls(
            actions=actions,
            log_p=log_p,
            entropy=entropy,
            entropy_norm=entropy_norm,
            batch_size=log_p.shape,
        )

    @property
    def mean_entropy(self) -> float:
        return self.entropy.mean().item()

    @property
    def mean_entropy_norm(self) -> float:
        return self.entropy_norm.mean().item()
