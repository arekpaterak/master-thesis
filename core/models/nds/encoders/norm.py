import torch
from torch import nn


def build_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    if norm_type == "instance":
        return InstanceNorm(hidden_dim)
    if norm_type == "layer":
        return nn.LayerNorm(hidden_dim)
    if norm_type == "rms":
        return nn.RMSNorm(hidden_dim)
    raise ValueError(f"Unknown norm_type: {norm_type!r}")


class InstanceNorm(nn.Module):
    """
    InstanceNorm1d for (B, N, D) tensors without manual transpose at call sites.
    The intent is to normalize across nodes, N axis.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(hidden_dim, affine=True, track_running_stats=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(-1, -2)).transpose(-1, -2)
