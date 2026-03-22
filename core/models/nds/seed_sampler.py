import torch


def sample_seed(
    B: int, rollout_size: int, z_dim: int, device: torch.device | None = None
) -> torch.Tensor:
    return torch.randint(0, 2, (B, rollout_size, z_dim), device=device).float()
