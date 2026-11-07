import torch


def sample_seed(
    *,
    batch_size: int,
    rollout_size: int,
    seed_dim: int,
    generator: torch.Generator,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.randint(
        0, 2, (batch_size, rollout_size, seed_dim), device=device, generator=generator
    ).float()
