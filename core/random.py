from __future__ import annotations

import torch


class SeededGenerator:
    """One ``torch.Generator`` per device, each seeded with ``seed`` when first used."""

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._generators: dict[torch.device, torch.Generator] = {}

    def on(self, device: torch.device | str) -> torch.Generator:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device not in self._generators:
            generator = torch.Generator(device=device)
            if self._seed is not None:
                generator.manual_seed(self._seed)
            self._generators[device] = generator
        return self._generators[device]


def sample_categorical(logits: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    """Draw one index per row of ``logits`` [..., N], as ``Categorical(logits=logits).sample()``."""
    probs = torch.softmax(logits - logits.logsumexp(dim=-1, keepdim=True), dim=-1)
    samples = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1, True, generator=generator)
    return samples.reshape(probs.shape[:-1])


def sample_gumbel(like: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    """Standard Gumbel noise shaped like ``like``, as ``torch.distributions.Gumbel(0, 1)``."""
    finfo = torch.finfo(like.dtype)
    low, high = finfo.tiny, 1 - finfo.eps
    uniform = low + torch.rand(
        like.shape, generator=generator, dtype=like.dtype, device=like.device
    ) * (high - low)
    return -(-uniform.log()).log()
