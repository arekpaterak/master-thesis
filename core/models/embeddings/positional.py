import math
from typing import Literal

import torch

FrequencySchedule = Literal["integer", "geometric"]

TWO_PI = 2 * math.pi


def circular_sinusoidal_encoding(
    phase: torch.Tensor,
    *,
    dim: int,
    direction_aware: bool = False,
    schedule: FrequencySchedule | None = None,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Maps a circular phase in [0, 2*pi] to a multi-frequency sinusoidal encoding.

    Two frequency schedules are supported:

    - ``"integer"``: omega_k = k + 1, the Fourier harmonics of the circle. The
      cosine-only variant is then exactly invariant to reversal of the phase
      (cos(k*(2*pi - x)) = cos(k*x)), satisfying the directional desideratum by
      construction.
    - ``"geometric"``: omega_k = base^(-2k/dim), the line-position schedule of
      Vaswani et al. (2017). Useful for the direction-aware variant; the cosine-only
      variant is then only approximately reversal-invariant (exact for k = 0 only).

    Args:
        phase: (...) circular phase in [0, 2*pi].
        dim: Output embedding dimension. Must be even.
        direction_aware: If True, returns interleaved sin/cos pairs (distinguishes a
            phase from its reflection; direction-sensitive variants). If False, returns
            cosine-only components with the frequency count doubled to keep ``dim`` fixed.
        schedule: Frequency schedule. Defaults to ``"integer"`` for the direction-invariant
            variant and ``"geometric"`` for the direction-aware variant.
        base: Geometric base for the ``"geometric"`` schedule.

    Returns:
        (..., dim) sinusoidal encoding.
    """
    assert dim % 2 == 0, "Positional encoding dim must be even."

    if schedule is None:
        schedule = "integer"

    num_frequencies = dim // 2 if direction_aware else dim
    k = torch.arange(num_frequencies, device=phase.device, dtype=phase.dtype)
    if schedule == "integer":
        frequencies = k + 1.0
    elif schedule == "geometric":
        frequencies = base ** (-2.0 * k / dim)
    else:
        raise ValueError(f"Unknown frequency schedule: {schedule!r}")

    angles = phase.unsqueeze(-1) * frequencies  # (..., num_frequencies)

    if direction_aware:
        return torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(-2)
    return angles.cos()
