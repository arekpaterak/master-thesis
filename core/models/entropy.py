from typing import NamedTuple

import torch


class EntropyResult(NamedTuple):
    entropy: torch.Tensor
    entropy_norm: torch.Tensor
    entropy_max: torch.Tensor


def entropy(log_p: torch.Tensor) -> EntropyResult:
    """
    Compute entropy of the node selection distribution at a single decoding step.

    Ranges from 0 (always picks the same node) to log(k) where k is the number
    of unmasked nodes.
    """
    p = log_p.exp()
    H = -(p * log_p).nansum(dim=-1)  # (batch,)
    k = torch.isfinite(log_p).sum(dim=-1)
    H_max = torch.log(k.float())
    H_norm = H / H_max.clamp(min=1e-8)
    return EntropyResult(entropy=H, entropy_norm=H_norm, entropy_max=H_max)
