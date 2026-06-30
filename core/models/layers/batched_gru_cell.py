from __future__ import annotations

import torch
from torch import nn


class BatchedGRUCell(nn.GRUCell):
    """nn.GRUCell that accepts and returns tensors with extra leading batch dimensions.

    nn.GRUCell operates on (batch, dim) tensors. This subclass flattens any leading
    dimensions (e.g. (B, R)) into a single batch for the cell and restores them on the
    way out, so callers can stay in their natural (B, R, dim) layout.

    Subclassing (rather than wrapping) keeps the parameter names identical to nn.GRUCell
    (weight_ih, weight_hh, bias_ih, bias_hh), so checkpoints and the pretrained-weight
    loader continue to load unchanged.
    """

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input, (*batch_dims, input_size).
            hidden: Previous hidden state, (*batch_dims, hidden_size), or None for zeros.

        Returns:
            Next hidden state, (*batch_dims, hidden_size).
        """
        batch_dims = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_size)
        hidden_flat = None if hidden is None else hidden.reshape(-1, self.hidden_size)
        next_hidden = super().forward(x_flat, hidden_flat)
        return next_hidden.reshape(*batch_dims, self.hidden_size)
