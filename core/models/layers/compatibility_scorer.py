import torch
from torch import nn as nn
from torch.nn import functional as F


class CompatibilityScorer(nn.Module):
    """
    Computes log-probability scores for a set of nodes relative to a context vector.

    Args:
        hidden_dim (int): The hidden dimension of the input embeddings. Defaults to 128.
        C (float): Logit clipping coefficient. Defaults to 10.0.
        use_projections (bool, optional): Whether to use projections for query and keys. Defaults to False.
    """

    def __init__(
        self, *, hidden_dim: int = 128, C: float = 10.0, use_projections: bool = True
    ) -> None:
        super().__init__()
        self.scale = hidden_dim**0.5
        self.C = C
        self.use_projections = use_projections

        if self.use_projections:
            self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.W_Q = nn.Identity()
            self.W_K = nn.Identity()

    def forward(
        self,
        *,
        h_c: torch.Tensor,
        h_nodes: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            h_c: Single context embedding to use as a query (..., hidden_dim)
            h_nodes: Node embeddings to use as keys (..., n, hidden_dim)
            mask: Boolean mask (..., n)
            temperature (float): Softmax temperature. Defaults to 1.0.

        Returns:
            Log-probabilities (..., n)
        """
        Q = self.W_Q(h_c).unsqueeze(-2)  # (..., 1, d_model)
        K = self.W_K(h_nodes)  # (..., n, d_model)

        u = torch.matmul(Q, K.transpose(-2, -1)).squeeze(-2) / self.scale  # (..., n)
        u = self.C * torch.tanh(u)
        u = u / temperature
        u = u.masked_fill(mask, float("-inf"))
        return F.log_softmax(u, dim=-1)
