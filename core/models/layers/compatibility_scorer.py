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
        Two call patterns are supported, distinguished by the rank of ``h_c`` vs ``h_nodes``:

        - Matched (``h_c`` rank == ``h_nodes`` rank - 1): one query per leading index.
          ``h_c`` (..., d), ``h_nodes`` (..., n, d) -> scores (..., n).
        - Shared keys (``h_c`` rank == ``h_nodes`` rank): the queries along the
          second-to-last axis of ``h_c`` all score against the same ``n`` keys.
          ``h_c`` (..., m, d), ``h_nodes`` (..., n, d) -> scores (..., m, n).
          This avoids broadcasting the keys over the query axis (e.g. rollouts), which
          would otherwise materialise a (..., m, n, d) tensor.

        Args:
            h_c: Context embedding(s) to use as a query, (..., d) or (..., m, d).
            h_nodes: Node embeddings to use as keys (..., n, d).
            mask: Boolean mask broadcastable to the score shape; True = masked out.
            temperature (float): Softmax temperature. Defaults to 1.0.

        Returns:
            Log-probabilities, (..., n) or (..., m, n).
        """
        Q = self.W_Q(h_c)
        K = self.W_K(h_nodes)

        if Q.dim() == K.dim():
            # Shared keys: (..., m, d) x (..., d, n) -> (..., m, n). No key broadcasting.
            u = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        else:
            # Matched: one query vector per leading index.
            u = torch.matmul(Q.unsqueeze(-2), K.transpose(-2, -1)).squeeze(-2) / self.scale

        u = self.C * torch.tanh(u)
        u = u / temperature
        u = u.masked_fill(mask, float("-inf"))
        return F.log_softmax(u, dim=-1)
