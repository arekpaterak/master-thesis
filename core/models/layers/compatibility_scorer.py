import torch
from torch import nn as nn
from torch.nn import functional as F

from core.models.layers.attention_scaling import AttentionScalingConfig


class CompatibilityScorer(nn.Module):
    """
    Computes log-probability scores for a set of nodes relative to a context vector.

    Args:
        hidden_dim (int): The hidden dimension of the input embeddings. Defaults to 128.
        C (float): Logit clipping coefficient. Defaults to 10.0.
        use_projections (bool, optional): Whether to use projections for query and keys. Defaults to False.
        attention_scaling: Size-aware scaling of the clipped logits; None disables it.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        C: float = 10.0,
        use_projections: bool = True,
        attention_scaling: AttentionScalingConfig | None = None,
    ) -> None:
        super().__init__()
        self.attention_scaling = attention_scaling
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
        context: torch.Tensor,
        node_embeddings: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
        log_softmax: bool = True,
    ) -> torch.Tensor:
        """
        Scores queries from the context against per-instance keys (the nodes).

        Every query in ``h_context`` scores against the same ``num_nodes`` nodes,
        so the nodes carry no query dims and are never replicated over them.

        The query dims ``*q`` are arbitrary:

        - ``()`` for the attention model,
        - ``(R,)`` for the AR decoder and the NAR simple decoder,
        - ``(R, M)`` for the slot decoder.

        Args:
            context: Query embedding(s), (B, *q, d).
            node_embeddings: Node embeddings used as keys, (B, num_nodes, d).
            mask: Boolean mask broadcastable to (B, *q, num_nodes); True = masked out.
            temperature (float): Softmax temperature. Defaults to 1.0.
            log_softmax (bool): If True (default), return log-probabilities. If False,
                return the raw masked logits (masked entries at -inf) without the final
                log_softmax, for callers that renormalize themselves .

        Returns:
            Log-probabilities if ``log_softmax`` else raw masked logits, (B, *q, num_nodes).
        """
        Q = self.W_Q(context)  # (B, *q, d)
        K = self.W_K(node_embeddings)  # (B, num_nodes, d)

        batch_size, d, num_nodes = Q.shape[0], Q.shape[-1], K.shape[-2]
        query_dims = Q.shape[1:-1]  # *q

        expected_mask_dim = len(query_dims) + 2
        if mask.dim() != expected_mask_dim:
            raise ValueError(
                f"mask must have dim {expected_mask_dim} (batch + {len(query_dims)} query dim(s) + nodes) "
                f"for query dims {tuple(query_dims)}, got mask shape {tuple(mask.shape)}"
            )
        if mask.shape[-1] != num_nodes:
            raise ValueError(
                f"mask's last dim must be num_nodes={num_nodes}, got mask shape {tuple(mask.shape)}."
            )

        # Fold the query dims (*q) into one dimension so the keys are never broadcast or expanded over the query dims.
        Q_flat = Q.reshape(batch_size, -1, d)  # (B, prod(*q), d)
        # (B, prod(*q), d) x (B, d, num_nodes) -> (B, prod(*q), num_nodes)
        scores = torch.matmul(Q_flat, K.transpose(-2, -1)) / self.scale
        u = scores.reshape(batch_size, *query_dims, num_nodes)  # (B, *q, num_nodes)

        u = self.C * torch.tanh(u)  # (B, *q, num_nodes)
        if self.attention_scaling is not None:
            # After clipping, so the bounded logits are what gets sharpened.
            u = u * self.attention_scaling.factor(num_nodes)
        u = u / temperature
        u = u.masked_fill(mask, float("-inf"))  # broadcast mask, fill forbidden keys
        if not log_softmax:
            return u  # (B, *q, num_nodes) raw masked logits
        return F.log_softmax(u, dim=-1)  # (B, *q, num_nodes)
