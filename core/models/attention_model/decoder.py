import torch
import torch.nn as nn

from core.models.layers.compatibility_scorer import CompatibilityScorer
from core.states.constructive import TSPConstructiveState


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_heads: int = 8,
        C: float = 10.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.C = C

        # Learnable input placeholders for the initial decoding step
        self.v_l = nn.Parameter(torch.zeros(hidden_dim))
        self.v_f = nn.Parameter(torch.zeros(hidden_dim))

        # MODIFICATION: Project 3*d_model context down to d_model upfront
        self.context_projection = nn.Linear(3 * hidden_dim, hidden_dim, bias=False)

        self.MHA = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            bias=False,
        )
        self.final_scorer = CompatibilityScorer(d_h=hidden_dim, C=C)

    def forward(
        self,
        *,
        node_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        state: TSPConstructiveState,
    ) -> torch.Tensor:
        batch_size, n, _ = node_embeddings.shape

        # Initial context embeddings for first and last node in the tour
        if state.is_initial:
            h_last = self.v_l.expand(batch_size, -1)
            h_first = self.v_f.expand(batch_size, -1)
        else:
            hidden_dim = node_embeddings.size(-1)
            h_first = node_embeddings.gather(
                dim=-2,
                index=state.first_node_idx.unsqueeze(-1).expand(batch_size, 1, hidden_dim),
            ).squeeze(-2)
            h_last = node_embeddings.gather(
                dim=-2,
                index=state.last_node_idx.unsqueeze(-1).expand(batch_size, 1, hidden_dim),
            ).squeeze(-2)

        # Initial context embedding
        context_embedding = torch.cat([graph_embedding, h_last, h_first], dim=-1)

        # Project context to hidden dim
        context_embedding = self.context_projection(context_embedding)

        action_mask = state.action_mask

        # MHA: context node attends over all unmasked nodes
        context_embedding, _ = self.MHA(
            query=context_embedding.unsqueeze(1),
            key=node_embeddings,
            value=node_embeddings,
            key_padding_mask=action_mask,
        )
        context_embedding = context_embedding.squeeze(1)  # (batch, d_model)

        # Final single-head layer
        log_p = self.final_scorer(h_c=context_embedding, h_nodes=node_embeddings, mask=action_mask)
        return log_p
