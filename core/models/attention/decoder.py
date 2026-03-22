import torch
import torch.nn as nn

from core.models.layers.single_head_scorer import CompatibilityScorer


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        d_h: int = 128,
        num_heads: int = 8,
        C: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.C = C

        # Learnable input placeholders for the initial decoding step
        self.v_l = nn.Parameter(torch.zeros(d_h))
        self.v_f = nn.Parameter(torch.zeros(d_h))

        # MODIFICATION: Project 3*d_model context down to d_model upfront
        self.context_projection = nn.Linear(3 * d_h, d_h, bias=False)

        self.MHA = nn.MultiheadAttention(
            embed_dim=d_h,
            num_heads=num_heads,
            batch_first=True,
            bias=False,
        )
        self.final_scorer = CompatibilityScorer(d_h=d_h, C=C)

    def forward(
        self, *, h_nodes: torch.Tensor, h_graph: torch.Tensor, tour_idx: list
    ) -> torch.Tensor:
        batch_size, n, _ = h_nodes.shape

        # Initial context embeddings for first and last node in the tour
        if len(tour_idx) == 0:
            h_last = self.v_l.expand(batch_size, -1)
            h_first = self.v_f.expand(batch_size, -1)
        else:
            h_first = h_nodes[torch.arange(batch_size), tour_idx[0]]
            h_last = h_nodes[torch.arange(batch_size), tour_idx[-1]]

        # Initial context embedding
        context_embedding = torch.cat([h_graph, h_last, h_first], dim=-1)

        # Project context to d_h
        context_embedding = self.context_projection(context_embedding)

        # Mask nodes that cannot be visited
        mask = torch.zeros(batch_size, n, dtype=torch.bool, device=h_nodes.device)
        for t_idx in tour_idx:
            mask[torch.arange(batch_size), t_idx] = True

        # MHA: context node attends over all unmasked nodes
        context_embedding, _ = self.MHA(
            query=context_embedding.unsqueeze(1),
            key=h_nodes,
            value=h_nodes,
            key_padding_mask=mask,
        )
        context_embedding = context_embedding.squeeze(1)  # (batch, d_model)

        # Final single-head layer
        log_p = self.final_scorer(h_c=context_embedding, h_nodes=h_nodes, mask=mask)
        return log_p
