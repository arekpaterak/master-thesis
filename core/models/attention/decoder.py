import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, d_h: int = 128, num_heads: int = 8, C: float = 10.0) -> None:
        super().__init__()
        self.d_h = d_h
        self.C = C

        self.v_l = nn.Parameter(torch.zeros(d_h))
        self.v_f = nn.Parameter(torch.zeros(d_h))

        # MODIFICATION: Project 3*d_model context down to d_model upfront
        self.context_projection = nn.Linear(3 * d_h, d_h, bias=False)

        self.mha = nn.MultiheadAttention(
            embed_dim=d_h,
            num_heads=num_heads,
            batch_first=True,
            bias=False,
        )
        self.final_scorer = SingleHeadScorer(d_h, C)

    def forward(self, h_nodes, h_graph, tour_idx) -> torch.Tensor:
        batch_size, n, _ = h_nodes.shape

        # Initial context embeddings for first and last node in the tour
        if len(tour_idx) == 0:
            h_last = self.v_l.expand(batch_size, -1)
            h_first = self.v_f.expand(batch_size, -1)
        else:
            h_first = h_nodes[torch.arange(batch_size), tour_idx[0]]
            h_last = h_nodes[torch.arange(batch_size), tour_idx[-1]]

        # Initial context embedding
        h_c = torch.cat([h_graph, h_last, h_first], dim=-1)

        # Project context to d_h
        h_c = self.context_projection(h_c)

        # Mask nodes that cannot be visited
        mask = torch.zeros(batch_size, n, dtype=torch.bool, device=h_nodes.device)
        for t_idx in tour_idx:
            mask[torch.arange(batch_size), t_idx] = True

        # MHA: context node attends over all unmasked nodes
        h_c, _ = self.mha(
            query=h_c.unsqueeze(1),
            key=h_nodes,
            value=h_nodes,
            key_padding_mask=mask,
        )
        h_c = h_c.squeeze(1)  # (batch, d_model)

        # Final single-head layer
        log_p = self.final_scorer(h_c, h_nodes, mask)
        return log_p


class SingleHeadScorer(nn.Module):
    def __init__(self, d_h: int = 128, C: float = 10.0) -> None:
        super().__init__()
        self.scale = d_h**0.5
        self.C = C
        self.W_Q = nn.Linear(d_h, d_h, bias=False)
        self.W_K = nn.Linear(d_h, d_h, bias=False)

    def forward(self, h_c, h_nodes, mask) -> torch.Tensor:
        """
        Args:
            h_c: (batch, d_model)
            h_nodes: (batch, n, d_model)
            mask: (batch, n) bool
        Returns:
            log_p: (batch, n)
        """
        Q = self.W_Q(h_c).unsqueeze(1)  # (batch, 1, d_model)
        K = self.W_K(h_nodes)  # (batch, n, d_model)
        u = torch.matmul(Q, K.transpose(-2, -1)).squeeze(1) / self.scale  # (batch, n)
        u = self.C * torch.tanh(u)
        u = u.masked_fill(mask, float("-inf"))
        return F.log_softmax(u, dim=-1)
