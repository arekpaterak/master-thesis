from typing import NamedTuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from core.models.attention.decoder import Decoder
from core.models.attention.encoder import Encoder
from core.models.attention.graph import make_complete_graph
from core.models.entropy import entropy
from core.problems.tsp import TSPInstance


class RolloutResult(NamedTuple):
    tour: torch.Tensor
    log_p: torch.Tensor
    entropy: float | None
    entropy_norm: float | None


class AttentionModel(nn.Module):
    def __init__(self, d_x: int = 2, d_h: int = 128, num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        self.encoder = Encoder(
            d_x=d_x,
            d_h=d_h,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.decoder = Decoder(d_h=d_h, num_heads=num_heads)

    def forward(
        self,
        instances: TSPInstance,
        *,
        greedy: bool = False,
        partial_tour: torch.Tensor | None = None,
    ) -> RolloutResult:
        """
        Args:
            instances: (batch, n, d_x)
            greedy: if True, argmax; if False, sample
            partial_tour: (batch, k) optional pre-placed nodes. Assumes same number of pre-placed nodes across batch.
        Returns:
            RolloutResult
        """
        batch_size, n, _ = instances.locations.shape

        data = Batch.from_data_list(
            [make_complete_graph(instances.locations[b]) for b in range(batch_size)]
        )
        assert data["edge_index"].shape[0] == 2, (
            "Expected edge_index to have shape (2, total_edges)"
        )
        h_nodes, h_graph = self.encoder(
            x=data["x"], edge_index=data["edge_index"], batch=data["batch"]
        )
        h_nodes = h_nodes.view(batch_size, n, -1)

        # Initialize tour_idx from partial tour if provided
        if partial_tour is None:
            tour_idx = []
        else:
            k = partial_tour.shape[1]
            tour_idx = [partial_tour[:, i] for i in range(k)]

        log_p_of_selected = []
        entropies, entropies_norm = [], []

        steps_remaining = n - len(tour_idx)
        for _ in range(steps_remaining):
            log_p: torch.Tensor = self.decoder(h_nodes=h_nodes, h_graph=h_graph, tour_idx=tour_idx)
            selected = log_p.argmax(-1) if greedy else log_p.exp().multinomial(1).squeeze(1)
            log_p_of_selected.append(log_p[torch.arange(batch_size), selected])
            tour_idx.append(selected)

            step_entropy = entropy(log_p)
            entropies.append(step_entropy.entropy)
            entropies_norm.append(step_entropy.entropy_norm)

        mean_entropy = torch.stack(entropies, dim=1).mean().item()
        mean_entropy_norm = torch.stack(entropies_norm, dim=1).mean().item()

        # full tour: partial nodes + newly decoded nodes
        full_tour = torch.stack(tour_idx, dim=1)  # (batch, n)

        return RolloutResult(
            tour=full_tour,
            log_p=torch.stack(log_p_of_selected, dim=1),
            entropy=mean_entropy,
            entropy_norm=mean_entropy_norm,
        )
