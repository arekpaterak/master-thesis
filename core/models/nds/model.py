import torch
import torch.nn as nn

from core.models.nds.decoder import DecoderCache, DecoderOutput, DecoderState, NDSDecoder
from core.models.nds.encoder import NDSEncoder
from core.problems.cvrp import CVRPSolution


class NDSModel(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        poly_dim: int = 256,
        seed_dim: int = 10,
    ) -> None:
        super().__init__()

        self.z_dim = seed_dim

        self.encoder = NDSEncoder(hidden_dim=hidden_dim, num_heads=num_heads, ff_dim=ff_dim)
        self.decoder = NDSDecoder(
            d_h=hidden_dim, num_heads=num_heads, seed_dim=seed_dim, d_poly=poly_dim
        )

    def pre_forward(
        self,
        *,
        solution: CVRPSolution,
        seed: torch.Tensor,
        rollout_size: int,
    ) -> tuple[DecoderState, DecoderCache]:
        B = solution.batch_size[0]
        N = solution.instance.num_nodes

        instance = solution.instance

        encoded_nodes = self.encoder(
            depot_features=instance.depot_location,
            customer_features=instance.customer_features,
            neighbours_index=solution.neighbours_per_customer(),
            tour_index=solution.route_per_customer(),
        )
        cache = self.decoder.build_cache(h_nodes=encoded_nodes, seed=seed)
        initial_state = self.decoder.build_initial_state(
            B=B,
            rollout_size=rollout_size,
            num_nodes=N,
        )
        return initial_state, cache

    def forward(
        self,
        *,
        state: DecoderState,
        cache: DecoderCache,
    ) -> DecoderOutput:
        return self.decoder(state=state, cache=cache)
