import torch
import torch.nn as nn

from core.models.nds.decoder import DecoderCache, DecoderState, NDSDecoder
from core.models.nds.encoder import NDSEncoder


class NDSModel(nn.Module):
    def __init__(
        self,
        *,
        d_h: int = 128,
        num_heads: int = 8,
        d_ff: int = 512,
        d_poly: int = 256,
        z_dim: int = 10,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim

        self.encoder = NDSEncoder(d_h=d_h, num_heads=num_heads, d_ff=d_ff)
        self.decoder = NDSDecoder(d_h=d_h, num_heads=num_heads, z_dim=z_dim, d_poly=d_poly)

    def pre_forward(
        self,
        *,
        depot_location: torch.Tensor,
        customer_features: torch.Tensor,
        neighbours_index: torch.Tensor,
        tour_index: torch.Tensor,
        seed: torch.Tensor,
        rollout_size: int,
    ) -> tuple[DecoderState, DecoderCache]:
        B, N, _ = customer_features.shape

        encoded_nodes = self.encoder(
            depot_location=depot_location,
            customer_features=customer_features,
            neighbours_index=neighbours_index,
            tour_index=tour_index,
        )
        cache = self.decoder.build_cache(h_nodes=encoded_nodes, seed=seed)
        initial_state = self.decoder.build_initial_state(
            B=B,
            rollout_size=rollout_size,
            num_customers=N,
        )
        return initial_state, cache

    def forward(
        self,
        *,
        state: DecoderState,
        cache: DecoderCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(state=state, cache=cache)
