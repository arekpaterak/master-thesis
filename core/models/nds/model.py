import torch
import torch.nn as nn

from core.models.nds.decoder import DecoderCache, DecoderOutput, DecoderState, NDSDecoder
from core.models.nds.encoder import NDSEncoder


class NDSModel(nn.Module):
    def __init__(
        self,
        *,
        problem_name: str,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        poly_dim: int = 256,
        seed_dim: int = 10,
    ) -> None:
        super().__init__()

        self.z_dim = seed_dim

        self.encoder = NDSEncoder(
            problem_name=problem_name, hidden_dim=hidden_dim, num_heads=num_heads, ff_dim=ff_dim
        )
        self.decoder = NDSDecoder(
            hidden_dim=hidden_dim, num_heads=num_heads, seed_dim=seed_dim, poly_dim=poly_dim
        )

    def pre_forward(
        self,
        *,
        seed: torch.Tensor,
        rollout_size: int,
        **features: torch.Tensor,
    ) -> tuple[DecoderState, DecoderCache]:
        encoded_nodes = self.encoder(**features)

        B = encoded_nodes.shape[0]
        N = encoded_nodes.shape[1]

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
