import torch
import torch.nn as nn

from core.models.nds.decoder import DecoderCache, DecoderOutput, DecoderState, NDSDecoder
from core.models.nds.encoder import NDSEncoder
from core.states.destroy import CVRPDestroyState, DestroyState, TSPDestroyState


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

        self.problem_name = problem_name
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
    ) -> tuple[DestroyState, DecoderState, DecoderCache]:
        encoded_nodes = self.encoder(**features)

        B = encoded_nodes.shape[0]
        N = encoded_nodes.shape[1]

        destroy_state_cls: type[DestroyState]
        if self.problem_name == "cvrp":
            destroy_state_cls = CVRPDestroyState
        elif self.problem_name == "tsp":
            destroy_state_cls = TSPDestroyState
        else:
            raise ValueError(f"Unsupported problem_name: {self.problem_name}")

        cache = self.decoder.build_cache(h_nodes=encoded_nodes, seed=seed)
        problem_state = destroy_state_cls.build_initial(
            batch_size=B,
            rollout_size=rollout_size,
            num_nodes=N,
            device=encoded_nodes.device,
        )
        initial_state = self.decoder.build_initial_state(
            B=B,
            rollout_size=rollout_size,
        )
        return problem_state, initial_state, cache

    def forward(
        self,
        *,
        problem_state: DestroyState,
        decoder_state: DecoderState,
        cache: DecoderCache,
    ) -> DecoderOutput:
        return self.decoder.forward(
            problem_state=problem_state, decoder_state=decoder_state, cache=cache
        )
