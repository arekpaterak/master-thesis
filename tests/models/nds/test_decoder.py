import torch

from core.models.nds.decoder import NDSDecoder
from core.models.nds.seed_sampler import sample_seed


def test_one_step_of_decoder() -> None:
    B, rollout_size, N = 2, 10, 4
    z_dim = 10

    decoder = NDSDecoder()

    embedded_nodes = torch.randn(B, N + 1, 128)
    z = sample_seed(batch_size=B, rollout_size=rollout_size, seed_dim=z_dim)

    state = decoder.build_initial_state(B=B, rollout_size=rollout_size, num_nodes=N + 1)
    cache = decoder.build_cache(h_nodes=embedded_nodes, seed=z)

    log_p, gru_hidden_state = decoder.forward(cache=cache, state=state)

    assert log_p.shape == (B, rollout_size, N + 1)
    assert gru_hidden_state.shape == (B * rollout_size, 128)
