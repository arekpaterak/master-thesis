import pytest
import torch

from core.models.nds.decoders.ar import AutoregressiveNDSDecoder
from core.models.nds.seed_sampler import sample_seed
from core.states.destroy import CVRPDestroyState

B, rollout_size, N, seed_dim = 2, 10, 4, 10


@pytest.mark.parametrize("use_hidden_decoding_state", [True, False])
def test_decoder_step(use_hidden_decoding_state: bool) -> None:
    decoder = AutoregressiveNDSDecoder(use_hidden_decoding_state=use_hidden_decoding_state)
    embedded_nodes = torch.randn(B, N + 1, 128)
    z = sample_seed(batch_size=B, rollout_size=rollout_size, seed_dim=seed_dim)

    problem_state = CVRPDestroyState.build_initial(
        batch_size=B, rollout_size=rollout_size, num_nodes=N + 1, device=embedded_nodes.device
    )
    decoder_state = decoder.build_initial_state(batch_size=B, rollout_size=rollout_size)
    cache = decoder.build_cache(h_nodes=embedded_nodes, seed=z)

    log_p, gru_hidden_state = decoder.forward(
        problem_state=problem_state, cache=cache, decoder_state=decoder_state
    )

    assert log_p.shape == (B, rollout_size, N + 1)
    assert gru_hidden_state.shape == (B, rollout_size, 128)
    if use_hidden_decoding_state:
        assert torch.not_equal(gru_hidden_state, decoder_state.gru_hidden).all()
    else:
        assert torch.allclose(gru_hidden_state, decoder_state.gru_hidden)
