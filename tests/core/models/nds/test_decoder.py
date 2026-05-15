import torch

from core.models.nds.decoder import NDSDecoder
from core.models.nds.seed_sampler import sample_seed
from core.states.destroy import CVRPDestroyState


def test_one_step_of_decoder() -> None:
    B, rollout_size, N = 2, 10, 4
    seed_dim = 10

    decoder = NDSDecoder()

    embedded_nodes = torch.randn(B, N + 1, 128)
    z = sample_seed(batch_size=B, rollout_size=rollout_size, seed_dim=seed_dim)

    problem_state = CVRPDestroyState.build_initial(
        batch_size=B,
        rollout_size=rollout_size,
        num_nodes=N + 1,
        device=embedded_nodes.device,
    )
    decoder_state = decoder.build_initial_state(B=B, rollout_size=rollout_size)
    cache = decoder.build_cache(h_nodes=embedded_nodes, seed=z)

    log_p, gru_hidden_state = decoder.forward(
        problem_state=problem_state,
        cache=cache,
        decoder_state=decoder_state,
    )

    assert log_p.shape == (B, rollout_size, N + 1)
    assert gru_hidden_state.shape == (B * rollout_size, 128)
    assert torch.not_equal(gru_hidden_state, decoder_state.gru_hidden).all(), (
        "GRU hidden state should be updated after forward pass"
    )


def test_without_hidden_state() -> None:
    B, rollout_size, N = 2, 10, 4
    seed_dim = 10

    decoder = NDSDecoder(use_hidden_decoding_state=False)

    embedded_nodes = torch.randn(B, N + 1, 128)
    z = sample_seed(batch_size=B, rollout_size=rollout_size, seed_dim=seed_dim)

    problem_state = CVRPDestroyState.build_initial(
        batch_size=B,
        rollout_size=rollout_size,
        num_nodes=N + 1,
        device=embedded_nodes.device,
    )
    decoder_state = decoder.build_initial_state(B=B, rollout_size=rollout_size)
    cache = decoder.build_cache(h_nodes=embedded_nodes, seed=z)

    log_p, gru_hidden_state = decoder.forward(
        problem_state=problem_state,
        cache=cache,
        decoder_state=decoder_state,
    )
    assert log_p.shape == (B, rollout_size, N + 1)
    assert torch.allclose(gru_hidden_state, decoder_state.gru_hidden), (
        "GRU hidden state should not be updated after forward pass"
    )
