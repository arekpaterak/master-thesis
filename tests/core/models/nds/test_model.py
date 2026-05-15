from core.models.nds.model import NDSModel
from core.models.nds.seed_sampler import sample_seed
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.cvrp.features import CVRPSolutionFeatureExtractor
from core.problems.cvrp.solve import one_customer_per_route


def test_model() -> None:
    B, R, N = 4, 10, 20
    sampler = CVRPSampler(CVRPConfig.for_n(N))
    instance = sampler(B)
    solution = one_customer_per_route(instance)
    seed = sample_seed(batch_size=B, rollout_size=R, seed_dim=10)

    model = NDSModel(problem_name="cvrp")

    extractor = CVRPSolutionFeatureExtractor()
    features = extractor(solution)

    problem_state, decoder_state, cache = model.pre_forward(
        seed=seed,
        rollout_size=R,
        **features,
    )

    log_p, gru_hidden_state = model.forward(
        problem_state=problem_state,
        decoder_state=decoder_state,
        cache=cache,
    )

    assert log_p.shape == (B, R, N + 1)
    assert gru_hidden_state.shape == (B * R, 128)
