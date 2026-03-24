from core.models.nds.model import NDSModel
from core.models.nds.preprocessing import build_encoder_input_indexes_from_sequence
from core.models.nds.seed_sampler import sample_seed
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.cvrp.solve import one_customer_per_route


def test_model() -> None:
    sampler = CVRPSampler(CVRPConfig.for_n(20))
    instance = sampler(4)
    solution = one_customer_per_route(instance)
    neighbours_index, tour_index = build_encoder_input_indexes_from_sequence(
        solution.sequence, num_customers=instance.num_customers
    )
    seed = sample_seed(batch_size=4, rollout_size=10, seed_dim=10)

    model = NDSModel()

    state, cache = model.pre_forward(
        depot_location=instance.depot_location,
        customer_features=instance.customer_features,
        neighbours_index=neighbours_index,
        tour_index=tour_index,
        seed=seed,
        rollout_size=10,
    )

    log_p, gru_hidden_state = model.forward(state=state, cache=cache)

    assert log_p.shape == (4, 10, 21)
    assert gru_hidden_state.shape == (4 * 10, 128)
