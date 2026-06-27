from core.models.nds.models.joint import JointNDSModel
from core.problems.cvrp import CVRP, CVRPSampler
from core.problems.cvrp.solve import one_customer_per_route


def test_model() -> None:
    B, R, N, k = 4, 10, 20, 5
    sampler = CVRPSampler(CVRP.for_n(N))
    instance = sampler(B)
    solution = one_customer_per_route(instance)

    model = JointNDSModel(problem_name="cvrp")

    rollout = model.rollout(solution, rollouts_per_solution=R, deconstruction_size=k)

    assert rollout.actions.shape == (B, R, k)
    assert rollout.log_p.shape == (B, R)
