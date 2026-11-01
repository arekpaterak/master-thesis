from core.models.nds.models.joint import JointNDSModel
from core.problems.cvrp import CVRPDistribution, CVRPSampler
from core.problems.cvrp.solve import one_customer_per_route


def test_model() -> None:
    B, R, N, M = 4, 10, 20, 5
    sampler = CVRPSampler(CVRPDistribution.for_n(N))
    instance = sampler(B)
    solution = one_customer_per_route(instance)

    model = JointNDSModel(problem_name="cvrp")

    rollout = model.rollout(solution, rollouts_per_solution=R, deconstruction_size=M)

    assert rollout.actions.shape == (B, R, M)
    assert rollout.log_p.shape == (B, R)
