from core.lns.config import SearchParams, StartSolutionParams
from core.lns.improvement_step import ImprovementStep, build_random_node_destroy_step
from core.lns.start_solution import with_warmup
from core.problems.cvrp import CVRPDistribution, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.lns.components import LNS
from core.problems.cvrp.solve import one_customer_per_route_graph


def _search(steps: int) -> SearchParams:
    return SearchParams(
        repair_type="greedy_insert",
        improvement_step_type="numba_sequential",
        start_solution_params=StartSolutionParams(
            num_improvement_steps=steps, deconstruction_size=10
        ),
    )


def test_build_random_destroy_step_returns_requested_variant() -> None:
    step = build_random_node_destroy_step(_search(50), lns=LNS, seed=0)
    assert isinstance(step, ImprovementStep)
    assert step._mode == "sequential"


def test_warmup_improves_and_stays_feasible() -> None:
    instance = CVRPSampler(CVRPDistribution.for_n(20), seed=42)(4)
    start_fn = with_warmup(one_customer_per_route_graph, lns=LNS, search=_search(30))

    trivial = one_customer_per_route_graph(instance)
    warmed: CVRPGraphSolution = start_fn(instance)

    assert (warmed.objective() < trivial.objective()).all()
    assert warmed.is_complete.all()
    assert warmed.is_feasible.all()
    assert (warmed.node_degree() == 2).all()


def test_warmup_disabled_returns_base_callable() -> None:
    start_fn = with_warmup(one_customer_per_route_graph, lns=LNS, search=_search(0))
    assert start_fn is one_customer_per_route_graph
