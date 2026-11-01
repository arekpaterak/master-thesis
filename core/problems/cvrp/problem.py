from core.problems.base import Problem
from core.problems.cvrp.dataset import load_cvrp_dataset
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.instance import CVRPInstance
from core.problems.cvrp.sampler import CVRPSampler
from core.problems.cvrp.solve import one_customer_per_route_graph

PROBLEM = Problem[CVRPInstance, CVRPGraphSolution](
    name="cvrp",
    sampler=CVRPSampler,
    load_dataset=load_cvrp_dataset,
    initial_solution=one_customer_per_route_graph,
)
