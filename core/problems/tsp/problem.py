from core.problems.base import Problem
from core.problems.tsp.dataset import load_tsp_dataset
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.instance import TSPInstance
from core.problems.tsp.sampler import TSPSampler
from core.problems.tsp.solve import random_permutation_graph

PROBLEM = Problem[TSPInstance, TSPGraphSolution](
    name="tsp",
    sampler=TSPSampler,
    load_dataset=load_tsp_dataset,
    initial_solution=random_permutation_graph,
)
