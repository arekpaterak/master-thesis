from core.lns.components import LNSComponents
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.instance import TSPInstance
from core.problems.tsp.lns.apply_repair import TSPGreedyInsertApplyRepair
from core.problems.tsp.lns.operators.repair.insert import TSPGreedyInsertRepair
from core.problems.tsp.lns.operators.repair.reconnect import TSPGreedyReconnectRepair
from core.problems.tsp.problem import PROBLEM

LNS = LNSComponents[TSPInstance, TSPGraphSolution](
    problem=PROBLEM,
    repairs={
        "greedy_insert": TSPGreedyInsertRepair,
        "greedy_reconnect": TSPGreedyReconnectRepair,
    },
    apply_and_repairs={"greedy_insert": TSPGreedyInsertApplyRepair},
)
