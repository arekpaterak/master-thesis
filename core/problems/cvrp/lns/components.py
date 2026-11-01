from core.lns.components import LNSComponents
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.instance import CVRPInstance
from core.problems.cvrp.lns.apply_repair import CVRPGreedyInsertApplyRepair
from core.problems.cvrp.lns.operators.repair.insert import CVRPGreedyInsertRepair
from core.problems.cvrp.lns.operators.repair.reconnect import CVRPGreedyReconnectRepair
from core.problems.cvrp.problem import PROBLEM

LNS = LNSComponents[CVRPInstance, CVRPGraphSolution](
    problem=PROBLEM,
    repairs={
        "greedy_insert": CVRPGreedyInsertRepair,
        "greedy_reconnect": CVRPGreedyReconnectRepair,
    },
    apply_and_repairs={"greedy_insert": CVRPGreedyInsertApplyRepair},
)
