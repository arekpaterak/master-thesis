from core.lns.components import LNSComponents
from core.problems.cmstp.instance import CMSTPInstance
from core.problems.cmstp.lns.apply_repair import CMSTPGreedyInsertApplyRepair
from core.problems.cmstp.lns.operators.repair.insert import CMSTPGreedyInsertRepair
from core.problems.cmstp.problem import PROBLEM
from core.problems.cmstp.solution import CMSTPSolution

LNS = LNSComponents[CMSTPInstance, CMSTPSolution](
    problem=PROBLEM,
    repairs={"greedy_insert": CMSTPGreedyInsertRepair},
    apply_and_repairs={"greedy_insert": CMSTPGreedyInsertApplyRepair},
)
