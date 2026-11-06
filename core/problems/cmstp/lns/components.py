from core.lns.components import LNSComponents
from core.problems.cmstp.instance import CMSTPInstance
from core.problems.cmstp.lns.apply_repair import CMSTPGreedyInsertApplyRepair
from core.problems.cmstp.problem import PROBLEM
from core.problems.cmstp.solution import CMSTPSolution

# No torch repairs: CMSTPSolution cannot remove nodes outside the numba apply-and-repair.
LNS = LNSComponents[CMSTPInstance, CMSTPSolution](
    problem=PROBLEM,
    repairs={},
    apply_and_repairs={"greedy_insert": CMSTPGreedyInsertApplyRepair},
)
