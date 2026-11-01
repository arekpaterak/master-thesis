from core.problems.base import Problem
from core.problems.cmstp.construction import esau_williams, star
from core.problems.cmstp.dataset import load_cmstp_dataset
from core.problems.cmstp.instance import CMSTPInstance
from core.problems.cmstp.sampler import CMSTPSampler
from core.problems.cmstp.solution import CMSTPSolution

PROBLEM = Problem[CMSTPInstance, CMSTPSolution](
    name="cmstp",
    sampler=CMSTPSampler,
    load_dataset=load_cmstp_dataset,
    initial_solution=star,
    constructions={"star": star, "esau_williams": esau_williams},
)
