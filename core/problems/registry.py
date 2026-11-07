from pathlib import Path

from core.lns.components import LNSComponents
from core.models.nds.components import NDSComponents
from core.models.nds.config import ModelConfig
from core.models.nds.models import NDSModel
from core.problems.base import Problem
from core.problems.cmstp.lns.components import LNS as CMSTP_LNS
from core.problems.cmstp.nds.components import NDS as CMSTP_NDS
from core.problems.cmstp.problem import PROBLEM as CMSTP
from core.problems.cvrp.lns.components import LNS as CVRP_LNS
from core.problems.cvrp.nds.components import NDS as CVRP_NDS
from core.problems.cvrp.problem import PROBLEM as CVRP
from core.problems.dataset import DatasetSampler
from core.problems.names import ProblemName
from core.problems.problem_config import ProblemConfig
from core.problems.sampler import Sampler
from core.problems.tsp.lns.components import LNS as TSP_LNS
from core.problems.tsp.nds.components import NDS as TSP_NDS
from core.problems.tsp.problem import PROBLEM as TSP
from core.resolve import resolve

PROBLEMS: dict[ProblemName, Problem] = {"tsp": TSP, "cvrp": CVRP, "cmstp": CMSTP}
LNS: dict[ProblemName, LNSComponents] = {"tsp": TSP_LNS, "cvrp": CVRP_LNS, "cmstp": CMSTP_LNS}
NDS: dict[ProblemName, NDSComponents] = {"tsp": TSP_NDS, "cvrp": CVRP_NDS, "cmstp": CMSTP_NDS}


def problem(name: str) -> Problem:
    return resolve(PROBLEMS, name, kind="problem")


def lns(name: str) -> LNSComponents:
    return resolve(LNS, name, kind="LNS components")


def nds(name: str) -> NDSComponents:
    return resolve(NDS, name, kind="NDS components")


def build_nds_model(config: ModelConfig) -> NDSModel:
    return config.build(nds(config.problem_name))


def build_sampler(config: ProblemConfig, *, seed: int | None = None) -> Sampler:
    return problem(config.name).sampler(config.to_distribution(), seed=seed)


def build_dataset_sampler(problem_name: str, path: str | Path) -> DatasetSampler:
    return DatasetSampler(problem(problem_name).load_dataset(path))
