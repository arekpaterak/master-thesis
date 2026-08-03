from typing import Annotated

from pydantic import Field, TypeAdapter

# Import samplers so they register themselves with Sampler._registry.
from core.problems.cmstp.config import CMSTP, CMSTPConfig
from core.problems.cmstp.sampler import CMSTPSampler
from core.problems.cvrp.config import CVRP, CVRPConfig
from core.problems.cvrp.sampler import CVRPSampler
from core.problems.solve import generate_initial_solution
from core.problems.tsp.config import TSP, TSPConfig
from core.problems.tsp.sampler import TSPSampler

# TODO: Move from __init__
ProblemConfig = Annotated[CVRPConfig | TSPConfig | CMSTPConfig, Field(discriminator="name")]

_problem_config_adapter = TypeAdapter(ProblemConfig)


def make_problem_config(name: str = "cvrp", size: int = 20, **fields: object) -> ProblemConfig:
    """Build the matching ProblemConfig member from a problem name, size and extra fields."""
    return _problem_config_adapter.validate_python({"name": name, "size": size, **fields})


def depot_offset(problem_name: str) -> int:
    """Number of leading non-customer nodes in a ``(B, N + offset, D)`` node tensor."""
    match problem_name:
        case "cvrp":
            return 1
        case "tsp":
            return 0
        case "cmstp":
            return 1
        case _:
            raise ValueError(f"Unsupported problem_name: {problem_name}")


__all__ = [
    "CMSTP",
    "CVRP",
    "TSP",
    "CMSTPConfig",
    "CMSTPSampler",
    "CVRPConfig",
    "CVRPSampler",
    "ProblemConfig",
    "depot_offset",
    "TSPConfig",
    "TSPSampler",
    "generate_initial_solution",
    "make_problem_config",
]
