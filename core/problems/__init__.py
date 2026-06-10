from typing import Annotated

from pydantic import Field, TypeAdapter

# Import samplers so they register themselves with Sampler._registry.
from core.problems.cvrp.config import CVRP, CVRPConfig
from core.problems.cvrp.sampler import CVRPSampler
from core.problems.solve import initial_solution
from core.problems.tsp.config import TSP, TSPConfig
from core.problems.tsp.sampler import TSPSampler

ProblemConfig = Annotated[CVRPConfig | TSPConfig, Field(discriminator="name")]

_problem_config_adapter = TypeAdapter(ProblemConfig)


def make_problem_config(name: str = "cvrp", size: int = 20, **fields: object) -> ProblemConfig:
    """Build the matching ProblemConfig member from a problem name, size and extra fields."""
    return _problem_config_adapter.validate_python({"name": name, "size": size, **fields})


__all__ = [
    "CVRP",
    "TSP",
    "CVRPConfig",
    "CVRPSampler",
    "ProblemConfig",
    "TSPConfig",
    "TSPSampler",
    "initial_solution",
    "make_problem_config",
]
