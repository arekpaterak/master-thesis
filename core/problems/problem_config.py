from typing import Annotated

from pydantic import Field, TypeAdapter

from core.problems.cmstp.config import CMSTPConfig
from core.problems.cvrp.config import CVRPConfig
from core.problems.tsp.config import TSPConfig

ProblemConfig = Annotated[CVRPConfig | TSPConfig | CMSTPConfig, Field(discriminator="name")]

_problem_config_adapter = TypeAdapter(ProblemConfig)


def make_problem_config(name: str = "cvrp", size: int = 20, **fields: object) -> ProblemConfig:
    """Build the matching ProblemConfig member from a problem name, size and extra fields."""
    return _problem_config_adapter.validate_python({"name": name, "size": size, **fields})
