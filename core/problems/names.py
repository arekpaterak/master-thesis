from typing import Literal, get_args

ProblemName = Literal["tsp", "cvrp", "cmstp"]
PROBLEM_NAMES: tuple[ProblemName, ...] = get_args(ProblemName)
