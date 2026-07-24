# Import solve so it registers the default start solution with the singledispatch.
from core.problems.cmst import solve as _solve  # noqa: F401
from core.problems.cmst.instance import CMSTInstance
from core.problems.cmst.solution import CMSTSolution

__all__ = [
    "CMSTInstance",
    "CMSTSolution",
]
