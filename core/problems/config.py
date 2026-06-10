from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from core.problems.cvrp.config import CVRP
    from core.problems.tsp.config import TSP


class BaseProblemConfig(BaseModel):
    """Shared input fields for every problem.

    Concrete problem configs (e.g. ``CVRPConfig``, ``TSPConfig``) subclass this, narrow
    ``name`` to a discriminating literal, add problem-specific fields, and build their
    domain problem definition via ``to_problem``.
    """

    name: str
    size: int = Field(default=20, description="Problem size (number of customers/locations).")

    def to_problem(self) -> CVRP | TSP:
        raise NotImplementedError
