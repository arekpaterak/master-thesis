from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from core.problems.cmstp.config import CMSTPDistribution
    from core.problems.cvrp.config import CVRPDistribution
    from core.problems.tsp.config import TSPDistribution


class BaseProblemConfig(BaseModel):
    """Shared input fields for every problem.

    Concrete problem configs (e.g. ``CVRPConfig``, ``TSPConfig``) subclass this, narrow
    ``name`` to a discriminating literal, add problem-specific fields, and build their
    instance distribution via ``to_distribution``.
    """

    name: str
    size: int = Field(
        default=20,
        title="Problem size",
        description="Problem size (number of customers/locations).",
    )

    def to_distribution(self) -> CVRPDistribution | TSPDistribution | CMSTPDistribution:
        raise NotImplementedError
