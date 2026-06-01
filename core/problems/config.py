from __future__ import annotations

from pydantic import BaseModel, Field


class ProblemParams(BaseModel):
    """Configuration for the optimisation problem."""

    name: str = Field(default="cvrp", description="Problem to solve.")
    size: int = Field(default=20, description="Number of customers.")
