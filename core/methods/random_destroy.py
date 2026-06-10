from __future__ import annotations

from pydantic import BaseModel, Field

from core.lns.config import SearchParams
from core.lns.improvement_step import ParallelImprovementStep, SequentialImprovementStep
from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.methods.base import Method
from core.problems import ProblemConfig
from core.problems.setup import resolve_repair


class RandomDestroyMethodConfig(BaseModel):
    search: SearchParams = Field(default_factory=SearchParams)
    seed: int | None = Field(default=None, description="Optional seed for the random destroy.")


class RandomDestroyMethod:
    @classmethod
    def build(
        cls,
        config: RandomDestroyMethodConfig,
        *,
        problem: ProblemConfig,
        generate_start_solution,
    ) -> Method:
        repair = resolve_repair(
            problem.name,
            config.search.repair_type,
            deconstruction_size=config.search.deconstruction_size,
        )
        destroy = RandomNodeDestroy(
            destruction_size=config.search.deconstruction_size,
            seed=config.seed,
            stitch=repair.requires_stitch,
        )
        step_cls = (
            SequentialImprovementStep
            if config.search.improvement_step_type == "sequential"
            else ParallelImprovementStep
        )
        improvement_step = step_cls(
            destroy=destroy,
            repair=repair,
            num_rollouts=config.search.num_rollouts,
            n_repairs=config.search.n_repairs,
        )

        name = _build_name(config.search)

        return Method(
            name=name,
            search=config.search,
            generate_start_solution=generate_start_solution,
            improvement_step=improvement_step,
        )


def _build_name(search: SearchParams) -> str:
    return (
        f"random-destroy-{search.deconstruction_size}"
        f"+{search.repair_type}x{search.n_repairs}"
        f"-{search.improvement_step_type}x{search.num_improvement_steps}"
    )


RANDOM_DESTROY_PRESETS: dict[str, RandomDestroyMethodConfig] = {
    "source": RandomDestroyMethodConfig(
        search=SearchParams(
            repair_type="greedy_insert",
            num_rollouts=1,
            deconstruction_size=15,
            n_repairs=5,
            improvement_step_type="sequential",
        )
    ),
    "parallel": RandomDestroyMethodConfig(
        search=SearchParams(
            repair_type="greedy_insert",
            num_rollouts=100,
            deconstruction_size=15,
            n_repairs=1,
            improvement_step_type="parallel",
        )
    ),
}
