from __future__ import annotations

from pydantic import BaseModel, Field

from core.lns.acceptance_criterion import build_criterion
from core.lns.config import SearchParams
from core.lns.improvement_step import ParallelImprovementStep, SequentialImprovementStep
from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.lns.operators.repair import RepairOperator
from core.methods.base import Method


class RandomDestroyMethodConfig(BaseModel):
    """Config for a random-node-destroy baseline (no model)."""

    search: SearchParams = Field(
        default_factory=lambda: SearchParams(repair_type="greedy_reconnect", num_rollouts=1),
        description="Per-method search knobs.",
    )
    seed: int | None = Field(default=None, description="Optional seed for the random destroyer.")
    name_override: str | None = Field(
        default=None, description="Override the auto-derived method name."
    )


class RandomDestroyMethod:
    @classmethod
    def build(
        cls,
        config: RandomDestroyMethodConfig,
        *,
        repair: RepairOperator,
        generate_start_solution,
    ) -> Method:
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
        criterion = build_criterion(
            config.search.acceptance, n_steps=config.search.num_improvement_steps
        )

        name = config.name_override or (
            f"random-d{config.search.deconstruction_size}-r{config.search.n_repairs}"
        )

        return Method(
            name=name,
            search=config.search,
            generate_start_solution=generate_start_solution,
            improvement_step=improvement_step,
            criterion=criterion,
        )


RANDOM_DESTROY_PRESETS: dict[str, RandomDestroyMethodConfig] = {
    "source": RandomDestroyMethodConfig(
        search=SearchParams(
            repair_type="greedy_reconnect",
            num_rollouts=1,
            deconstruction_size=15,
            n_repairs=5,
            improvement_step_type="sequential",
        )
    ),
    "parallel": RandomDestroyMethodConfig(
        search=SearchParams(
            repair_type="greedy_reconnect",
            num_rollouts=100,
            deconstruction_size=15,
            n_repairs=1,
            improvement_step_type="parallel",
        )
    ),
}
