from __future__ import annotations

import pytest
import torch

from core.eval_config import EvaluationConfig
from core.evaluation import BatchEvaluationResult, Evaluator, PlotSnapshotCallback
from core.lns.acceptance_criterion import GreedyAcceptanceConfig
from core.methods.base import Method
from core.methods.random_destroy import RANDOM_DESTROY_PRESETS, RandomDestroyMethod
from core.problems.config import ProblemParams
from core.problems.cvrp import CVRPSampler
from core.problems.cvrp.config import CVRPConfig
from core.problems.setup import resolve_start_solution
from core.solve import solve


def _build(
    preset: str, problem: ProblemParams, *, num_steps: int = 3, greedy: bool = False
) -> Method:
    base = RANDOM_DESTROY_PRESETS[preset]
    search = base.search.model_copy(
        update={
            "num_improvement_steps": num_steps,
            "acceptance": GreedyAcceptanceConfig() if greedy else base.search.acceptance,
        }
    )
    config = base.model_copy(update={"search": search, "seed": 0})
    return RandomDestroyMethod.build(
        config,
        problem=problem,
        generate_start_solution=resolve_start_solution(problem.name),
    )


@pytest.fixture
def tiny_eval_config() -> EvaluationConfig:
    return EvaluationConfig(
        problem=ProblemParams(name="cvrp", size=20),
        num_test_instances=4,
        batch_size=2,
        test_seed=7,
        num_plot_instances=0,
    )


@pytest.mark.parametrize("preset", ["source", "parallel"])
def test_evaluate_returns_well_shaped_result(
    preset: str, tiny_eval_config: EvaluationConfig
) -> None:
    method = _build(preset, tiny_eval_config.problem)
    result = Evaluator().evaluate(tiny_eval_config, method)

    n_captures = method.search.num_improvement_steps + 1
    assert result.method_name == method.name
    assert result.step_labels == list(range(n_captures))
    assert result.objective_per_step.shape == (n_captures, tiny_eval_config.num_test_instances)


def test_objective_non_increasing_with_greedy_acceptance(
    tiny_eval_config: EvaluationConfig,
) -> None:
    method = _build("source", tiny_eval_config.problem, greedy=True)
    result = Evaluator().evaluate(tiny_eval_config, method)
    diffs = result.objective_per_step[1:] - result.objective_per_step[:-1]
    assert torch.all(diffs <= 1e-6)


def test_batched_eval_concatenates_along_instance_dim(tiny_eval_config: EvaluationConfig) -> None:
    method = _build("source", tiny_eval_config.problem, greedy=True)
    cfg_single = tiny_eval_config.model_copy(update={"batch_size": 4})
    cfg_split = tiny_eval_config.model_copy(update={"batch_size": 2})

    r_single = Evaluator().evaluate(cfg_single, method)
    r_split = Evaluator().evaluate(cfg_split, method)

    assert r_single.objective_per_step.shape == r_split.objective_per_step.shape


def test_short_tail_batch_is_allowed(tiny_eval_config: EvaluationConfig) -> None:
    cfg = tiny_eval_config.model_copy(update={"num_test_instances": 5, "batch_size": 2})
    method = _build("source", cfg.problem)
    result = Evaluator().evaluate(cfg, method)
    assert result.objective_per_step.shape[1] == 5


def test_callbacks_fire_once_per_batch(tiny_eval_config: EvaluationConfig) -> None:
    method = _build("source", tiny_eval_config.problem)
    calls: list[tuple[int, BatchEvaluationResult]] = []

    Evaluator(on_batch_end=[lambda idx, br: calls.append((idx, br))]).evaluate(
        tiny_eval_config, method
    )

    expected_batches = tiny_eval_config.num_test_instances // tiny_eval_config.batch_size
    assert len(calls) == expected_batches
    assert [idx for idx, _ in calls] == list(range(expected_batches))
    n_captures = method.search.num_improvement_steps + 1
    for _idx, batch_result in calls:
        assert batch_result.objective_per_step.shape == (
            n_captures,
            tiny_eval_config.batch_size,
        )


def test_determinism_same_seed(tiny_eval_config: EvaluationConfig) -> None:
    method_a = _build("source", tiny_eval_config.problem, greedy=True)
    method_b = _build("source", tiny_eval_config.problem, greedy=True)
    r1 = Evaluator().evaluate(tiny_eval_config, method_a)
    r2 = Evaluator().evaluate(tiny_eval_config, method_b)
    torch.testing.assert_close(r1.objective_per_step, r2.objective_per_step)


def test_solve_returns_solution_with_matching_batch_shape(
    tiny_eval_config: EvaluationConfig,
) -> None:
    method = _build("source", tiny_eval_config.problem)
    sampler = CVRPSampler(
        CVRPConfig.for_n(tiny_eval_config.problem.size), seed=tiny_eval_config.test_seed
    )
    instance = sampler(2)
    final_solution = solve(method, instance)
    assert final_solution.batch_size == instance.batch_size


def test_plot_snapshot_callback_zero(tiny_eval_config: EvaluationConfig) -> None:
    method = _build("source", tiny_eval_config.problem)
    cb = PlotSnapshotCallback(num_plot_instances=0)
    Evaluator(on_step=[cb.on_step]).evaluate(tiny_eval_config, method)
    assert cb.snapshots == []


def test_plot_snapshot_callback_first_batch_only(tiny_eval_config: EvaluationConfig) -> None:
    cfg = tiny_eval_config.model_copy(update={"num_plot_instances": 2})
    method = _build("source", cfg.problem)
    cb = PlotSnapshotCallback(num_plot_instances=cfg.num_plot_instances)
    Evaluator(on_step=[cb.on_step]).evaluate(cfg, method)

    n_captures = method.search.num_improvement_steps + 1
    assert len(cb.snapshots) == 2  # min(num_plot_instances, first_batch_size=2)
    for trajectory in cb.snapshots:
        assert len(trajectory) == n_captures


def test_plot_snapshot_callback_warns_when_batch_too_small(
    tiny_eval_config: EvaluationConfig,
) -> None:
    # batch_size=2 < num_plot_instances=5; expect warning + capped capture.
    cfg = tiny_eval_config.model_copy(update={"num_plot_instances": 5})
    method = _build("source", cfg.problem)
    cb = PlotSnapshotCallback(num_plot_instances=cfg.num_plot_instances)
    with pytest.warns(UserWarning, match="first batch_size=2"):
        Evaluator(on_step=[cb.on_step]).evaluate(cfg, method)
    assert len(cb.snapshots) == 2  # capped to first batch_size


def test_solve_on_step_fires_once_per_step_plus_initial() -> None:
    problem = ProblemParams(name="cvrp", size=20)
    method = _build("source", problem, num_steps=4)
    sampler = CVRPSampler(CVRPConfig.for_n(problem.size), seed=0)
    seen_steps: list[int] = []

    def on_step(step: int, _solution) -> None:
        seen_steps.append(step)

    solve(method, sampler(2), on_step=on_step)
    assert seen_steps == [0, 1, 2, 3, 4]
