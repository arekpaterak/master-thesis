from __future__ import annotations

from core.lns.operators.repair.greedy_reconnect import CVRPGreedyReconnectRepair
from core.models.nds.eval_config import EvaluationConfig
from core.models.nds.evaluation import EvaluationResult, configure_evaluation_step
from core.models.nds.models.ar import AutoregressiveNDSModel
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.cvrp.solve import one_customer_per_route_graph


def test_configure_eval_step_shapes() -> None:
    batch_size, num_steps, plot_k = 4, 3, 2
    sampler = CVRPSampler(CVRPConfig.for_n(20), seed=7)
    model = AutoregressiveNDSModel(problem_name="cvrp")

    eval_config = EvaluationConfig(
        num_test_instances=batch_size,
        test_seed=7,
        num_improvement_steps=num_steps,
        rollouts_per_step=4,
        n_repairs=1,
        num_plot_instances=plot_k,
    )

    eval_step = configure_evaluation_step(
        generate_start_solution=one_customer_per_route_graph,
        repair=CVRPGreedyReconnectRepair(),
        deconstruction_size=5,
        eval_config=eval_config,
        capture_every_n_improvement_steps=1,
    )

    instance = sampler(batch_size)
    result = eval_step(model, instance)

    assert isinstance(result, EvaluationResult)
    # step 0 + each of the captured LNS steps
    expected_n_captured = num_steps + 1
    assert len(result.step_labels) == expected_n_captured
    assert result.step_labels[0] == 0
    assert result.step_labels[-1] == num_steps

    assert len(result.per_instance_objective) == expected_n_captured
    for row in result.per_instance_objective:
        assert len(row) == batch_size

    assert len(result.cumulative_time_s) == expected_n_captured
    assert result.cumulative_time_s[0] == 0.0
    assert all(b >= a for a, b in zip(result.cumulative_time_s, result.cumulative_time_s[1:]))

    assert len(result.plot_snapshots) == plot_k
    for snapshots in result.plot_snapshots:
        assert len(snapshots) == expected_n_captured


def test_configure_eval_step_plot_count_clamps_to_batch_size() -> None:
    sampler = CVRPSampler(CVRPConfig.for_n(20), seed=0)
    model = AutoregressiveNDSModel(problem_name="cvrp")
    eval_config = EvaluationConfig(
        num_test_instances=2,
        num_improvement_steps=1,
        rollouts_per_step=2,
        n_repairs=1,
        num_plot_instances=10,
    )
    eval_step = configure_evaluation_step(
        generate_start_solution=one_customer_per_route_graph,
        repair=CVRPGreedyReconnectRepair(),
        deconstruction_size=5,
        eval_config=eval_config,
    )
    result = eval_step(model, sampler(2))
    assert len(result.plot_snapshots) == 2
