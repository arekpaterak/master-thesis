from __future__ import annotations

import time

import torch

from core.lns.acceptance_criterion import SimulatedAnnealingAcceptanceConfig
from core.lns.config import SearchParams, StartSolutionParams
from core.methods import build_method
from core.methods.random_destroy import RandomDestroyMethodConfig
from core.problems import make_problem_config
from core.problems.dataset import DatasetSampler
from core.solve import solve

DATASET = "data/synthetic/cvrp/vrp100_test_seed1234.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_method(
    step_type: str,
    n_aug: int,
    rollouts: int = 100,
    repairs: int = 5,
    steps: int = 100,
    warmup_steps: int = 50,
):
    search = SearchParams(
        repair_type="greedy_insert",
        num_improvement_steps=steps,
        num_rollouts=rollouts,
        deconstruction_size=15,
        n_repairs=repairs,
        improvement_step_type=step_type,
        first_repair_uses_rollout_order=True,
        insert_in_affected_routes_only=True,
        n_augmentations=n_aug,
        acceptance=SimulatedAnnealingAcceptanceConfig(),
        start_solution_params=StartSolutionParams(num_improvement_steps=warmup_steps),
    )
    cfg = RandomDestroyMethodConfig(seed=0, search=search)
    problem = make_problem_config(name="cvrp", size=100)
    return build_method(cfg, problem=problem, device=DEVICE)


def warmup(sampler, step_types: list[str]) -> None:
    """Trigger CUDA init and numba JIT cheaply, once per step type.

    A couple of search steps with the start-solution warmup disabled is enough to
    compile every kernel; the full 100+50 step solve is not needed here.
    """
    for step_type in step_types:
        method = make_method(step_type, n_aug=8, steps=2, warmup_steps=0)
        time_batch(method, sampler, 8)


def time_batch(method, sampler, batch_size: int) -> float:
    instance = sampler(batch_size).to(DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        solve(method, instance)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def run():
    sampler = DatasetSampler.from_file("cvrp", DATASET)
    print(f"device={DEVICE}, dataset instances={len(sampler)}\n")

    configs = [
        ("numba_sequential", 1, 1),
        ("numba_sequential", 1, 10),
        ("numba_sequential", 1, 100),
        ("numba_sequential", 1, 200),
        ("numba_sequential", 8, 1),
        ("numba_sequential", 8, 10),
        ("numba_sequential", 8, 100),
        ("numba_sequential", 8, 200),
    ]

    step_types = list(dict.fromkeys(step_type for step_type, _, _ in configs))
    print("warming up (CUDA init / numba JIT)...\n")
    warmup(sampler, step_types)

    print(
        f"{'step_type':>18} {'n_aug':>5} {'batch_size':>10} {'batch_time_s':>12} {'instances/s':>11} {'10k_estimated_s':>15}"
    )
    for step_type, n_aug, batch in configs:
        try:
            method = make_method(step_type, n_aug)
            elapsed = time_batch(method, sampler, batch)
            per_inst = elapsed / batch
            inst_s = 1.0 / per_inst
            est10k = per_inst * 10000
            print(
                f"{step_type:>18} {n_aug:>5} {batch:>10} {elapsed:>12.2f} {inst_s:>11.1f} {est10k:>15.0f}"
            )
        except Exception as e:
            print(f"{step_type:>18} {n_aug:>5} {batch:>6}  FAILED: {type(e).__name__}: {e}")
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
