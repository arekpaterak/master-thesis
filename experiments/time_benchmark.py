"""Benchmark the wall-clock time of an LNS search per destroy method.

Builds each destroy method (random + the NDS/LNS-native model variants) with
untrained weights -- the forward cost is identical to a trained model -- and runs a
full ``num_steps`` search, recording the cumulative time at selected checkpoint steps
(10, 20, ..., 100). The whole search is repeated ``repeats`` times to report a mean
and std per checkpoint; each repeat rebuilds the method so the LNS-native one-time
instance encoding is paid (and timed) on every run -- it is a real per-search cost.

Warmup runs random destroy only: it compiles the numba kernel and initialises CUDA
without warming any model cache, so the timed search still pays the model costs.

Rows are upserted into ``experiments/results/tables/time_benchmark.csv`` keyed by
``(method, step_type, step)`` -- re-running overwrites matching rows.

Run with::

    uv run python experiments/time_benchmark.py
    uv run python experiments/time_benchmark.py --methods nds-ar --methods nds-slot
    uv run python experiments/time_benchmark.py --step-types numba_sequential --batch-size 8
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
import typer
from tqdm import tqdm

from core.lns.acceptance_criterion import (
    AcceptanceCriterion,
    GreedyAcceptanceConfig,
    SimulatedAnnealingAcceptanceConfig,
    build_criterion,
)
from core.lns.config import SearchParams
from core.lns.improvement_step import ImprovementStep, build_random_destroy_step
from core.lns.operators.repair import RepairOperator
from core.models.nds.config import JointModelParams, ModelParams, SplitModelParams
from core.models.nds.decoders.config import (
    AutoregressiveDecoderConfig,
    SimpleOnePassDecoderConfig,
    SlotDecoderConfig,
)
from core.models.nds.factory import build_model
from core.models.nds.improvement_step import build_nds_improvement_step
from core.problems import make_problem_config
from core.problems.instance import BaseProblemInstance
from core.problems.sampler import Sampler
from core.problems.solution import BaseSolution
from core.problems.solve import generate_initial_solution

OUTPUT_DEFAULT = Path(__file__).parent / "results" / "tables" / "time_benchmark.csv"

CSV_FIELDS = [
    "method",
    "step_type",
    "step",
    "objective",
    "objective_std",
    "cumulative_time_ms",
    "cumulative_time_std_ms",
    "mean_step_ms",
    "mean_step_std_ms",
    "problem",
    "problem_size",
    "batch_size",
    "num_parameters",
    "deconstruction_size",
    "num_rollouts",
    "n_repairs",
    "repair_type",
    "acceptance",
    "use_cuda",
    "repeats",
    "seed",
    "torch_version",
]

# Columns identifying a unique measurement; re-running overwrites the matching row.
KEY_FIELDS = (
    "method",
    "step_type",
    "step",
    "problem",
    "problem_size",
    "batch_size",
    "deconstruction_size",
)


@dataclass(frozen=True)
class MethodSpec:
    """A destroy method to benchmark. ``model_params is None`` means random destroy."""

    name: str
    model_params: ModelParams | None


def _method_specs(deconstruction_size: int) -> list[MethodSpec]:
    return [
        MethodSpec("random-destroy", None),
        MethodSpec(
            "nds-ar",
            JointModelParams(decoder=AutoregressiveDecoderConfig(use_hidden_decoding_state=True)),
        ),
        MethodSpec(
            "nds-ar-no-gru",
            JointModelParams(decoder=AutoregressiveDecoderConfig(use_hidden_decoding_state=False)),
        ),
        MethodSpec("nds-simple", JointModelParams(decoder=SimpleOnePassDecoderConfig())),
        MethodSpec(
            "nds-slot", JointModelParams(decoder=SlotDecoderConfig(num_slots=deconstruction_size))
        ),
        MethodSpec("lns-native-simple", SplitModelParams(decoder=SimpleOnePassDecoderConfig())),
        MethodSpec(
            "lns-native-slot",
            SplitModelParams(decoder=SlotDecoderConfig(num_slots=deconstruction_size)),
        ),
        MethodSpec("lns-native-ar", SplitModelParams(decoder=AutoregressiveDecoderConfig())),
    ]


def _build_step(
    spec: MethodSpec,
    search: SearchParams,
    *,
    problem_name: str,
    device: torch.device,
    seed: int,
) -> ImprovementStep:
    """Build a fresh improvement step. NDS models start with an empty instance cache,
    so the first timed step pays the (real) one-time encoding cost."""
    if spec.model_params is None:
        return build_random_destroy_step(search, problem_name=problem_name, seed=seed)

    model = build_model(spec.model_params, problem_name).to(device)
    model.eval()
    repair = RepairOperator.create(
        problem_name, search.repair_type, deconstruction_size=search.deconstruction_size
    )
    return build_nds_improvement_step(search, model=model, repair=repair, seed=seed)


def _count_parameters(spec: MethodSpec, problem_name: str) -> int:
    if spec.model_params is None:
        return 0
    return sum(p.numel() for p in build_model(spec.model_params, problem_name).parameters())


def _make_search(
    *,
    step_type: str,
    repair_type: str,
    num_steps: int,
    num_rollouts: int,
    deconstruction_size: int,
    n_repairs: int,
    acceptance,
) -> SearchParams:
    return SearchParams(
        repair_type=repair_type,
        num_improvement_steps=num_steps,
        num_rollouts=num_rollouts,
        deconstruction_size=deconstruction_size,
        n_repairs=n_repairs,
        improvement_step_type=step_type,
        acceptance=acceptance,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _run_search(
    step: ImprovementStep,
    solution: BaseSolution,
    criterion: AcceptanceCriterion,
    *,
    num_steps: int,
    checkpoints: list[int],
    device: torch.device,
) -> tuple[dict[int, float], dict[int, float]]:
    """Run the full search once, returning cumulative seconds and objective at checkpoints."""
    checkpoint_set = set(checkpoints)
    criterion.reset()
    cumulative: dict[int, float] = {}
    objective: dict[int, float] = {}
    with torch.no_grad():
        _sync(device)
        start = perf_counter()
        for s in tqdm(range(1, num_steps + 1), desc="search", leave=False):
            solution = step(solution, criterion)
            criterion.step()
            if s in checkpoint_set:
                _sync(device)
                cumulative[s] = perf_counter() - start
                objective[s] = solution.objective().mean().item()
    return cumulative, objective


def _warmup(
    instance: BaseProblemInstance,
    *,
    problem_name: str,
    step_types: list[str],
    repair_type: str,
    deconstruction_size: int,
    num_rollouts: int,
    n_repairs: int,
    acceptance,
    warmup: int,
    device: torch.device,
    seed: int,
) -> None:
    """Compile the numba kernel and initialise CUDA using random destroy only.

    Deliberately does not touch any model, so NDS instance-encoding caches stay cold
    for the timed runs."""
    for step_type in step_types:
        search = _make_search(
            step_type=step_type,
            repair_type=repair_type,
            num_steps=warmup,
            num_rollouts=num_rollouts,
            deconstruction_size=deconstruction_size,
            n_repairs=n_repairs,
            acceptance=acceptance,
        )
        step = build_random_destroy_step(search, problem_name=problem_name, seed=seed)
        criterion = build_criterion(search.acceptance, n_steps=max(warmup, 1))
        solution = generate_initial_solution(instance)
        criterion.reset()
        with torch.no_grad():
            for _ in range(warmup):
                solution = step(solution, criterion)
                criterion.step()
    _sync(device)


def _upsert_rows(output: Path, new_rows: list[dict[str, object]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if output.exists():
        with output.open(newline="") as f:
            existing = list(csv.DictReader(f))

    new_keys = {tuple(str(row[k]) for k in KEY_FIELDS) for row in new_rows}
    kept = [r for r in existing if tuple(r.get(k, "") for k in KEY_FIELDS) not in new_keys]

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in [*kept, *new_rows]:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    size: int = typer.Option(100, help="CVRP problem size (number of customers)."),
    batch_size: int = typer.Option(1, help="Instances per batch (more = less noisy timings)."),
    deconstruction_size: int = typer.Option(15, help="Nodes removed per rollout."),
    num_rollouts: int = typer.Option(100, help="Rollouts per improvement step."),
    n_repairs: int = typer.Option(5, help="Repair attempts per rollout; best is kept."),
    repair_type: str = typer.Option("greedy_insert", help="Repair operator."),
    num_steps: int = typer.Option(100, help="Total improvement steps per search."),
    checkpoints: list[int] = typer.Option(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="Steps at which the cumulative time is recorded.",
    ),
    step_types: list[str] = typer.Option(
        ["parallel", "numba_sequential"], help="Improvement step variant(s) to time."
    ),
    methods: list[str] = typer.Option(None, help="Method name(s) to run. Omit to run all."),
    acceptance: str = typer.Option("greedy", help="Acceptance criterion: greedy or sa."),
    warmup: int = typer.Option(
        3, help="Untimed random-destroy steps to compile numba / init CUDA."
    ),
    repeats: int = typer.Option(5, help="Full-search repetitions; reported as mean +/- std."),
    use_cuda: bool = typer.Option(True, help="Run on CUDA. Set --no-use-cuda to force CPU."),
    seed: int = typer.Option(0, help="Seed for instance sampling and destroy."),
    output: Path = typer.Option(OUTPUT_DEFAULT, help="CSV file to upsert rows into."),
) -> None:
    torch_device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    acceptance_config = (
        GreedyAcceptanceConfig() if acceptance == "greedy" else SimulatedAnnealingAcceptanceConfig()
    )
    checkpoints = sorted(c for c in checkpoints if c <= num_steps)

    problem_name = "cvrp"
    problem_config = make_problem_config(name=problem_name, size=size)
    sampler = Sampler.from_config(problem_config, seed=seed)
    instance = sampler(batch_size).to(torch_device)

    all_specs = _method_specs(deconstruction_size)
    if methods:
        wanted = set(methods)
        unknown = wanted - {spec.name for spec in all_specs}
        if unknown:
            raise typer.BadParameter(f"Unknown method(s): {', '.join(sorted(unknown))}")
        all_specs = [spec for spec in all_specs if spec.name in wanted]

    typer.echo(
        f"Benchmarking {problem_name}{size} | batch={batch_size} | device={torch_device} | "
        f"rollouts={num_rollouts} | n_repairs={n_repairs} | steps={num_steps} | repeats={repeats}"
    )

    _warmup(
        instance,
        problem_name=problem_name,
        step_types=step_types,
        repair_type=repair_type,
        deconstruction_size=deconstruction_size,
        num_rollouts=num_rollouts,
        n_repairs=n_repairs,
        acceptance=acceptance_config,
        warmup=warmup,
        device=torch_device,
        seed=seed,
    )

    for spec in all_specs:
        num_parameters = _count_parameters(spec, problem_name)
        for step_type in step_types:
            search = _make_search(
                step_type=step_type,
                repair_type=repair_type,
                num_steps=num_steps,
                num_rollouts=num_rollouts,
                deconstruction_size=deconstruction_size,
                n_repairs=n_repairs,
                acceptance=acceptance_config,
            )

            time_samples_per_checkpoint: dict[int, list[float]] = {c: [] for c in checkpoints}
            objective_samples_per_checkpoint: dict[int, list[float]] = {c: [] for c in checkpoints}
            for _ in range(repeats):
                step = _build_step(
                    spec, search, problem_name=problem_name, device=torch_device, seed=seed
                )
                solution = generate_initial_solution(instance)
                criterion = build_criterion(search.acceptance, n_steps=num_steps)
                cumulative, objective = _run_search(
                    step,
                    solution,
                    criterion,
                    num_steps=num_steps,
                    checkpoints=checkpoints,
                    device=torch_device,
                )
                for c in checkpoints:
                    time_samples_per_checkpoint[c].append(cumulative[c])
                    objective_samples_per_checkpoint[c].append(objective[c])

            new_rows: list[dict[str, object]] = []
            for c in checkpoints:
                time_samples = time_samples_per_checkpoint[c]
                objective_samples = objective_samples_per_checkpoint[c]
                mean_ms = statistics.mean(time_samples) * 1e3
                std_ms = statistics.stdev(time_samples) * 1e3 if len(time_samples) > 1 else 0.0
                mean_objective = statistics.mean(objective_samples)
                std_objective = (
                    statistics.stdev(objective_samples) if len(objective_samples) > 1 else 0.0
                )
                new_rows.append(
                    {
                        "method": spec.name,
                        "step_type": step_type,
                        "step": c,
                        "objective": round(mean_objective, 6),
                        "objective_std": round(std_objective, 6),
                        "cumulative_time_ms": round(mean_ms, 4),
                        "cumulative_time_std_ms": round(std_ms, 4),
                        "mean_step_ms": round(mean_ms / c, 4),
                        "mean_step_std_ms": round(std_ms / c, 4),
                        "problem": problem_name,
                        "problem_size": size,
                        "batch_size": batch_size,
                        "num_parameters": num_parameters,
                        "deconstruction_size": deconstruction_size,
                        "num_rollouts": num_rollouts,
                        "n_repairs": n_repairs,
                        "repair_type": repair_type,
                        "acceptance": acceptance,
                        "use_cuda": torch_device.type == "cuda",
                        "repeats": repeats,
                        "seed": seed,
                        "torch_version": torch.__version__,
                    }
                )
            _upsert_rows(output, new_rows)

            final = new_rows[-1]
            typer.echo(
                f"  {spec.name:<18} {step_type:<16} "
                f"total {final['cumulative_time_ms']:>9} ms "
                f"+/-{final['cumulative_time_std_ms']:<8} | "
                f"{final['mean_step_ms']} +/-{final['mean_step_std_ms']} ms/step"
            )

    typer.echo(f"Upserted results into {output}")


if __name__ == "__main__":
    app()
