"""Reproducible low-dimensional DTLZ/ZDT comparison for the four BO solvers.

Example (a quick smoke benchmark):
    python benchmark.py --problems zdt1 dtlz2 --trials 2 --budget 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import math
import platform
from pathlib import Path
import subprocess
from time import perf_counter
import traceback
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from solvers import (
    SolverResult,
    TIMING_KEYS,
    chebyshev_bo,
    composite_chebyshev_bo,
    composite_mobo,
    simplex_weights,
    standard_mobo,
)

Tensor = torch.Tensor

METHOD_LABELS = {
    "standard_qlogehvi": "Standard qLogEHVI",
    "composite_qlogehvi": "Composite qLogEHVI",
    "objective_gp_stch": "Objective-GP STCH",
    "composite_stch": "Composite STCH",
}
COMPARISONS = {
    "qlogehvi": ("standard_qlogehvi", "composite_qlogehvi"),
    "stch": ("objective_gp_stch", "composite_stch"),
}
RESULT_TIMING_KEYS = (*TIMING_KEYS, "hypervolume_seconds", "total_seconds")


@dataclass(frozen=True)
class CompositeProblem:
    name: str
    dim: int
    objectives: int
    ref_point: Tensor
    ideal: Tensor
    components: Callable[[Tensor], Tensor]
    compose: Callable[[Tensor, Tensor], Tensor]

    def evaluate(self, X: Tensor) -> Tensor:
        return self.compose(self.components(X), X)


def _zdt(name: str, dim: int = 6) -> CompositeProblem:
    # Model a radius whose square reconstructs the valid ZDT domain g >= 1.
    def components(X: Tensor) -> Tensor:
        g = 1.0 + 9.0 * X[..., 1:].mean(dim=-1)
        return torch.sqrt((g - 1.0).clamp_min(0.0)).unsqueeze(-1)

    def compose(C: Tensor, X: Tensor) -> Tensor:
        g = 1.0 + C[..., 0].square()
        # Expand exact x_1 across MC sample dimensions through broadcasting.
        f1 = X[..., 0] + torch.zeros_like(g)
        ratio = (f1 / g).clamp_min(0.0)
        if name == "zdt1":
            h = 1.0 - ratio.clamp_min(1e-12).sqrt()
        elif name == "zdt2":
            h = 1.0 - ratio.square()
        elif name == "zdt3":
            h = (
                1.0
                - ratio.clamp_min(1e-12).sqrt()
                - ratio * torch.sin(10.0 * torch.pi * f1)
            )
        else:
            raise ValueError(name)
        return torch.stack((f1, g * h), dim=-1)

    ideal = torch.tensor(
        [0.0, -0.7733690123] if name == "zdt3" else [0.0, 0.0],
        dtype=torch.double,
    )
    return CompositeProblem(
        name, dim, 2, torch.tensor([1.1, 11.0], dtype=torch.double),
        ideal, components, compose,
    )


def _dtlz2(dim: int = 6, objectives: int = 2) -> CompositeProblem:
    # Model sqrt(g); angular variables remain exact candidate coordinates.
    k = dim - objectives + 1

    def components(X: Tensor) -> Tensor:
        g = (X[..., -k:] - 0.5).square().sum(dim=-1, keepdim=True)
        return torch.sqrt(g)

    def compose(C: Tensor, X: Tensor) -> Tensor:
        g = C[..., 0].square()
        angles = X[..., : objectives - 1]
        # Give exact angles the posterior sample dimensions of g.
        angles = angles + torch.zeros_like(g).unsqueeze(-1)
        values = []
        for i in range(objectives):
            value = 1.0 + g
            for j in range(objectives - i - 1):
                value = value * torch.cos(angles[..., j] * torch.pi / 2)
            if i > 0:
                value = value * torch.sin(angles[..., objectives - i - 1] * torch.pi / 2)
            values.append(value)
        return torch.stack(values, dim=-1)

    return CompositeProblem(
        "dtlz2", dim, objectives,
        torch.full((objectives,), 2.5, dtype=torch.double),
        torch.zeros(objectives, dtype=torch.double), components, compose,
    )


def get_problem(name: str, dim: int) -> CompositeProblem:
    return _dtlz2(dim) if name == "dtlz2" else _zdt(name, dim)


def hypervolume(Y: Tensor, ref_point: Tensor) -> float:
    """Dominated HV for minimization data (negated for BoTorch)."""

    feasible = torch.all(Y <= ref_point, dim=-1)
    if not feasible.any():
        return 0.0
    max_y = -Y[feasible]
    max_y = max_y[is_non_dominated(max_y)]
    return float(Hypervolume(ref_point=-ref_point).compute(max_y))


def hypervolume_trace(Y: Tensor, ref_point: Tensor) -> np.ndarray:
    """Cumulative hypervolume after each expensive function evaluation."""

    return np.asarray(
        [hypervolume(Y[: i + 1], ref_point) for i in range(len(Y))], dtype=float
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _valid_result(path: Path, expected: dict) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "schema_version", "problem", "method", "trial", "seed", "config",
            "metadata", "X", "Y", "components", "weights", "run_ids",
            "hypervolume", "wall_seconds", "timing", "failed",
        }
        config_keys = {
            "dim", "budget", "initial", "weights", "temperature", "seed",
            "raw_samples", "restarts",
        }

        def number(value: object) -> bool:
            return type(value) in (int, float) and math.isfinite(value)

        def vector(values: object, length: int, nonnegative: bool = False) -> bool:
            return (
                isinstance(values, list)
                and len(values) == length
                and all(number(value) and (not nonnegative or value >= 0) for value in values)
            )

        def matrix(values: object, rows: int, columns: int | None = None) -> bool:
            return (
                isinstance(values, list)
                and len(values) == rows
                and rows > 0
                and all(isinstance(row, list) for row in values)
                and (width := len(values[0])) > 0
                and (columns is None or width == columns)
                and all(
                    len(row) == width and all(number(value) for value in row)
                    for row in values
                )
            )

        config = payload["config"]
        metadata = payload["metadata"]
        timing = payload["timing"]
        method = payload["method"]
        budget = expected["budget"]
        trial = int(path.stem.removeprefix("trial"))
        return (
            isinstance(payload, dict)
            and set(payload) == required
            and type(payload["schema_version"]) is int
            and payload["schema_version"] == 1
            and payload["problem"] == path.parent.parent.name
            and payload["method"] == path.parent.name
            and payload["method"] in METHOD_LABELS
            and type(payload["trial"]) is int
            and payload["trial"] == trial >= 0
            and type(payload["seed"]) is int
            and isinstance(expected, dict)
            and isinstance(config, dict)
            and set(config) == set(expected) == config_keys
            and config == expected
            and all(type(config[key]) is type(expected[key]) for key in config)
            and all(number(value) for value in config.values())
            and all(
                type(config[key]) is int and config[key] > 0
                for key in (
                    "dim", "budget", "initial", "weights", "raw_samples",
                    "restarts",
                )
            )
            and type(config["seed"]) is int
            and budget >= config["initial"]
            and budget % config["weights"] == 0
            and payload["seed"] == config["seed"]
            and isinstance(metadata, dict)
            and {"python", "packages", "git_commit"} <= metadata.keys()
            and isinstance(metadata["python"], str)
            and isinstance(metadata["packages"], dict)
            and isinstance(metadata["git_commit"], str)
            and all(
                isinstance(key, str) and (value is None or isinstance(value, str))
                for key, value in metadata["packages"].items()
            )
            and payload["failed"] is None
            and matrix(payload["X"], budget, config["dim"])
            and matrix(payload["Y"], budget, 2)
            and (
                matrix(payload["components"], budget)
                if method.startswith("composite_")
                else payload["components"] is None
            )
            and (
                (
                    matrix(payload["weights"], config["weights"], 2)
                    and isinstance(payload["run_ids"], list)
                    and all(type(value) is int for value in payload["run_ids"])
                    and payload["run_ids"]
                    == list(range(config["weights"]))
                    * (budget // config["weights"])
                )
                if method.endswith("_stch")
                else payload["weights"] is None and payload["run_ids"] is None
            )
            and vector(payload["hypervolume"], budget)
            and vector(payload["wall_seconds"], budget, nonnegative=True)
            and isinstance(timing, dict)
            and set(RESULT_TIMING_KEYS) <= timing.keys()
            and all(number(value) and value >= 0 for value in timing.values())
        )
    except Exception:
        return False


def _result_payload(
    problem_name: str,
    method: str,
    trial: int,
    config: dict,
    result: SolverResult | None,
    hypervolume_values: list[float],
    metadata: dict,
    failed: str | None,
) -> dict:
    return {
        "schema_version": 1,
        "problem": problem_name,
        "method": method,
        "trial": trial,
        "seed": config["seed"],
        "config": config,
        "metadata": metadata,
        "X": result.X.tolist() if result else [],
        "Y": result.Y.tolist() if result else [],
        "components": (
            result.components.tolist()
            if result and result.components is not None
            else None
        ),
        "weights": (
            result.weights.tolist()
            if result and result.weights is not None
            else None
        ),
        "run_ids": (
            result.run_ids.tolist()
            if result and result.run_ids is not None
            else None
        ),
        "hypervolume": hypervolume_values,
        "wall_seconds": result.wall_seconds if result else [],
        "timing": dict(result.timing) if result else {},
        "failed": failed,
    }


def _run_metadata() -> dict:
    packages = {}
    for package in ("numpy", "torch", "botorch", "gpytorch", "matplotlib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        commit = ""
    return {
        "python": platform.python_version(),
        "packages": packages,
        "git_commit": commit or "unknown",
    }


def _job_config(args: argparse.Namespace, problem: str, trial: int) -> dict:
    return {
        "dim": args.dim,
        "budget": 30 if problem == "zdt2" else args.budget,
        "initial": args.initial,
        "weights": args.weights,
        "temperature": args.temperature,
        "seed": args.seed + trial,
        "raw_samples": args.raw_samples,
        "restarts": args.restarts,
    }


def load_traces(args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    selected_trial = getattr(args, "trial", None)
    trials = range(args.trials) if selected_trial is None else (selected_trial,)
    selected_method = getattr(args, "method", None)
    selected_methods = (
        tuple(METHOD_LABELS) if selected_method is None else (selected_method,)
    )
    traces = {}
    for problem in args.problems:
        records = {method: {} for method in selected_methods}
        for method in selected_methods:
            for trial in trials:
                path = args.results_dir / problem / method / f"trial{trial}.json"
                expected = _job_config(args, problem, trial)
                if not _valid_result(path, expected):
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    values = np.asarray(payload["hypervolume"], dtype=float)
                except (OSError, TypeError, ValueError):
                    continue
                records[method][trial] = (payload["config"], values)

        methods = {}
        for pair in COMPARISONS.values():
            if any(method not in records for method in pair):
                continue
            paired_trials = sorted(set(records[pair[0]]) & set(records[pair[1]]))
            paired_trials = [
                trial
                for trial in paired_trials
                if records[pair[0]][trial][0] == records[pair[1]][trial][0]
            ]
            if not paired_trials:
                continue
            for method in pair:
                methods[METHOD_LABELS[method]] = np.stack(
                    [records[method][trial][1] for trial in paired_trials]
                )
        traces[problem] = methods
    return traces


def run(args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    """Run selected jobs atomically and load completed traces from disk."""

    selected_trial = getattr(args, "trial", None)
    if selected_trial is not None and not 0 <= selected_trial < args.trials:
        raise ValueError("trial must be zero-based and less than trials")
    trials = range(args.trials) if selected_trial is None else (selected_trial,)
    selected_method = getattr(args, "method", None)
    if selected_method is not None and selected_method not in METHOD_LABELS:
        raise ValueError(f"unknown method: {selected_method}")
    methods = tuple(METHOD_LABELS) if selected_method is None else (selected_method,)
    metadata = _run_metadata()

    for problem_name in args.problems:
        base_config = _job_config(args, problem_name, 0)
        budget = base_config["budget"]
        initial = base_config["initial"]
        if budget < initial:
            raise ValueError("budget must be at least the number of initial points")
        if budget % args.weights:
            raise ValueError("budget must be divisible by weights for equal scalarization runs")
        scalar_run_budget = budget // args.weights
        if scalar_run_budget < initial:
            raise ValueError("budget / weights must be at least the initial-point count")
        problem = get_problem(problem_name, args.dim)
        for trial in trials:
            seed = args.seed + trial
            weights = simplex_weights(args.weights, problem.objectives, seed=seed)
            common = dict(
                seed=seed, raw_samples=args.raw_samples,
                num_restarts=args.restarts,
            )
            config = _job_config(args, problem_name, trial)
            jobs = {
                "standard_qlogehvi": lambda: standard_mobo(
                    problem.evaluate, problem.dim, problem.ref_point,
                    n_init=initial, n_iter=budget - initial, **common,
                ),
                "composite_qlogehvi": lambda: composite_mobo(
                    problem.evaluate, problem.components, problem.compose,
                    problem.dim, problem.ref_point, n_init=initial,
                    n_iter=budget - initial, **common,
                ),
                "objective_gp_stch": lambda: chebyshev_bo(
                    problem.evaluate, problem.dim, weights, problem.ideal,
                    temperature=args.temperature, n_init=initial,
                    n_iter=scalar_run_budget - initial, **common,
                ),
                "composite_stch": lambda: composite_chebyshev_bo(
                    problem.evaluate, problem.components, problem.compose,
                    problem.dim, weights, problem.ideal,
                    temperature=args.temperature, n_init=initial,
                    n_iter=scalar_run_budget - initial, **common,
                ),
            }
            for method in methods:
                path = (
                    args.results_dir
                    / problem.name
                    / method
                    / f"trial{trial}.json"
                )
                if _valid_result(path, config):
                    print(
                        f"{problem.name:6s} trial={trial:02d} "
                        f"{METHOD_LABELS[method]:20s} resumed"
                    )
                    continue

                started = perf_counter()
                result = None
                hypervolume_values = []
                failed = None
                timing = dict.fromkeys(RESULT_TIMING_KEYS, 0.0)
                try:
                    result = jobs[method]()
                    timing.update(result.timing)
                    hypervolume_started = perf_counter()
                    try:
                        trace = hypervolume_trace(result.Y, problem.ref_point)
                    finally:
                        timing["hypervolume_seconds"] = (
                            perf_counter() - hypervolume_started
                        )
                    hypervolume_values = trace.tolist()
                    if len(trace) != budget:
                        raise RuntimeError(
                            f"{method} used {len(trace)}, expected {budget}"
                        )
                    timing["total_seconds"] = perf_counter() - started
                    payload = _result_payload(
                        problem.name,
                        method,
                        trial,
                        config,
                        result,
                        hypervolume_values,
                        metadata,
                        None,
                    )
                    payload["timing"] = timing
                    _atomic_write_json(path, payload)
                except Exception:
                    failed = traceback.format_exc()
                    timing = dict.fromkeys(RESULT_TIMING_KEYS, 0.0)
                    timing["total_seconds"] = perf_counter() - started
                    payload = _result_payload(
                        problem.name,
                        method,
                        trial,
                        config,
                        None,
                        [],
                        metadata,
                        failed,
                    )
                    payload["timing"] = timing
                    try:
                        _atomic_write_json(path, payload)
                    except Exception:
                        failed += (
                            "\nFailure artifact write failed:\n"
                            + traceback.format_exc()
                        )

                if failed is not None:
                    print(
                        f"{problem.name:6s} trial={trial:02d} "
                        f"{METHOD_LABELS[method]:20s} FAILED"
                    )
                    continue
                print(
                    f"{problem.name:6s} trial={trial:02d} "
                    f"{METHOD_LABELS[method]:20s} "
                    f"HV={hypervolume_values[-1]:.6f} "
                    f"time={timing['total_seconds']:.1f}s"
                )
    return load_traces(args)


def plot_results(
    traces: dict[str, dict[str, np.ndarray]],
    output: Path,
    initial: int,
    weights: int = 2,
) -> None:
    """Write two separate pairwise-comparison figures per benchmark."""

    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix or ".png"
    prefix = output.stem
    colors = ("#0072B2", "#D55E00")

    for problem, methods in traces.items():
        for comparison_name, method_keys in COMPARISONS.items():
            method_names = tuple(METHOD_LABELS[key] for key in method_keys)
            if any(method not in methods for method in method_names):
                continue
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            for color, method in zip(colors, method_names):
                values = methods[method]
                evaluations = np.arange(1, values.shape[1] + 1)
                mean = np.mean(values, axis=0)
                sem = (
                    np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
                    if values.shape[0] > 1
                    else np.zeros_like(mean)
                )
                ax.plot(evaluations, mean, color=color, linewidth=2, label=method)
                ax.fill_between(
                    evaluations, mean - sem, mean + sem, color=color, alpha=0.2
                )
            ax.axvline(
                (
                    initial * weights
                    if comparison_name == "stch"
                    else initial
                ),
                color="0.45", linestyle="--", linewidth=1,
                label="End of initial design",
            )
            ax.set_title(f"{problem.upper()}: {method_names[0]} vs {method_names[1]}")
            ax.set_xlabel("Total function evaluations")
            ax.set_ylabel("Dominated hypervolume")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            destination = output.with_name(
                f"{prefix}_{problem}_{comparison_name}{suffix}"
            )
            fig.savefig(destination, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote plot to {destination}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problems", nargs="+", default=["zdt1", "zdt2", "zdt3", "dtlz2"], choices=["zdt1", "zdt2", "zdt3", "dtlz2"])
    p.add_argument("--dim", type=int, default=6)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--trial", type=int, help="run one zero-based trial")
    p.add_argument("--method", choices=tuple(METHOD_LABELS), help="run one method")
    p.add_argument("--budget", type=int, default=40, help="total evaluations per method and trial")
    p.add_argument("--initial", type=int, default=5, choices=[5], help="fixed at five as specified")
    p.add_argument("--weights", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw-samples", type=int, default=128)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--output", type=Path, default=Path("hypervolume_vs_evaluations.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_only:
        traces = load_traces(args)
    else:
        traces = run(args)
        if args.trial is not None or args.method is not None:
            return
    plot_results(traces, args.output, args.initial, args.weights)


if __name__ == "__main__":
    main()
