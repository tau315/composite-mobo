"""Reproducible low-dimensional DTLZ/ZDT comparison for the four BO solvers.

Example (a quick smoke benchmark):
    python benchmark.py --problems zdt1 dtlz2 --trials 2 --budget 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from solvers import (
    SolverResult,
    chebyshev_bo,
    composite_chebyshev_bo,
    composite_mobo,
    simplex_weights,
    standard_mobo,
)

Tensor = torch.Tensor


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
    # Model log(g), an invertible observation transform of the unknown inner
    # function. Reconstructing with exp guarantees positive posterior samples,
    # which is essential because the known ZDT outer maps divide by g.
    def components(X: Tensor) -> Tensor:
        g = 1.0 + 9.0 * X[..., 1:].mean(dim=-1)
        return torch.log(g).unsqueeze(-1)

    def compose(C: Tensor, X: Tensor) -> Tensor:
        g = torch.exp(C[..., 0])
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

    return CompositeProblem(
        name, dim, 2, torch.tensor([1.1, 11.0], dtype=torch.double),
        torch.zeros(2, dtype=torch.double), components, compose,
    )


def _dtlz2(dim: int = 6, objectives: int = 2) -> CompositeProblem:
    # Only the unknown radial inner function g is modeled. Angular variables are
    # exact candidate coordinates supplied to the known hyperspherical outer map.
    k = dim - objectives + 1

    def components(X: Tensor) -> Tensor:
        g = (X[..., -k:] - 0.5).square().sum(dim=-1, keepdim=True)
        return g

    def compose(C: Tensor, X: Tensor) -> Tensor:
        g = C[..., 0]
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


def run(args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    """Run all trials and return arrays shaped (trials, evaluations)."""

    traces: dict[str, dict[str, list[np.ndarray]]] = {}
    for problem_name in args.problems:
        # Requested ZDT2 protocol; other problems use the command-line defaults.
        budget = 30 if problem_name == "zdt2" else args.budget
        initial = 3 if problem_name == "zdt2" else args.initial
        if budget < initial:
            raise ValueError("budget must be at least the number of initial points")
        if budget % args.weights:
            raise ValueError("budget must be divisible by weights for equal scalarization runs")
        scalar_run_budget = budget // args.weights
        if scalar_run_budget < initial:
            raise ValueError("budget / weights must be at least the initial-point count")
        problem = get_problem(problem_name, args.dim)
        traces[problem.name] = {
            "Standard qLogEHVI": [],
            "Composite qLogEHVI": [],
            "Objective-GP STCH": [],
            "Composite STCH": [],
        }
        for trial in range(args.trials):
            seed = args.seed + trial
            weights = simplex_weights(args.weights, problem.objectives, seed=seed)
            common = dict(
                seed=seed, raw_samples=args.raw_samples,
                num_restarts=args.restarts,
            )
            jobs = {
                "Standard qLogEHVI": lambda: standard_mobo(
                    problem.evaluate, problem.dim, problem.ref_point,
                    n_init=initial, n_iter=budget - initial, **common,
                ),
                "Composite qLogEHVI": lambda: composite_mobo(
                    problem.evaluate, problem.components, problem.compose,
                    problem.dim, problem.ref_point, n_init=initial,
                    n_iter=budget - initial, **common,
                ),
                "Objective-GP STCH": lambda: chebyshev_bo(
                    problem.evaluate, problem.dim, weights, problem.ideal,
                    temperature=args.temperature, n_init=initial,
                    n_iter=scalar_run_budget - initial, **common,
                ),
                "Composite STCH": lambda: composite_chebyshev_bo(
                    problem.evaluate, problem.components, problem.compose,
                    problem.dim, weights, problem.ideal,
                    temperature=args.temperature, n_init=initial,
                    n_iter=scalar_run_budget - initial, **common,
                ),
            }
            for method, job in jobs.items():
                start = perf_counter()
                result: SolverResult = job()
                trace = hypervolume_trace(result.Y, problem.ref_point)
                if len(trace) != budget:
                    raise RuntimeError(f"{method} used {len(trace)}, expected {budget}")
                traces[problem.name][method].append(trace)
                print(
                    f"{problem.name:6s} trial={trial + 1:02d}/{args.trials} "
                    f"{method:20s} HV={trace[-1]:.6f} "
                    f"time={perf_counter() - start:.1f}s"
                )
    return {
        problem: {method: np.stack(values) for method, values in methods.items()}
        for problem, methods in traces.items()
    }


def plot_results(
    traces: dict[str, dict[str, np.ndarray]], output: Path, initial: int
) -> None:
    """Write two separate pairwise-comparison figures per benchmark."""

    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix or ".png"
    prefix = output.stem
    comparisons = {
        "qlogehvi": ("Standard qLogEHVI", "Composite qLogEHVI"),
        "stch": ("Objective-GP STCH", "Composite STCH"),
    }
    colors = ("#0072B2", "#D55E00")

    for problem, methods in traces.items():
        problem_initial = 3 if problem == "zdt2" else initial
        for comparison_name, method_names in comparisons.items():
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
                problem_initial, color="0.45", linestyle="--", linewidth=1,
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
    p.add_argument("--budget", type=int, default=40, help="total evaluations per method and trial")
    p.add_argument("--initial", type=int, default=5, choices=[5], help="fixed at five as specified")
    p.add_argument("--weights", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw-samples", type=int, default=128)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--output", type=Path, default=Path("hypervolume_vs_evaluations.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    traces = run(args)
    plot_results(traces, args.output, args.initial)


if __name__ == "__main__":
    main()
