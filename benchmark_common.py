"""Shared experiment runner and mathematical helpers for benchmark scripts.

Each benchmark lives in its own executable ``benchmark_*.py`` file.  This
module only centralizes the experimental protocol, hypervolume calculation,
plotting, and reusable test-function primitives so the six scripts cannot
silently drift to different budgets or plotting conventions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
from pathlib import Path
import time
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from solvers import (
    MORBOConfig,
    SolverResult,
    chebyshev_bo,
    composite_chebyshev_bo,
    composite_mobo,
    composite_morbo,
    composite_spherical_chebyshev_bo,
    morbo,
    simplex_weights,
    spherical_chebyshev_bo,
    standard_mobo,
)

Tensor = torch.Tensor
Evaluator = Callable[[Tensor], Tensor]
Composer = Callable[[Tensor], Tensor]
Suite = Literal["low", "high"]


ACKLEY_UPPER_BOUND = 20.0 + np.e - np.exp(-1.0)


@dataclass(frozen=True)
class BenchmarkProblem:
    """A deterministic composite multi-objective minimization problem."""

    name: str
    slug: str
    dim: int
    num_objectives: int
    suite: Suite
    evaluate_components: Evaluator
    compose: Composer
    ideal: Tensor
    ref_point: Tensor
    exact_max_hypervolume: float | None = None

    def evaluate(self, X: Tensor) -> Tensor:
        return self.compose(self.evaluate_components(X))

    def validate(self) -> None:
        if self.dim < 1 or self.num_objectives < 2:
            raise ValueError("invalid benchmark dimensions")
        probe = (
            torch.quasirandom.SobolEngine(self.dim, scramble=True, seed=1729)
            .draw(8)
            .double()
        )
        components = self.evaluate_components(probe).double()
        objectives = self.compose(components).double()
        if components.ndim != 2 or components.shape[0] != len(probe):
            raise ValueError("components must have shape n x number_of_components")
        if objectives.shape != (len(probe), self.num_objectives):
            raise ValueError("objectives must have shape n x number_of_objectives")
        if not torch.isfinite(components).all() or not torch.isfinite(objectives).all():
            raise ValueError("benchmark returned a non-finite value")
        if self.ideal.shape != (self.num_objectives,):
            raise ValueError("ideal point has the wrong shape")
        if self.ref_point.shape != (self.num_objectives,):
            raise ValueError("reference point has the wrong shape")
        if not (self.ref_point > self.ideal).all():
            raise ValueError("reference point must be worse than the ideal point")

    @property
    def plotted_max_hypervolume(self) -> float:
        """Exact maximum when known, otherwise the ideal/reference box ceiling."""

        if self.exact_max_hypervolume is not None:
            return self.exact_max_hypervolume
        return float((self.ref_point - self.ideal).prod())

    @property
    def max_hypervolume_label(self) -> str:
        return "Maximum HV" if self.exact_max_hypervolume is not None else "HV ceiling"


def orthogonal_matrix(dim: int, seed: int) -> Tensor:
    """Return a fixed deterministic dense orthogonal matrix."""

    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(dim, dim, generator=generator, dtype=torch.double)
    q, r = torch.linalg.qr(raw)
    signs = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0)
    return q * signs


def orthonormal_rows(rows: int, dim: int, seed: int) -> Tensor:
    """Return ``rows`` fixed dense orthonormal directions in R^dim."""

    if rows > dim:
        raise ValueError("the number of rows cannot exceed the dimension")
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(dim, rows, generator=generator, dtype=torch.double)
    q, r = torch.linalg.qr(raw, mode="reduced")
    signs = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0)
    return (q * signs).T.contiguous()


def transformed_inputs(
    X: Tensor, center: Tensor, rotation: Tensor, scale: float
) -> Tensor:
    """Shift, densely rotate, and scale points from the unit cube."""

    return scale * ((X.double() - center.double()) @ rotation.double().T)


def ackley_components(Z: Tensor) -> Tensor:
    """The two canonical intermediate quantities of the Ackley function."""

    return torch.stack(
        (
            Z.square().mean(dim=-1),
            torch.cos(2.0 * torch.pi * Z).mean(dim=-1),
        ),
        dim=-1,
    )


def compose_ackley(H: Tensor, *, normalize: bool = True) -> Tensor:
    """Apply the known Ackley outer map to two intermediate quantities."""

    mean_square = H[..., 0].clamp_min(0)
    mean_cosine = H[..., 1].clamp(-1.0, 1.0)
    value = (
        -20.0 * torch.exp(-0.2 * mean_square.sqrt())
        - torch.exp(mean_cosine)
        + 20.0
        + torch.e
    )
    return value / ACKLEY_UPPER_BOUND if normalize else value


def griewank_components(Z: Tensor) -> Tensor:
    """The quadratic and cosine-product intermediates of Griewank."""

    indices = torch.arange(1, Z.shape[-1] + 1, dtype=Z.dtype, device=Z.device)
    return torch.stack(
        (
            Z.square().sum(dim=-1) / 4000.0,
            torch.cos(Z / indices.sqrt()).prod(dim=-1),
        ),
        dim=-1,
    )


def compose_griewank(H: Tensor, upper_bound: float) -> Tensor:
    """Apply and safely normalize the known Griewank outer map."""

    quadratic = H[..., 0].clamp_min(0)
    cosine_product = H[..., 1].clamp(-1.0, 1.0)
    return (1.0 + quadratic - cosine_product) / upper_bound


def griewank_upper_bound(center: Tensor, scale: float) -> float:
    """A conservative bound over the unit cube, invariant to rotation."""

    furthest_squared = torch.maximum(center.square(), (1.0 - center).square()).sum()
    return float(2.0 + scale**2 * furthest_squared / 4000.0)


def compose_langermann(
    H: Tensor, coefficients: Tensor, targets: Tensor | None = None
) -> Tensor:
    """Normalized minimization form of the generalized Langermann outer map.

    If ``targets`` is omitted, H contains squared distances.  Otherwise H
    contains projections and the squared distances are ``(H - targets)^2``.
    The theoretical range [-sum(c), sum(c)] is mapped to [0, 1].
    """

    coefficients = coefficients.to(dtype=H.dtype, device=H.device)
    distances = (
        H.clamp_min(0)
        if targets is None
        else (H - targets.to(dtype=H.dtype, device=H.device)).square()
    )
    raw = -(
        coefficients
        * torch.exp(-distances / torch.pi)
        * torch.cos(torch.pi * distances)
    ).sum(dim=-1)
    total = coefficients.sum()
    return (raw + total) / (2.0 * total)


def dominated_hypervolume_trace(Y: Tensor, ref_point: Tensor) -> np.ndarray:
    """Compute prefix dominated hypervolume for minimization objectives."""

    values = -Y.detach().double().cpu()
    ref = -ref_point.detach().double().cpu()
    hypervolume = Hypervolume(ref_point=ref)
    trace = np.zeros(len(values), dtype=np.float64)
    for end in range(1, len(values) + 1):
        prefix = values[:end]
        valid = (prefix > ref).all(dim=-1)
        if valid.any():
            front = prefix[valid]
            front = front[is_non_dominated(front)]
            trace[end - 1] = float(hypervolume.compute(front))
    return trace


def _argument_parser(problem: BenchmarkProblem) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run 20-trial BO comparisons on {problem.name}."
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--weights", type=int, default=8)
    parser.add_argument("--per-weight", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--raw-samples", type=int, default=128 if problem.suite == "low" else 256
    )
    parser.add_argument(
        "--restarts", type=int, default=8 if problem.suite == "low" else 10
    )
    parser.add_argument("--morbo-raw-samples", type=int, default=512)
    parser.add_argument("--trust-regions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"hypervolume_{problem.slug}.png"),
    )
    parser.add_argument("--show", action="store_true", help="also open the plot window")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one very small trial for installation/smoke testing only",
    )
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    positive = (
        "trials",
        "initial",
        "iterations",
        "weights",
        "per_weight",
        "raw_samples",
        "restarts",
        "morbo_raw_samples",
        "trust_regions",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")


def _quick_arguments(args: argparse.Namespace) -> None:
    args.trials = 1
    args.initial = 3
    args.iterations = 1
    args.weights = 2
    args.per_weight = 1
    args.raw_samples = 16
    args.restarts = 2
    args.morbo_raw_samples = 32
    args.trust_regions = 2


def _solver_jobs(
    problem: BenchmarkProblem, args: argparse.Namespace, seed: int
) -> tuple[
    list[tuple[str, str, Callable[[], SolverResult]]], list[tuple[str, list[str]]]
]:
    weights = simplex_weights(args.weights, problem.num_objectives, seed=314159)
    scalar_common = {
        "n_init": args.initial,
        "n_per_scalarization": args.per_weight,
        "temperature": args.temperature,
        "seed": seed,
        "raw_samples": args.raw_samples,
        "num_restarts": args.restarts,
    }
    if problem.suite == "low":
        sequential_common = {
            "n_init": args.initial,
            "n_iter": args.iterations,
            "seed": seed,
            "raw_samples": args.raw_samples,
            "num_restarts": args.restarts,
        }
        jobs = [
            (
                "Direct qLogEHVI",
                "qlogehvi",
                lambda: standard_mobo(
                    problem.evaluate,
                    problem.dim,
                    problem.ref_point,
                    **sequential_common,
                ),
            ),
            (
                "Composite qLogEHVI",
                "qlogehvi",
                lambda: composite_mobo(
                    problem.evaluate,
                    problem.evaluate_components,
                    problem.compose,
                    problem.dim,
                    problem.ref_point,
                    **sequential_common,
                ),
            ),
            (
                "Objective-GP STCH",
                "stch",
                lambda: chebyshev_bo(
                    problem.evaluate,
                    problem.dim,
                    weights,
                    problem.ideal,
                    **scalar_common,
                ),
            ),
            (
                "Composite STCH",
                "stch",
                lambda: composite_chebyshev_bo(
                    problem.evaluate,
                    problem.evaluate_components,
                    problem.compose,
                    problem.dim,
                    weights,
                    problem.ideal,
                    **scalar_common,
                ),
            ),
        ]
        panels = [
            ("qLogEHVI", ["Direct qLogEHVI", "Composite qLogEHVI"]),
            ("Smooth Tchebycheff", ["Objective-GP STCH", "Composite STCH"]),
        ]
        return jobs, panels

    morbo_config = MORBOConfig(
        n_trust_regions=args.trust_regions, raw_samples=args.morbo_raw_samples
    )
    morbo_common = {
        "n_init": args.initial,
        "n_iter": args.iterations,
        "seed": seed,
        "config": morbo_config,
    }
    jobs = [
        (
            "Spherical objective STCH",
            "stch",
            lambda: spherical_chebyshev_bo(
                problem.evaluate,
                problem.dim,
                weights,
                problem.ideal,
                **scalar_common,
            ),
        ),
        (
            "Spherical composite STCH",
            "stch",
            lambda: composite_spherical_chebyshev_bo(
                problem.evaluate,
                problem.evaluate_components,
                problem.compose,
                problem.dim,
                weights,
                problem.ideal,
                **scalar_common,
            ),
        ),
        (
            "MORBO",
            "morbo",
            lambda: morbo(
                problem.evaluate,
                problem.dim,
                problem.ref_point,
                **morbo_common,
            ),
        ),
        (
            "Composite MORBO",
            "morbo",
            lambda: composite_morbo(
                problem.evaluate,
                problem.evaluate_components,
                problem.compose,
                problem.dim,
                problem.ref_point,
                **morbo_common,
            ),
        ),
    ]
    panels = [
        (
            "Spherical-linear smooth Tchebycheff",
            ["Spherical objective STCH", "Spherical composite STCH"],
        ),
        ("MORBO", ["MORBO", "Composite MORBO"]),
    ]
    return jobs, panels


def _plot_traces(
    problem: BenchmarkProblem,
    traces: dict[str, list[np.ndarray]],
    panels: list[tuple[str, list[str]]],
    n_initial: int,
    output: Path,
    show: bool,
) -> None:
    colors = {
        panels[0][1][0]: "#1f77b4",
        panels[0][1][1]: "#ff7f0e",
        panels[1][1][0]: "#1f77b4",
        panels[1][1][1]: "#ff7f0e",
    }
    fig, axes = plt.subplots(
        1, 2, figsize=(12.4, 4.8), constrained_layout=True, sharey=True
    )
    for ax, (panel_title, method_names) in zip(axes, panels):
        for method_name in method_names:
            trial_values = np.stack(traces[method_name], axis=0)
            mean = trial_values.mean(axis=0)
            if len(trial_values) > 1:
                sem = trial_values.std(axis=0, ddof=1) / np.sqrt(len(trial_values))
            else:
                sem = np.zeros_like(mean)
            evaluations = np.arange(1, len(mean) + 1)
            color = colors[method_name]
            ax.plot(evaluations, mean, color=color, linewidth=2.2, label=method_name)
            ax.fill_between(
                evaluations,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        ax.axvline(
            n_initial,
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="End of initial design",
        )
        ax.axhline(
            problem.plotted_max_hypervolume,
            color="#2f2f2f",
            linestyle=":",
            linewidth=1.4,
            label=(
                f"{problem.max_hypervolume_label} = "
                f"{problem.plotted_max_hypervolume:.4f}"
            ),
        )
        ax.set_title(panel_title, pad=9)
        ax.set_xlabel("Total function evaluations")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.01)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    axes[0].set_ylabel("Dominated hypervolume")
    fig.suptitle(problem.name, fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {output.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_benchmark(problem: BenchmarkProblem) -> None:
    """Command-line entry point shared by every standalone benchmark file."""

    parser = _argument_parser(problem)
    args = parser.parse_args()
    if args.quick:
        _quick_arguments(args)
    _validate_arguments(args)
    problem.validate()

    traces: dict[str, list[np.ndarray]] = {}
    panels: list[tuple[str, list[str]]] | None = None
    for trial in range(args.trials):
        seed = args.seed + 10_007 * trial
        jobs, panels = _solver_jobs(problem, args, seed)
        for method_name, _, job in jobs:
            started = time.perf_counter()
            result = job()
            trace = dominated_hypervolume_trace(result.Y, problem.ref_point)
            traces.setdefault(method_name, []).append(trace)
            elapsed = time.perf_counter() - started
            print(
                f"{problem.slug:<30} trial={trial + 1:02d}/{args.trials:02d} "
                f"{method_name:<28} HV={trace[-1]:.6f} time={elapsed:.1f}s",
                flush=True,
            )
            del result
            gc.collect()

    if panels is None:
        raise RuntimeError("no solver jobs were configured")
    _plot_traces(
        problem,
        traces,
        panels,
        args.initial,
        args.output,
        args.show,
    )
