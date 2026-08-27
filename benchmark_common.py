"""Shared experiment runner and mathematical helpers for benchmark scripts.

Each benchmark lives in its own executable ``benchmark_*.py`` file. This
module centralizes the experimental protocol, hypervolume calculation, and
plotting so the scripts cannot silently drift to different conventions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
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
    batched_morbo,
    chebyshev_bo,
    composite_chebyshev_bo,
    composite_mobo,
    composite_batched_morbo,
    composite_spherical_chebyshev_bo,
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
    num_components: int | None = None
    exact_max_hypervolume: float | None = None
    prepare: Callable[[], None] | None = None
    clear_evaluation_cache: Callable[[], None] | None = None

    def evaluate(self, X: Tensor) -> Tensor:
        return self.compose(self.evaluate_components(X))

    def validate(self) -> None:
        """Validate metadata without spending an unrecorded simulator call."""

        if self.dim < 1 or self.num_objectives < 2:
            raise ValueError("invalid benchmark dimensions")
        if self.num_components is not None and self.num_components < 1:
            raise ValueError("num_components must be positive")
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
        description=f"Run 10-trial BO comparisons on {problem.name}."
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--initial", type=int, default=5 if problem.suite == "low" else 20
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        default=45 if problem.suite == "low" else 120,
        help="exact total expensive design evaluations per method",
    )
    parser.add_argument(
        "--weights",
        type=int,
        default=5,
        help="number of smooth-Tchebycheff scalarization weights",
    )
    parser.add_argument(
        "--per-weight",
        type=int,
        default=None,
        help="override evaluations per STCH weight; must exactly fill the budget",
    )
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--raw-samples", type=int, default=128 if problem.suite == "low" else 256
    )
    parser.add_argument(
        "--restarts", type=int, default=8 if problem.suite == "low" else 10
    )
    parser.add_argument("--morbo-raw-samples", type=int, default=512)
    parser.add_argument("--trust-regions", type=int, default=5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="joint MORBO batch size (high-dimensional suite only)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results") / problem.slug,
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
        "evaluations",
        "weights",
        "per_weight",
        "raw_samples",
        "restarts",
        "morbo_raw_samples",
        "trust_regions",
        "batch_size",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")


def _quick_arguments(args: argparse.Namespace) -> None:
    args.trials = 1
    args.initial = 3
    args.evaluations = 5
    args.weights = 2
    args.per_weight = 1
    args.raw_samples = 16
    args.restarts = 2
    args.morbo_raw_samples = 32
    args.trust_regions = 2
    args.batch_size = 1


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
            "n_iter": args.evaluations - args.initial,
            "seed": seed,
            "raw_samples": args.raw_samples,
            "num_restarts": args.restarts,
        }
        jobs = [
            (
                "Hypervolume",
                "qlogehvi",
                lambda: standard_mobo(
                    problem.evaluate,
                    problem.dim,
                    problem.ref_point,
                    **sequential_common,
                ),
            ),
            (
                "Composite hypervolume",
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
                "Tchebycheff",
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
                "Composite Tchebycheff",
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
            ("Hypervolume", ["Hypervolume", "Composite hypervolume"]),
            (
                "Smooth Tchebycheff",
                ["Tchebycheff", "Composite Tchebycheff"],
            ),
        ]
        return jobs, panels

    morbo_config = MORBOConfig(
        n_trust_regions=args.trust_regions, raw_samples=args.morbo_raw_samples
    )
    morbo_common = {
        "n_init": args.initial,
        "n_iter": (args.evaluations - args.initial) // args.batch_size,
        "seed": seed,
        "config": morbo_config,
        "batch_size": args.batch_size,
    }
    jobs = [
        (
            "Spherical Tchebycheff",
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
            "Composite spherical Tchebycheff",
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
            lambda: batched_morbo(
                problem.evaluate,
                problem.dim,
                problem.ref_point,
                **morbo_common,
            ),
        ),
        (
            "Composite MORBO",
            "morbo",
            lambda: composite_batched_morbo(
                problem.evaluate,
                problem.evaluate_components,
                problem.compose,
                problem.dim,
                problem.ref_point,
                num_components=problem.num_components,
                **morbo_common,
            ),
        ),
    ]
    panels = [
        (
            "Spherical-linear smooth Tchebycheff",
            ["Spherical Tchebycheff", "Composite spherical Tchebycheff"],
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
    family_colors = {
        panels[0][0]: "#2ca02c" if "Tchebycheff" in panels[0][0] else "#9467bd",
        panels[1][0]: "#2ca02c" if "Tchebycheff" in panels[1][0] else "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(10.8, 6.4), constrained_layout=True)
    longest_trace = 0
    for family_name, method_names in panels:
        color = family_colors[family_name]
        for method_name in method_names:
            trial_values = np.stack(traces[method_name], axis=0)
            mean = trial_values.mean(axis=0)
            if len(trial_values) > 1:
                sem = trial_values.std(axis=0, ddof=1) / np.sqrt(len(trial_values))
            else:
                sem = np.zeros_like(mean)
            evaluations = np.arange(1, len(mean) + 1)
            longest_trace = max(longest_trace, len(mean))
            linestyle = "--" if "Composite" in method_name else "-"
            ax.plot(
                evaluations,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=2.4,
                label=method_name,
            )
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
        alpha=0.75,
        label="End of initial design",
    )
    ax.axhline(
        problem.plotted_max_hypervolume,
        color="#2f2f2f",
        linestyle="-.",
        linewidth=1.3,
        alpha=0.8,
        label=(
            f"{problem.max_hypervolume_label} = "
            f"{problem.plotted_max_hypervolume:.4f}"
        ),
    )
    ax.set_title(problem.name, fontsize=14, pad=10)
    ax.set_xlabel("Total function evaluations")
    ax.set_ylabel("Dominated hypervolume")
    ax.set_xlim(1, max(longest_trace, 2))
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.01)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
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
    if args.per_weight is None:
        remaining = args.evaluations - args.initial
        if remaining % args.weights:
            raise ValueError(
                "--evaluations minus --initial must be divisible by --weights"
            )
        args.per_weight = remaining // args.weights
    _validate_arguments(args)
    if args.initial >= args.evaluations:
        raise ValueError("--initial must be smaller than --evaluations")
    expected_scalar_budget = args.initial + args.weights * args.per_weight
    if expected_scalar_budget != args.evaluations:
        raise ValueError(
            "--initial + --weights * --per-weight must equal --evaluations"
        )
    if problem.suite == "high":
        remaining = args.evaluations - args.initial
        if remaining % args.batch_size:
            raise ValueError(
                "high-dimensional --evaluations minus --initial must be "
                "divisible by --batch-size"
            )
    problem.validate()
    # Dataset loading and fixed benchmark-constant construction are setup,
    # not BO runtime. Scientific evaluators can perform that work once here.
    if problem.prepare is not None:
        problem.prepare()

    traces: dict[str, list[np.ndarray]] = {}
    designs: dict[str, list[np.ndarray]] = {}
    objectives: dict[str, list[np.ndarray]] = {}
    components: dict[str, list[np.ndarray]] = {}
    run_ids: dict[str, list[np.ndarray]] = {}
    scalarization_weights: dict[str, list[np.ndarray]] = {}
    runtimes: dict[str, list[float]] = {}
    panels: list[tuple[str, list[str]]] | None = None
    for trial in range(args.trials):
        seed = args.seed + 10_007 * trial
        jobs, panels = _solver_jobs(problem, args, seed)
        initial_designs_by_family: dict[str, torch.Tensor] = {}
        for method_name, family_name, job in jobs:
            # Never let a method inherit simulator results from the method run
            # before it. Within-method memoization is still available for an
            # optimizer that deliberately revisits the same design.
            if problem.clear_evaluation_cache is not None:
                problem.clear_evaluation_cache()
            started = time.perf_counter()
            result = job()
            if len(result.Y) != args.evaluations:
                raise RuntimeError(
                    f"{method_name} used {len(result.Y)} evaluations; "
                    f"expected exactly {args.evaluations}"
                )
            if result.Y.shape != (args.evaluations, problem.num_objectives):
                raise RuntimeError(
                    f"{method_name} returned objective shape {tuple(result.Y.shape)}"
                )
            if not torch.isfinite(result.Y).all():
                raise RuntimeError(f"{method_name} returned a non-finite objective")
            if result.X.shape != (args.evaluations, problem.dim):
                raise RuntimeError(
                    f"{method_name} returned design shape {tuple(result.X.shape)}"
                )
            if family_name not in initial_designs_by_family:
                initial_designs_by_family[family_name] = (
                    result.X[: args.initial].detach().cpu()
                )
            elif not torch.equal(
                initial_designs_by_family[family_name],
                result.X[: args.initial].detach().cpu(),
            ):
                raise RuntimeError(
                    f"{method_name} did not use its paired method's initial design"
                )
            if result.components is not None and problem.num_components is not None:
                expected = (args.evaluations, problem.num_components)
                if result.components.shape != expected:
                    raise RuntimeError(
                        f"{method_name} returned component shape "
                        f"{tuple(result.components.shape)}; expected {expected}"
                    )
            trace = dominated_hypervolume_trace(result.Y, problem.ref_point)
            traces.setdefault(method_name, []).append(trace)
            designs.setdefault(method_name, []).append(
                result.X.detach().double().cpu().numpy()
            )
            objectives.setdefault(method_name, []).append(
                result.Y.detach().double().cpu().numpy()
            )
            if result.components is not None:
                components.setdefault(method_name, []).append(
                    result.components.detach().double().cpu().numpy()
                )
            if result.run_ids is not None:
                run_ids.setdefault(method_name, []).append(
                    result.run_ids.detach().cpu().numpy()
                )
            if result.weights is not None:
                scalarization_weights.setdefault(method_name, []).append(
                    result.weights.detach().double().cpu().numpy()
                )
            elapsed = time.perf_counter() - started
            runtimes.setdefault(method_name, []).append(elapsed)
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
        args.output_dir / "hypervolume_vs_evaluations.png",
        args.show,
    )
    method_names = list(traces)
    archive_path = args.output_dir / "benchmark_data.npz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "problem": problem.name,
        "slug": problem.slug,
        "suite": problem.suite,
        "dimension": problem.dim,
        "num_objectives": problem.num_objectives,
        "num_components": problem.num_components,
        "trials": args.trials,
        "initial_evaluations": args.initial,
        "total_evaluations": args.evaluations,
        "weights": args.weights,
        "evaluations_per_weight": args.per_weight,
        "temperature": args.temperature,
        "scalarization_trace_order": "round_robin_by_weight",
        "acquisition_raw_samples": args.raw_samples,
        "acquisition_restarts": args.restarts,
        "acquisition_optimizer_maxiter": 200,
        "base_seed": args.seed,
        "trial_seed_stride": 10_007,
        "morbo_batch_size": args.batch_size if problem.suite == "high" else None,
        "morbo_raw_samples": (
            args.morbo_raw_samples if problem.suite == "high" else None
        ),
        "morbo_trust_regions": (
            args.trust_regions if problem.suite == "high" else None
        ),
        "morbo_success_streak": 10_000 if problem.suite == "high" else None,
        "morbo_failure_streak": (
            max(problem.dim // 3, 10) if problem.suite == "high" else None
        ),
        "morbo_min_tr_size": (
            max(1, min(args.initial - 1, 20))
            if problem.suite == "high"
            else None
        ),
        "morbo_engine": "vendored batched MORBO" if problem.suite == "high" else None,
        "evaluation_cache_reset_per_method": True,
        "methods": method_names,
    }
    archive_arrays: dict[str, np.ndarray] = {
        "method_names": np.asarray(method_names),
        "evaluation": np.arange(1, args.evaluations + 1, dtype=np.int64),
        "hypervolume": np.stack([np.stack(traces[name]) for name in method_names]),
        "designs": np.stack([np.stack(designs[name]) for name in method_names]),
        "objectives": np.stack([np.stack(objectives[name]) for name in method_names]),
        "runtimes_seconds": np.stack(
            [np.asarray(runtimes[name], dtype=np.float64) for name in method_names]
        ),
        "seeds": np.asarray(
            [args.seed + 10_007 * trial for trial in range(args.trials)],
            dtype=np.int64,
        ),
        "ideal": problem.ideal.detach().double().cpu().numpy(),
        "reference_point": problem.ref_point.detach().double().cpu().numpy(),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for method_name in method_names:
        key = "_".join(method_name.lower().split())
        if method_name in components:
            archive_arrays[f"components__{key}"] = np.stack(components[method_name])
        if method_name in run_ids:
            archive_arrays[f"run_ids__{key}"] = np.stack(run_ids[method_name])
        if method_name in scalarization_weights:
            archive_arrays[f"weights__{key}"] = np.stack(
                scalarization_weights[method_name]
            )
    np.savez_compressed(archive_path, **archive_arrays)
    print(f"Saved NumPy data: {archive_path.resolve()}")
