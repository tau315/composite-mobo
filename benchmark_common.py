"""Shared experiment runner and mathematical helpers for benchmark scripts.

Each benchmark lives in its own executable ``benchmark_*.py`` file. This
module centralizes the experimental protocol, hypervolume calculation, and
plotting so the scripts cannot silently drift to different conventions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import traceback
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from solvers import (
    MORBOConfig,
    SolverResult,
    TIMING_KEYS,
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

RESULT_TIMING_KEYS = (*TIMING_KEYS, "hypervolume_seconds", "total_seconds")
PACKAGE_NAMES = ("numpy", "torch", "botorch", "gpytorch", "matplotlib")
VALIDATION_RTOL = 1e-12
VALIDATION_ATOL = 1e-12
SCHEMA_VERSION = 3
CONFIG_KEYS = (
    "dim",
    "num_objectives",
    "suite",
    "initial",
    "iterations",
    "weights",
    "per_weight",
    "temperature",
    "seed",
    "raw_samples",
    "restarts",
    "morbo_raw_samples",
    "trust_regions",
)


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


def method_key(method_name: str) -> str:
    """Stable on-disk directory name for a solver's display label."""

    return re.sub(r"[^a-z0-9]+", "_", method_name.lower()).strip("_")


def expected_evaluations(args: argparse.Namespace, family: str) -> int:
    """Total expensive evaluations a method family spends in one trial."""

    if family == "stch":
        return args.initial + args.weights * args.per_weight
    return args.initial + args.iterations


def _run_metadata() -> dict:
    packages = {}
    for package in PACKAGE_NAMES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    commit = os.environ.get("COMPOSITE_MOBO_COMMIT", "").strip()
    if not commit:
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


def _job_config(problem: BenchmarkProblem, args: argparse.Namespace, seed: int) -> dict:
    """Every knob that changes a trial's numbers, and nothing that does not."""

    return {
        "dim": problem.dim,
        "num_objectives": problem.num_objectives,
        "suite": problem.suite,
        "initial": args.initial,
        "iterations": args.iterations,
        "weights": args.weights,
        "per_weight": args.per_weight,
        "temperature": args.temperature,
        "seed": seed,
        "raw_samples": args.raw_samples,
        "restarts": args.restarts,
        "morbo_raw_samples": args.morbo_raw_samples,
        "trust_regions": args.trust_regions,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _result_payload(
    problem: BenchmarkProblem,
    method: str,
    family: str,
    trial: int,
    config: dict,
    result: SolverResult | None,
    hypervolume_values: list[float],
    metadata: dict,
    failed: str | None,
) -> dict:
    complete = result is not None and failed is None
    acquisition_fallbacks = getattr(result, "acquisition_fallbacks", 0)
    if type(acquisition_fallbacks) is not int or acquisition_fallbacks < 0:
        acquisition_fallbacks = 0
    return {
        "schema_version": SCHEMA_VERSION,
        "problem": problem.slug,
        "method": method,
        "family": family,
        "composite": "composite" in method,
        "trial": trial,
        "seed": config["seed"],
        "config": config,
        "metadata": metadata,
        "X": result.X.tolist() if complete else [],
        "Y": result.Y.tolist() if complete else [],
        "components": (
            result.components.tolist()
            if complete and result.components is not None
            else None
        ),
        "weights": (
            result.weights.tolist()
            if complete and result.weights is not None
            else None
        ),
        "run_ids": (
            result.run_ids.tolist()
            if complete and result.run_ids is not None
            else None
        ),
        "hypervolume": hypervolume_values,
        "wall_seconds": result.wall_seconds if complete else [],
        "timing": dict(result.timing) if result else {},
        "acquisition_fallbacks": acquisition_fallbacks,
        "failed": failed,
    }


def _validated_payload(
    path: Path,
    problem: BenchmarkProblem,
    expected: dict,
    metadata: dict | None = None,
    *,
    allow_foreign_metadata: bool = False,
) -> dict | None:
    """Return the artifact only if it re-derives exactly from this config.

    Everything expensive is recomputed from the stored designs: components,
    objectives, scalarization weights, and the hypervolume trace. An artifact
    that fails any check is treated as absent so the job simply reruns.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "schema_version", "problem", "method", "family", "composite",
            "trial", "seed", "config", "metadata", "X", "Y", "components",
            "weights", "run_ids", "hypervolume", "wall_seconds", "timing",
            "acquisition_fallbacks", "failed",
        }

        def number(value: object) -> bool:
            return type(value) in (int, float) and math.isfinite(value)

        def vector(values: object, length: int, nonnegative: bool = False) -> bool:
            return (
                isinstance(values, list)
                and len(values) == length
                and all(
                    number(value) and (not nonnegative or value >= 0)
                    for value in values
                )
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

        expected_metadata = metadata
        if not allow_foreign_metadata and expected_metadata is None:
            expected_metadata = _run_metadata()
        trial = int(path.stem.removeprefix("trial"))
        if not (
            isinstance(payload, dict)
            and set(payload) == required
            and type(payload["schema_version"]) is int
            and payload["schema_version"] == SCHEMA_VERSION
            and payload["problem"] == problem.slug
            and payload["problem"] == path.parent.parent.name
            and payload["method"] == path.parent.name
            and payload["family"] in ("qlogehvi", "stch", "morbo")
            and payload["composite"] is ("composite" in payload["method"])
            and type(payload["trial"]) is int
            and payload["trial"] == trial >= 0
            and type(payload["seed"]) is int
            and isinstance(expected, dict)
            and isinstance(payload["config"], dict)
            and set(payload["config"]) == set(expected) == set(CONFIG_KEYS)
            and payload["config"] == expected
            and all(
                type(payload["config"][key]) is type(expected[key])
                for key in payload["config"]
            )
        ):
            return None

        config = payload["config"]
        family = payload["family"]
        timing = payload["timing"]
        stch = family == "stch"
        counts = (
            "dim", "num_objectives", "initial", "iterations", "weights",
            "per_weight", "raw_samples", "restarts", "morbo_raw_samples",
            "trust_regions",
        )
        evaluations = (
            config["initial"] + config["weights"] * config["per_weight"]
            if stch
            else config["initial"] + config["iterations"]
        )
        if not (
            payload["seed"] == config["seed"]
            and all(type(config[key]) is int and config[key] > 0 for key in counts)
            and type(config["seed"]) is int
            and config["seed"] >= 0
            and config["suite"] in ("low", "high")
            and type(config["temperature"]) is float
            and config["temperature"] > 0
            and config["dim"] == problem.dim
            and config["num_objectives"] == problem.num_objectives
            and config["suite"] == problem.suite
            and isinstance(payload["metadata"], dict)
            and set(payload["metadata"]) == {"python", "packages", "git_commit"}
            and (allow_foreign_metadata or payload["metadata"] == expected_metadata)
            and isinstance(payload["metadata"]["python"], str)
            and bool(payload["metadata"]["python"].strip())
            and isinstance(payload["metadata"]["packages"], dict)
            and set(payload["metadata"]["packages"]) == set(PACKAGE_NAMES)
            and isinstance(payload["metadata"]["git_commit"], str)
            and bool(payload["metadata"]["git_commit"].strip())
            and all(
                isinstance(value, str) and bool(value.strip())
                for value in payload["metadata"]["packages"].values()
            )
            and payload["failed"] is None
            and matrix(payload["X"], evaluations, config["dim"])
            and matrix(payload["Y"], evaluations, config["num_objectives"])
            and vector(payload["hypervolume"], evaluations, nonnegative=True)
            and vector(payload["wall_seconds"], evaluations, nonnegative=True)
            and isinstance(timing, dict)
            and set(timing) == set(RESULT_TIMING_KEYS)
            and all(number(value) and value >= 0 for value in timing.values())
            and type(payload["acquisition_fallbacks"]) is int
            and payload["acquisition_fallbacks"] >= 0
        ):
            return None

        X = torch.tensor(payload["X"], dtype=torch.double)
        Y = torch.tensor(payload["Y"], dtype=torch.double)
        if not torch.all((0 <= X) & (X <= 1)):
            return None
        if payload["composite"]:
            expected_components = problem.evaluate_components(X).double()
            if not matrix(
                payload["components"], evaluations, expected_components.shape[-1]
            ):
                return None
            components = torch.tensor(payload["components"], dtype=torch.double)
            if not torch.allclose(
                components,
                expected_components,
                rtol=VALIDATION_RTOL,
                atol=VALIDATION_ATOL,
            ):
                return None
            expected_y = problem.compose(components)
        else:
            if payload["components"] is not None:
                return None
            expected_y = problem.evaluate(X)
        if not torch.allclose(
            Y, expected_y.double(), rtol=VALIDATION_RTOL, atol=VALIDATION_ATOL
        ):
            return None

        if stch:
            expected_weights = simplex_weights(
                config["weights"], problem.num_objectives, seed=314159
            )
            # One shared initial design (run id -1) then one block per weight.
            expected_ids = [-1] * config["initial"]
            for weight_id in range(config["weights"]):
                expected_ids += [weight_id] * config["per_weight"]
            if not (
                matrix(
                    payload["weights"], config["weights"], problem.num_objectives
                )
                and torch.allclose(
                    torch.tensor(payload["weights"], dtype=torch.double),
                    expected_weights,
                    rtol=VALIDATION_RTOL,
                    atol=VALIDATION_ATOL,
                )
                and isinstance(payload["run_ids"], list)
                and all(type(value) is int for value in payload["run_ids"])
                and payload["run_ids"] == expected_ids
            ):
                return None
        elif payload["weights"] is not None or payload["run_ids"] is not None:
            return None

        values = np.asarray(payload["hypervolume"], dtype=float)
        if np.any(np.diff(values) < -VALIDATION_ATOL) or not np.allclose(
            values,
            dominated_hypervolume_trace(Y, problem.ref_point),
            rtol=VALIDATION_RTOL,
            atol=VALIDATION_ATOL,
        ):
            return None
        return payload
    except Exception:
        return None


def _argument_parser(problem: BenchmarkProblem) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run 20-trial BO comparisons on {problem.name}."
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument(
        "--evaluations",
        type=int,
        default=50 if problem.suite == "low" else 400,
        help=(
            "target total expensive evaluations per method; STCH uses the "
            "largest equal allocation across all weights that does not exceed "
            "this target"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "override adaptive evaluations for qLogEHVI/MORBO "
            "(default: --evaluations minus --initial)"
        ),
    )
    parser.add_argument(
        "--weights",
        type=int,
        default=4 if problem.suite == "low" else 10,
        help="number of smooth-Tchebycheff scalarization weights",
    )
    parser.add_argument(
        "--per-weight",
        type=int,
        default=10 if problem.suite == "low" else None,
        help=(
            "override adaptive evaluations per STCH weight "
            "(low-dimensional default: 10; high-dimensional default: largest "
            "equal allocation within --evaluations)"
        ),
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"hypervolume_{problem.slug}.png"),
    )
    parser.add_argument("--show", action="store_true", help="also open the plot window")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="directory holding one resumable JSON artifact per solver run",
    )
    parser.add_argument(
        "--trial", type=int, default=None, help="run only this zero-based trial"
    )
    parser.add_argument(
        "--method", default=None, help="run only this method key (see --list-methods)"
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="print this benchmark's method keys and exit",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="plot from existing artifacts without running any solver",
    )
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
    args.evaluations = 5
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
    family_colors = {
        panels[0][0]: "#2ca02c" if "Tchebycheff" in panels[0][0] else "#9467bd",
        panels[1][0]: "#2ca02c" if "Tchebycheff" in panels[1][0] else "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(10.8, 6.4), constrained_layout=True)
    longest_trace = 0
    for family_name, method_names in panels:
        color = family_colors[family_name]
        for method_name in method_names:
            if method_name not in traces:
                continue
            trial_values = np.stack(traces[method_name], axis=0)
            mean = trial_values.mean(axis=0)
            if len(trial_values) > 1:
                sem = trial_values.std(axis=0, ddof=1) / np.sqrt(len(trial_values))
            else:
                sem = np.zeros_like(mean)
            evaluations = np.arange(1, len(mean) + 1)
            longest_trace = max(longest_trace, len(mean))
            linestyle = ":" if "Composite" in method_name else "-"
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


def load_traces(
    problem: BenchmarkProblem,
    args: argparse.Namespace,
    panels: list[tuple[str, list[str]]],
) -> dict[str, list[np.ndarray]]:
    """Read valid artifacts, keeping only trials each panel can compare.

    A direct/composite pair is only plotted for trials where both runs exist,
    agree on configuration and provenance, and started from the same initial
    design -- otherwise the curves would not be answering the same question.
    """

    trials = range(args.trials) if args.trial is None else (args.trial,)
    records: dict[str, dict[int, dict]] = {}
    for _, method_names in panels:
        for method_name in method_names:
            key = method_key(method_name)
            found: dict[int, dict] = {}
            for trial in trials:
                path = args.results_dir / problem.slug / key / f"trial{trial}.json"
                expected = _job_config(problem, args, args.seed + 10_007 * trial)
                payload = _validated_payload(
                    path, problem, expected, allow_foreign_metadata=True
                )
                if payload is not None:
                    found[trial] = payload
            records[method_name] = found

    traces: dict[str, list[np.ndarray]] = {}
    for _, method_names in panels:
        paired = sorted(
            set.intersection(*(set(records[name]) for name in method_names))
        )
        compatible = []
        family_metadata = None
        for trial in paired:
            payloads = [records[name][trial] for name in method_names]
            first = payloads[0]
            if (
                all(other["config"] == first["config"] for other in payloads)
                and all(other["metadata"] == first["metadata"] for other in payloads)
                and all(other["weights"] == first["weights"] for other in payloads)
                and all(
                    other["X"][: args.initial] == first["X"][: args.initial]
                    for other in payloads
                )
                and (family_metadata is None or first["metadata"] == family_metadata)
            ):
                compatible.append(trial)
                family_metadata = first["metadata"]
        if not compatible:
            continue
        for method_name in method_names:
            traces[method_name] = [
                np.asarray(records[method_name][trial]["hypervolume"], dtype=float)
                for trial in compatible
            ]
    return traces


def _run_jobs(problem: BenchmarkProblem, args: argparse.Namespace) -> None:
    """Run every selected job, writing one resumable artifact per run."""

    if args.trial is not None and not 0 <= args.trial < args.trials:
        raise ValueError("--trial must be zero-based and less than --trials")
    trials = range(args.trials) if args.trial is None else (args.trial,)
    reference_jobs, _ = _solver_jobs(problem, args, args.seed)
    keys = {method_key(name) for name, _, _ in reference_jobs}
    if args.method is not None and args.method not in keys:
        raise ValueError(f"unknown --method: {args.method} (choose from {sorted(keys)})")

    metadata = _run_metadata()
    failure_count = 0
    for trial in trials:
        seed = args.seed + 10_007 * trial
        config = _job_config(problem, args, seed)
        jobs, _ = _solver_jobs(problem, args, seed)
        for method_name, family, job in jobs:
            key = method_key(method_name)
            if args.method is not None and key != args.method:
                continue
            path = args.results_dir / problem.slug / key / f"trial{trial}.json"
            if _validated_payload(path, problem, config, metadata) is not None:
                print(
                    f"{problem.slug:<24} trial={trial:02d} {method_name:<28} resumed",
                    flush=True,
                )
                continue

            started = time.perf_counter()
            result = None
            failed = None
            timing = dict.fromkeys(RESULT_TIMING_KEYS, 0.0)
            hypervolume_values: list[float] = []
            try:
                result = job()
                timing.update(result.timing)
                hypervolume_started = time.perf_counter()
                try:
                    trace = dominated_hypervolume_trace(result.Y, problem.ref_point)
                finally:
                    timing["hypervolume_seconds"] = (
                        time.perf_counter() - hypervolume_started
                    )
                hypervolume_values = trace.tolist()
                expected = expected_evaluations(args, family)
                if len(trace) != expected:
                    raise RuntimeError(
                        f"{key} used {len(trace)} evaluations, expected {expected}"
                    )
                timing["total_seconds"] = time.perf_counter() - started
                payload = _result_payload(
                    problem, key, family, trial, config, result,
                    hypervolume_values, metadata, None,
                )
                payload["timing"] = timing
                _atomic_write_json(path, payload)
            except Exception:
                failed = traceback.format_exc()
                timing = {
                    name: (
                        float(value)
                        if type(value) in (int, float)
                        and math.isfinite(value)
                        and value >= 0
                        else 0.0
                    )
                    for name, value in (
                        (name, timing.get(name, 0.0)) for name in RESULT_TIMING_KEYS
                    )
                }
                timing["total_seconds"] = time.perf_counter() - started
                payload = _result_payload(
                    problem, key, family, trial, config, result, [], metadata, failed,
                )
                payload["timing"] = timing
                try:
                    _atomic_write_json(path, payload)
                except Exception:
                    failed += (
                        "\nFailure artifact write failed:\n" + traceback.format_exc()
                    )
            finally:
                del result
                gc.collect()

            if failed is not None:
                failure_count += 1
                print(
                    f"{problem.slug:<24} trial={trial:02d} {method_name:<28} FAILED",
                    flush=True,
                )
                continue
            print(
                f"{problem.slug:<24} trial={trial:02d} {method_name:<28} "
                f"HV={hypervolume_values[-1]:.6f} "
                f"time={timing['total_seconds']:.1f}s",
                flush=True,
            )
    if failure_count:
        noun = "job" if failure_count == 1 else "jobs"
        raise RuntimeError(f"{failure_count} benchmark {noun} failed")


def run_benchmark(problem: BenchmarkProblem) -> None:
    """Command-line entry point shared by every standalone benchmark file."""

    parser = _argument_parser(problem)
    args = parser.parse_args()
    if args.quick:
        _quick_arguments(args)
    if args.iterations is None:
        args.iterations = args.evaluations - args.initial
    if args.per_weight is None:
        args.per_weight = (args.evaluations - args.initial) // args.weights
    _validate_arguments(args)
    problem.validate()

    _, panels = _solver_jobs(problem, args, args.seed)
    if args.list_methods:
        for _, method_names in panels:
            for method_name in method_names:
                print(f"{method_key(method_name):<28} {method_name}")
        return

    if not args.summary_only:
        _run_jobs(problem, args)
        # A single-job worker writes its artifact and stops; plotting belongs
        # to whichever process can see every method.
        if args.trial is not None or args.method is not None:
            return

    traces = load_traces(problem, args, panels)
    if not traces:
        raise RuntimeError("no comparable artifacts were found")
    _plot_traces(problem, traces, panels, args.initial, args.output, args.show)
