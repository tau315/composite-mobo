"""Predict whether a benchmark's composite structure is worth exploiting.

Composite modeling can only help when predicting the objectives *through* the
known map is more accurate than predicting them directly. This script measures
exactly that, with no optimization loop involved: fit independent GPs on a Sobol
training set, then score held-out objective predictions two ways.

    direct     : one GP per objective, predict f
    composite  : one GP per component, predict f as g(component posterior mean)

The reported ``advantage`` is the fraction of direct RMSE removed by going
through g. Positive means the composite surrogate is the better model of f,
which is a necessary condition for composite BO to win.

**A single split is not enough.** One Sobol seed gave SNAr +45.6%; another gave
-57.0% on the same problem, because averaging over objectives hid one objective
improving 43% while the other degraded 110%. Every number here is therefore
reported across several seeds, with the spread and the per-objective breakdown,
so an unstable screen cannot pass as a confident one.

Usage:
    python diagnose_composite.py                     # every benchmark
    python diagnose_composite.py benchmark_snar      # selected modules
"""

from __future__ import annotations

import glob
import importlib
import os
import sys

import numpy as np
import torch

from benchmark_common import BenchmarkProblem
from solvers import _independent_gp

Tensor = torch.Tensor


def _standardized_rmse(predicted: Tensor, actual: Tensor) -> Tensor:
    """Per-column RMSE divided by that column's spread on the test set.

    Scale-free, so objectives with wildly different units stay comparable and
    the numbers mean the same thing across benchmarks.
    """

    error = (predicted - actual).square().mean(dim=0).sqrt()
    spread = actual.std(dim=0).clamp_min(1e-12)
    return error / spread


def diagnose(
    problem: BenchmarkProblem, n_train: int = 32, n_test: int = 128, seed: int = 0
) -> dict[str, float]:
    """Compare direct and through-g surrogate accuracy on held-out designs."""

    engine = torch.quasirandom.SobolEngine(problem.dim, scramble=True, seed=seed)
    X = engine.draw(n_train + n_test).double()
    X_train, X_test = X[:n_train], X[n_train:]

    C_train = problem.evaluate_components(X_train).double()
    C_test = problem.evaluate_components(X_test).double()
    Y_train = problem.composed(C_train, X_train).double()
    Y_test = problem.composed(C_test, X_test).double()

    # Fitting needs gradients; only the predictions are taken under no_grad.
    direct_model = _independent_gp(X_train, Y_train)
    component_model = _independent_gp(X_train, C_train)
    with torch.no_grad():
        direct_prediction = direct_model.posterior(X_test).mean
        component_prediction = component_model.posterior(X_test).mean
        composite_prediction = problem.composed(component_prediction, X_test).double()

    direct_per_objective = _standardized_rmse(direct_prediction, Y_test)
    composite_per_objective = _standardized_rmse(composite_prediction, Y_test)
    direct_rmse = direct_per_objective.mean()
    composite_rmse = composite_per_objective.mean()
    component_rmse = _standardized_rmse(component_prediction, C_test).mean()

    return {
        "dim": problem.dim,
        "objectives": problem.num_objectives,
        "components": C_train.shape[-1],
        "direct_rmse": float(direct_rmse),
        "composite_rmse": float(composite_rmse),
        "component_rmse": float(component_rmse),
        # Fraction of the direct model's error that routing through g removes.
        # Clamping the denominator keeps a near-perfect direct model from
        # turning a negligible absolute difference into a huge ratio.
        "advantage": float(1.0 - composite_rmse / direct_rmse.clamp_min(1e-12)),
        "per_objective_advantage": [
            float(1.0 - c / d.clamp_min(1e-12))
            for d, c in zip(direct_per_objective, composite_per_objective)
        ],
    }


def diagnose_repeated(
    problem: BenchmarkProblem,
    seeds: int = 5,
    n_train: int = 32,
    n_test: int = 128,
) -> dict:
    """Repeat the screen over independent Sobol splits and report the spread.

    The worst per-objective advantage is reported alongside the mean because a
    benchmark can look good on average while one objective is badly degraded --
    which is exactly how a single split reported +45.6% for a problem that
    another split scored -57.0%.
    """

    reports = [diagnose(problem, n_train=n_train, n_test=n_test, seed=s) for s in range(seeds)]
    advantages = np.array([r["advantage"] for r in reports])
    worst_objective = min(min(r["per_objective_advantage"]) for r in reports)
    return {
        **reports[0],
        "advantage": float(advantages.mean()),
        "advantage_std": float(advantages.std(ddof=1)) if seeds > 1 else 0.0,
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
        "worst_objective_advantage": worst_objective,
        "seeds": seeds,
    }


def _benchmark_modules(names: list[str]) -> list[str]:
    if names:
        return names
    stems = [os.path.splitext(f)[0] for f in sorted(glob.glob("benchmark_*.py"))]
    return [s for s in stems if s != "benchmark_common"]


def main() -> None:
    header = (
        f"{'benchmark':<30}{'d':>6}{'m':>4}{'p':>5}"
        f"{'advantage (mean+-sd)':>22}{'range':>20}{'worst obj':>11}"
    )
    print(header)
    print("-" * len(header))
    for name in _benchmark_modules(sys.argv[1:]):
        problem = importlib.import_module(name).PROBLEM
        r = diagnose_repeated(problem)
        spread = f"[{r['advantage_min']:+.0%}, {r['advantage_max']:+.0%}]"
        print(
            f"{problem.slug:<30}{r['dim']:>6}{r['objectives']:>4}{r['components']:>5}"
            f"{r['advantage']:>15.1%} +-{r['advantage_std']:>5.1%}"
            f"{spread:>20}{r['worst_objective_advantage']:>11.1%}",
            flush=True,
        )


def demo() -> None:
    """Self-check: a benchmark whose g does the hard work must score positive."""

    def components(X: Tensor) -> Tensor:
        # A smooth, easily-learned intermediate.
        return torch.stack((X[..., 0], X[..., 1:].square().sum(dim=-1)), dim=-1)

    def compose(H: Tensor) -> Tensor:
        # A violently nonlinear known map: a direct GP on this is hopeless,
        # a GP on H is trivial.
        oscillation = torch.cos(12.0 * torch.pi * H[..., 0]) * (1.0 + H[..., 1])
        return torch.stack((oscillation, -oscillation + H[..., 1]), dim=-1)

    easy_g = BenchmarkProblem(
        name="identity-g control",
        slug="control",
        dim=4,
        num_objectives=2,
        suite="low",
        evaluate_components=components,
        compose=lambda H: torch.stack((H[..., 0], H[..., 1]), dim=-1),
        ideal=torch.zeros(2, dtype=torch.double),
        ref_point=torch.full((2,), 5.0, dtype=torch.double),
    )
    hard_g = BenchmarkProblem(
        name="nonlinear-g case",
        slug="nonlinear",
        dim=4,
        num_objectives=2,
        suite="low",
        evaluate_components=components,
        compose=compose,
        ideal=torch.full((2,), -5.0, dtype=torch.double),
        ref_point=torch.full((2,), 5.0, dtype=torch.double),
    )
    control = diagnose(easy_g)
    nonlinear = diagnose(hard_g)
    # When g is the identity the two surrogates are the same model, so there is
    # nothing to gain; when g hides the oscillation there is a lot.
    assert abs(control["advantage"]) < 0.05, control
    assert nonlinear["advantage"] > 0.5, nonlinear
    print(f"control advantage {control['advantage']:+.1%} (expected ~0)")
    print(f"nonlinear advantage {nonlinear['advantage']:+.1%} (expected large)")
    print("demo OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        sys.argv.remove("--demo")
        demo()
    else:
        main()
