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

Usage:
    python diagnose_composite.py                     # every benchmark
    python diagnose_composite.py benchmark_dtlz2     # selected modules
"""

from __future__ import annotations

import glob
import importlib
import os
import sys

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
    Y_train = problem.compose(C_train).double()
    Y_test = problem.compose(C_test).double()

    # Fitting needs gradients; only the predictions are taken under no_grad.
    direct_model = _independent_gp(X_train, Y_train)
    component_model = _independent_gp(X_train, C_train)
    with torch.no_grad():
        direct_prediction = direct_model.posterior(X_test).mean
        component_prediction = component_model.posterior(X_test).mean
        composite_prediction = problem.compose(component_prediction).double()

    direct_rmse = _standardized_rmse(direct_prediction, Y_test).mean()
    composite_rmse = _standardized_rmse(composite_prediction, Y_test).mean()
    component_rmse = _standardized_rmse(component_prediction, C_test).mean()

    return {
        "dim": problem.dim,
        "objectives": problem.num_objectives,
        "components": C_train.shape[-1],
        "direct_rmse": float(direct_rmse),
        "composite_rmse": float(composite_rmse),
        "component_rmse": float(component_rmse),
        # Fraction of the direct model's error that routing through g removes.
        "advantage": float(1.0 - composite_rmse / direct_rmse.clamp_min(1e-12)),
    }


def _benchmark_modules(names: list[str]) -> list[str]:
    if names:
        return names
    stems = [os.path.splitext(f)[0] for f in sorted(glob.glob("benchmark_*.py"))]
    return [s for s in stems if s != "benchmark_common"]


def main() -> None:
    header = (
        f"{'benchmark':<30}{'suite':<6}{'d':>6}{'m':>4}{'p':>5}"
        f"{'direct':>9}{'composite':>11}{'advantage':>11}"
    )
    print(header)
    print("-" * len(header))
    for name in _benchmark_modules(sys.argv[1:]):
        problem = importlib.import_module(name).PROBLEM
        report = diagnose(problem)
        print(
            f"{problem.slug:<30}{problem.suite:<6}{report['dim']:>6}"
            f"{report['objectives']:>4}{report['components']:>5}"
            f"{report['direct_rmse']:>9.3f}{report['composite_rmse']:>11.3f}"
            f"{report['advantage']:>10.1%}",
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
