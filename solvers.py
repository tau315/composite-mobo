"""Bayesian-optimization solvers for direct and composite multi-objective problems.

All public solvers assume that ``evaluate(X)`` returns objectives to *minimize*.
Composite solvers additionally receive ``evaluate_components(X)`` and a known,
differentiable ``compose(C, X)`` map satisfying
``compose(components(X), X) == f(X)``. The GP models only unknown intermediate
outputs; analytically known functions of ``X`` remain exact in the outer map.
Internally objectives are negated because BoTorch's acquisition functions use a
maximization convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import sys
from typing import Callable, Optional

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
import botorch.acquisition.multi_objective.logei as _multi_objective_logei
from botorch.acquisition.multi_objective.objective import (
    GenericMCMultiOutputObjective,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions.errors import OptimizationGradientError
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood

Tensor = torch.Tensor
Evaluator = Callable[[Tensor], Tensor]
Composer = Callable[[Tensor, Tensor], Tensor]

# BoTorch 0.18 tries to JIT-build an optional fused qLogEHVI kernel. On Windows
# without the MSVC compiler this produces a long subprocess traceback before
# correctly falling back to the identical pure-Python calculation. Select that
# fallback up front. This private sentinel can be removed once BoTorch exposes a
# public switch for disabling the optional extension.
if sys.platform == "win32" and shutil.which("cl") is None:
    _multi_objective_logei._load_attempted = True


@dataclass
class SolverResult:
    """Evaluated design and objective data returned by every solver."""

    X: Tensor
    Y: Tensor
    components: Optional[Tensor] = None
    weights: Optional[Tensor] = None
    run_ids: Optional[Tensor] = None


def smooth_tchebycheff(
    Y: Tensor, weights: Tensor, ideal: Tensor, temperature: float = 0.05
) -> Tensor:
    """Smooth Tchebycheff loss for minimization.

    Computes ``t log sum_i exp(w_i (y_i-z_i*) / t)``. As temperature tends to
    zero this converges to ``max_i w_i(y_i-z_i*)``.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return temperature * torch.logsumexp(
        weights * (Y - ideal) / temperature, dim=-1
    )


def simplex_weights(n: int, m: int, *, seed: int = 0) -> Tensor:
    """Deterministic boundary-covering weights for 2-D, Dirichlet otherwise."""

    if n < 1 or m < 2:
        raise ValueError("n >= 1 and m >= 2 are required")
    if m == 2:
        w = torch.linspace(0.05, 0.95, n, dtype=torch.double)
        return torch.stack((w, 1.0 - w), dim=-1)
    generator = torch.Generator().manual_seed(seed)
    e = -torch.log(torch.rand(n, m, generator=generator, dtype=torch.double))
    return e / e.sum(dim=-1, keepdim=True)


def _sobol(n: int, d: int, seed: int) -> Tensor:
    return torch.quasirandom.SobolEngine(d, scramble=True, seed=seed).draw(n).double()


def _single_gp(X: Tensor, y: Tensor) -> SingleTaskGP:
    model = SingleTaskGP(X, y, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def _independent_gp(X: Tensor, Y: Tensor) -> ModelListGP:
    models = [
        SingleTaskGP(X, Y[:, i : i + 1], outcome_transform=Standardize(m=1))
        for i in range(Y.shape[-1])
    ]
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def _optimize(acq, d: int, raw_samples: int, num_restarts: int) -> Tensor:
    bounds = torch.stack((torch.zeros(d), torch.ones(d))).double()
    try:
        X, _ = optimize_acqf(
            acq,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )
        return X.detach()
    except OptimizationGradientError:
        # A nonlinear composite map can occasionally make a local acquisition
        # gradient non-finite. Preserve the BO run by selecting the best finite
        # acquisition value from a fresh, space-filling Sobol candidate set.
        candidates = draw_sobol_samples(
            bounds=bounds, n=max(raw_samples * num_restarts, 256), q=1
        )
        with torch.no_grad():
            values = acq(candidates)
        values = torch.nan_to_num(values, nan=-torch.inf, neginf=-torch.inf)
        if not torch.isfinite(values).any():
            raise RuntimeError("acquisition was non-finite on every fallback candidate")
        return candidates[values.argmax()].detach()


def _check_composition(C: Tensor, X: Tensor, Y: Tensor, compose: Composer) -> None:
    reconstructed = compose(C, X)
    if reconstructed.shape != Y.shape or not torch.allclose(
        reconstructed, Y, atol=1e-7, rtol=1e-5
    ):
        raise ValueError("compose(evaluate_components(X), X) must equal evaluate(X)")


def standard_mobo(
    evaluate: Evaluator,
    dim: int,
    ref_point: Tensor,
    *,
    n_init: int = 8,
    n_iter: int = 24,
    seed: int = 0,
    raw_samples: int = 128,
    num_restarts: int = 8,
) -> SolverResult:
    """Independent objective GPs followed by numerically stable sequential qLogEHVI."""

    torch.manual_seed(seed)
    X = _sobol(n_init, dim, seed)
    Y = evaluate(X).double()
    ref_max = -torch.as_tensor(ref_point, dtype=torch.double)
    for _ in range(n_iter):
        model = _independent_gp(X, -Y)
        partitioning = NondominatedPartitioning(ref_point=ref_max, Y=-Y)
        acq = qLogExpectedHypervolumeImprovement(
            model=model, ref_point=ref_max.tolist(), partitioning=partitioning
        )
        x = _optimize(acq, dim, raw_samples, num_restarts)
        X, Y = torch.cat((X, x)), torch.cat((Y, evaluate(x).double()))
    return SolverResult(X=X, Y=Y)


def composite_mobo(
    evaluate: Evaluator,
    evaluate_components: Evaluator,
    compose: Composer,
    dim: int,
    ref_point: Tensor,
    *,
    n_init: int = 8,
    n_iter: int = 24,
    seed: int = 0,
    raw_samples: int = 128,
    num_restarts: int = 8,
) -> SolverResult:
    """Intermediate-node GPs and qLogEHVI on composed samples (MO-BOCF)."""

    torch.manual_seed(seed)
    X = _sobol(n_init, dim, seed)
    C, Y = evaluate_components(X).double(), evaluate(X).double()
    _check_composition(C, X, Y, compose)
    ref_max = -torch.as_tensor(ref_point, dtype=torch.double)
    objective = GenericMCMultiOutputObjective(
        lambda samples, X=None: -compose(samples, X)
    )
    for _ in range(n_iter):
        model = _independent_gp(X, C)
        partitioning = NondominatedPartitioning(ref_point=ref_max, Y=-Y)
        acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_max.tolist(),
            partitioning=partitioning,
            objective=objective,
        )
        x = _optimize(acq, dim, raw_samples, num_restarts)
        c, y = evaluate_components(x).double(), evaluate(x).double()
        X, C, Y = torch.cat((X, x)), torch.cat((C, c)), torch.cat((Y, y))
    return SolverResult(X=X, Y=Y, components=C)


def _scalarized_runs(
    evaluate: Evaluator,
    dim: int,
    weights: Tensor,
    ideal: Tensor,
    temperature: float,
    n_init: int,
    n_iter: int,
    seed: int,
    raw_samples: int,
    num_restarts: int,
    evaluate_components: Optional[Evaluator] = None,
    compose: Optional[Composer] = None,
) -> SolverResult:
    all_x, all_y, all_c, ids = [], [], [], []
    for run_id, weight in enumerate(weights.double()):
        run_seed = seed + 104729 * run_id
        torch.manual_seed(run_seed)
        X = _sobol(n_init, dim, run_seed)
        Y = evaluate(X).double()
        if evaluate_components is None:
            C = None
        else:
            C = evaluate_components(X).double()
            _check_composition(C, X, Y, compose)  # type: ignore[arg-type]
        for _ in range(n_iter):
            if C is None:
                # Paper-style CBO: model f_1,...,f_m directly, then apply the
                # known STCH map to joint posterior samples inside EI.
                model = _independent_gp(X, Y)
                objective = GenericMCObjective(
                    lambda samples, X=None, w=weight: -smooth_tchebycheff(
                        samples, w, ideal, temperature
                    )
                )
                observed_utility = -smooth_tchebycheff(Y, weight, ideal, temperature)
                acq = qLogExpectedImprovement(
                    model=model, best_f=observed_utility.max(), objective=objective
                )
            else:
                model = _independent_gp(X, C)
                objective = GenericMCObjective(
                    lambda samples, X=None, w=weight: -smooth_tchebycheff(
                        compose(samples, X), w, ideal, temperature  # type: ignore[misc]
                    )
                )
                observed_utility = -smooth_tchebycheff(Y, weight, ideal, temperature)
                acq = qLogExpectedImprovement(
                    model=model, best_f=observed_utility.max(), objective=objective
                )
            x = _optimize(acq, dim, raw_samples, num_restarts)
            y = evaluate(x).double()
            X, Y = torch.cat((X, x)), torch.cat((Y, y))
            if C is not None:
                C = torch.cat((C, evaluate_components(x).double()))  # type: ignore[misc]
        all_x.append(X)
        all_y.append(Y)
        ids.append(torch.full((len(X),), run_id, dtype=torch.long))
        if C is not None:
            all_c.append(C)
    return SolverResult(
        X=torch.cat(all_x),
        Y=torch.cat(all_y),
        components=torch.cat(all_c) if all_c else None,
        weights=weights,
        run_ids=torch.cat(ids),
    )


def chebyshev_bo(
    evaluate: Evaluator,
    dim: int,
    weights: Tensor,
    ideal: Tensor,
    *,
    temperature: float = 0.05,
    n_init: int = 6,
    n_iter: int = 12,
    seed: int = 0,
    raw_samples: int = 128,
    num_restarts: int = 8,
) -> SolverResult:
    """Objective GPs and EI through STCH posterior samples, one run per weight."""

    return _scalarized_runs(
        evaluate, dim, weights, ideal.double(), temperature, n_init, n_iter,
        seed, raw_samples, num_restarts,
    )


def composite_chebyshev_bo(
    evaluate: Evaluator,
    evaluate_components: Evaluator,
    compose: Composer,
    dim: int,
    weights: Tensor,
    ideal: Tensor,
    *,
    temperature: float = 0.05,
    n_init: int = 6,
    n_iter: int = 12,
    seed: int = 0,
    raw_samples: int = 128,
    num_restarts: int = 8,
) -> SolverResult:
    """Node GPs and EI on the double-composed smooth-Tchebycheff posterior."""

    return _scalarized_runs(
        evaluate, dim, weights, ideal.double(), temperature, n_init, n_iter,
        seed, raw_samples, num_restarts, evaluate_components, compose,
    )
