"""Bayesian-optimization solvers for direct and composite multi-objective problems.

All public solvers assume that ``evaluate(X)`` returns objectives to *minimize*.
Composite solvers additionally receive ``evaluate_components(X)`` and a known,
differentiable ``compose(H)`` map satisfying
``compose(evaluate_components(X)) == evaluate(X)``. A benchmark whose known map
also needs exact design coordinates declares ``compose(H, X)`` instead, so that
quantities known without error are never handed to a GP; see ``composer``.
Internally objectives are negated because BoTorch's acquisition functions use a
maximization convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import math
import os
import sys
from time import perf_counter
from typing import Callable, Optional, Sequence

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
from botorch.exceptions.errors import (
    CandidateGenerationError,
    OptimizationGradientError,
)
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior, LogNormalPrior

# `morbo/` (vendored alongside this file) is a full port of the published
# MORBO algorithm (Daulton et al., UAI 2022), with genuine coordinated
# parallel-batch trust-region candidate selection. `batched_morbo`/
# `composite_batched_morbo` below drive it directly, as an alternative to
# `morbo`/`composite_morbo`'s own from-scratch, one-point-per-iteration
# core further down in this file.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from morbo.run_one_replication import run_one_replication

Tensor = torch.Tensor
Evaluator = Callable[[Tensor], Tensor]
Composer = Callable[[Tensor], Tensor]

TIMING_KEYS = (
    "initial_design_seconds",
    "initial_evaluate_seconds",
    "initial_compose_seconds",
    "gp_fit_seconds",
    "acquisition_build_seconds",
    "acquisition_optimize_seconds",
    "bo_evaluate_seconds",
    "bo_compose_seconds",
    "solver_total_seconds",
)


def _new_timing() -> dict[str, float]:
    return dict.fromkeys(TIMING_KEYS, 0.0)


def _timed(timing, key, fn, *args):
    started = perf_counter()
    value = fn(*args)
    timing[key] += perf_counter() - started
    return value


# BoTorch 0.18 tries to JIT-build an optional fused qLogEHVI kernel on first use.
# Always select the identical pure-Python fallback instead, for two reasons.
#
# On Windows without MSVC the build produces a long subprocess traceback before
# falling back anyway. On a heterogeneous Slurm cluster it is worse than noisy:
# the kernel is compiled by whichever node happens to run first, cached in a
# shared home directory, and then loaded by every other node -- so an older CPU
# dies with SIGILL ("Illegal instruction"). Even where it does load, having some
# workers on the fused path and others on the fallback would make their results
# disagree, and resume validation recomputes objectives to 1e-12.
#
# The fallback is the same calculation, so the only cost is a marginal speedup
# we do not need. This private sentinel can go once BoTorch exposes a public
# switch for the optional extension.
_multi_objective_logei._load_attempted = True


@dataclass
class SolverResult:
    """Evaluated design and objective data returned by every solver."""

    X: Tensor
    Y: Tensor
    components: Optional[Tensor] = None
    weights: Optional[Tensor] = None
    run_ids: Optional[Tensor] = None
    acquisition_fallbacks: int = 0
    timing: dict[str, float] = field(default_factory=_new_timing)
    wall_seconds: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ObjectivewiseComposer:
    """Compose objectives from arbitrary groups of intermediate functions.

    ``component_groups[i]`` identifies the h_ij values used by objective i, and
    ``objective_maps[i]`` implements f_i = g_i(h_i1, ..., h_ik_i).
    Groups may have different sizes and may overlap. Each column of H is still
    modeled by its own independent GP.
    """

    component_groups: Sequence[Sequence[int]]
    objective_maps: Sequence[Callable[[Tensor], Tensor]]

    def __post_init__(self) -> None:
        if len(self.component_groups) != len(self.objective_maps):
            raise ValueError("one component group is required per objective")
        if any(len(group) == 0 for group in self.component_groups):
            raise ValueError("every objective requires at least one component")
        if any(index < 0 for group in self.component_groups for index in group):
            raise ValueError("component indices must be non-negative")

    def __call__(self, H: Tensor) -> Tensor:
        outputs = [
            outer(H[..., list(group)])
            for group, outer in zip(self.component_groups, self.objective_maps)
        ]
        return torch.stack(outputs, dim=-1)


def smooth_tchebycheff(
    Y: Tensor, weights: Tensor, ideal: Tensor, temperature: float = 0.05
) -> Tensor:
    """Smooth Tchebycheff loss for minimization.

    Computes ``t log sum_i exp(w_i (y_i-z_i*) / t)``. As temperature tends to
    zero this converges to ``max_i w_i(y_i-z_i*)``.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return temperature * torch.logsumexp(weights * (Y - ideal) / temperature, dim=-1)


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


def _optimize(
    acq, d: int, raw_samples: int, num_restarts: int
) -> tuple[Tensor, bool]:
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
        return X.detach(), False
    except (CandidateGenerationError, OptimizationGradientError):
        # Preserve the BO run by selecting the best finite acquisition value
        # from a fresh, space-filling Sobol candidate set.
        candidates = draw_sobol_samples(
            bounds=bounds, n=max(raw_samples * num_restarts, 256), q=1
        )
        with torch.no_grad():
            values = acq(candidates)
        finite = torch.isfinite(values)
        if not finite.any():
            raise RuntimeError("acquisition was non-finite on every fallback candidate")
        values = torch.where(finite, values, torch.full_like(values, -torch.inf))
        return candidates[values.argmax()].detach(), True


def _accepts_inputs(compose: Composer) -> bool:
    """True when ``compose`` wants the exact design inputs as a second argument."""

    try:
        parameters = list(inspect.signature(compose).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind is p.VAR_POSITIONAL for p in parameters):
        return True
    positional = [
        p
        for p in parameters
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def composer(compose: Composer) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
    """Normalize a known map to a uniform ``(H, X)`` callable.

    Coordinates of ``X`` that ``g`` needs are known exactly, so handing them to
    a GP throws away information for nothing. A benchmark that can use them
    declares a second parameter and receives the candidate designs; one that
    cannot is called with ``H`` alone. Every call site goes through here so the
    capability cannot be silently lost again.

    Implementations that take ``X`` must broadcast it against the leading Monte
    Carlo sample dimensions of ``H``.
    """

    if _accepts_inputs(compose):
        return compose
    return lambda H, X=None: compose(H)


def _check_composition(
    C: Tensor, Y: Tensor, X: Tensor, compose_fn: Callable[..., Tensor]
) -> None:
    reconstructed = compose_fn(C, X)
    if reconstructed.shape != Y.shape or not torch.allclose(
        reconstructed, Y, atol=1e-7, rtol=1e-5
    ):
        raise ValueError("compose(evaluate_components(X), X) must equal evaluate(X)")


def standard_mobo(
    evaluate: Evaluator,
    dim: int,
    ref_point: Tensor,
    *,
    n_init: int = 5,
    n_iter: int = 40,
    seed: int = 0,
    raw_samples: int = 128,
    num_restarts: int = 8,
    mc_samples: int = 512,
) -> SolverResult:
    """Independent objective GPs followed by numerically stable sequential qLogEHVI."""

    solver_started = perf_counter()
    timing = _new_timing()
    torch.manual_seed(seed)
    initial_started = perf_counter()
    X = _timed(timing, "initial_design_seconds", _sobol, n_init, dim, seed)
    Y = _timed(timing, "initial_evaluate_seconds", evaluate, X).double()
    initial_wall = (perf_counter() - initial_started) / len(X)
    wall_seconds = [initial_wall] * len(X)
    acquisition_fallbacks = 0
    ref_max = -torch.as_tensor(ref_point, dtype=torch.double)
    for _ in range(n_iter):
        iteration_started = perf_counter()
        model = _timed(timing, "gp_fit_seconds", _independent_gp, X, -Y)
        acquisition_started = perf_counter()
        partitioning = NondominatedPartitioning(ref_point=ref_max, Y=-Y)
        acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_max.tolist(),
            partitioning=partitioning,
            sampler=SobolQMCNormalSampler(
                torch.Size([mc_samples]), seed=seed * 100003 + len(X)
            ),
        )
        timing["acquisition_build_seconds"] += (
            perf_counter() - acquisition_started
        )
        (x, used_fallback) = _timed(
            timing,
            "acquisition_optimize_seconds",
            _optimize,
            acq,
            dim,
            raw_samples,
            num_restarts,
        )
        acquisition_fallbacks += int(used_fallback)
        y = _timed(timing, "bo_evaluate_seconds", evaluate, x).double()
        X, Y = torch.cat((X, x)), torch.cat((Y, y))
        wall_seconds.append(perf_counter() - iteration_started)
    timing["solver_total_seconds"] = perf_counter() - solver_started
    return SolverResult(
        X=X,
        Y=Y,
        acquisition_fallbacks=acquisition_fallbacks,
        timing=timing,
        wall_seconds=wall_seconds,
    )


def composite_mobo(
    evaluate: Evaluator,
    evaluate_components: Evaluator,
    compose: Composer,
    dim: int,
    ref_point: Tensor,
    *,
    n_init: int = 5,
    n_iter: int = 40,
    seed: int = 0,
    raw_samples: int = 128,
    num_restarts: int = 8,
    mc_samples: int = 512,
) -> SolverResult:
    """Intermediate-node GPs and qLogEHVI on composed samples (MO-BOCF)."""

    solver_started = perf_counter()
    timing = _new_timing()
    torch.manual_seed(seed)
    initial_started = perf_counter()
    X = _timed(timing, "initial_design_seconds", _sobol, n_init, dim, seed)
    C = _timed(
        timing, "initial_evaluate_seconds", evaluate_components, X
    ).double()
    Y = _timed(timing, "initial_evaluate_seconds", evaluate, X).double()
    compose_fn = composer(compose)
    _timed(timing, "initial_compose_seconds", _check_composition, C, Y, X, compose_fn)
    initial_wall = (perf_counter() - initial_started) / len(X)
    wall_seconds = [initial_wall] * len(X)
    acquisition_fallbacks = 0
    ref_max = -torch.as_tensor(ref_point, dtype=torch.double)
    objective = GenericMCMultiOutputObjective(
        lambda samples, X=None: -compose_fn(samples, X)
    )
    for _ in range(n_iter):
        iteration_started = perf_counter()
        model = _timed(timing, "gp_fit_seconds", _independent_gp, X, C)
        acquisition_started = perf_counter()
        partitioning = NondominatedPartitioning(ref_point=ref_max, Y=-Y)
        acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_max.tolist(),
            partitioning=partitioning,
            objective=objective,
            sampler=SobolQMCNormalSampler(
                torch.Size([mc_samples]), seed=seed * 100003 + len(X)
            ),
        )
        timing["acquisition_build_seconds"] += (
            perf_counter() - acquisition_started
        )
        (x, used_fallback) = _timed(
            timing,
            "acquisition_optimize_seconds",
            _optimize,
            acq,
            dim,
            raw_samples,
            num_restarts,
        )
        acquisition_fallbacks += int(used_fallback)
        c = _timed(
            timing, "bo_evaluate_seconds", evaluate_components, x
        ).double()
        # compose(c) == evaluate(x) is checked once on the initial design, so
        # the loop reuses it instead of paying a second expensive evaluation.
        y = _timed(timing, "bo_compose_seconds", compose_fn, c, x).double()
        X, C, Y = torch.cat((X, x)), torch.cat((C, c)), torch.cat((Y, y))
        wall_seconds.append(perf_counter() - iteration_started)
    timing["solver_total_seconds"] = perf_counter() - solver_started
    return SolverResult(
        X=X,
        Y=Y,
        components=C,
        acquisition_fallbacks=acquisition_fallbacks,
        timing=timing,
        wall_seconds=wall_seconds,
    )


def _scalarized_runs(
    evaluate: Evaluator,
    dim: int,
    weights: Tensor,
    ideal: Tensor,
    temperature: float,
    n_init: int,
    n_per_scalarization: int,
    seed: int,
    raw_samples: int,
    num_restarts: int,
    mc_samples: int,
    evaluate_components: Optional[Evaluator] = None,
    compose: Optional[Composer] = None,
) -> SolverResult:
    """Branch every weight from one shared, once-evaluated initial design."""

    solver_started = perf_counter()
    timing = _new_timing()
    acquisition_fallbacks = 0
    compose_fn = composer(compose) if compose is not None else None
    torch.manual_seed(seed)
    initial_started = perf_counter()
    X_initial = _timed(timing, "initial_design_seconds", _sobol, n_init, dim, seed)
    Y_initial = _timed(
        timing, "initial_evaluate_seconds", evaluate, X_initial
    ).double()
    if evaluate_components is None:
        C_initial = None
    else:
        C_initial = _timed(
            timing, "initial_evaluate_seconds", evaluate_components, X_initial
        ).double()
        _timed(
            timing,
            "initial_compose_seconds",
            _check_composition,
            C_initial,
            Y_initial,
            X_initial,
            compose_fn,
        )
    initial_wall = (perf_counter() - initial_started) / n_init
    wall_seconds = [initial_wall] * n_init

    all_x, all_y = [X_initial], [Y_initial]
    all_c = [C_initial] if C_initial is not None else []
    ids = [torch.full((n_init,), -1, dtype=torch.long)]
    for weight_id, weight in enumerate(weights.double()):
        torch.manual_seed(seed + 104729 * weight_id)
        X, Y = X_initial.clone(), Y_initial.clone()
        C = C_initial.clone() if C_initial is not None else None
        for _ in range(n_per_scalarization):
            iteration_started = perf_counter()
            if C is None:
                # Paper-style CBO: model f_1,...,f_m directly, then apply the
                # known STCH map to joint posterior samples inside EI.
                model = _timed(timing, "gp_fit_seconds", _independent_gp, X, Y)
                acquisition_started = perf_counter()
                objective = GenericMCObjective(
                    lambda samples, X=None, w=weight: (
                        -smooth_tchebycheff(samples, w, ideal, temperature)
                    )
                )
            else:
                model = _timed(timing, "gp_fit_seconds", _independent_gp, X, C)
                acquisition_started = perf_counter()
                objective = GenericMCObjective(
                    lambda samples, X=None, w=weight: (
                        -smooth_tchebycheff(
                            compose_fn(samples, X),
                            w,
                            ideal,
                            temperature,  # type: ignore[misc]
                        )
                    )
                )
            observed_utility = -smooth_tchebycheff(Y, weight, ideal, temperature)
            acq = qLogExpectedImprovement(
                model=model,
                best_f=observed_utility.max(),
                objective=objective,
                sampler=SobolQMCNormalSampler(
                    torch.Size([mc_samples]),
                    seed=seed * 100003 + 104729 * weight_id + len(X),
                ),
            )
            timing["acquisition_build_seconds"] += (
                perf_counter() - acquisition_started
            )
            (x, used_fallback) = _timed(
                timing,
                "acquisition_optimize_seconds",
                _optimize,
                acq,
                dim,
                raw_samples,
                num_restarts,
            )
            acquisition_fallbacks += int(used_fallback)
            if C is None:
                y = _timed(timing, "bo_evaluate_seconds", evaluate, x).double()
            else:
                c = _timed(
                    timing, "bo_evaluate_seconds", evaluate_components, x
                ).double()
                y = _timed(timing, "bo_compose_seconds", compose_fn, c, x).double()
                C = torch.cat((C, c))
            X, Y = torch.cat((X, x)), torch.cat((Y, y))
            wall_seconds.append(perf_counter() - iteration_started)
        all_x.append(X[n_init:])
        all_y.append(Y[n_init:])
        ids.append(torch.full((n_per_scalarization,), weight_id, dtype=torch.long))
        if C is not None:
            all_c.append(C[n_init:])
    X = torch.cat(all_x)
    Y = torch.cat(all_y)
    C = torch.cat(all_c) if all_c else None
    run_ids = torch.cat(ids)
    timing["solver_total_seconds"] = perf_counter() - solver_started
    return SolverResult(
        X=X,
        Y=Y,
        components=C,
        weights=weights,
        run_ids=run_ids,
        acquisition_fallbacks=acquisition_fallbacks,
        timing=timing,
        wall_seconds=wall_seconds,
    )


def chebyshev_bo(
    evaluate: Evaluator,
    dim: int,
    weights: Tensor,
    ideal: Tensor,
    *,
    temperature: float = 0.05,
    n_init: int = 5,
    n_per_scalarization: int = 5,
    seed: int = 0,
    raw_samples: int = 128,
    num_restarts: int = 8,
    mc_samples: int = 512,
) -> SolverResult:
    """Objective GPs and EI through STCH posterior samples, one run per weight."""

    return _scalarized_runs(
        evaluate,
        dim,
        weights,
        ideal.double(),
        temperature,
        n_init,
        n_per_scalarization,
        seed,
        raw_samples,
        num_restarts,
        mc_samples,
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
    n_init: int = 5,
    n_per_scalarization: int = 5,
    seed: int = 0,
    raw_samples: int = 128,
    num_restarts: int = 8,
    mc_samples: int = 512,
) -> SolverResult:
    """Node GPs and EI on the double-composed smooth-Tchebycheff posterior."""

    return _scalarized_runs(
        evaluate,
        dim,
        weights,
        ideal.double(),
        temperature,
        n_init,
        n_per_scalarization,
        seed,
        raw_samples,
        num_restarts,
        mc_samples,
        evaluate_components,
        compose,
    )


# ---------------------------------------------------------------------------
# High-dimensional spherical-linear BO (Doumont et al., AISTATS 2026)
# ---------------------------------------------------------------------------


def inverse_stereographic_projection(X: Tensor) -> Tensor:
    """Map R^d bijectively to the unit sphere S^d (paper Eq. 4)."""

    norm2 = X.square().sum(dim=-1, keepdim=True)
    return torch.cat((2.0 * X, norm2 - 1.0), dim=-1) / (1.0 + norm2)


class SphericalLinearKernel(Kernel):
    """Paper-faithful spherical linear kernel.

    Inputs are centered, divided by ARD scales and a decoupled global scale,
    inverse-stereographically projected, and evaluated with
    k(x,x') = b0 + b1 P(z)^T P(z'), where (b0,b1) lies on the simplex.
    """

    has_lengthscale = True

    def __init__(self, dim: int, bounds: tuple[float, float] = (0.0, 1.0)):
        prior = LogNormalPrior(math.sqrt(2.0), math.sqrt(3.0))
        super().__init__(
            ard_num_dims=dim,
            lengthscale_prior=prior,
            lengthscale_constraint=GreaterThan(
                2.5e-2, transform=None, initial_value=prior.mode
            ),
        )
        self.register_buffer("centers", torch.full((dim,), sum(bounds) / 2))
        self.register_buffer("widths", torch.full((dim,), bounds[1] - bounds[0]))
        self.register_parameter("raw_coeffs", torch.nn.Parameter(torch.zeros(2)))
        self.register_parameter("raw_global_scale", torch.nn.Parameter(torch.zeros(1)))

    @property
    def coefficients(self) -> Tensor:
        return torch.softmax(self.raw_coeffs, dim=-1)

    def _features(self, X: Tensor) -> Tensor:
        scaled = (X - self.centers) / self.lengthscale
        max_norm2 = (
            (self.widths / (2.0 * self.lengthscale.squeeze(-2))).square().sum(-1)
        )
        global_scale = torch.sqrt(torch.sigmoid(self.raw_global_scale) * max_norm2)
        projected = inverse_stereographic_projection(scaled / global_scale)
        b0, b1 = self.coefficients
        return torch.cat(
            (projected * b1.sqrt(), b0.sqrt().expand(*projected.shape[:-1], 1)),
            dim=-1,
        )

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params):
        phi1, phi2 = self._features(x1), self._features(x2)
        if diag:
            return (phi1 * phi2).sum(dim=-1)
        return phi1 @ phi2.transpose(-1, -2)


def _spherical_gp(X: Tensor, Y: Tensor) -> ModelListGP:
    """Independent paper-style spherical-linear GPs for every output."""

    models = []
    for i in range(Y.shape[-1]):
        noise_prior = LogNormalPrior(-4.0, 1.0)
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(1e-4, initial_value=noise_prior.mode),
        )
        models.append(
            SingleTaskGP(
                X,
                Y[:, i : i + 1],
                mean_module=ConstantMean(),
                covar_module=SphericalLinearKernel(X.shape[-1]),
                likelihood=likelihood,
                outcome_transform=Standardize(m=1),
            )
        )
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def _high_dim_scalarized_runs(
    evaluate: Evaluator,
    dim: int,
    weights: Tensor,
    ideal: Tensor,
    *,
    temperature: float = 0.05,
    n_init: int = 5,
    n_per_scalarization: int = 5,
    seed: int = 0,
    raw_samples: int = 256,
    num_restarts: int = 10,
    evaluate_components: Optional[Evaluator] = None,
    compose: Optional[Composer] = None,
) -> SolverResult:
    """Spherical-linear STCH with one shared initial design."""
    solver_started = perf_counter()
    timing = _new_timing()
    acquisition_fallbacks = 0
    compose_fn = composer(compose) if compose is not None else None
    torch.manual_seed(seed)
    initial_started = perf_counter()
    X_initial = _timed(timing, "initial_design_seconds", _sobol, n_init, dim, seed)
    Y_initial = _timed(
        timing, "initial_evaluate_seconds", evaluate, X_initial
    ).double()
    if evaluate_components is None:
        C_initial = None
    else:
        C_initial = _timed(
            timing, "initial_evaluate_seconds", evaluate_components, X_initial
        ).double()
        _timed(
            timing,
            "initial_compose_seconds",
            _check_composition,
            C_initial,
            Y_initial,
            X_initial,
            compose_fn,
        )
    initial_wall = (perf_counter() - initial_started) / n_init
    wall_seconds = [initial_wall] * n_init

    all_x, all_y = [X_initial], [Y_initial]
    all_c = [C_initial] if C_initial is not None else []
    all_ids = [torch.full((n_init,), -1, dtype=torch.long)]
    for weight_id, weight in enumerate(weights.double()):
        torch.manual_seed(seed + 104729 * weight_id)
        X, Y = X_initial.clone(), Y_initial.clone()
        C = C_initial.clone() if C_initial is not None else None
        for _ in range(n_per_scalarization):
            iteration_started = perf_counter()
            training_values = Y if C is None else C
            model = _timed(timing, "gp_fit_seconds", _spherical_gp, X, training_values)
            acquisition_started = perf_counter()
            if C is None:
                objective = GenericMCObjective(
                    lambda samples, X=None, w=weight: (
                        -smooth_tchebycheff(samples, w, ideal, temperature)
                    )
                )
            else:
                objective = GenericMCObjective(
                    lambda samples, X=None, w=weight: (
                        -smooth_tchebycheff(
                            compose_fn(samples, X),
                            w,
                            ideal,
                            temperature,  # type: ignore[misc]
                        )
                    )
                )
            observed_utility = -smooth_tchebycheff(Y, weight, ideal, temperature)
            acq = qLogExpectedImprovement(
                model=model, best_f=observed_utility.max(), objective=objective
            )
            timing["acquisition_build_seconds"] += (
                perf_counter() - acquisition_started
            )
            x, used_fallback = _timed(
                timing,
                "acquisition_optimize_seconds",
                _optimize,
                acq,
                dim,
                raw_samples,
                num_restarts,
            )
            acquisition_fallbacks += int(used_fallback)
            y = _timed(timing, "bo_evaluate_seconds", evaluate, x).double()
            X, Y = torch.cat((X, x)), torch.cat((Y, y))
            if C is not None:
                c = _timed(
                    timing, "bo_evaluate_seconds", evaluate_components, x
                ).double()
                C = torch.cat((C, c))  # type: ignore[misc]
            wall_seconds.append(perf_counter() - iteration_started)
        all_x.append(X[n_init:])
        all_y.append(Y[n_init:])
        all_ids.append(torch.full((n_per_scalarization,), weight_id, dtype=torch.long))
        if C is not None:
            all_c.append(C[n_init:])
    timing["solver_total_seconds"] = perf_counter() - solver_started
    return SolverResult(
        X=torch.cat(all_x),
        Y=torch.cat(all_y),
        components=torch.cat(all_c) if all_c else None,
        weights=weights,
        run_ids=torch.cat(all_ids),
        acquisition_fallbacks=acquisition_fallbacks,
        timing=timing,
        wall_seconds=wall_seconds,
    )


def spherical_chebyshev_bo(
    evaluate: Evaluator, dim: int, weights: Tensor, ideal: Tensor, **kwargs
) -> SolverResult:
    """Spherical-linear STCH with caller-provided simplex weights."""
    return _high_dim_scalarized_runs(evaluate, dim, weights, ideal.double(), **kwargs)


def composite_spherical_chebyshev_bo(
    evaluate: Evaluator,
    evaluate_components: Evaluator,
    compose: Composer,
    dim: int,
    weights: Tensor,
    ideal: Tensor,
    **kwargs,
) -> SolverResult:
    """Spherical-linear GPs on h_ij, followed by composition and STCH."""
    return _high_dim_scalarized_runs(
        evaluate,
        dim,
        weights,
        ideal.double(),
        evaluate_components=evaluate_components,
        compose=compose,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# MORBO: coordinated multi-trust-region multi-objective optimization
# ---------------------------------------------------------------------------


@dataclass
class MORBOConfig:
    n_trust_regions: int = 5
    length_init: float = 0.8
    length_min: float = 0.01
    length_max: float = 1.6
    success_streak: int = 10_000
    failure_streak: Optional[int] = None
    raw_samples: int = 512


def _morbo_gp(X: Tensor, Y: Tensor) -> ModelListGP:
    models = []
    for i in range(Y.shape[-1]):
        kernel = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=X.shape[-1],
                lengthscale_constraint=Interval(0.05, 4.0),
            )
        )
        likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(1e-6),
            noise_prior=GammaPrior(0.9, 10.0),
        )
        models.append(
            SingleTaskGP(
                X,
                Y[:, i : i + 1],
                covar_module=kernel,
                likelihood=likelihood,
                outcome_transform=Standardize(m=1),
            )
        )
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def _hv_max(Y: Tensor, ref: Tensor) -> float:
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from botorch.utils.multi_objective.pareto import is_non_dominated

    valid = (Y > ref).all(dim=-1)
    if not valid.any():
        return 0.0
    P = Y[valid]
    P = P[is_non_dominated(P)]
    return float(Hypervolume(ref_point=ref).compute(P))


def _morbo(
    evaluate: Evaluator,
    dim: int,
    ref_point: Tensor,
    *,
    n_init: int = 5,
    n_iter: int = 40,
    seed: int = 0,
    config: Optional[MORBOConfig] = None,
    evaluate_components: Optional[Evaluator] = None,
    compose: Optional[Composer] = None,
) -> SolverResult:
    """Sequential MORBO core with coordinated local Thompson/HVI selection."""
    solver_started = perf_counter()
    timing = _new_timing()
    cfg = config or MORBOConfig()
    failure_streak = cfg.failure_streak or max(dim // 3, 10)
    torch.manual_seed(seed)
    initial_started = perf_counter()
    X = _timed(timing, "initial_design_seconds", _sobol, n_init, dim, seed)
    Y = _timed(timing, "initial_evaluate_seconds", evaluate, X).double()
    C = (
        _timed(timing, "initial_evaluate_seconds", evaluate_components, X).double()
        if evaluate_components
        else None
    )
    compose_fn = composer(compose) if compose is not None else None
    if C is not None:
        _timed(
            timing, "initial_compose_seconds", _check_composition, C, Y, X, compose_fn
        )
    initial_wall = (perf_counter() - initial_started) / n_init
    wall_seconds = [initial_wall] * n_init
    ref = -torch.as_tensor(ref_point, dtype=torch.double)
    ntr = cfg.n_trust_regions
    pareto_ids = torch.where(
        torch.tensor(
            [
                not ((Y <= Y[i]).all(-1) & (Y < Y[i]).any(-1)).any()
                for i in range(len(Y))
            ]
        )
    )[0]
    centers = X[pareto_ids[:ntr]].clone()
    while len(centers) < ntr:
        centers = torch.cat((centers, X[torch.randint(len(X), (1,))]))
    lengths = torch.full((ntr,), cfg.length_init, dtype=torch.double)
    failures = torch.zeros(ntr, dtype=torch.long)
    successes = torch.zeros_like(failures)
    previous_hv = _hv_max(-Y, ref)
    for step in range(n_iter):
        iteration_started = perf_counter()
        best_x, best_score, best_tr = None, float("-inf"), 0
        for tr in range(ntr):
            local = ((X - centers[tr]).abs() <= lengths[tr]).all(-1)
            if local.sum() < min(3, len(X)):
                local = torch.ones(len(X), dtype=torch.bool)
            train = Y[local] if C is None else C[local]
            model = _timed(
                timing,
                "gp_fit_seconds",
                _morbo_gp,
                X[local],
                -train if C is None else train,
            )
            selection_started = perf_counter()
            sobol = _sobol(cfg.raw_samples, dim, seed + 7919 * (step + 1) + tr)
            lo = (centers[tr] - lengths[tr] / 2).clamp(0, 1)
            hi = (centers[tr] + lengths[tr] / 2).clamp(0, 1)
            cand = lo + (hi - lo) * sobol
            prob = min(20.0 / dim, 1.0)
            mask = torch.rand_like(cand) < prob
            empty = ~mask.any(-1)
            mask[empty, torch.randint(dim, (int(empty.sum()),))] = True
            cand = torch.where(mask, cand, centers[tr])
            with torch.no_grad():
                sample = model.posterior(cand).rsample().squeeze(0)
            obj_sample = (
                sample if C is None else -compose_fn(sample, cand)  # type: ignore[misc]
            )
            base = _hv_max(-Y, ref)
            scores = torch.tensor(
                [_hv_max(torch.cat((-Y, v[None])), ref) - base for v in obj_sample]
            )
            idx = int(scores.argmax())
            if float(scores[idx]) > best_score:
                best_x, best_score, best_tr = (
                    cand[idx : idx + 1],
                    float(scores[idx]),
                    tr,
                )
            # Trust-region candidate generation and hypervolume scoring is
            # MORBO's analogue of building and optimizing an acquisition.
            timing["acquisition_optimize_seconds"] += (
                perf_counter() - selection_started
            )
        x = best_x
        y = _timed(timing, "bo_evaluate_seconds", evaluate, x).double()  # type: ignore[arg-type]
        X, Y = torch.cat((X, x)), torch.cat((Y, y))
        if C is not None:
            c = _timed(timing, "bo_evaluate_seconds", evaluate_components, x).double()
            C = torch.cat((C, c))  # type: ignore[misc]
        new_hv = _hv_max(-Y, ref)
        if new_hv > previous_hv + 1e-3 * max(abs(previous_hv), 1.0):
            successes[best_tr] += 1
            failures[best_tr] = 0
            if successes[best_tr] >= cfg.success_streak:
                lengths[best_tr] = min(2 * lengths[best_tr], cfg.length_max)
                successes[best_tr] = 0
        else:
            failures[best_tr] += 1
            successes[best_tr] = 0
            if failures[best_tr] >= failure_streak:
                lengths[best_tr] /= 2
                failures[best_tr] = 0
        centers[best_tr] = x.squeeze(0)
        if lengths[best_tr] < cfg.length_min:
            centers[best_tr] = _sobol(1, dim, seed + 99991 + step).squeeze(0)
            lengths[best_tr] = cfg.length_init
        previous_hv = new_hv
        wall_seconds.append(perf_counter() - iteration_started)
    timing["solver_total_seconds"] = perf_counter() - solver_started
    return SolverResult(
        X=X, Y=Y, components=C, timing=timing, wall_seconds=wall_seconds
    )


def morbo(evaluate: Evaluator, dim: int, ref_point: Tensor, **kwargs) -> SolverResult:
    """Direct-objective MORBO."""
    return _morbo(evaluate, dim, ref_point, **kwargs)


def composite_morbo(
    evaluate: Evaluator,
    evaluate_components: Evaluator,
    compose: Composer,
    dim: int,
    ref_point: Tensor,
    **kwargs,
) -> SolverResult:
    """MORBO with local GPs on objective-specific composite subfunctions."""
    return _morbo(
        evaluate,
        dim,
        ref_point,
        evaluate_components=evaluate_components,
        compose=compose,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# batched_morbo / composite_batched_morbo: same calling convention as
# morbo/composite_morbo above, but driven by the vendored morbo/ package
# (a full port of Daulton et al.'s published algorithm) instead of _morbo's
# from-scratch core. The key difference: _morbo proposes and evaluates one
# point at a time, whereas the vendored engine jointly selects a batch of
# candidates across all trust regions per iteration, then evaluates the
# whole batch before updating any trust region -- matching the paper's
# actual coordinated parallel-batch selection.
#
# Compatibility note: the vendored morbo/ package was developed and tested
# against botorch 0.9.5 / gpytorch 1.11. If this file's own botorch version
# (newer, per the JIT-kernel workaround above) turns out to be incompatible
# with morbo/'s expectations, that would surface as an import or runtime
# error the first time one of these two functions is called.
# ---------------------------------------------------------------------------


def _batched_morbo(
    evaluate: Evaluator,
    dim: int,
    ref_point: Tensor,
    *,
    n_init: int = 5,
    n_iter: int = 40,
    seed: int = 0,
    config: Optional[MORBOConfig] = None,
    batch_size: Optional[int] = None,
    min_tr_size: Optional[int] = None,
    evaluate_components: Optional[Evaluator] = None,
    compose: Optional[Composer] = None,
) -> SolverResult:
    cfg = config or MORBOConfig()
    ref_point_t = torch.as_tensor(ref_point, dtype=torch.double)
    # `run_one_replication`'s `max_reference_point` is in the vendored
    # engine's maximize convention; this file's `ref_point` (like its
    # `evaluate` outputs) is minimize-convention, so the sign flip mirrors
    # what `_morbo`/`dominated_hypervolume_trace` do with `ref = -ref_point`.
    max_reference_point = (-ref_point_t).tolist()
    # A batched engine can't reuse "one iteration = one evaluation"; by
    # default each iteration proposes one candidate per trust region
    # (matching the paper's per-TR batch element), evaluated jointly.
    bs = batch_size if batch_size is not None else cfg.n_trust_regions
    max_evals = n_init + n_iter * bs
    # The vendored engine's own default min_tr_size (250) assumes the large
    # init budgets its own experiments use; this file's benchmarks default
    # to n_init=5, well under that, so pick something that always satisfies
    # the engine's "n_initial_points > min_tr_size" requirement instead of
    # forcing every caller to pass this explicitly.
    mts = min_tr_size if min_tr_size is not None else max(1, min(n_init - 1, 20))

    if evaluate_components is not None:
        if compose is None:
            raise ValueError("evaluate_components requires compose.")

        def raw_evaluate_components(X: Tensor) -> Tensor:
            return evaluate_components(X).double()

        # The vendored engine never passes designs alongside the responses, so
        # an exact-input map cannot be honoured on this path.
        if _accepts_inputs(compose):
            raise ValueError(
                "batched_morbo cannot supply exact inputs to a compose(H, X) map"
            )

        def raw_compose(H: Tensor) -> Tensor:
            return compose(H).double()

        extra = dict(
            raw_evaluate_components=raw_evaluate_components,
            raw_compose=raw_compose,
        )
    else:
        def raw_evaluate(X: Tensor) -> Tensor:
            return evaluate(X).double()

        extra = dict(raw_evaluate=raw_evaluate)

    solver_started = perf_counter()
    outputs = []
    run_one_replication(
        seed=seed,
        label="morbo",
        max_evals=max_evals,
        evalfn="Callable",
        dim=dim,
        batch_size=bs,
        n_initial_points=n_init,
        min_tr_size=mts,
        n_trust_regions=cfg.n_trust_regions,
        length_init=cfg.length_init,
        length_min=cfg.length_min,
        length_max=cfg.length_max,
        success_streak=cfg.success_streak,
        failure_streak=cfg.failure_streak,
        raw_samples=cfg.raw_samples,
        max_reference_point=max_reference_point,
        save_callback=lambda output: outputs.append(output),
        save_during_opt=False,
        verbose=False,
        **extra,
    )
    result = outputs[-1]
    X = result["X_history"]
    # `metric_history` is the raw callable's own output, recorded before the
    # vendored engine's internal negation: for the direct-objective case
    # that internal negation IS applied (so it's flipped back here to
    # recover `evaluate`'s own minimize-convention values, exactly, with no
    # extra `evaluate` calls); for the composite case the internal negation
    # is skipped instead (the engine's own composite reduction handles it
    # as part of composing), so `metric_history` is already the raw,
    # minimize-convention components `evaluate_components` returned.
    # ponytail: the vendored engine reports no per-phase or per-evaluation
    # timings, so the total is amortized evenly. Push timers into morbo/ if a
    # phase breakdown is ever needed here.
    timing = _new_timing()
    timing["solver_total_seconds"] = perf_counter() - solver_started
    wall_seconds = [timing["solver_total_seconds"] / len(X)] * len(X)
    if evaluate_components is not None:
        C = result["metric_history"]
        Y = compose(C).double()
        return SolverResult(
            X=X, Y=Y, components=C, timing=timing, wall_seconds=wall_seconds
        )
    Y = -result["metric_history"]
    return SolverResult(X=X, Y=Y, timing=timing, wall_seconds=wall_seconds)


def batched_morbo(
    evaluate: Evaluator, dim: int, ref_point: Tensor, **kwargs
) -> SolverResult:
    """Direct-objective MORBO, driven by the vendored paper-accurate engine.

    Same signature as ``morbo`` above (``evaluate``, ``dim``, ``ref_point``,
    plus ``n_init``/``n_iter``/``seed``/``config`` via ``kwargs``), with two
    additions: an optional ``batch_size`` kwarg (default
    ``config.n_trust_regions``, i.e. one candidate per trust region per
    iteration) and an optional ``min_tr_size`` kwarg (default scales with
    ``n_init``; see ``_batched_morbo``). See the section note above for why
    ``n_iter`` means "batch iterations" here rather than "evaluations".
    """
    return _batched_morbo(evaluate, dim, ref_point, **kwargs)


def composite_batched_morbo(
    evaluate: Evaluator,
    evaluate_components: Evaluator,
    compose: Composer,
    dim: int,
    ref_point: Tensor,
    **kwargs,
) -> SolverResult:
    """MORBO with local GPs on composite subfunctions, vendored engine.

    Same signature as ``composite_morbo`` above, plus the same optional
    ``batch_size``/``min_tr_size`` kwargs described in ``batched_morbo``.
    """
    return _batched_morbo(
        evaluate,
        dim,
        ref_point,
        evaluate_components=evaluate_components,
        compose=compose,
        **kwargs,
    )
