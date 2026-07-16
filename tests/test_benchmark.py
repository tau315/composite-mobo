import math

import pytest
import torch
from botorch.exceptions.errors import CandidateGenerationError

import benchmark
import solvers


def _tiny_problem_counter():
    calls = {"rows": 0, "mc_compose": 0}

    def components(X):
        calls["rows"] += len(X)
        return X[..., :1]

    def compose(C, X):
        if C.ndim > X.ndim:
            calls["mc_compose"] += 1
        exact = (X[..., 1:2] - 0.5).square() + torch.zeros_like(C)
        return torch.cat((C.square(), exact), dim=-1)

    return calls, components, compose


def test_problem_composition_is_exact():
    X = torch.rand(8, 6, dtype=torch.double)
    for name in ("zdt1", "zdt2", "zdt3", "dtlz2"):
        problem = benchmark.get_problem(name, 6)
        assert torch.allclose(
            problem.compose(problem.components(X), X),
            problem.evaluate(X),
        )


def test_composite_samples_respect_g_domain():
    X = torch.zeros(2, 6, dtype=torch.double)
    C = torch.tensor([[-2.0], [2.0]], dtype=torch.double)
    zdt = benchmark.get_problem("zdt1", 6).compose(C, X)
    dtlz = benchmark.get_problem("dtlz2", 6).compose(C, X)
    assert torch.all(zdt[:, 1] >= 1.0)
    assert torch.all(dtlz[:, 0] >= 1.0)


def test_zdt3_uses_true_ideal():
    ideal = benchmark.get_problem("zdt3", 6).ideal
    assert torch.allclose(
        ideal,
        torch.tensor([0.0, -0.7733690123], dtype=torch.double),
    )


def test_composite_calls_components_once_per_point_with_mc_sample_dims():
    calls, components, compose = _tiny_problem_counter()
    result = solvers.composite_mobo(
        lambda X: compose(components(X), X),
        components,
        compose,
        2,
        torch.tensor([2.0, 2.0], dtype=torch.double),
        n_init=2,
        n_iter=1,
        raw_samples=16,
        num_restarts=2,
    )
    assert calls["rows"] == len(result.X) == 3
    assert calls["mc_compose"] > 0
    assert torch.allclose(result.Y, compose(result.components, result.X))


def test_composite_stch_calls_components_once_per_point():
    calls, components, compose = _tiny_problem_counter()
    result = solvers.composite_chebyshev_bo(
        lambda X: compose(components(X), X),
        components,
        compose,
        2,
        torch.tensor([[0.5, 0.5]], dtype=torch.double),
        torch.zeros(2, dtype=torch.double),
        n_init=2,
        n_iter=1,
        raw_samples=16,
        num_restarts=2,
    )
    assert calls["rows"] == len(result.X) == 3
    assert torch.allclose(result.Y, compose(result.components, result.X))


def test_solver_timings_are_complete_and_finite():
    calls, components, compose = _tiny_problem_counter()
    result = solvers.composite_mobo(
        lambda X: compose(components(X), X),
        components,
        compose,
        2,
        torch.tensor([2.0, 2.0], dtype=torch.double),
        n_init=2,
        n_iter=1,
        raw_samples=16,
        num_restarts=2,
    )
    assert set(result.timing) == set(solvers.TIMING_KEYS)
    assert all(math.isfinite(v) and v >= 0 for v in result.timing.values())
    assert len(result.wall_seconds) == len(result.X)
    assert result.wall_seconds[0] == result.wall_seconds[1]
    bo_phases = (
        "gp_fit_seconds",
        "acquisition_build_seconds",
        "acquisition_optimize_seconds",
        "bo_evaluate_seconds",
        "bo_compose_seconds",
    )
    assert result.wall_seconds[-1] >= sum(result.timing[key] for key in bo_phases)


def test_all_solver_paths_return_timing_and_wall_values():
    _, components, compose = _tiny_problem_counter()
    evaluate = lambda X: compose(components(X), X)
    ref_point = torch.tensor([2.0, 2.0], dtype=torch.double)
    weights = torch.tensor([[0.5, 0.5]], dtype=torch.double)
    ideal = torch.zeros(2, dtype=torch.double)
    results = (
        solvers.standard_mobo(evaluate, 2, ref_point, n_init=2, n_iter=0),
        solvers.composite_mobo(
            evaluate, components, compose, 2, ref_point, n_init=2, n_iter=0
        ),
        solvers.chebyshev_bo(
            evaluate, 2, weights, ideal, n_init=2, n_iter=0
        ),
        solvers.composite_chebyshev_bo(
            evaluate,
            components,
            compose,
            2,
            weights,
            ideal,
            n_init=2,
            n_iter=0,
        ),
    )
    for result in results:
        assert set(result.timing) == set(solvers.TIMING_KEYS)
        assert len(result.wall_seconds) == len(result.X)


def test_stch_results_interleave_weight_runs():
    weights = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.double)
    evaluate = lambda X: torch.cat((X[:, :1], X[:, 1:2]), dim=-1)
    result = solvers.chebyshev_bo(
        evaluate,
        2,
        weights,
        torch.zeros(2, dtype=torch.double),
        n_init=2,
        n_iter=0,
    )
    assert result.run_ids.tolist() == [0, 1, 0, 1]


def test_acquisition_fallback_rejects_all_nonfinite_values(monkeypatch):
    candidates = torch.tensor(
        [[[0.1, 0.2]], [[0.4, 0.5]], [[0.8, 0.9]]], dtype=torch.double
    )

    def fail_generation(*args, **kwargs):
        raise CandidateGenerationError("failed")

    monkeypatch.setattr(solvers, "optimize_acqf", fail_generation)
    monkeypatch.setattr(
        solvers, "draw_sobol_samples", lambda **kwargs: candidates
    )

    values = torch.tensor([torch.inf, 1.0, torch.nan], dtype=torch.double)
    chosen = solvers._optimize(lambda X: values, 2, 2, 2)
    assert torch.equal(chosen, candidates[1])

    with pytest.raises(
        RuntimeError, match="non-finite on every fallback candidate"
    ):
        solvers._optimize(
            lambda X: torch.tensor(
                [torch.inf, torch.nan, -torch.inf], dtype=torch.double
            ),
            2,
            2,
            2,
        )
