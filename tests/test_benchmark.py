from argparse import Namespace
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pytest
import torch
from botorch.exceptions.errors import CandidateGenerationError

import benchmark
import solvers


def _benchmark_args(tmp_path, **overrides):
    values = {
        "problems": ["zdt1"],
        "dim": 2,
        "trials": 1,
        "trial": None,
        "method": None,
        "budget": 2,
        "initial": 1,
        "weights": 1,
        "temperature": 0.05,
        "seed": 7,
        "raw_samples": 4,
        "restarts": 1,
        "mc_samples": 8,
        "results_dir": tmp_path,
        "summary_only": False,
        "output": tmp_path / "hypervolume.png",
    }
    values.update(overrides)
    return Namespace(**values)


def _fake_solver_result(
    method="standard_qlogehvi", budget=2, dim=2, weights=1,
    problem_name="zdt1",
):
    problem = benchmark.get_problem(problem_name, dim)
    X = torch.zeros((budget, dim), dtype=torch.double)
    return solvers.SolverResult(
        X=X,
        Y=problem.evaluate(X),
        components=(
            problem.components(X)
            if method.startswith("composite_")
            else None
        ),
        weights=(
            solvers.simplex_weights(weights, 2, seed=7)
            if method.endswith("_stch")
            else None
        ),
        run_ids=(
            torch.arange(weights).repeat(budget // weights)
            if method.endswith("_stch")
            else None
        ),
        timing=dict.fromkeys(solvers.TIMING_KEYS, 0.0),
        wall_seconds=[0.01] * budget,
    )


def _artifact_config(**overrides):
    config = {
        "dim": 2,
        "budget": 2,
        "initial": 1,
        "weights": 1,
        "temperature": 0.05,
        "seed": 7,
        "raw_samples": 4,
        "restarts": 1,
        "mc_samples": 8,
    }
    config.update(overrides)
    return config


def _artifact(config=None, **overrides):
    config = config or _artifact_config()
    budget = config["budget"]
    X = torch.zeros((budget, config["dim"]), dtype=torch.double)
    X[:, 0] = torch.linspace(0.0, 0.75, budget, dtype=torch.double)
    problem = benchmark.get_problem("zdt1", config["dim"])
    Y = problem.evaluate(X)
    payload = {
        "schema_version": 2,
        "problem": "zdt1",
        "method": "standard_qlogehvi",
        "trial": config["seed"] - 7,
        "seed": config["seed"],
        "config": config,
        "metadata": benchmark._run_metadata(),
        "X": X.tolist(),
        "Y": Y.tolist(),
        "components": None,
        "weights": None,
        "run_ids": None,
        "hypervolume": benchmark.hypervolume_trace(
            Y, problem.ref_point
        ).tolist(),
        "wall_seconds": [0.01] * budget,
        "timing": dict.fromkeys(
            (*solvers.TIMING_KEYS, "hypervolume_seconds", "total_seconds"),
            0.0,
        ),
        "acquisition_fallbacks": 0,
        "failed": None,
    }
    payload.update(overrides)
    return payload


def _method_artifact(method, config=None, **overrides):
    config = config or _artifact_config(budget=4, weights=2)
    budget = config["budget"]
    payload = _artifact(config, method=method)
    if method.startswith("composite_"):
        problem = benchmark.get_problem(payload["problem"], config["dim"])
        payload["components"] = problem.components(
            torch.tensor(payload["X"], dtype=torch.double)
        ).tolist()
    if method.endswith("_stch"):
        payload["weights"] = solvers.simplex_weights(
            config["weights"], 2, seed=config["seed"]
        ).tolist()
        payload["run_ids"] = list(range(config["weights"])) * (
            budget // config["weights"]
        )
    payload.update(overrides)
    return payload


def _set_design(payload, X):
    """Update every deterministic artifact field after changing its design."""

    problem = benchmark.get_problem(
        payload["problem"], payload["config"]["dim"]
    )
    X = torch.as_tensor(X, dtype=torch.double)
    Y = problem.evaluate(X)
    payload["X"] = X.tolist()
    payload["Y"] = Y.tolist()
    payload["hypervolume"] = benchmark.hypervolume_trace(
        Y, problem.ref_point
    ).tolist()
    if payload["method"].startswith("composite_"):
        payload["components"] = problem.components(X).tolist()
    return payload


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
    chosen, fallback = solvers._optimize(lambda X: values, 2, 2, 2)
    assert torch.equal(chosen, candidates[1])
    assert fallback

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


def test_all_acquisitions_use_matched_seeded_mc_samplers_and_count_fallbacks(
    monkeypatch,
):
    qlog_samplers = []
    qei_samplers = []

    def sampler(sample_shape, seed):
        return (tuple(sample_shape), seed)

    def qlog_acquisition(**kwargs):
        qlog_samplers.append(kwargs["sampler"])
        return object()

    def qei_acquisition(**kwargs):
        qei_samplers.append(kwargs["sampler"])
        return object()

    monkeypatch.setattr(solvers, "SobolQMCNormalSampler", sampler)
    monkeypatch.setattr(solvers, "_independent_gp", lambda X, Y: object())
    monkeypatch.setattr(
        solvers, "NondominatedPartitioning", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        solvers, "qLogExpectedHypervolumeImprovement", qlog_acquisition
    )
    monkeypatch.setattr(solvers, "qLogExpectedImprovement", qei_acquisition)
    monkeypatch.setattr(
        solvers,
        "_optimize",
        lambda *args: (torch.full((1, 2), 0.25, dtype=torch.double), True),
    )

    _, components, compose = _tiny_problem_counter()
    evaluate = lambda X: compose(components(X), X)
    ref_point = torch.tensor([2.0, 2.0], dtype=torch.double)
    weights = torch.tensor(
        [[0.25, 0.75], [0.75, 0.25]], dtype=torch.double
    )
    ideal = torch.zeros(2, dtype=torch.double)
    qlog_results = (
        solvers.standard_mobo(
            evaluate, 2, ref_point, n_init=2, n_iter=2,
            seed=11, mc_samples=13,
        ),
        solvers.composite_mobo(
            evaluate, components, compose, 2, ref_point,
            n_init=2, n_iter=2, seed=11, mc_samples=13,
        ),
    )
    stch_results = (
        solvers.chebyshev_bo(
            evaluate, 2, weights, ideal, n_init=1, n_iter=1,
            seed=11, mc_samples=13,
        ),
        solvers.composite_chebyshev_bo(
            evaluate, components, compose, 2, weights, ideal,
            n_init=1, n_iter=1, seed=11, mc_samples=13,
        ),
    )

    assert qlog_samplers == [((13,), 13), ((13,), 14)] * 2
    assert qei_samplers == [((13,), 12), ((13,), 104741)] * 2
    assert [result.acquisition_fallbacks for result in qlog_results] == [2, 2]
    assert [result.acquisition_fallbacks for result in stch_results] == [2, 2]


def test_atomic_artifact_and_strict_resume_validation(tmp_path):
    path = tmp_path / "zdt1" / "standard_qlogehvi" / "trial0.json"
    config = _artifact_config()
    payload = _artifact(config)

    benchmark._atomic_write_json(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert benchmark._validated_payload(path, config) == payload
    assert benchmark._valid_result(path, config)
    assert not benchmark._valid_result(path, _artifact_config(budget=3))
    assert not benchmark._valid_result(path, _artifact_config(seed=8))


@pytest.mark.parametrize("field", ("python", "packages", "git_commit"))
def test_resume_validation_requires_exact_current_provenance(tmp_path, field):
    path = tmp_path / "zdt1" / "standard_qlogehvi" / "trial0.json"
    payload = _artifact()
    payload["metadata"] = deepcopy(payload["metadata"])
    payload["metadata"][field] = (
        {"torch": "stale"} if field == "packages" else "stale"
    )
    benchmark._atomic_write_json(path, payload)

    assert not benchmark._valid_result(path, payload["config"])


def test_resume_validation_recomputes_problem_outputs_and_hypervolume(tmp_path):
    path = tmp_path / "zdt1" / "composite_stch" / "trial0.json"
    config = _artifact_config(budget=4, weights=2)
    valid = _method_artifact("composite_stch", config)
    corruptions = []

    payload = deepcopy(valid)
    payload["X"][0][0] = 1.01
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["Y"][0][0] += 0.01
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["components"][0][0] += 0.01
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["weights"][0] = [0.5, 0.5]
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["hypervolume"][1] = payload["hypervolume"][0] - 0.01
    corruptions.append(payload)
    payload = deepcopy(valid)
    payload["hypervolume"][-1] += 0.01
    corruptions.append(payload)

    for payload in corruptions:
        benchmark._atomic_write_json(path, payload)
        assert not benchmark._valid_result(path, config)


@pytest.mark.parametrize("method", tuple(benchmark.METHOD_LABELS))
def test_resume_validation_accepts_method_result_shapes(tmp_path, method):
    config = _artifact_config(budget=4, weights=2)
    path = tmp_path / "zdt1" / method / "trial0.json"
    benchmark._atomic_write_json(path, _method_artifact(method, config))

    assert benchmark._valid_result(path, config)


@pytest.mark.parametrize(
    ("method", "overrides"),
    (
        ("standard_qlogehvi", {"Y": [[0.0, 1.0, 2.0]] * 4}),
        ("composite_qlogehvi", {"components": None}),
        ("composite_qlogehvi", {"components": [[0.0, 1.0]] * 4}),
        ("composite_stch", {"components": [[0.0]] * 3}),
        (
            "composite_qlogehvi",
            {"components": [[0.0], [1.0], [float("nan")], [3.0]]},
        ),
        ("standard_qlogehvi", {"components": [[0.0]] * 4}),
        ("objective_gp_stch", {"components": [[0.0]] * 4}),
        ("objective_gp_stch", {"weights": None}),
        ("composite_stch", {"weights": [[0.5, 0.5]]}),
        ("objective_gp_stch", {"weights": [[0.2, 0.3, 0.5]] * 2}),
        ("composite_stch", {"run_ids": None}),
        ("objective_gp_stch", {"run_ids": [0, 0, 1, 1]}),
        ("objective_gp_stch", {"run_ids": [0, 1, 0]}),
        ("standard_qlogehvi", {"weights": [[0.5, 0.5]] * 2}),
        ("composite_qlogehvi", {"run_ids": [0, 1, 0, 1]}),
    ),
)
def test_resume_validation_rejects_method_incompatible_results(
    tmp_path, method, overrides
):
    config = _artifact_config(budget=4, weights=2)
    path = tmp_path / "zdt1" / method / "trial0.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(_method_artifact(method, config, **overrides)),
        encoding="utf-8",
    )

    assert not benchmark._valid_result(path, config)


@pytest.mark.parametrize("method", ("objective_gp_stch", "composite_stch"))
def test_resume_validation_rejects_nondivisible_budget(tmp_path, method):
    config = _artifact_config(budget=3, weights=2)
    path = tmp_path / "zdt1" / method / "trial0.json"
    payload = _method_artifact(method, config)
    if method.endswith("_stch"):
        payload["run_ids"] = [0, 1, 0]
    benchmark._atomic_write_json(path, payload)

    assert not benchmark._valid_result(path, config)


def test_resume_validation_never_raises_and_rejects_malformed_artifacts(tmp_path):
    path = tmp_path / "zdt1" / "standard_qlogehvi" / "trial0.json"
    config = _artifact_config()
    invalid_payloads = [
        [],
        {key: value for key, value in _artifact(config).items() if key != "metadata"},
        _artifact(config, problem="zdt2"),
        _artifact(config, X=[[0.0, 0.0]]),
        _artifact(config, Y=[[0.0, 1.0], [2.0]]),
        _artifact(config, hypervolume=[0.0, float("nan")]),
        _artifact(config, wall_seconds=[0.01, -0.01]),
        _artifact(config, acquisition_fallbacks=-1),
        _artifact(config, acquisition_fallbacks=1.5),
        _artifact(config, schema_version=1),
        _artifact(
            config,
            timing={"total_seconds": 0.0},
        ),
        _artifact(
            config,
            timing={
                **_artifact(config)["timing"],
                "total_seconds": float("inf"),
            },
        ),
    ]

    for payload in invalid_payloads:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert not benchmark._valid_result(path, config)


def test_failed_or_corrupt_artifact_is_not_valid_for_resume(tmp_path):
    path = tmp_path / "zdt1" / "standard_qlogehvi" / "trial0.json"
    benchmark._atomic_write_json(
        path,
        _artifact(failed="boom"),
    )
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{", encoding="utf-8")

    assert not benchmark._valid_result(path, _artifact_config())
    assert not benchmark._valid_result(corrupt, _artifact_config())


def test_selected_job_resumes_only_with_the_full_expected_config(
    tmp_path, monkeypatch
):
    calls = []

    def succeed(*args, **kwargs):
        calls.append("standard_qlogehvi")
        result = _fake_solver_result()
        result.acquisition_fallbacks = 3
        return result

    def unexpected(*args, **kwargs):
        pytest.fail("unselected method ran")

    monkeypatch.setattr(benchmark, "standard_mobo", succeed)
    monkeypatch.setattr(benchmark, "composite_mobo", unexpected)
    monkeypatch.setattr(benchmark, "chebyshev_bo", unexpected)
    monkeypatch.setattr(benchmark, "composite_chebyshev_bo", unexpected)
    args = _benchmark_args(
        tmp_path,
        trials=3,
        trial=1,
        method="standard_qlogehvi",
    )

    benchmark.run(args)
    benchmark.run(args)

    path = tmp_path / "zdt1" / "standard_qlogehvi" / "trial1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert calls == ["standard_qlogehvi"]
    assert payload["seed"] == 8
    assert payload["config"] == {
        "dim": 2,
        "budget": 2,
        "initial": 1,
        "weights": 1,
        "temperature": 0.05,
        "seed": 8,
        "raw_samples": 4,
        "restarts": 1,
        "mc_samples": 8,
    }
    assert payload["acquisition_fallbacks"] == 3
    assert not (tmp_path / "zdt1" / "standard_qlogehvi" / "trial0.json").exists()

    args.raw_samples = 8
    benchmark.run(args)
    assert calls == ["standard_qlogehvi", "standard_qlogehvi"]


def test_one_job_failure_writes_traceback_and_does_not_abort_siblings(
    tmp_path, monkeypatch
):
    calls = []

    def fail(*args, **kwargs):
        calls.append("standard_qlogehvi")
        raise RuntimeError("boom")

    def succeed(method):
        def job(*args, **kwargs):
            calls.append(method)
            return _fake_solver_result(method)

        return job

    monkeypatch.setattr(benchmark, "standard_mobo", fail)
    monkeypatch.setattr(
        benchmark, "composite_mobo", succeed("composite_qlogehvi")
    )
    monkeypatch.setattr(
        benchmark, "chebyshev_bo", succeed("objective_gp_stch")
    )
    monkeypatch.setattr(
        benchmark, "composite_chebyshev_bo", succeed("composite_stch")
    )

    with pytest.raises(RuntimeError, match="1 benchmark job failed"):
        benchmark.run(_benchmark_args(tmp_path))

    assert calls == list(benchmark.METHOD_LABELS)
    for method in benchmark.METHOD_LABELS:
        path = tmp_path / "zdt1" / method / "trial0.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert path.exists()
        assert payload["metadata"]["git_commit"]
        assert payload["metadata"]["packages"]
        assert set(solvers.TIMING_KEYS) <= set(payload["timing"])
        assert {"hypervolume_seconds", "total_seconds"} <= set(
            payload["timing"]
        )
        if method == "standard_qlogehvi":
            assert "Traceback" in payload["failed"]
            assert "RuntimeError: boom" in payload["failed"]
            assert not benchmark._valid_result(path, payload["config"])
        else:
            assert payload["failed"] is None
            assert payload["components"] == (
                [[0.0], [0.0]] if method.startswith("composite_") else None
            )
            assert payload["weights"] == (
                [[0.05, 0.95]] if method.endswith("_stch") else None
            )
            assert payload["run_ids"] == (
                [0, 0] if method.endswith("_stch") else None
            )
            assert payload["wall_seconds"] == [0.01, 0.01]
            assert benchmark._valid_result(path, payload["config"])


def test_nonfinite_result_writes_finite_failure_and_continues_siblings(
    tmp_path, monkeypatch
):
    calls = []
    bad = _fake_solver_result()
    bad.X[0, 0] = torch.nan

    def solver(method, result):
        def job(*args, **kwargs):
            calls.append(method)
            return result

        return job

    monkeypatch.setattr(
        benchmark,
        "standard_mobo",
        solver("standard_qlogehvi", bad),
    )
    monkeypatch.setattr(
        benchmark,
        "composite_mobo",
        solver(
            "composite_qlogehvi",
            _fake_solver_result("composite_qlogehvi"),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "chebyshev_bo",
        solver("objective_gp_stch", _fake_solver_result("objective_gp_stch")),
    )
    monkeypatch.setattr(
        benchmark,
        "composite_chebyshev_bo",
        solver("composite_stch", _fake_solver_result("composite_stch")),
    )

    with pytest.raises(RuntimeError, match="1 benchmark job failed"):
        benchmark.run(_benchmark_args(tmp_path))

    assert calls == list(benchmark.METHOD_LABELS)
    failed_path = tmp_path / "zdt1" / "standard_qlogehvi" / "trial0.json"
    payload = json.loads(
        failed_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail(f"nonfinite JSON: {value}"),
    )
    assert "ValueError" in payload["failed"]
    assert payload["X"] == []
    assert not benchmark._valid_result(failed_path, payload["config"])


def test_post_solver_failure_preserves_completed_timings(tmp_path, monkeypatch):
    result = _fake_solver_result(budget=1)
    result.timing["gp_fit_seconds"] = 1.25
    result.timing["solver_total_seconds"] = 1.5
    monkeypatch.setattr(benchmark, "standard_mobo", lambda *args, **kwargs: result)

    args = _benchmark_args(tmp_path, method="standard_qlogehvi")
    with pytest.raises(RuntimeError, match="1 benchmark job failed"):
        benchmark.run(args)

    payload = json.loads(
        (tmp_path / "zdt1" / "standard_qlogehvi" / "trial0.json").read_text()
    )
    assert "used 1, expected 2" in payload["failed"]
    assert payload["timing"]["gp_fit_seconds"] == 1.25
    assert payload["timing"]["solver_total_seconds"] == 1.5
    assert payload["timing"]["total_seconds"] > 0


def test_qlog_worker_ignores_stch_budget_constraints(tmp_path, monkeypatch):
    calls = []

    def succeed(*args, **kwargs):
        calls.append((kwargs["n_init"], kwargs["n_iter"]))
        return _fake_solver_result(budget=3)

    monkeypatch.setattr(benchmark, "standard_mobo", succeed)
    args = _benchmark_args(
        tmp_path,
        method="standard_qlogehvi",
        budget=3,
        initial=2,
        weights=2,
    )

    benchmark.run(args)

    assert calls == [(2, 1)]
    path = tmp_path / "zdt1" / "standard_qlogehvi" / "trial0.json"
    assert benchmark._valid_result(path, benchmark._job_config(args, "zdt1", 0))


def _write_pair_artifact(tmp_path, method, trial, payload):
    benchmark._atomic_write_json(
        tmp_path / "zdt1" / method / f"trial{trial}.json",
        payload,
    )


def test_load_traces_pairs_trials_and_filters_requested_config(tmp_path):
    requested = _benchmark_args(tmp_path, trials=3)
    exact0 = _artifact_config(seed=7)
    exact1 = _artifact_config(seed=8)
    stale2 = _artifact_config(seed=9, raw_samples=8)

    _write_pair_artifact(
        tmp_path, "standard_qlogehvi", 0,
        _method_artifact("standard_qlogehvi", exact0),
    )
    direct = _method_artifact("standard_qlogehvi", exact1)
    composite = _method_artifact("composite_qlogehvi", exact1)
    _write_pair_artifact(
        tmp_path, "standard_qlogehvi", 1, direct
    )
    _write_pair_artifact(
        tmp_path, "composite_qlogehvi", 1, composite
    )
    for method in ("standard_qlogehvi", "composite_qlogehvi"):
        _write_pair_artifact(
            tmp_path, method, 2, _method_artifact(method, stale2)
        )

    traces = benchmark.load_traces(requested)

    assert np.array_equal(
        traces["zdt1"]["Standard qLogEHVI"], [direct["hypervolume"]]
    )
    assert np.array_equal(
        traces["zdt1"]["Composite qLogEHVI"],
        [composite["hypervolume"]],
    )


def test_load_traces_reads_each_valid_artifact_once(tmp_path, monkeypatch):
    config = _artifact_config()
    for method in ("standard_qlogehvi", "composite_qlogehvi"):
        _write_pair_artifact(
            tmp_path, method, 0, _method_artifact(method, config)
        )
    reads = {}
    original = Path.read_text

    def read_once(path, *args, **kwargs):
        if path.suffix == ".json":
            reads[path] = reads.get(path, 0) + 1
            if reads[path] > 1:
                raise AssertionError(f"reread {path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_once)

    traces = benchmark.load_traces(_benchmark_args(tmp_path))

    assert set(traces["zdt1"]) == {
        "Standard qLogEHVI",
        "Composite qLogEHVI",
    }
    assert set(reads.values()) == {1}


@pytest.mark.parametrize(
    ("pair", "initial_rows"),
    (("qlogehvi", 1), ("stch", 2)),
)
def test_load_traces_requires_family_matched_initial_designs(
    tmp_path, pair, initial_rows
):
    config = _artifact_config(budget=4, initial=1, weights=2)
    args = _benchmark_args(tmp_path, budget=4, initial=1, weights=2)
    direct_method, composite_method = benchmark.COMPARISONS[pair]
    direct = _method_artifact(direct_method, config)
    composite = _method_artifact(composite_method, config)
    post_initial = torch.tensor(composite["X"], dtype=torch.double)
    post_initial[-1, 1] = 0.5
    _set_design(composite, post_initial)
    _write_pair_artifact(tmp_path, direct_method, 0, direct)
    _write_pair_artifact(tmp_path, composite_method, 0, composite)

    paired = benchmark.load_traces(args)

    assert set(paired["zdt1"]) == {
        benchmark.METHOD_LABELS[direct_method],
        benchmark.METHOD_LABELS[composite_method],
    }

    mismatched = torch.tensor(composite["X"], dtype=torch.double)
    mismatched[initial_rows - 1, 0] += 0.01
    _set_design(composite, mismatched)
    _write_pair_artifact(tmp_path, composite_method, 0, composite)

    assert benchmark.load_traces(args)["zdt1"] == {}


def test_cli_accepts_zero_based_job_selectors_and_results_dir(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--problems",
            "zdt1",
            "--trial",
            "0",
            "--method",
            "composite_stch",
            "--results-dir",
            str(tmp_path),
            "--summary-only",
        ],
    )

    args = benchmark.parse_args()

    assert args.trial == 0
    assert args.method == "composite_stch"
    assert args.results_dir == tmp_path
    assert args.summary_only


def test_cli_defaults_and_zdt2_job_protocol(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark.py"])

    args = benchmark.parse_args()

    assert (
        args.trials,
        args.initial,
        args.budget,
        args.weights,
        args.temperature,
        args.seed,
        args.raw_samples,
        args.restarts,
        args.mc_samples,
    ) == (20, 5, 40, 2, 0.05, 0, 128, 8, 512)
    config = benchmark._job_config(args, "zdt2", 0)
    assert (
        config["budget"], config["initial"], config["mc_samples"]
    ) == (30, 5, 512)


def test_summary_only_reads_disk_without_running_jobs(tmp_path, monkeypatch):
    args = _benchmark_args(tmp_path, summary_only=True)
    traces = {"zdt1": {}}
    seen = []
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark, "run", lambda _: pytest.fail("summary-only ran jobs")
    )
    monkeypatch.setattr(
        benchmark,
        "load_traces",
        lambda requested: traces if requested is args else pytest.fail(),
    )
    monkeypatch.setattr(
        benchmark,
        "plot_results",
        lambda loaded, output, initial, weights: seen.append(loaded),
    )

    benchmark.main()

    assert seen == [traces]


def test_selected_worker_does_not_plot(tmp_path, monkeypatch):
    args = _benchmark_args(tmp_path, trial=0)
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(benchmark, "run", lambda _: {})
    monkeypatch.setattr(
        benchmark, "plot_results", lambda *args: pytest.fail("worker plotted")
    )

    benchmark.main()


@pytest.mark.parametrize(
    ("problem", "expected"),
    (("zdt1", [5, 10]), ("zdt2", [5, 10])),
)
def test_plot_uses_global_five_point_initial_markers(
    problem, expected, tmp_path, monkeypatch
):
    markers = []
    original_axvline = Axes.axvline

    def record_marker(self, x=0, *args, **kwargs):
        markers.append(x)
        return original_axvline(self, x, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvline", record_marker)
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    values = np.zeros((1, 12))
    traces = {
        problem: {label: values for label in benchmark.METHOD_LABELS.values()}
    }

    benchmark.plot_results(traces, tmp_path / "plot.png", 5, 2)

    assert markers == expected


def test_zdt2_uses_five_initial_points_and_thirty_total(tmp_path, monkeypatch):
    calls = {}

    def solver(method):
        def job(*args, **kwargs):
            calls[method] = (
                kwargs["n_init"], kwargs["n_iter"], kwargs["mc_samples"]
            )
            return _fake_solver_result(method, budget=30, weights=2)

        return job

    monkeypatch.setattr(
        benchmark, "standard_mobo", solver("standard_qlogehvi")
    )
    monkeypatch.setattr(
        benchmark, "composite_mobo", solver("composite_qlogehvi")
    )
    monkeypatch.setattr(
        benchmark, "chebyshev_bo", solver("objective_gp_stch")
    )
    monkeypatch.setattr(
        benchmark, "composite_chebyshev_bo", solver("composite_stch")
    )

    benchmark.run(
        _benchmark_args(
            tmp_path,
            problems=["zdt2"],
            budget=99,
            initial=5,
            weights=2,
            mc_samples=17,
        )
    )

    assert calls == {
        "standard_qlogehvi": (5, 25, 17),
        "composite_qlogehvi": (5, 25, 17),
        "objective_gp_stch": (5, 10, 17),
        "composite_stch": (5, 10, 17),
    }
