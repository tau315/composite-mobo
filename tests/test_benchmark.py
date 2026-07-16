from argparse import Namespace
import json
import math
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
        "results_dir": tmp_path,
        "summary_only": False,
        "output": tmp_path / "hypervolume.png",
    }
    values.update(overrides)
    return Namespace(**values)


def _fake_solver_result(
    method="standard_qlogehvi", budget=2, dim=2, weights=1
):
    return solvers.SolverResult(
        X=torch.zeros((budget, dim), dtype=torch.double),
        Y=torch.column_stack(
            (
                torch.linspace(1.0, 0.0, budget, dtype=torch.double),
                torch.linspace(0.0, 1.0, budget, dtype=torch.double),
            )
        ),
        components=(
            torch.zeros((budget, 1), dtype=torch.double)
            if method.startswith("composite_")
            else None
        ),
        weights=(
            torch.full((weights, 2), 0.5, dtype=torch.double)
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
    }
    config.update(overrides)
    return config


def _artifact(config=None, **overrides):
    config = config or _artifact_config()
    budget = config["budget"]
    payload = {
        "schema_version": 1,
        "problem": "zdt1",
        "method": "standard_qlogehvi",
        "trial": config["seed"] - 7,
        "seed": config["seed"],
        "config": config,
        "metadata": {
            "python": "3.12.0",
            "packages": {"torch": "2.0"},
            "git_commit": "abc123",
        },
        "X": [[float(i), 0.0] for i in range(budget)],
        "Y": [[float(i), 1.0] for i in range(budget)],
        "components": None,
        "weights": None,
        "run_ids": None,
        "hypervolume": [float(i) for i in range(budget)],
        "wall_seconds": [0.01] * budget,
        "timing": dict.fromkeys(
            (*solvers.TIMING_KEYS, "hypervolume_seconds", "total_seconds"),
            0.0,
        ),
        "failed": None,
    }
    payload.update(overrides)
    return payload


def _method_artifact(method, config=None, **overrides):
    config = config or _artifact_config(budget=4, weights=2)
    budget = config["budget"]
    payload = _artifact(config, method=method)
    if method.startswith("composite_"):
        payload["components"] = [[float(i)] for i in range(budget)]
    if method.endswith("_stch"):
        payload["weights"] = [[0.5, 0.5]] * config["weights"]
        payload["run_ids"] = list(range(config["weights"])) * (
            budget // config["weights"]
        )
    payload.update(overrides)
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


def test_atomic_artifact_and_strict_resume_validation(tmp_path):
    path = tmp_path / "zdt1" / "standard_qlogehvi" / "trial0.json"
    config = _artifact_config()
    payload = _artifact(config)

    benchmark._atomic_write_json(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert benchmark._valid_result(path, config)
    assert not benchmark._valid_result(path, _artifact_config(budget=3))
    assert not benchmark._valid_result(path, _artifact_config(seed=8))


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


@pytest.mark.parametrize("method", tuple(benchmark.METHOD_LABELS))
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
        return _fake_solver_result()

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
    }
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
                [[0.5, 0.5]] if method.endswith("_stch") else None
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


def _write_pair_artifact(tmp_path, method, trial, config, values):
    benchmark._atomic_write_json(
        tmp_path / "zdt1" / method / f"trial{trial}.json",
        _method_artifact(
            method,
            config,
            trial=trial,
            hypervolume=values,
        ),
    )


def test_load_traces_pairs_trials_and_filters_requested_config(tmp_path):
    requested = _benchmark_args(tmp_path, trials=3)
    exact0 = _artifact_config(seed=7)
    exact1 = _artifact_config(seed=8)
    stale2 = _artifact_config(seed=9, raw_samples=8)

    _write_pair_artifact(
        tmp_path, "standard_qlogehvi", 0, exact0, [0.0, 1.0]
    )
    _write_pair_artifact(
        tmp_path, "standard_qlogehvi", 1, exact1, [1.0, 2.0]
    )
    _write_pair_artifact(
        tmp_path, "composite_qlogehvi", 1, exact1, [3.0, 4.0]
    )
    for method in ("standard_qlogehvi", "composite_qlogehvi"):
        _write_pair_artifact(tmp_path, method, 2, stale2, [5.0, 6.0])

    traces = benchmark.load_traces(requested)

    assert np.array_equal(
        traces["zdt1"]["Standard qLogEHVI"], [[1.0, 2.0]]
    )
    assert np.array_equal(
        traces["zdt1"]["Composite qLogEHVI"], [[3.0, 4.0]]
    )


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
    ) == (20, 5, 40, 2, 0.05, 0, 128, 8)
    config = benchmark._job_config(args, "zdt2", 0)
    assert (config["budget"], config["initial"]) == (30, 5)


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
            calls[method] = (kwargs["n_init"], kwargs["n_iter"])
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
        )
    )

    assert calls == {
        "standard_qlogehvi": (5, 25),
        "composite_qlogehvi": (5, 25),
        "objective_gp_stch": (5, 10),
        "composite_stch": (5, 10),
    }
