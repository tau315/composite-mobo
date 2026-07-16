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


def _fake_solver_result(budget=2, dim=2):
    return solvers.SolverResult(
        X=torch.zeros((budget, dim), dtype=torch.double),
        Y=torch.column_stack(
            (
                torch.linspace(1.0, 0.0, budget, dtype=torch.double),
                torch.linspace(0.0, 1.0, budget, dtype=torch.double),
            )
        ),
        components=torch.zeros((budget, 1), dtype=torch.double),
        weights=torch.tensor([[0.5, 0.5]], dtype=torch.double),
        run_ids=torch.zeros(budget, dtype=torch.long),
        timing=dict.fromkeys(solvers.TIMING_KEYS, 0.0),
        wall_seconds=[0.01] * budget,
    )


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
    payload = {
        "problem": "zdt1",
        "method": "standard_qlogehvi",
        "trial": 0,
        "config": {"budget": 2, "seed": 7},
        "X": [[0.0], [1.0]],
        "Y": [[1.0, 1.0], [0.0, 2.0]],
        "hypervolume": [0.0, 1.0],
        "failed": None,
        "timing": {"total_seconds": 1.0},
    }

    benchmark._atomic_write_json(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert benchmark._valid_result(path, {"budget": 2, "seed": 7})
    assert not benchmark._valid_result(path, {"budget": 3, "seed": 7})
    assert not benchmark._valid_result(path, {"budget": 2})


def test_failed_or_corrupt_artifact_is_not_valid_for_resume(tmp_path):
    path = tmp_path / "failed.json"
    benchmark._atomic_write_json(
        path,
        {
            "config": {"budget": 0},
            "X": [],
            "Y": [],
            "hypervolume": [],
            "failed": "boom",
        },
    )
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{", encoding="utf-8")

    assert not benchmark._valid_result(path, {"budget": 0})
    assert not benchmark._valid_result(corrupt, {"budget": 0})


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
            return _fake_solver_result()

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
            assert payload["components"] == [[0.0], [0.0]]
            assert payload["weights"] == [[0.5, 0.5]]
            assert payload["run_ids"] == [0, 0]
            assert payload["wall_seconds"] == [0.01, 0.01]
            assert benchmark._valid_result(path, payload["config"])


def test_load_traces_reads_only_successful_disk_artifacts(tmp_path):
    for trial in range(2):
        benchmark._atomic_write_json(
            tmp_path / "zdt1" / "standard_qlogehvi" / f"trial{trial}.json",
            {
                "trial": trial,
                "config": {"budget": 2},
                "X": [[0.0], [1.0]],
                "Y": [[1.0, 0.0], [0.0, 1.0]],
                "hypervolume": [float(trial), float(trial + 1)],
                "failed": None,
            },
        )
    benchmark._atomic_write_json(
        tmp_path / "zdt1" / "composite_qlogehvi" / "trial0.json",
        {
            "trial": 0,
            "config": {"budget": 0},
            "X": [],
            "Y": [],
            "hypervolume": [],
            "failed": "boom",
        },
    )

    traces = benchmark.load_traces(tmp_path, ["zdt1"])

    assert set(traces["zdt1"]) == {"Standard qLogEHVI"}
    assert np.array_equal(
        traces["zdt1"]["Standard qLogEHVI"],
        np.asarray([[0.0, 1.0], [1.0, 2.0]]),
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


def test_summary_only_reads_disk_without_running_jobs(tmp_path, monkeypatch):
    args = _benchmark_args(tmp_path, summary_only=True)
    traces = {"zdt1": {}}
    seen = []
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark, "run", lambda _: pytest.fail("summary-only ran jobs")
    )
    monkeypatch.setattr(
        benchmark, "load_traces", lambda results_dir, problems: traces
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
    (("zdt1", [5, 10]), ("zdt2", [3, 6])),
)
def test_plot_uses_family_specific_initial_markers(
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
