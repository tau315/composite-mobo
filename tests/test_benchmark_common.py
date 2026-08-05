from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from botorch.exceptions.errors import CandidateGenerationError

import benchmark_common
import solvers
from benchmark_common import BenchmarkProblem


def _tiny_problem_counter():
    """A trivially cheap composite problem plus per-call counters."""

    calls = {"rows": 0, "mc_compose": 0}

    def components(X):
        calls["rows"] += len(X)
        return torch.stack((X[..., 0], (X[..., 1] - 0.5).square()), dim=-1)

    def compose(H):
        if H.ndim > 2:
            calls["mc_compose"] += 1
        return torch.stack((H[..., 0], H[..., 1]), dim=-1)

    return calls, components, compose


def _problem(suite="low"):
    _, components, compose = _tiny_problem_counter()
    return BenchmarkProblem(
        name="Tiny composite problem",
        slug="tiny",
        dim=2,
        num_objectives=2,
        suite=suite,
        evaluate_components=components,
        compose=compose,
        ideal=torch.zeros(2, dtype=torch.double),
        ref_point=torch.full((2,), 1.5, dtype=torch.double),
    )


def _args(problem, tmp_path, extra=()):
    """Parse the benchmark CLI exactly the way ``run_benchmark`` does."""

    argv = [
        "--trials", "1",
        "--initial", "1",
        "--iterations", "1",
        "--weights", "1",
        "--per-weight", "1",
        "--raw-samples", "4",
        "--restarts", "1",
        "--morbo-raw-samples", "8",
        "--trust-regions", "2",
        "--seed", "7",
        "--results-dir", str(tmp_path),
        "--output", str(tmp_path / "plot.png"),
        *extra,
    ]
    args = benchmark_common._argument_parser(problem).parse_args(argv)
    if args.iterations is None:
        args.iterations = args.evaluations - args.initial
    if args.per_weight is None:
        args.per_weight = (args.evaluations - args.initial) // args.weights
    return args


def _artifact(problem, args, method, family, trial=0, seed=None):
    """A synthetic but fully self-consistent artifact for ``method``."""

    seed = args.seed if seed is None else seed
    evaluations = benchmark_common.expected_evaluations(args, family)
    X = torch.zeros((evaluations, problem.dim), dtype=torch.double)
    X[:, 0] = torch.linspace(0.0, 0.75, evaluations, dtype=torch.double)
    composite = "composite" in method
    components = problem.evaluate_components(X).double()
    Y = problem.compose(components).double() if composite else problem.evaluate(X)
    payload = {
        "schema_version": benchmark_common.SCHEMA_VERSION,
        "problem": problem.slug,
        "method": method,
        "family": family,
        "composite": composite,
        "trial": trial,
        "seed": seed,
        "config": benchmark_common._job_config(problem, args, seed),
        "metadata": benchmark_common._run_metadata(),
        "X": X.tolist(),
        "Y": Y.tolist(),
        "components": components.tolist() if composite else None,
        "weights": None,
        "run_ids": None,
        "hypervolume": benchmark_common.dominated_hypervolume_trace(
            Y, problem.ref_point
        ).tolist(),
        "wall_seconds": [0.01] * evaluations,
        "timing": dict.fromkeys(benchmark_common.RESULT_TIMING_KEYS, 0.0),
        "acquisition_fallbacks": 0,
        "failed": None,
    }
    if family == "stch":
        payload["weights"] = solvers.simplex_weights(
            args.weights, problem.num_objectives, seed=314159
        ).tolist()
        run_ids = [-1] * args.initial
        for weight_id in range(args.weights):
            run_ids += [weight_id] * args.per_weight
        payload["run_ids"] = run_ids
    return payload


def _write(tmp_path, problem, payload):
    path = (
        tmp_path
        / problem.slug
        / payload["method"]
        / f"trial{payload['trial']}.json"
    )
    benchmark_common._atomic_write_json(path, payload)
    return path


# --------------------------------------------------------------------------
# Solver instrumentation
# --------------------------------------------------------------------------


def test_solver_timings_are_complete_and_finite():
    _, components, compose = _tiny_problem_counter()
    result = solvers.composite_mobo(
        lambda X: compose(components(X)),
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


def test_every_solver_reports_timing_and_one_wall_time_per_evaluation():
    """Resume validation requires this of *all* solvers, not just the fast ones."""

    _, components, compose = _tiny_problem_counter()
    evaluate = lambda X: compose(components(X))
    ref_point = torch.tensor([2.0, 2.0], dtype=torch.double)
    weights = torch.tensor([[0.5, 0.5]], dtype=torch.double)
    ideal = torch.zeros(2, dtype=torch.double)
    config = solvers.MORBOConfig(n_trust_regions=2, raw_samples=8)
    results = (
        solvers.standard_mobo(evaluate, 2, ref_point, n_init=2, n_iter=0),
        solvers.composite_mobo(
            evaluate, components, compose, 2, ref_point, n_init=2, n_iter=0
        ),
        solvers.chebyshev_bo(
            evaluate, 2, weights, ideal, n_init=2, n_per_scalarization=1,
            raw_samples=8, num_restarts=2,
        ),
        solvers.composite_chebyshev_bo(
            evaluate, components, compose, 2, weights, ideal,
            n_init=2, n_per_scalarization=1, raw_samples=8, num_restarts=2,
        ),
        solvers.spherical_chebyshev_bo(
            evaluate, 2, weights, ideal, n_init=2, n_per_scalarization=1,
            raw_samples=8, num_restarts=2,
        ),
        solvers.composite_spherical_chebyshev_bo(
            evaluate, components, compose, 2, weights, ideal,
            n_init=2, n_per_scalarization=1, raw_samples=8, num_restarts=2,
        ),
        solvers.morbo(evaluate, 2, ref_point, n_init=3, n_iter=1, config=config),
        solvers.composite_morbo(
            evaluate, components, compose, 2, ref_point,
            n_init=3, n_iter=1, config=config,
        ),
    )
    for result in results:
        assert set(result.timing) == set(solvers.TIMING_KEYS)
        assert all(math.isfinite(v) and v >= 0 for v in result.timing.values())
        assert len(result.wall_seconds) == len(result.X)
        assert result.timing["solver_total_seconds"] > 0


def test_scalarized_runs_share_one_initial_design_across_weights():
    weights = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.double)
    evaluate = lambda X: torch.cat((X[:, :1], X[:, 1:2]), dim=-1)
    result = solvers.chebyshev_bo(
        evaluate,
        2,
        weights,
        torch.zeros(2, dtype=torch.double),
        n_init=2,
        n_per_scalarization=1,
        raw_samples=8,
        num_restarts=2,
    )
    # -1 marks the shared design, then one contiguous block per weight.
    assert result.run_ids.tolist() == [-1, -1, 0, 1]
    assert len(result.X) == len(result.wall_seconds) == 4


def test_acquisition_fallback_rejects_all_nonfinite_values(monkeypatch):
    candidates = torch.tensor(
        [[[0.1, 0.2]], [[0.4, 0.5]], [[0.8, 0.9]]], dtype=torch.double
    )

    def fail_generation(*args, **kwargs):
        raise CandidateGenerationError("failed")

    monkeypatch.setattr(solvers, "optimize_acqf", fail_generation)
    monkeypatch.setattr(solvers, "draw_sobol_samples", lambda **kwargs: candidates)

    values = torch.tensor([torch.inf, 1.0, torch.nan], dtype=torch.double)
    chosen, fallback = solvers._optimize(lambda X: values, 2, 2, 2)
    assert torch.equal(chosen, candidates[1])
    assert fallback

    with pytest.raises(RuntimeError, match="non-finite on every fallback candidate"):
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
    monkeypatch.setattr(solvers, "NondominatedPartitioning", lambda **kwargs: object())
    monkeypatch.setattr(solvers, "qLogExpectedHypervolumeImprovement", qlog_acquisition)
    monkeypatch.setattr(solvers, "qLogExpectedImprovement", qei_acquisition)
    monkeypatch.setattr(
        solvers,
        "_optimize",
        lambda *args: (torch.full((1, 2), 0.25, dtype=torch.double), True),
    )

    _, components, compose = _tiny_problem_counter()
    evaluate = lambda X: compose(components(X))
    ref_point = torch.tensor([2.0, 2.0], dtype=torch.double)
    weights = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.double)
    ideal = torch.zeros(2, dtype=torch.double)
    qlog_results = (
        solvers.standard_mobo(
            evaluate, 2, ref_point, n_init=2, n_iter=2, seed=11, mc_samples=13
        ),
        solvers.composite_mobo(
            evaluate, components, compose, 2, ref_point,
            n_init=2, n_iter=2, seed=11, mc_samples=13,
        ),
    )
    stch_results = (
        solvers.chebyshev_bo(
            evaluate, 2, weights, ideal, n_init=1, n_per_scalarization=1,
            seed=11, mc_samples=13,
        ),
        solvers.composite_chebyshev_bo(
            evaluate, components, compose, 2, weights, ideal,
            n_init=1, n_per_scalarization=1, seed=11, mc_samples=13,
        ),
    )

    # Every stream is a pure function of (solver seed, weight index, design
    # size), so a direct and a composite run see identical MC noise.
    assert qlog_samplers == [((13,), 1100035), ((13,), 1100036)] * 2
    assert qei_samplers == [((13,), 1100034), ((13,), 1204763)] * 2
    assert [result.acquisition_fallbacks for result in qlog_results] == [2, 2]
    assert [result.acquisition_fallbacks for result in stch_results] == [2, 2]


def test_composite_solvers_evaluate_components_once_per_candidate():
    calls, components, compose = _tiny_problem_counter()
    evaluate = lambda X: compose(components(X))
    solvers.composite_mobo(
        evaluate,
        components,
        compose,
        2,
        torch.tensor([2.0, 2.0], dtype=torch.double),
        n_init=2,
        n_iter=1,
        raw_samples=8,
        num_restarts=2,
    )
    # Two initial points (once through evaluate, once directly) plus one
    # candidate; the composed objective never re-evaluates the components.
    assert calls["rows"] == 5
    assert calls["mc_compose"] > 0


# --------------------------------------------------------------------------
# Artifacts and resume validation
# --------------------------------------------------------------------------


def test_atomic_write_leaves_no_partial_file(tmp_path):
    problem = _problem()
    args = _args(problem, tmp_path)
    payload = _artifact(problem, args, "direct_qlogehvi", "qlogehvi")
    path = _write(tmp_path, problem, payload)
    assert path.exists()
    assert not list(path.parent.glob("*.tmp"))
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_valid_artifact_resumes_and_recomputes_every_derived_field(tmp_path):
    problem = _problem()
    args = _args(problem, tmp_path)
    for method, family in (
        ("direct_qlogehvi", "qlogehvi"),
        ("composite_qlogehvi", "qlogehvi"),
        ("objective_gp_stch", "stch"),
        ("composite_stch", "stch"),
    ):
        payload = _artifact(problem, args, method, family)
        path = _write(tmp_path, problem, payload)
        config = benchmark_common._job_config(problem, args, args.seed)
        assert (
            benchmark_common._validated_payload(path, problem, config) is not None
        ), method


@pytest.mark.parametrize(
    "field", ["Y", "components", "hypervolume", "weights", "run_ids"]
)
def test_resume_validation_rejects_tampered_derived_fields(tmp_path, field):
    problem = _problem()
    args = _args(problem, tmp_path)
    payload = _artifact(problem, args, "composite_stch", "stch")
    assert payload[field] is not None
    if isinstance(payload[field][0], list):
        payload[field][0][0] += 0.5
    else:
        payload[field][0] += 1
    path = _write(tmp_path, problem, payload)
    config = benchmark_common._job_config(problem, args, args.seed)
    assert benchmark_common._validated_payload(path, problem, config) is None


@pytest.mark.parametrize("key", ["dim", "initial", "seed", "raw_samples", "weights"])
def test_resume_validation_requires_the_exact_current_config(tmp_path, key):
    problem = _problem()
    args = _args(problem, tmp_path)
    payload = _artifact(problem, args, "direct_qlogehvi", "qlogehvi")
    payload["config"][key] += 1
    path = _write(tmp_path, problem, payload)
    config = benchmark_common._job_config(problem, args, args.seed)
    assert benchmark_common._validated_payload(path, problem, config) is None


@pytest.mark.parametrize("field", ["python", "packages", "git_commit"])
def test_resume_validation_requires_matching_provenance(tmp_path, field):
    problem = _problem()
    args = _args(problem, tmp_path)
    payload = _artifact(problem, args, "direct_qlogehvi", "qlogehvi")
    foreign = deepcopy(payload["metadata"])
    foreign[field] = (
        {name: "0.0.0" for name in payload["metadata"]["packages"]}
        if field == "packages"
        else "other"
    )
    payload["metadata"] = foreign
    path = _write(tmp_path, problem, payload)
    config = benchmark_common._job_config(problem, args, args.seed)
    assert benchmark_common._validated_payload(path, problem, config) is None
    # Aggregation across machines opts out of the provenance check explicitly.
    assert (
        benchmark_common._validated_payload(
            path, problem, config, allow_foreign_metadata=True
        )
        is not None
    )


def test_run_metadata_prefers_the_cluster_supplied_commit(monkeypatch):
    monkeypatch.setenv("COMPOSITE_MOBO_COMMIT", "deadbeef")
    assert benchmark_common._run_metadata()["git_commit"] == "deadbeef"


def test_resume_validation_never_raises_on_malformed_artifacts(tmp_path):
    problem = _problem()
    args = _args(problem, tmp_path)
    config = benchmark_common._job_config(problem, args, args.seed)
    directory = tmp_path / problem.slug / "direct_qlogehvi"
    directory.mkdir(parents=True)
    path = directory / "trial0.json"
    good = _artifact(problem, args, "direct_qlogehvi", "qlogehvi")
    for text in (
        "",
        "not json",
        "[]",
        "null",
        json.dumps({"schema_version": benchmark_common.SCHEMA_VERSION}),
        json.dumps({**good, "X": [[float("nan"), 0.0]]}),
        json.dumps({**good, "X": [[2.0, 0.0]]}),
        json.dumps({**good, "wall_seconds": []}),
        json.dumps({**good, "timing": {}}),
        json.dumps({**good, "acquisition_fallbacks": -1}),
        json.dumps({**good, "schema_version": 1}),
        json.dumps({**good, "composite": True}),
    ):
        path.write_text(text, encoding="utf-8")
        assert benchmark_common._validated_payload(path, problem, config) is None
    assert benchmark_common._validated_payload(
        tmp_path / "missing" / "m" / "trial0.json", problem, config
    ) is None


def test_failed_artifact_is_never_valid_for_resume(tmp_path):
    problem = _problem()
    args = _args(problem, tmp_path)
    payload = _artifact(problem, args, "direct_qlogehvi", "qlogehvi")
    payload["failed"] = "Traceback ..."
    path = _write(tmp_path, problem, payload)
    config = benchmark_common._job_config(problem, args, args.seed)
    assert benchmark_common._validated_payload(path, problem, config) is None


# --------------------------------------------------------------------------
# Running jobs
# --------------------------------------------------------------------------


def _run_cli(problem, tmp_path, extra=(), monkeypatch=None):
    argv = [
        "prog",
        "--trials", "1",
        "--initial", "1",
        "--iterations", "1",
        "--weights", "1",
        "--per-weight", "1",
        "--raw-samples", "4",
        "--restarts", "1",
        "--morbo-raw-samples", "8",
        "--trust-regions", "2",
        "--seed", "7",
        "--results-dir", str(tmp_path),
        "--output", str(tmp_path / "plot.png"),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    benchmark_common.run_benchmark(problem)


def test_worker_selectors_write_one_artifact_and_do_not_plot(tmp_path, monkeypatch):
    problem = _problem()
    _run_cli(
        problem,
        tmp_path,
        ["--trial", "0", "--method", "composite_qlogehvi"],
        monkeypatch,
    )
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json"))
    assert written == ["tiny/composite_qlogehvi/trial0.json"]
    assert not (tmp_path / "plot.png").exists()


def test_unknown_method_and_out_of_range_trial_are_rejected(tmp_path, monkeypatch):
    problem = _problem()
    with pytest.raises(ValueError, match="unknown --method"):
        _run_cli(problem, tmp_path, ["--method", "nope"], monkeypatch)
    with pytest.raises(ValueError, match="zero-based"):
        _run_cli(problem, tmp_path, ["--trial", "1"], monkeypatch)


def test_completed_jobs_resume_instead_of_rerunning(tmp_path, monkeypatch, capsys):
    problem = _problem()
    _run_cli(problem, tmp_path, [], monkeypatch)
    capsys.readouterr()
    _run_cli(problem, tmp_path, [], monkeypatch)
    output = capsys.readouterr().out
    assert output.count("resumed") == 4


def test_one_failing_job_writes_a_traceback_and_does_not_abort_siblings(
    tmp_path, monkeypatch
):
    problem = _problem()
    original = benchmark_common.standard_mobo

    def explode(*args, **kwargs):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(benchmark_common, "standard_mobo", explode)
    with pytest.raises(RuntimeError, match="1 benchmark job failed"):
        _run_cli(problem, tmp_path, [], monkeypatch)
    monkeypatch.setattr(benchmark_common, "standard_mobo", original)

    failed = json.loads(
        (tmp_path / "tiny/direct_qlogehvi/trial0.json").read_text(encoding="utf-8")
    )
    assert "solver exploded" in failed["failed"]
    assert failed["X"] == [] and failed["hypervolume"] == []
    assert failed["timing"]["total_seconds"] > 0
    # The three siblings still ran and are individually resumable.
    for method in ("composite_qlogehvi", "objective_gp_stch", "composite_stch"):
        assert (tmp_path / "tiny" / method / "trial0.json").exists()


def test_wrong_evaluation_count_is_recorded_as_a_failure(tmp_path, monkeypatch):
    problem = _problem()

    def short_run(*args, **kwargs):
        X = torch.zeros((1, 2), dtype=torch.double)
        return solvers.SolverResult(
            X=X,
            Y=problem.evaluate(X),
            timing=dict.fromkeys(solvers.TIMING_KEYS, 0.0),
            wall_seconds=[0.0],
        )

    monkeypatch.setattr(benchmark_common, "standard_mobo", short_run)
    with pytest.raises(RuntimeError, match="1 benchmark job failed"):
        _run_cli(problem, tmp_path, [], monkeypatch)
    failed = json.loads(
        (tmp_path / "tiny/direct_qlogehvi/trial0.json").read_text(encoding="utf-8")
    )
    assert "expected 2" in failed["failed"]


# --------------------------------------------------------------------------
# Loading and pairing traces
# --------------------------------------------------------------------------


def _panels(problem, args):
    _, panels = benchmark_common._solver_jobs(problem, args, args.seed)
    return panels


def test_load_traces_returns_one_array_per_trial_per_method(tmp_path):
    problem = _problem()
    args = _args(problem, tmp_path)
    for method, family in (
        ("direct_qlogehvi", "qlogehvi"),
        ("composite_qlogehvi", "qlogehvi"),
        ("objective_gp_stch", "stch"),
        ("composite_stch", "stch"),
    ):
        _write(tmp_path, problem, _artifact(problem, args, method, family))
    traces = benchmark_common.load_traces(problem, args, _panels(problem, args))
    assert set(traces) == {
        "Direct qLogEHVI",
        "Composite qLogEHVI",
        "Objective-GP STCH",
        "Composite STCH",
    }
    assert all(len(values) == 1 for values in traces.values())


def test_load_traces_drops_a_panel_whose_pair_is_incomplete(tmp_path):
    problem = _problem()
    args = _args(problem, tmp_path)
    _write(tmp_path, problem, _artifact(problem, args, "direct_qlogehvi", "qlogehvi"))
    _write(tmp_path, problem, _artifact(problem, args, "objective_gp_stch", "stch"))
    _write(tmp_path, problem, _artifact(problem, args, "composite_stch", "stch"))
    traces = benchmark_common.load_traces(problem, args, _panels(problem, args))
    assert "Direct qLogEHVI" not in traces
    assert set(traces) == {"Objective-GP STCH", "Composite STCH"}


def test_load_traces_requires_a_shared_initial_design_within_a_panel(tmp_path):
    problem = _problem()
    args = _args(problem, tmp_path)
    direct = _artifact(problem, args, "direct_qlogehvi", "qlogehvi")
    composite = _artifact(problem, args, "composite_qlogehvi", "qlogehvi")
    # Same config and provenance, but a different starting point: the two
    # curves would not be answering the same question.
    X = torch.tensor(composite["X"], dtype=torch.double)
    X[0, 1] = 0.9
    components = problem.evaluate_components(X).double()
    Y = problem.compose(components).double()
    composite["X"] = X.tolist()
    composite["components"] = components.tolist()
    composite["Y"] = Y.tolist()
    composite["hypervolume"] = benchmark_common.dominated_hypervolume_trace(
        Y, problem.ref_point
    ).tolist()
    _write(tmp_path, problem, direct)
    _write(tmp_path, problem, composite)
    traces = benchmark_common.load_traces(problem, args, _panels(problem, args))
    assert "Direct qLogEHVI" not in traces


def test_load_traces_does_not_mix_provenance_within_a_panel(tmp_path):
    problem = _problem()
    args = _args(problem, tmp_path)
    direct = _artifact(problem, args, "direct_qlogehvi", "qlogehvi")
    composite = _artifact(problem, args, "composite_qlogehvi", "qlogehvi")
    composite["metadata"] = {**composite["metadata"], "git_commit": "other"}
    _write(tmp_path, problem, direct)
    _write(tmp_path, problem, composite)
    traces = benchmark_common.load_traces(problem, args, _panels(problem, args))
    assert "Direct qLogEHVI" not in traces


def test_summary_only_plots_from_disk_without_running_any_solver(
    tmp_path, monkeypatch
):
    problem = _problem()
    args = _args(problem, tmp_path)
    for method, family in (
        ("direct_qlogehvi", "qlogehvi"),
        ("composite_qlogehvi", "qlogehvi"),
        ("objective_gp_stch", "stch"),
        ("composite_stch", "stch"),
    ):
        _write(tmp_path, problem, _artifact(problem, args, method, family))

    def forbidden(*args, **kwargs):
        raise AssertionError("no solver may run under --summary-only")

    monkeypatch.setattr(benchmark_common, "standard_mobo", forbidden)
    monkeypatch.setattr(benchmark_common, "chebyshev_bo", forbidden)
    _run_cli(problem, tmp_path, ["--summary-only"], monkeypatch)
    assert (tmp_path / "plot.png").exists()


def test_summary_only_without_artifacts_fails_loudly(tmp_path, monkeypatch):
    problem = _problem()
    with pytest.raises(RuntimeError, match="no comparable artifacts"):
        _run_cli(problem, tmp_path, ["--summary-only"], monkeypatch)


def test_list_methods_prints_the_on_disk_keys(tmp_path, monkeypatch, capsys):
    _run_cli(_problem(), tmp_path, ["--list-methods"], monkeypatch)
    keys = [line.split()[0] for line in capsys.readouterr().out.splitlines()]
    assert keys == [
        "direct_qlogehvi",
        "composite_qlogehvi",
        "objective_gp_stch",
        "composite_stch",
    ]
    assert not list(tmp_path.rglob("*.json"))


def test_high_suite_methods_round_trip_through_artifacts(tmp_path, monkeypatch):
    problem = _problem(suite="high")
    _run_cli(problem, tmp_path, [], monkeypatch)
    written = sorted(p.parent.name for p in tmp_path.rglob("*.json"))
    assert written == [
        "composite_morbo",
        "morbo",
        "spherical_composite_stch",
        "spherical_objective_stch",
    ]
    assert (tmp_path / "plot.png").exists()


def test_method_keys_are_stable_slugs():
    assert benchmark_common.method_key("Objective-GP STCH") == "objective_gp_stch"
    assert benchmark_common.method_key("Direct qLogEHVI") == "direct_qlogehvi"
    assert benchmark_common.method_key("Composite MORBO") == "composite_morbo"
