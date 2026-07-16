# Benchmark Readiness and Timers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the four-method ZDT/DTLZ benchmark correctly budgeted, timed, restartable, failure-isolated, and suitable for parallel 20-trial execution.

**Architecture:** Keep the existing two-file design. `solvers.py` owns solver-phase timing and exactly-once oracle use; `benchmark.py` owns problem transforms, one-run artifacts, resume, selectors, hypervolume timing, and plots. JSON files under `results/{problem}/{method}/trialN.json` are the source of truth.

**Tech Stack:** Python 3.10+, PyTorch, BoTorch 0.18, GPyTorch, NumPy, Matplotlib, pytest, standard-library JSON and filesystem operations.

## Global Constraints

- Preserve four existing method families, problem formulas, total budgets, and matched seeds.
- Default trials remain 20; ZDT1/ZDT3/DTLZ2 use five initial and 40 total evaluations; ZDT2 uses three initial and 30 total evaluations.
- Count one component-oracle call per composite observation.
- Use continuous BoTorch acquisition optimization; no internal multiprocessing framework.
- Primary comparisons are direct versus composite within qLogEHVI and STCH families.
- Delete `docs/superpowers/specs/2026-07-15-benchmark-readiness-timers-design.md` after implementation.

---

### Task 1: Physically Valid Composite Problems

**Files:**
- Modify: `benchmark.py:33-104`
- Create: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: existing `CompositeProblem`, `_zdt`, `_dtlz2`, and `get_problem`.
- Produces: exact component maps `r=sqrt(g-1)` for ZDT and `r=sqrt(g)` for DTLZ2; composers reconstruct `g=1+r²` and `g=r²`; ZDT3 ideal `(0, -0.7733690123)`.

- [ ] **Step 1: Write failing composition/domain tests**

```python
import torch

import benchmark


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
    assert torch.allclose(ideal, torch.tensor([0.0, -0.7733690123], dtype=torch.double))
```

- [ ] **Step 2: Run tests and verify intended failures**

Run: `python -m pytest tests/test_benchmark.py -q`

Expected: domain and ZDT3 ideal assertions fail against current `log(g)`/raw-`g` maps and `(0, 0)` ideal.

- [ ] **Step 3: Implement exact valid transforms**

```python
# ZDT components / compose
g = 1.0 + 9.0 * X[..., 1:].mean(dim=-1)
return torch.sqrt((g - 1.0).clamp_min(0.0)).unsqueeze(-1)

g = 1.0 + C[..., 0].square()

# ZDT ideal
ideal = torch.tensor(
    [0.0, -0.7733690123] if name == "zdt3" else [0.0, 0.0],
    dtype=torch.double,
)

# DTLZ components / compose
g = (X[..., -k:] - 0.5).square().sum(dim=-1, keepdim=True)
return torch.sqrt(g)

g = C[..., 0].square()
```

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_benchmark.py -q`

Expected: three tests pass.

- [ ] **Step 5: Commit**

```bash
git add benchmark.py tests/test_benchmark.py
git commit -m "fix: constrain composite components"
```

### Task 2: Exact Evaluation Accounting and Phase Timers

**Files:**
- Modify: `solvers.py:19-324`
- Modify: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `TIMING_KEYS`, `_new_timing() -> dict[str, float]`, `_timed(timing, key, callable, *args)`, and `SolverResult.timing` / `SolverResult.wall_seconds`.
- `standard_mobo`, `composite_mobo`, `chebyshev_bo`, and `composite_chebyshev_bo` retain public arguments.
- Composite paths evaluate components once, then call `compose(C, X)`.

- [ ] **Step 1: Write failing accounting, timing, and STCH-order tests**

```python
import math
import torch

import solvers


def _tiny_problem_counter():
    calls = {"rows": 0}
    def components(X):
        calls["rows"] += len(X)
        return X[:, :1]
    def compose(C, X):
        return torch.cat((C.square(), (X[:, 1:2] - 0.5).square()), dim=-1)
    return calls, components, compose


def test_composite_calls_components_once_per_point():
    calls, components, compose = _tiny_problem_counter()
    result = solvers.composite_mobo(
        lambda X: compose(components(X), X), components, compose, 2,
        torch.tensor([2.0, 2.0], dtype=torch.double),
        n_init=2, n_iter=1, raw_samples=16, num_restarts=2,
    )
    assert calls["rows"] == len(result.X) == 3


def test_solver_timings_are_complete_and_finite():
    calls, components, compose = _tiny_problem_counter()
    result = solvers.composite_mobo(
        lambda X: compose(components(X), X), components, compose, 2,
        torch.tensor([2.0, 2.0], dtype=torch.double),
        n_init=2, n_iter=1, raw_samples=16, num_restarts=2,
    )
    assert set(result.timing) == set(solvers.TIMING_KEYS)
    assert all(math.isfinite(v) and v >= 0 for v in result.timing.values())
    assert len(result.wall_seconds) == len(result.X)


def test_stch_results_interleave_weight_runs():
    weights = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.double)
    evaluate = lambda X: torch.cat((X[:, :1], X[:, 1:2]), dim=-1)
    result = solvers.chebyshev_bo(
        evaluate, 2, weights, torch.zeros(2, dtype=torch.double),
        n_init=2, n_iter=0,
    )
    assert result.run_ids.tolist() == [0, 1, 0, 1]
```

- [ ] **Step 2: Run tests and verify failures**

Run: `python -m pytest tests/test_benchmark.py -q`

Expected: duplicate count, missing timing fields, and weight-major ordering fail.

- [ ] **Step 3: Add minimal timing helper and result fields**

```python
from dataclasses import dataclass, field
from time import perf_counter

TIMING_KEYS = (
    "initial_design_seconds", "initial_evaluate_seconds",
    "initial_compose_seconds", "gp_fit_seconds",
    "acquisition_build_seconds", "acquisition_optimize_seconds",
    "bo_evaluate_seconds", "bo_compose_seconds", "solver_total_seconds",
)

def _new_timing():
    return dict.fromkeys(TIMING_KEYS, 0.0)

def _timed(timing, key, fn, *args):
    started = perf_counter()
    value = fn(*args)
    timing[key] += perf_counter() - started
    return value

@dataclass
class SolverResult:
    X: Tensor
    Y: Tensor
    components: Optional[Tensor] = None
    weights: Optional[Tensor] = None
    run_ids: Optional[Tensor] = None
    timing: dict[str, float] = field(default_factory=_new_timing)
    wall_seconds: list[float] = field(default_factory=list)
```

- [ ] **Step 4: Instrument all solver phases and remove duplicate calls**

For composite initialization and BO evaluations, use:

```python
C = _timed(timing, "initial_evaluate_seconds", evaluate_components, X).double()
Y = _timed(timing, "initial_compose_seconds", compose, C, X).double()

c = _timed(timing, "bo_evaluate_seconds", evaluate_components, x).double()
y = _timed(timing, "bo_compose_seconds", compose, c, x).double()
```

Time GP fitting, acquisition construction, and `_optimize` separately. Store amortized initial-batch wall time for each initial point and one total iteration duration for each BO point. In `_scalarized_runs`, accumulate timing across weights and interleave tensors with:

```python
def _interleave(tensors):
    trailing = tensors[0].shape[1:]
    return torch.stack(tensors, dim=1).reshape(-1, *trailing)

X = _interleave(all_x)
Y = _interleave(all_y)
C = _interleave(all_c) if all_c else None
run_ids = torch.arange(len(weights)).repeat(len(all_x[0]))
wall_seconds = torch.tensor(all_wall).T.reshape(-1).tolist()
```

- [ ] **Step 5: Harden acquisition fallback**

```python
from botorch.exceptions.errors import CandidateGenerationError, OptimizationGradientError

except (CandidateGenerationError, OptimizationGradientError):
    candidates = draw_sobol_samples(
        bounds=bounds, n=max(raw_samples * num_restarts, 256), q=1
    )
    with torch.no_grad():
        values = acq(candidates)
    finite = torch.isfinite(values)
    if not finite.any():
        raise RuntimeError("acquisition was non-finite on every fallback candidate")
    values = torch.where(finite, values, torch.full_like(values, -torch.inf))
```

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_benchmark.py -q`

Expected: all Task 1-2 tests pass.

- [ ] **Step 7: Commit**

```bash
git add solvers.py tests/test_benchmark.py
git commit -m "feat: time solver phases"
```

### Task 3: Atomic Artifacts, Resume, and Job Selectors

**Files:**
- Modify: `benchmark.py:9-275`
- Modify: `tests/test_benchmark.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `METHOD_LABELS`, `_atomic_write_json(path, payload)`, `_valid_result(path, expected)`, `_result_payload(problem_name, method, trial, config, result, hypervolume_values, metadata, failed)`, and `load_traces(results_dir, problems)`.
- Adds CLI flags `--trial`, `--method`, `--results-dir`, and `--summary-only`.
- One artifact path: `results/{problem}/{method}/trial{trial}.json`.

- [ ] **Step 1: Write failing persistence tests**

```python
import json


def test_atomic_artifact_and_resume_validation(tmp_path):
    path = tmp_path / "zdt1" / "standard_qlogehvi" / "trial0.json"
    payload = {
        "problem": "zdt1", "method": "standard_qlogehvi", "trial": 0,
        "config": {"budget": 2}, "X": [[0.0], [1.0]],
        "Y": [[1.0, 1.0], [0.0, 2.0]], "hypervolume": [0.0, 1.0],
        "failed": None, "timing": {"total_seconds": 1.0},
    }
    benchmark._atomic_write_json(path, payload)
    assert json.loads(path.read_text()) == payload
    assert benchmark._valid_result(path, {"budget": 2})
    assert not benchmark._valid_result(path, {"budget": 3})


def test_failed_artifact_is_not_valid_for_resume(tmp_path):
    path = tmp_path / "failed.json"
    benchmark._atomic_write_json(path, {
        "config": {"budget": 0}, "X": [], "Y": [], "hypervolume": [],
        "failed": "boom",
    })
    assert not benchmark._valid_result(path, {"budget": 0})
```

- [ ] **Step 2: Run tests and verify missing-helper failures**

Run: `python -m pytest tests/test_benchmark.py -q`

Expected: `AttributeError` for `_atomic_write_json`.

- [ ] **Step 3: Implement atomic JSON and strict resume validation**

```python
def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

def _valid_result(path: Path, expected: dict) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    budget = expected["budget"]
    return (
        payload.get("failed") is None
        and payload.get("config", {}).get("budget") == budget
        and len(payload.get("X", [])) == budget
        and len(payload.get("Y", [])) == budget
        and len(payload.get("hypervolume", [])) == budget
    )
```

- [ ] **Step 4: Refactor `run` into isolated resumable units**

For each selected tuple, build the existing solver job, start the total timer, catch `Exception`, compute hypervolume timing for successful results, and atomically write success or failure. Include full config, seed, package versions, commit SHA, tensors converted with `.tolist()`, `timing`, and `wall_seconds`. Continue after failures. Skip only `_valid_result(path, config)` artifacts. Use this payload shape:

```python
payload = {
    "schema_version": 1,
    "problem": problem_name,
    "method": method,
    "trial": trial,
    "seed": seed,
    "config": config,
    "metadata": metadata,
    "X": result.X.tolist() if result else [],
    "Y": result.Y.tolist() if result else [],
    "components": result.components.tolist() if result and result.components is not None else None,
    "weights": result.weights.tolist() if result and result.weights is not None else None,
    "run_ids": result.run_ids.tolist() if result and result.run_ids is not None else None,
    "hypervolume": hypervolume_values,
    "wall_seconds": result.wall_seconds if result else [],
    "timing": timing,
    "failed": failed,
}
```

- [ ] **Step 5: Add selectors and disk-backed plotting**

```python
p.add_argument("--trial", type=int, help="run one zero-based trial")
p.add_argument("--method", choices=tuple(METHOD_LABELS), help="run one method")
p.add_argument("--results-dir", type=Path, default=Path("results"))
p.add_argument("--summary-only", action="store_true")
```

`load_traces` reads successful artifacts and stacks equal-length HV arrays. Workers selected by `--trial` or `--method` write artifacts without plotting. Full sequential runs and `--summary-only` generate pairwise figures from disk.

- [ ] **Step 6: Correct STCH initial markers**

In `plot_results`, use `initial` for qLogEHVI pairs and `initial * weights` for STCH pairs; retain ZDT2's three-point base initial count.

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_benchmark.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add benchmark.py tests/test_benchmark.py .gitignore
git commit -m "feat: persist resumable benchmark runs"
```

### Task 4: Documentation, Spec Cleanup, and End-to-End Verification

**Files:**
- Modify: `README.md`
- Delete: `docs/superpowers/specs/2026-07-15-benchmark-readiness-timers-design.md`
- Keep: `docs/superpowers/plans/2026-07-15-benchmark-readiness-timers.md`

**Interfaces:**
- Documents artifact schema, timer meanings, exact run unit, paired-comparison interpretation, resume, worker selectors, and summary command.

- [ ] **Step 1: Update README formulas and commands**

Document square-root component representations, corrected ZDT3 ideal, and:

```powershell
python benchmark.py --problems zdt1 --trial 0 --method composite_qlogehvi
python benchmark.py --problems zdt1 --summary-only
```

State that one run is one `(problem, trial, method)` artifact. qLogEHVI runs use one five-point initial design plus 35 BO points; STCH runs pool two independent five-initial-plus-15-BO subruns. Primary comparisons stay within acquisition families.

- [ ] **Step 2: Delete approved design spec**

Use `apply_patch` to delete `docs/superpowers/specs/2026-07-15-benchmark-readiness-timers-design.md`.

- [ ] **Step 3: Run full automated verification**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 4: Run and resume a multi-method smoke benchmark**

Run:

```powershell
python benchmark.py --problems zdt1 --trials 1 --budget 10 --raw-samples 16 --restarts 2 --results-dir smoke-results --output smoke.png
python benchmark.py --problems zdt1 --trials 1 --budget 10 --raw-samples 16 --restarts 2 --results-dir smoke-results --output smoke.png
```

Expected: first command writes four result JSON files and two PNGs; second reports four skips and regenerates plots from disk.

- [ ] **Step 5: Validate artifacts**

Run:

```powershell
python -c "import json,glob,math; ps=glob.glob('smoke-results/zdt1/*/trial0.json'); assert len(ps)==4; ds=[json.load(open(p)) for p in ps]; assert all(len(d['X'])==10 and d['failed'] is None for d in ds); assert all(all(math.isfinite(v) and v>=0 for v in d['timing'].values()) for d in ds); print('validated',len(ds))"
```

Expected: `validated 4`.

- [ ] **Step 6: Remove smoke artifacts, inspect diff, and commit**

Delete only verified `smoke-results/` and generated `smoke_*.png`, then run `git diff --check` and `git status --short`.

```bash
git add README.md docs/superpowers
git commit -m "docs: explain benchmark measurements"
```

- [ ] **Step 7: Push branch**

```bash
git push -u origin codex/benchmark-ready-timers
```
