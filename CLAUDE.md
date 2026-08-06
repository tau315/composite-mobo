# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is for

This is a research codebase for a paper, not a library. It exists to answer one
question: **does exploiting known composite structure `f_i(x) = g_i(h_i(x))`
improve multi-objective Bayesian optimization?** Target venue is the AI4Mat
workshop at NeurIPS 2026; the outline is in
`paper/Outline for MOBOCF workshop paper.md`.

Every solver comes in a **matched pair** — a direct method that GP-models the
final objectives, and a composite counterpart that GP-models the intermediate
responses and pushes posterior samples through the known `g`. The pairing is the
experiment. Work that does not end in comparable paired numbers on disk does not
advance the goal.

## Commands

There is no build step, no packaging, and no linter config. Tests import repo
modules by running pytest from the repo root.

```powershell
python -m pytest -q                                     # full suite (~35 s)
python -m pytest tests/test_benchmark_common.py -q       # one file
python -m pytest tests/test_benchmark_common.py::test_method_keys_are_stable_slugs -q

python benchmark_dtlz2.py --quick                        # smoke-run any benchmark
python benchmark_dtlz2.py --list-methods                 # this benchmark's method keys
python benchmark_dtlz2.py --trial 0 --method composite_stch --results-dir results
python benchmark_dtlz2.py --summary-only --results-dir results

python diagnose_composite.py                             # screen every benchmark
python diagnose_composite.py benchmark_dtlz2             # screen selected modules
python diagnose_composite.py --demo                      # self-check
```

`--quick` shrinks a benchmark to one tiny trial and is the right first move when
touching the runner. A full default run is 20 trials and is genuinely expensive
(low-dim 50 evaluations, high-dim 400, times 4 methods).

Cluster runs go through `run_unicorn.sh` (a Slurm array of single-run workers
plus an aggregation step), submitted from the login node. To drive the cluster
from a laptop:

```bash
scripts/unicorn.sh 'sinfo'                       # any remote command
git bundle create - HEAD | scripts/unicorn.sh 'cat > ~/repo.bundle'
```

That wrapper uses SSH key authentication only and refuses password fallback, so
a missing key fails loudly instead of hanging. Host, user, and key path are
overridable via `UNICORN_HOST` / `UNICORN_USER` / `UNICORN_KEY`; the private key
is the one thing that deliberately lives outside the repo. Cornell VPN is
required — without it the host times out rather than refusing.

## Architecture

### The three layers

`solvers.py` owns every BO algorithm and knows nothing about benchmarks.
`benchmark_common.py` owns the experimental protocol — problem definition,
artifacts, resume, plotting. Each `benchmark_<name>.py` is a thin script that
builds one `BenchmarkProblem` and calls `run_benchmark(PROBLEM)`. Adding a
benchmark means adding one file; it inherits the whole protocol.

Every benchmark module must expose a module-level `PROBLEM`. `run_unicorn.sh`
and the aggregation step import it by module stem to recover the slug.

### The composite contract

```python
evaluate_components(X) -> H      # intermediate responses, observed for free
compose(H) -> Y                  # known, cheap, deterministic, differentiable
```

`compose` takes **one argument**. It used to take `(H, X)` so that exact input
coordinates could be reused inside the map; that signature is gone. Any result
file or script predating the change is incompatible — see "Orphaned results".

`compose` runs on Monte Carlo posterior samples, so it must broadcast over
leading sample dimensions. Index the last axis (`H[..., 0]`), never assume rank.

`BenchmarkProblem.validate()` and `_check_composition` enforce
`compose(evaluate_components(X)) == evaluate(X)` on a Sobol probe before any
trial runs.

### Solver families and suites

`_solver_jobs` in `benchmark_common.py` picks the method set from
`problem.suite`, and it is the single source of truth for which solvers run:

- `low` — `direct_qlogehvi`, `composite_qlogehvi`, `objective_gp_stch`, `composite_stch`
- `high` — `spherical_objective_stch`, `spherical_composite_stch`, `morbo`, `composite_morbo`

`batched_morbo` / `composite_batched_morbo` exist in `solvers.py`, backed by the
vendored `morbo/` port of Daulton et al., but **are not wired into any
benchmark**. The `ablation_results/` data was generated with them, which is why
it is not comparable to anything the current runner produces.

Method keys used on disk and by `--method` are slugs of the display labels
(`"Objective-GP STCH"` → `objective_gp_stch`). There is no hand-maintained table;
`method_key()` derives them.

### Artifacts and resume — the part that bites

Each run writes `<results-dir>/<slug>/<method key>/trial<N>.json` atomically.
Resume revalidates strictly: schema version, path identity, full numeric config,
Python/package/git provenance, and **recomputed** components, objectives,
weights, run IDs, and hypervolume. Anything failing any check is treated as
absent and simply rerun, so a stale artifact can never contaminate a figure.

Two consequences that are easy to trip over:

1. **Every solver must populate `SolverResult.timing` and one `wall_seconds`
   entry per expensive evaluation.** Validation requires
   `len(wall_seconds) == evaluations`. A solver that leaves it empty produces
   artifacts that never resume and never load, and the failure surfaces as
   "no comparable artifacts were found" rather than anything pointing at the
   solver.
2. **Evaluation counts differ by family.** Sequential methods spend
   `initial + iterations`; STCH spends `initial + weights * per_weight` because
   all weights branch from one shared initial design. `expected_evaluations()`
   is the authority.

`load_traces` only pairs trials whose direct and composite runs agree on config,
provenance, weights, and initial design — a panel whose pair cannot satisfy that
is dropped from the figure rather than plotted against a mismatched partner.

### Screening before implementing

`diagnose_composite.py` fits GPs on a Sobol design and compares held-out
objective predictions from a direct GP against a component GP pushed through
`g`, reporting **advantage** — the fraction of direct RMSE that routing through
`g` removes. It runs in seconds with no BO loop.

The established finding: composite modeling only pays off when `g` takes a
*smooth* intermediate and **creates** sharp structure. When `g` averages or
integrates it smooths, making `f` easier to model than `h`, and composite
modeling is strictly worse. In high dimensions the advantage additionally
requires the intermediate to depend on a low-dimensional active subspace.
Advantage tracks the measured BO outcomes across the existing benchmarks
(DTLZ2 6d +65% → +5.3% BO gain; penicillin 0% → −4.6%).

Screen any candidate benchmark with this before implementing it. Below roughly
20% advantage, do not expect a publishable effect.

Known hazard: a ratio-shaped `g` has a pole, and a component GP posterior sample
can cross it. WeldedBeam's `g = 2.1952/K` diverges this way. Clamp denominators
in any `compose` that divides.

## Orphaned results — do not cite as current evidence

- `results/production-31f21c3/` — ZDT + DTLZ2, written by a deleted `benchmark.py`
  under an older schema and the two-argument `compose`. Its reader no longer exists.
- `ablation_results/` — 16 problems, real 20-trial paired statistics, but the
  generating script is not in the repo, it used `batched_morbo`, several of its
  problems have no benchmark script, and the spherical-STCH runs are incomplete
  (1–16 trials, which is why they have no `summary.json`).

Neither was produced by the current pipeline. Re-run under the present protocol
before quoting numbers from either.

## Conventions

Flat module layout at the repo root; benchmarks are scripts, not a package.
Match the surrounding style — module docstrings explain *why*, comments explain
non-obvious modeling choices rather than restating code. Solvers minimize
objectives and negate internally for BoTorch's maximization convention; keep
that boundary inside `solvers.py`.

`morbo/llm_candidates.py` imports the optional `anthropic` package and is
imported lazily from `morbo/gen.py` for that reason. Keep it lazy.

Delegate literature search and other token-heavy exploration to the Codex CLI —
see `codex-use.md` for the invocation patterns.
