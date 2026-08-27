# Composite multi-objective Bayesian optimization benchmarks

This repository tests one question: does modeling known intermediate responses
and composing them into objectives improve dominated hypervolume at a fixed
number of expensive design evaluations?

## Canonical paper suite

These are the five benchmark entry points. Other `benchmark_*.py` files are
legacy implementations and are not part of the paper protocol.

| Script | Kind | d | Objectives | Default evaluations |
|---|---|---:|---:|---:|
| `benchmark_dtlz2_6d.py` | low-D synthetic | 6 | 2 | 45 |
| `benchmark_nanoparticle_discrete.py` | low-D scientific | 6 | 3 | 45 |
| `benchmark_rcm40_balanced.py` | moderate-D scientific | 34 | 3 | 45 |
| `benchmark_langermann_ackley_600d.py` | high-D synthetic | 600 | 2 | 120 |
| `benchmark_cort_tg119_composition.py` | high-D scientific | 418 | 3 | 120 |

Low-dimensional scripts compare objective-GP versus composite versions of
qLogEHVI (labeled Hypervolume) and smooth Tchebycheff BO. High-dimensional
scripts compare objective versus composite spherical-linear Tchebycheff BO and
objective versus composite MORBO. MORBO uses the vendored joint-batch engine in
`morbo/`; the separate handwritten sequential implementation in `solvers.py`
is not used by the canonical runner.

Every paired method receives the same seeded initial design and exactly the
same number of design evaluations. A component vector and the objectives
composed from it count as one evaluation; composite solvers never call the
simulator again to reconstruct the objective.

Scientific-oracle evaluation caches are cleared before every method so a
composite run cannot reuse simulator values from its direct counterpart.
CORT's one-time matrix loading and fixed normalization setup occur before
solver timing.

## Running

The code requires Python with PyTorch, BoTorch, GPyTorch, NumPy, SciPy, and
Matplotlib. Run the complete paper suite from the repository root with:

```powershell
python benchmark_dtlz2_6d.py
python benchmark_nanoparticle_discrete.py
python benchmark_rcm40_balanced.py
python benchmark_langermann_ackley_600d.py
python benchmark_cort_tg119_composition.py
```

Each command defaults to 10 trials and the paper budget in the table. The two
high-dimensional commands are intentionally expensive. Use `--quick` for a
five-evaluation installation smoke test, for example:

```powershell
python benchmark_dtlz2_6d.py --quick
python benchmark_nanoparticle_discrete.py --quick
python benchmark_rcm40_balanced.py --quick
python benchmark_langermann_ackley_600d.py --quick
python benchmark_cort_tg119_composition.py --quick
```

Use `--help` to override trials, budget, initial design size, scalarization
weights, MORBO batch size, seed, or output directory. Budget overrides must
satisfy the divisibility checks printed by the runner. For example, this runs
one complete 45-evaluation DTLZ2 trial in a separate directory:

```powershell
python benchmark_dtlz2_6d.py --trials 1 --output-dir benchmark_results/dtlz2_test
```

Each run creates two files under `benchmark_results/<problem>/`:

- `hypervolume_vs_evaluations.png`: mean dominated hypervolume with standard
  error bands across trials.
- `benchmark_data.npz`: method names, evaluation indices, every trial's full
  hypervolume trace, every evaluated design and objective vector, composite
  component observations, scalarization weights/run IDs, runtimes, seeds,
  ideal and reference points, and JSON-encoded protocol metadata.

The `.npz` hypervolume array has shape
`(method, trial, evaluation)`; the objective array has shape
`(method, trial, evaluation, objective)`; and the design array has shape
`(method, trial, evaluation, dimension)`. Component and scalarization arrays
use method-labeled keys such as `components__composite_hypervolume`.

## Scientific formulations

- **Nanoparticle:** multilayer Mie scattering at exactly 450, 550, and 650 nm.
  The three nonlinear objectives are unwanted-scattering fractions. There are
  no wavelength-band sums or integrals in this version. The target and mean
  off-target scattering are repeated in three objective-specific component
  pairs, for six independently modeled component outputs.
- **Balanced RCM40:** IEEE-14 admittance-matrix physics with the original 34
  voltage and generator variables. The official problem's 26 power-balance
  equalities are represented by a mean-squared residual objective so every
  variable matters. This is an explicit bound-constrained reformulation of
  RCM40, not the untouched constrained CEC problem.
- **CORT TG-119:** 418 beamlet fluences act through public precomputed sparse
  dose-influence matrices. The six observed dose statistics are composed into
  a target hinge penalty and quadratic core/normal-tissue penalties. The first
  run downloads the public TG-119 archive unless `CORT_TG119_DIR` points at an
  extracted copy.

The 600-D Langermann--Ackley synthetic case uses six deterministic dense
orthonormal projections, duplicated into two independent objective-specific
groups. It therefore has 600 ambient variables and six effective directions;
the composite model fits 12 component GPs.

Penicillin and SNAr are excluded from the canonical suite because their current
oracles time-step dynamical systems. The old nanoparticle benchmark is excluded
because it sums over 201 wavelengths. The old RCM40/RCM46 scripts are excluded
because they omit the original equality constraints, leaving generator inputs
irrelevant.
