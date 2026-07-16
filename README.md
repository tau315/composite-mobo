# Composite-Function Multi-Objective Bayesian Optimization

This project tests whether exploiting a known composite objective structure can
improve sample efficiency in low-dimensional, multi-objective Bayesian
optimization (MOBO).

The central comparison is between learning final objectives directly,

$$
\mathbf{x}\longrightarrow \bigl(f_1(\mathbf{x}),f_2(\mathbf{x})\bigr),
$$

and learning an observable inner response before applying a known outer map,

$$
\mathbf{x}\longrightarrow h(\mathbf{x})
\longrightarrow
\Phi\bigl(h(\mathbf{x}),\mathbf{x}\bigr)
=\bigl(f_1(\mathbf{x}),f_2(\mathbf{x})\bigr).
$$

The benchmark measures Pareto-front recovery using dominated hypervolume versus
the total number of expensive function evaluations.

## Methods

Four solvers are implemented in `solvers.py`.

Artifacts and worker commands use these stable method keys:

| Method key | Plot label |
| --- | --- |
| `standard_qlogehvi` | Standard qLogEHVI |
| `composite_qlogehvi` | Composite qLogEHVI |
| `objective_gp_stch` | Objective-GP STCH |
| `composite_stch` | Composite STCH |

### Standard qLogEHVI

`standard_mobo` fits one independent Gaussian process to each final objective.
It selects new points with BoTorch's numerically stable
`qLogExpectedHypervolumeImprovement` acquisition function.

### Composite qLogEHVI

`composite_mobo` fits Gaussian processes to observable intermediate responses.
Monte Carlo component-posterior samples are passed through the known objective
composition, and qLogEHVI is evaluated on the resulting non-Gaussian objective
samples.

Known quantities such as a candidate coordinate `x1` remain exact and are not
given artificial GP uncertainty.

### Objective-GP smooth Tchebycheff BO

`chebyshev_bo` fits the final objectives directly. For each preference weight,
posterior objective samples are transformed using

$$
S_{\tau,\mathbf{w}}(\mathbf{f})
=\tau\log\sum_i
\exp\left(\frac{w_i(f_i-z_i^\star)}{\tau}\right).
$$

Because the benchmarks are minimization problems, the acquisition utility is
`-S`. A separate qLogEI run is performed for each weight.

### Composite smooth Tchebycheff BO

`composite_chebyshev_bo` models the intermediate responses and propagates their
posterior samples through both the known objective map and smooth Tchebycheff
scalarization:

$$
\mathbf{x}\rightarrow h(\mathbf{x})
\rightarrow\mathbf{f}(\mathbf{x})
\rightarrow S_{\tau,\mathbf{w}}(\mathbf{f}(\mathbf{x})).
$$

This is the fully nested composite method.

## Benchmarks

`benchmark.py` provides four deterministic, unconstrained benchmark problems.
All inputs are bounded to `[0, 1]^d`; the default dimension is six.

### ZDT1, ZDT2, and ZDT3

The ZDT problems share

$$
g(\mathbf{x})=1+\frac{9}{d-1}\sum_{j=2}^{d}x_j,
\qquad f_1(\mathbf{x})=x_1.
$$

The direct methods fit GPs to `f1` and `f2`. The composite methods fit one GP
to the radius-like component

$$
r(\mathbf{x})=\sqrt{g(\mathbf{x})-1}
$$

and reconstruct $g=1+r^2$. Squaring keeps posterior samples in the valid
$g\geq1$ domain. This is not the same prior as placing a GP directly on $g$.

- ZDT1 has a continuous convex Pareto front.
- ZDT2 has a continuous non-convex Pareto front.
- ZDT3 has a disconnected Pareto front and uses the componentwise ideal point
  $(0,-0.7733690123)$ for STCH.

### DTLZ2

For two objectives and six inputs,

$$
g(\mathbf{x})=\sum_{j=2}^{6}(x_j-0.5)^2,
$$

$$
f_1=(1+g)\cos(\pi x_1/2),
\qquad
f_2=(1+g)\sin(\pi x_1/2).
$$

The direct methods fit GPs to `f1` and `f2`. The composite methods fit one GP
to $r=\sqrt{g}$, reconstruct $g=r^2$, and use the exact candidate coordinate
`x1` in the known outer map. The Pareto front is the positive quadrant of the
unit circle.

## Experimental protocol

The default experiment uses:

- 20 independent trials
- Trial seeds `0, 1, ..., 19`
- Matched seeds for corresponding direct and composite methods
- Scrambled Sobol initial designs
- Two scalarization weights: `(0.05, 0.95)` and `(0.95, 0.05)`
- Smooth-Tchebycheff temperature `0.05`
- Ideal point `(0, 0)`, except ZDT3's `(0, -0.7733690123)`
- 128 raw acquisition samples and 8 optimization restarts

Every problem uses five initial points per solver run. The total accounting is:

| Problems | qLogEHVI artifact | STCH artifact |
| --- | --- | --- |
| ZDT1, ZDT3, DTLZ2 | 5 initial + 35 BO = 40 | 2 independent weights x (5 initial + 15 BO) = 40 |
| ZDT2 | 5 initial + 25 BO = 30 | 2 independent weights x (5 initial + 10 BO) = 30 |

An expensive direct-objective or component evaluation counts once; applying a
known composition does not add an evaluation. STCH artifacts interleave the two
weight runs by local evaluation index, so their pooled initial-design boundary
is 10 evaluations. Corresponding direct and composite methods use matched
seeds, initial designs, weights, and per-weight seeds.

For every problem, all methods and trials use the same fixed hypervolume
reference point:

- ZDT: `(1.1, 11.0)`
- DTLZ2: `(2.5, 2.5)`

Larger dominated hypervolume is better.

## Measurements and plots

The benchmark computes cumulative dominated hypervolume after every function
evaluation. A trial contributes to a comparison only when both methods have a
valid artifact for the same trial and their full configurations match. Curves
are the mean across those paired trials; shading is mean plus or minus one
standard error.

The two controlled comparisons hold the acquisition family fixed:

- `standard_qlogehvi` versus `composite_qlogehvi`
- `objective_gp_stch` versus `composite_stch`

Comparisons across qLogEHVI and STCH are descriptive only: the acquisition
rules and STCH's pooled independent weight runs differ. The standard-error
bands are also descriptive, not formal significance tests.

For example, ZDT3 produces:

```text
hypervolume_vs_evaluations_zdt3_qlogehvi.png
hypervolume_vs_evaluations_zdt3_stch.png
```

## Installation

The code requires Python and the following packages:

```text
torch
botorch
gpytorch
numpy
matplotlib
pymoo
```

Install them in a virtual environment, for example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch botorch gpytorch numpy matplotlib pymoo
```

## Running experiments

Run every benchmark:

```powershell
python benchmark.py
```

Run only DTLZ2:

```powershell
python benchmark.py --problems dtlz2
```

Run only ZDT3:

```powershell
python benchmark.py --problems zdt3
```

Run multiple selected problems:

```powershell
python benchmark.py --problems zdt1 zdt3 dtlz2
```

Run one independently schedulable worker job:

```powershell
python benchmark.py --problems zdt1 --trial 0 --method composite_qlogehvi
```

One job is one `(problem, trial, method)` run and writes one artifact to
`results/{problem}/{method}/trial{trial}.json`. A worker selected with `--trial`
or `--method` does not plot. Re-running a command resumes automatically: a
complete artifact with the exact requested configuration is reported as
`resumed` and skipped.

Failures do not stop sibling jobs. The failed artifact records a traceback in
`failed`; because failed artifacts are not resume-valid, re-running the same
command retries them. Writes are atomic.

Regenerate summaries and pairwise plots from valid artifacts without running
optimization:

```powershell
python benchmark.py --problems zdt1 --summary-only
```

Useful flags include:

```text
--trials          Number of independent trials
--trial           One zero-based trial worker
--method          One stable method-key worker
--budget          Total evaluations for non-ZDT2 problems (default: 40)
--initial         Initial points per solver run (fixed at 5)
--weights         Number of scalarization weights
--temperature     Smooth-Tchebycheff temperature
--seed            Base random seed
--raw-samples     Raw acquisition-optimization samples
--restarts        Acquisition-optimization restarts
--results-dir     Artifact root directory
--summary-only    Read artifacts and plot without running jobs
--output          Base output filename used to construct plot names
```

ZDT2's 30-evaluation budget is fixed in the benchmark code rather than
controlled by `--budget`; its initial count remains the global fixed value of
five.

### Artifact fields

Each JSON artifact records the schema version, problem, stable method key,
trial, seed, full configuration, Python/package/git metadata, evaluated `X`
and `Y`, optional composite components, optional STCH weights and run IDs,
cumulative `hypervolume`, per-observation `wall_seconds`, aggregate `timing`,
and `failed` state.

For initial points, `wall_seconds` repeats the initial batch time divided by
the number of points. Each BO entry is that iteration's wall time. STCH values
are interleaved across weight runs in the same order as `X` and `Y`.

| Timing key | Meaning |
| --- | --- |
| `initial_design_seconds` | Scrambled Sobol initial-design generation |
| `initial_evaluate_seconds` | Initial direct-objective or component oracle calls |
| `initial_compose_seconds` | Known composite map on initial components; zero for direct methods |
| `gp_fit_seconds` | All GP fits during BO |
| `acquisition_build_seconds` | Acquisition and supporting-object construction |
| `acquisition_optimize_seconds` | Acquisition optimization |
| `bo_evaluate_seconds` | Sequential direct-objective or component oracle calls |
| `bo_compose_seconds` | Known composite map after BO evaluations; zero for direct methods |
| `solver_total_seconds` | End-to-end solver time, including solver overhead |
| `hypervolume_seconds` | Post-solver cumulative-hypervolume calculation |
| `total_seconds` | Runner time through solver, hypervolume, and evaluation-count check, excluding artifact serialization |

Phase values are accumulated across the whole artifact, including both STCH
weight runs. They are subsets of the total timers rather than values to add to
`total_seconds`.

## Numerical safeguards

The implementation uses double precision and qLogEI/qLogEHVI for stable
acquisition calculations. If SciPy encounters a non-finite acquisition
gradient, the solver evaluates the acquisition on a fresh Sobol candidate set
and selects its best finite candidate.

On Windows without the MSVC compiler, the code skips BoTorch's optional fused
C++ qLogEHVI extension and uses the equivalent pure-Python implementation.

## Repository layout

```text
solvers.py             Four BO solvers and shared GP/acquisition utilities
benchmark.py           Benchmark definitions, trials, hypervolume, and plots
README.md              Project documentation
.gitignore             Generated and local files excluded from Git
```

## Current limitations

- Only low-dimensional, two-objective synthetic tests are currently included.
- Intermediate outputs are assumed observable at no additional evaluation cost.
- Output GPs are independent and do not model cross-output correlations.
- Composite component transforms induce different priors than modeling `g`
  directly.
- Two scalarization weights mostly target the ends of the Pareto front.
