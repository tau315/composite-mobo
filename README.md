# Composite Multi-Objective Bayesian Optimization

This project tests whether exploiting known composite structure improves
Pareto-front recovery in multi-objective Bayesian optimization (MOBO). Every
benchmark compares a direct solver that models final objectives against a
composite counterpart that models objective-specific intermediate functions.

The implemented structure is

$$
f_i(x)=g_i\left(h_{i1}(x),\ldots,h_{ik_i}(x)\right).
$$

Each intermediate column has its own independent GP. Different objectives may
use different numbers of intermediates. They may also use the same underlying
physical quantity, but that quantity is repeated in the component matrix so it
is still modeled independently for each objective.

All solvers minimize objectives on the normalized input cube
$[0,1]^d$. The benchmark files perform any required conversion to physical
units.

## Files

Every benchmark is a standalone script:

| Script | Benchmark | Objectives | Input dimension | Suite |
|---|---|---:|---:|---|
| `benchmark_dtlz2.py` | DTLZ2 | 2 | 6 | Low-dimensional |
| `benchmark_snar.py` | Summit SNAr reaction | 2 | 4 | Low-dimensional scientific |
| `benchmark_nanoparticle_rgb.py` | RGB-selective multilayer nanoparticle | 3 | 6 | Low-dimensional scientific |
| `benchmark_penicillin.py` | Penicillin fed-batch fermentation | 3 | 7 | Low-dimensional scientific |
| `benchmark_dtlz2_100d.py` | DTLZ2 | 2 | 100 | High-dimensional |
| `benchmark_dtlz2_600d.py` | DTLZ2 | 2 | 600 | High-dimensional |
| `benchmark_cort_tg119.py` | CORT TG119 radiotherapy | 3 | 418 | High-dimensional scientific |
| `benchmark_rcm40.py` | RCM40 optimal power flow | 2 | 34 | High-dimensional scientific |
| `benchmark_rcm46.py` | RCM46 optimal power flow | 4 | 34 | High-dimensional scientific |

Shared experiment, hypervolume, and plotting code is in
`benchmark_common.py`. All BO algorithms remain in `solvers.py`. The `morbo/`
directory contains supporting code for the vendored batched MORBO
implementation; the default benchmark runner uses the sequential coordinated
MORBO implementation exported by `solvers.py`.

## Installation

The required Python packages are:

```powershell
python -m pip install numpy scipy matplotlib torch gpytorch botorch
```

The code runs on CPU by default. A CUDA-enabled PyTorch installation can be
used for custom extensions, but the benchmark scripts do not require a GPU.

## Running the benchmarks

Run any benchmark directly:

```powershell
python benchmark_dtlz2.py
python benchmark_snar.py
python benchmark_nanoparticle_rgb.py
python benchmark_dtlz2_100d.py
python benchmark_dtlz2_600d.py
python benchmark_cort_tg119.py
```

Each full run uses 20 independent trials and writes one Matplotlib PNG. No CSV
file is generated.

The low-dimensional target budget is 50 expensive evaluations. Direct
qLogEHVI uses 5 initial plus 45 adaptive evaluations. STCH uses four
scalarization weights with ten adaptive evaluations per weight: 5 shared
initial plus $4\times10$ adaptive evaluations, or 45 total.

The high-dimensional target budget is 400 expensive evaluations. MORBO uses 5
initial plus 395 adaptive evaluations. Spherical STCH uses ten scalarization
weights with the largest equal allocation inside the target: 5 initial plus
$10\times39$ adaptive evaluations, or 395 total.

These are intentionally substantial experiments, especially with 20 trials.
Use `--quick` first to verify an installation:

```powershell
python benchmark_snar.py --quick
python benchmark_cort_tg119.py --quick
```

Useful overrides include:

```powershell
python benchmark_dtlz2.py --trials 5 --evaluations 30
python benchmark_dtlz2_100d.py --trials 2 --evaluations 50
python benchmark_snar.py --output results/snar.png --show
```

Available controls include `--trials`, `--evaluations`, `--initial`,
`--weights`, `--per-weight`, `--raw-samples`, `--restarts`, `--seed`,
`--output`, and `--show`. `--iterations` can explicitly override the number of
adaptive qLogEHVI or MORBO evaluations.

## Resumable artifacts

Every solver run writes one JSON artifact to
`--results-dir/<benchmark slug>/<method key>/trial<N>.json` (default
`results/`). Rerunning a benchmark revalidates each artifact and skips the runs
that are already complete, so an interrupted 20-trial experiment resumes rather
than restarting. Writes are atomic: an artifact is written to a temporary file
and renamed, so a killed process never leaves a half-written result behind.

Method keys are stable slugs of the plot labels. `--list-methods` prints the
set a benchmark will use:

| Suite | Method keys |
| --- | --- |
| Low-dimensional | `direct_qlogehvi`, `composite_qlogehvi`, `objective_gp_stch`, `composite_stch` |
| High-dimensional | `spherical_objective_stch`, `spherical_composite_stch`, `morbo`, `composite_morbo` |

Resume is deliberately strict. An artifact is only reused when it re-derives
exactly: schema version, benchmark slug and path identity, the full numeric
configuration, Python/package/git provenance, and freshly recomputed
components, objectives, scalarization weights, run IDs, and cumulative
hypervolume. Anything that fails is treated as absent and simply rerun, so a
stale or partially corrupt result can never contaminate a figure.

A failing run is recorded, not swallowed: its artifact stores the traceback,
its siblings still run, and the process exits non-zero at the end with a count
of the failures.

`--trial` and `--method` restrict a process to a single run, which is what lets
one experiment fan out across a cluster array. A worker writes its artifact and
stops without plotting; `--summary-only` later plots from whatever is on disk
without running any solver.

```powershell
python benchmark_dtlz2.py --list-methods
python benchmark_dtlz2.py --trial 0 --method composite_stch --results-dir results
python benchmark_dtlz2.py --summary-only --results-dir results
```

Artifacts also carry per-run timing: `solvers.TIMING_KEYS` phase totals
(initial design, expensive evaluations, composition, GP fitting, acquisition
construction and optimization), a wall-clock figure per expensive evaluation,
and a count of acquisition-optimizer fallbacks. `run_unicorn.sh` submits the
whole experiment as a Slurm array of single-run workers plus an aggregation
step that revalidates every artifact and writes a timing/fallback summary.

## Plots and evaluation protocol

Every graph shows dominated hypervolume versus total expensive function
evaluations:

- purple identifies qLogEHVI in low dimensions and MORBO in high dimensions;
- green identifies smooth Tchebycheff methods;
- solid lines are direct/objective-modeling solvers;
- dotted lines are composite/component-modeling solvers;
- shaded regions are one standard error over independent trials;
- the vertical dashed line marks the end of the shared initial design;
- the horizontal dash-dot line is the exact maximum hypervolume when known,
  or the ideal/reference-box ceiling otherwise.

Every method within one benchmark uses the same fixed reference point and the
same initial Sobol design for a given trial seed. Trial seeds differ, so the 20
trials have independent initial designs.

A trial only enters a curve when both members of its comparison pair produced
valid artifacts that agree on configuration, provenance, scalarization weights,
and initial design. A pair that cannot satisfy that is dropped from the figure
rather than plotted against a mismatched partner.

Hypervolume is computed for minimization objectives. Objective vectors are
internally negated when passed to BoTorch, which uses maximization.

## Low-dimensional solvers

### Direct qLogEHVI

`standard_mobo` fits one exact GP to each final objective and selects the next
point with qLogEHVI.

### Composite qLogEHVI

`composite_mobo` fits one exact GP to each intermediate function. Posterior
component samples are passed through the known outer maps before qLogEHVI is
calculated.

### Objective-GP STCH

`chebyshev_bo` fits final-objective GPs. For weight vector $w$ and ideal point
$z^\star$, it minimizes the smooth Tchebycheff scalarization

$$
S_{\tau,w}(f)
=
\tau\log\left[
\sum_i
\exp\left(
\frac{w_i(f_i-z_i^\star)}{\tau}
\right)
\right].
$$

The acquisition function is qLogEI applied to the negative scalarization.

### Composite STCH

`composite_chebyshev_bo` models intermediate functions and evaluates

$$
-S_{\tau,w}\left(
g_1(h_1),\ldots,g_m(h_m)
\right)
$$

inside qLogEI.

## High-dimensional solvers

### Spherical objective STCH

`spherical_chebyshev_bo` maps normalized inputs through inverse stereographic
projection and fits spherical-linear GPs to final objectives before applying
STCH.

### Spherical composite STCH

`composite_spherical_chebyshev_bo` uses the same spherical-linear model but
fits the intermediate functions and applies the known outer maps before
scalarization.

### MORBO

`morbo` uses multiple coordinated local trust regions, local ARD
Matérn-5/2 GPs, Thompson-sampled candidate sets, hypervolume-improvement
coordination, trust-region expansion/contraction, and restarts.

### Composite MORBO

`composite_morbo` uses the same trust-region logic but fits local GPs to the
intermediates. Thompson samples are composed into objective values before
hypervolume improvement is calculated.

## Benchmark definitions

### DTLZ2

For two objectives and $d$ inputs,

$$
g(x)=\sum_{j=2}^{d}(x_j-0.5)^2,
$$

$$
f_1(x)=(1+g(x))\cos\left(\frac{\pi x_1}{2}\right),
\qquad
f_2(x)=(1+g(x))\sin\left(\frac{\pi x_1}{2}\right).
$$

The component matrix repeats $g$ so the two objective-specific groups are

$$
h_1(x)=\left(g(x),\cos(\pi x_1/2)\right),
\qquad
h_2(x)=\left(g(x),\sin(\pi x_1/2)\right).
$$

The Pareto front satisfies $f_1^2+f_2^2=1$ in the positive quadrant.

The 6D reference point is $(2.5,2.5)$ and its exact maximum hypervolume is
$5.464602$. Canonical DTLZ2 values away from the front grow with dimension, so
the high-dimensional reference coordinates dominate the full input domains:

| Dimension | Reference coordinate | Exact maximum HV |
|---:|---:|---:|
| 100 | 25.85 | 667.437102 |
| 600 | 150.85 | 22754.937102 |

### Summit SNAr reaction

The four physical variables are residence time, pyrrolidine equivalents, inlet
concentration, and temperature. A vectorized RK4 integrator evaluates the
published five-species plug-flow kinetic model used by
[Summit](https://gosummit.readthedocs.io/en/latest/_modules/summit/benchmarks/snar.html).

The objectives are maximizing space-time yield and minimizing E-factor. They
are converted to normalized minimization objectives:

$$
f_1=1-\frac{\operatorname{STY}}{13000},
\qquad
f_2=\frac{E}{500}.
$$

The STY group models product outlet concentration and total flow. The E-factor
group independently models all five outlet concentrations and total flow. The
known mass-balance equations compose these quantities into the two objectives.

The fixed reference point is $(2.5,2.5)$ and the displayed ideal-box ceiling
is $6.25$.

### RGB-selective multilayer nanoparticle

The input contains six layer thicknesses in $[30,70]$ nm. The script implements
the 201-wavelength Mie-scattering simulator from the
[DeepBO nanoparticle study](https://arxiv.org/abs/2104.11667).

The three target bands are blue $[400,500)$ nm, green $[500,600)$ nm, and red
$[600,700)$ nm. For each band $c$,

$$
I_c(x)=\sum_{\lambda\in c}\sigma(\lambda;x),
\qquad
O_c(x)=\sum_{\lambda\notin c}\sigma(\lambda;x),
$$

$$
f_c(x)=\frac{O_c(x)}{I_c(x)+O_c(x)}.
$$

Each objective therefore has two independently modeled components,
$h_c=(I_c,O_c)$. Minimizing $f_c$ is equivalent to maximizing the corresponding
in-band/out-of-band selectivity ratio.

The fixed reference point is $(2.5,2.5,2.5)$ and the displayed ideal-box
ceiling is $15.625$.

### CORT TG119 radiotherapy

The public [CORT dataset](https://gigadb.org/dataset/100110) supplies sparse
dose-influence matrices for five beam angles, 418 beamlet controls, 7,429
target voxels, 1,280 core/OAR voxels, and 599,440 body voxels. The script
downloads the approximately 25 MB TG119 archive on first use into
the user cache (`%LOCALAPPDATA%\composite_mobo\cort_tg119` on Windows).

To use an existing extracted copy instead:

```powershell
$env:CORT_TG119_DIR = "C:\path\to\TG119"
python benchmark_cort_tg119.py
```

For normalized beamlet controls $x\in[0,1]^{418}$, physical fluence is $70x$
and voxel dose is

$$
d(x)=D(70x).
$$

The three objective-specific component groups are

$$
h_T=(D_{95}^{T},D_2^{T}),\qquad
h_C=(\overline d_C,D_2^C),\qquad
h_N=(\overline d_N,D_2^N).
$$

They compose target coverage/hotspot penalty, core exposure, and normal-tissue
exposure:

$$
\tilde f_T=
[1-D_{95}^{T}]_+^2
+\frac14[D_2^{T}-1.05]_+^2,
$$

$$
\tilde f_C=\frac12\overline d_C+\frac12D_2^C,
\qquad
\tilde f_N=\frac12\overline d_N+\frac12D_2^N.
$$

Each value is divided by a fixed domain upper scale computed from the
all-maximum-fluence plan. This keeps all three objectives on comparable scales
without changing Pareto dominance.

The fixed reference point is $(2.5,2.5,2.5)$ and the displayed ideal-box
ceiling is $15.625$.

## Solver interface

A direct benchmark supplies:

```python
evaluate(X) -> Y
```

A composite benchmark additionally supplies:

```python
evaluate_components(X) -> H
compose(H) -> Y
```

The runner validates that

```python
compose(evaluate_components(X)) == evaluate(X)
```

on a Sobol probe before starting any trials.

## Main exports from `solvers.py`

```text
standard_mobo
composite_mobo
chebyshev_bo
composite_chebyshev_bo
spherical_chebyshev_bo
composite_spherical_chebyshev_bo
morbo
composite_morbo
simplex_weights
smooth_tchebycheff
ObjectivewiseComposer
MORBOConfig
SolverResult
```

## Algorithm sources

- Doumont et al., *We Still Don't Understand High-Dimensional Bayesian
  Optimization*, AISTATS 2026:
  <https://github.com/colmont/linear-bo>
- Daulton et al., *Multi-Objective Bayesian Optimization over
  High-Dimensional Search Spaces*, UAI 2022:
  <https://github.com/facebookresearch/morbo>
