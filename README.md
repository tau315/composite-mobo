# Composite Multi-Objective Bayesian Optimization Solvers

This repository contains direct and composite multi-objective BO solvers plus
six reproducible benchmark scripts. Each benchmark is defined in its own file,
while `benchmark_common.py` keeps the trial, hypervolume, and plotting protocol
identical across problems.

## Problem interface

All objectives are minimized on `[0, 1]^d`.

Direct solvers require:

```python
evaluate(X) -> Y                 # n x m objectives
```

Composite solvers require:

```python
evaluate_components(X) -> H      # n x p intermediate values
compose(H) -> Y                  # n x m objectives
```

with `compose(evaluate_components(X)) == evaluate(X)`.

The intended structure is objective-specific:

$$
f_i(x)=g_i\left(h_{i1}(x),\ldots,h_{ik_i}(x)\right),
$$

Each objective may use a different number of intermediate functions. Component
groups may also overlap; every component column is modeled by its own independent
GP, and each known outer function receives the columns assigned to that objective.

Example:

```python
composer = ObjectivewiseComposer(
    component_groups=[[0, 1], [2, 3, 4]],
    objective_maps=[
        lambda H1: (torch.cos(H1[..., 0]) + 1) * torch.sin(H1[..., 1]),
        lambda H2: H2[..., 0]**2 + H2[..., 1] * H2[..., 2],
    ],
)
```

This represents

$$
f_1=g_1(h_{11},h_{12}),\qquad
f_2=g_2(h_{21},h_{22},h_{23}).
$$

## Benchmark suite

| Script | Problem | Components per objective | Solver comparison |
|---|---|---|---|
| `benchmark_dtlz2.py` | DTLZ2, 2 objectives, 6D | 2 + 2 | qLogEHVI and STCH |
| `benchmark_ackley_griewank_6d.py` | Ackley--Griewank, 2 objectives, 6D | 2 + 2 | qLogEHVI and STCH |
| `benchmark_five_ackley_6d.py` | Five shifted Ackley objectives, 6D | 2 each | qLogEHVI and STCH |
| `benchmark_langermann_ackley_6d.py` | Langermann--Ackley, 2 objectives, 6D | 3 + 2 | qLogEHVI and STCH |
| `benchmark_ackley_griewank_50d.py` | Ackley--Griewank, 2 objectives, 50D | 2 + 2 | spherical STCH and MORBO |
| `benchmark_projected_langermann_500d.py` | Projected Langermann, 2 objectives, 500D | 4 + 5 | spherical STCH and MORBO |

Running a script with no flags performs 20 independent trials and saves a
two-panel Matplotlib PNG. Each panel contains only one direct/composite pair,
with the trial mean as a solid line and its standard error as a shaded band.
No CSV file is produced.

```powershell
python benchmark_dtlz2.py
python benchmark_ackley_griewank_6d.py
python benchmark_five_ackley_6d.py
python benchmark_langermann_ackley_6d.py
python benchmark_ackley_griewank_50d.py
python benchmark_projected_langermann_500d.py
```

Use `--show` to open the plot after saving it. Use `--quick` only to verify an
installation with a tiny one-trial run. Other useful overrides include
`--trials`, `--iterations`, `--weights`, `--per-weight`, `--raw-samples`, and
`--output`.

### Fixed hypervolume reference points

Every objective uses the fixed reference coordinate `2.5`. Thus the reference
is `(2.5, 2.5)` for two-objective problems and `(2.5, ..., 2.5)` for the
five-objective problem. The same tensor is passed to the acquisition functions
and the plotted dominated-hypervolume metric.

DTLZ2 has the exact maximum

$$
2.5^2-\frac{\pi}{4}=5.464602.
$$

For the normalized custom objectives, the ideal/reference box ceiling is
`6.25` with two objectives and `97.65625` with five objectives. The custom
ideal vectors are not jointly attainable, so these are rigorous upper bounds
rather than exact attainable front hypervolumes. Every graph displays the
applicable exact maximum or ceiling as a dotted horizontal line. Its two panels
share one y-axis scale.

## Standard protocol

Public solvers default to:

- 5 scrambled-Sobol initial evaluations;
- 40 adaptive evaluations after initialization;
- double-precision tensors;
- sequential acquisition (`q=1`).

Scalarization studies should create eight weights:

```python
weights = simplex_weights(8, number_of_objectives, seed=seed)
```

Both low- and high-dimensional STCH solvers evaluate one shared five-point
initial design, then branch into eight scalarizations. Each weight adds five
adaptive points, for exactly

$$
5 + 8\times 5 = 45
$$

unique expensive evaluations. Scalarization branches share the initial data but
do not use points acquired by other weights when fitting their own GP.

## Low-dimensional solvers

### `standard_mobo`

Fits an independent exact GP to every final objective and selects points with
qLogEHVI.

### `composite_mobo`

Fits an independent exact GP to every intermediate function. Posterior
component samples are passed through `compose`, and qLogEHVI is evaluated on
the resulting objective samples.

### `chebyshev_bo`

Fits objective GPs and applies smooth Tchebycheff scalarization inside qLogEI:

$$
S_{\tau,w}(f)=\tau\log\sum_i
\exp\left(\frac{w_i(f_i-z_i^\star)}{\tau}\right).
$$

### `composite_chebyshev_bo`

Fits component GPs and evaluates

$$
-S_{\tau,w}\left(g_1(h_1),\ldots,g_m(h_m)\right)
$$

inside qLogEI.

## High-dimensional spherical-linear solvers

These use the model from Doumont et al., *We Still Don't Understand
High-Dimensional Bayesian Optimization*.

Inputs are centered, divided by ARD scales and a decoupled global scale, then
mapped by inverse stereographic projection:

$$
P(z)=\frac{[2z,\|z\|^2-1]}{1+\|z\|^2}.
$$

The kernel is

$$
k(x,x')=b_0+b_1P(z)^\top P(z'),\qquad b_0+b_1=1,
$$

with simplex coefficients, the paper's DSP-like unscaled ARD prior, a constant
mean, and its bounded-noise log-normal likelihood.

### `spherical_chebyshev_bo`

Fits spherical-linear GPs to final objectives, then applies STCH and qLogEI.

### `composite_spherical_chebyshev_bo`

Fits spherical-linear GPs to objective-specific intermediate functions, then
applies the known outer maps, STCH, and qLogEI.

## High-dimensional MORBO solvers

### `morbo`

Implements the coordinated MORBO core:

- five local trust regions by default;
- local independent ARD Matérn-5/2 GPs;
- initial trust-region length `0.8`, minimum `0.01`, maximum `1.6`;
- coordinate perturbation probability `min(20 / d, 1)`;
- local joint Thompson samples over discrete Sobol candidates;
- greedy coordination by sampled hypervolume improvement;
- failure tolerance `max(d / 3, 10)`;
- trust-region contraction and restart.

### `composite_morbo`

Uses the same coordinated trust-region procedure, but local GPs model the
intermediate functions. Thompson samples are passed through the objective-wise
composition before hypervolume improvement is calculated.

The implementation is sequential and captures MORBO's central trust-region,
Thompson-sampling, HVI coordination, and restart mechanisms. It is adapted to
the repository's `q=1`, 45-evaluation protocol; it is not a byte-for-byte copy
of the authors' large-batch reference runner.

## Main exports

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

## Sources

- Doumont et al., *We Still Don't Understand High-Dimensional Bayesian
  Optimization*, AISTATS 2026. Official code: https://github.com/colmont/linear-bo
- Daulton et al., *Multi-Objective Bayesian Optimization over High-Dimensional
  Search Spaces*, UAI 2022. Official code: https://github.com/facebookresearch/morbo
