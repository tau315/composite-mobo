# Composite-Function Multi-Objective Bayesian Optimization

This project tests whether exploiting a known composite objective structure can
improve sample efficiency in low-dimensional, multi-objective Bayesian
optimization (MOBO).

The central comparison is between learning final objectives directly,

\[
\mathbf{x}\longrightarrow \bigl(f_1(\mathbf{x}),f_2(\mathbf{x})\bigr),
\]

and learning an observable inner response before applying a known outer map,

\[
\mathbf{x}\longrightarrow h(\mathbf{x})
\longrightarrow
\Phi\bigl(h(\mathbf{x}),\mathbf{x}\bigr)
=\bigl(f_1(\mathbf{x}),f_2(\mathbf{x})\bigr).
\]

The benchmark measures Pareto-front recovery using dominated hypervolume versus
the total number of expensive function evaluations.

## Methods

Four solvers are implemented in `solvers.py`.

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

\[
S_{\tau,\mathbf{w}}(\mathbf{f})
=\tau\log\sum_i
\exp\left(\frac{w_i(f_i-z_i^\star)}{\tau}\right).
\]

Because the benchmarks are minimization problems, the acquisition utility is
`-S`. A separate qLogEI run is performed for each weight.

### Composite smooth Tchebycheff BO

`composite_chebyshev_bo` models the intermediate responses and propagates their
posterior samples through both the known objective map and smooth Tchebycheff
scalarization:

\[
\mathbf{x}\rightarrow h(\mathbf{x})
\rightarrow\mathbf{f}(\mathbf{x})
\rightarrow S_{\tau,\mathbf{w}}(\mathbf{f}(\mathbf{x})).
\]

This is the fully nested composite method.

## Benchmarks

`benchmark.py` provides four deterministic, unconstrained benchmark problems.
All inputs are bounded to `[0, 1]^d`; the default dimension is six.

### ZDT1, ZDT2, and ZDT3

The ZDT problems share

\[
g(\mathbf{x})=1+\frac{9}{d-1}\sum_{j=2}^{d}x_j,
\qquad f_1(\mathbf{x})=x_1.
\]

The direct methods fit GPs to `f1` and `f2`. The current composite methods fit
one GP to

\[
u(\mathbf{x})=\log g(\mathbf{x})
\]

and reconstruct `g = exp(u)`. This warped representation guarantees positive
posterior component samples for outer functions that divide by `g`. It is not
the same prior as placing a GP directly on `g`, and this distinction should be
reported when interpreting results.

- ZDT1 has a continuous convex Pareto front.
- ZDT2 has a continuous non-convex Pareto front.
- ZDT3 has a disconnected Pareto front.

### DTLZ2

For two objectives and six inputs,

\[
g(\mathbf{x})=\sum_{j=2}^{6}(x_j-0.5)^2,
\]

\[
f_1=(1+g)\cos(\pi x_1/2),
\qquad
f_2=(1+g)\sin(\pi x_1/2).
\]

The direct methods fit GPs to `f1` and `f2`. The composite methods fit one GP
directly to `g` and use the exact candidate coordinate `x1` in the known outer
map. The Pareto front is the positive quadrant of the unit circle.

## Experimental protocol

The default experiment uses:

- 20 independent trials
- Trial seeds `0, 1, ..., 19`
- Matched seeds for corresponding direct and composite methods
- Scrambled Sobol initial designs
- Two scalarization weights: `(0.05, 0.95)` and `(0.95, 0.05)`
- Smooth-Tchebycheff temperature `0.05`
- Ideal point `(0, 0)`

ZDT1, ZDT3, and DTLZ2 use 40 total evaluations per method and trial:

- qLogEHVI: 5 initial + 35 BO evaluations
- STCH: two independent runs of 5 initial + 15 BO evaluations

ZDT2 uses its requested special protocol of 30 total evaluations:

- qLogEHVI: 3 initial + 27 BO evaluations
- STCH: two independent runs of 3 initial + 12 BO evaluations

For every problem, all methods and trials use the same fixed hypervolume
reference point:

- ZDT: `(1.1, 11.0)`
- DTLZ2: `(2.5, 2.5)`

Larger dominated hypervolume is better.

## Plots

The benchmark computes cumulative hypervolume after every function evaluation.
Across trials, NumPy calculates the mean and standard error. Each solid curve is
the mean, and its shaded region is mean plus or minus one standard error.

Two plots are generated for every selected benchmark:

- Standard qLogEHVI versus Composite qLogEHVI
- Objective-GP STCH versus Composite STCH

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

Useful flags include:

```text
--trials          Number of independent trials
--budget          Total evaluations for non-ZDT2 problems
--weights         Number of scalarization weights
--temperature     Smooth-Tchebycheff temperature
--seed            Base random seed
--raw-samples     Raw acquisition-optimization samples
--restarts        Acquisition-optimization restarts
--output          Base output filename used to construct plot names
```

ZDT2's 30-evaluation and three-initial-point protocol is currently fixed in the
benchmark code rather than controlled by `--budget` and `--initial`.

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
project_writeup.tex    LaTeX description of the project and results section
README.md              Project documentation
.gitignore             Generated and local files excluded from Git
```

## Current limitations

- Only low-dimensional, two-objective synthetic tests are currently included.
- Intermediate outputs are assumed observable at no additional evaluation cost.
- Output GPs are independent and do not model cross-output correlations.
- ZDT composite models use `log(g)`, while DTLZ2 models `g` directly.
- Two scalarization weights mostly target the ends of the Pareto front.
- Standard-error bands are descriptive and are not formal significance tests.

