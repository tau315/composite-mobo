# Findings

The strict screen produced an awkward but useful result: I found **two strong high-dimensional candidates, one credible low-dimensional chemistry candidate, and several weaker controls/adaptations**. I did not find six native, ready-made scientific MOBO benchmarks satisfying every criterion. In particular, no second low-dimensional materials simulator cleanly exposes a modest intermediate and nonlinear known map.

“Wrapper” below means the simulator is real and released, but the MO objectives are an exact or physically natural decomposition of its published scalarization rather than an upstream benchmark configuration.

## Required-suite audit

| Source | Verdict |
|---|---|
| BoTorch / Ax | DiscBrake and WeldedBeam retained as cheap mechanistic controls. Other relevant built-ins expose final response surfaces or are synthetic. [BoTorch test functions](https://botorch.readthedocs.io/en/v0.15.0/test_functions.html) |
| Olympus | Rejected. Its chemistry/materials benchmarks are datasets or probabilistic emulators exposing final metrics; spectra and other raw scientific responses are generally not available to `g`. [Paper](https://arxiv.org/abs/2010.04153), [repo](https://github.com/aspuru-guzik-group/olympus) |
| Summit | Reizman–Suzuki retained with an exact-TON wrapper. SnAR is already in your suite. [Repo](https://github.com/sustainable-processes/summit) |
| Matbench / Matbench Discovery | Rejected: supervised property-prediction datasets and model leaderboards, not same-evaluation simulators. [Matbench](https://github.com/materialsproject/matbench), [Matbench Discovery](https://github.com/janosh/matbench-discovery) |
| MORBO | WeldedBeam retained. DTLZ is already covered; VehicleSafety has direct response surfaces; Rover’s useful response is a long trajectory rather than modest \(p\). [Paper](https://proceedings.mlr.press/v180/daulton22a.html), [benchmark list](https://github.com/facebookresearch/morbo) |
| HPO-B / LassoBench | Rejected: scalar HPO losses, no scientific response. A residual-vector Lasso decomposition would have \(p\) proportional to dataset size. [HPO-B](https://github.com/machinelearningnuremberg/HPO-B), [LassoBench](https://github.com/ksehic/LassoBench) |
| BOCF literature | The original experiments are single-objective: synthetic GP functions, Langermann, Rosenbrock, and environmental calibration with 12 concentration residuals. No native MO scientific survivor. [Astudillo & Frazier 2019](https://proceedings.mlr.press/v97/astudillo19a.html) |
| Recent materials MOBO | The open AgNP work overlaps your nanoparticle benchmark. Multi-BOWS is genuinely multiobjective photonics, but requires a commercial Ansys Lumerical license. [EGBO paper/code](https://www.nature.com/articles/s41524-024-01274-x), [Multi-BOWS code](https://github.com/jungtaekkim/Multi-BOWS) |

## Low-dimensional candidates

### Summit Reizman–Suzuki yield/TON, fixed catalyst

- **Source**: Reizman et al., *Reaction Chemistry & Engineering* 1, 658–666 (2016), [DOI and paper](https://pubs.rsc.org/en/content/articlelanding/2016/re/c6re00153j); Felton et al., *Summit*, [repository](https://github.com/sustainable-processes/summit).
- **d**: 3 continuous variables after fixing one catalyst, preferably P1-L1: residence time \(60\!-\!600\) s, temperature \(30\!-\!110^\circ\mathrm C\), and catalyst loading \(L=0.5\!-\!2.5\) mol%. The published benchmark also has an eight-level catalyst variable.
- **m**: 2: maximize reaction yield \(Y\) and turnover number, TON.
- **h**: \(h_1=Y\), \(p_1=1\); \(h_2=(Y,L)\), \(p_2=2\). Yield is the emulated experimental reaction response; loading is an exactly known experimental condition included to keep \(g_2\) a function of \(h_2\) alone.
- **g**: In minimization form,
  \[
  f_1=-Y,\qquad f_2=-Y/L.
  \]
  The source data’s TON agrees with \(Y/L\) up to reported-value rounding, with one apparent outlier. For a mathematically exact benchmark, ignore Summit’s separately trained TON output and recompute it from its yield emulator.
- **Why the composite structure should pay off here**: The direct TON surface includes a reciprocal loading distortion that the yield GP does not need to learn. This is a clean test of your nonlinearity hypothesis with only one nontrivial response GP, although the yield objective itself receives no structural advantage.
- **Availability**: `pip install summit`; pretrained models and four 96–97-row CSV files are vendored. The four Reizman model directories total roughly 620 KB in the checked repository.
- **Cost per evaluation**: Expected milliseconds on CPU after model loading; not independently timed because current Summit has an older dependency stack.
- **Suite**: low-dim.
- **Risk**: **Verified simulator, adapted benchmark.** Fixing the catalyst is necessary for a continuous cube and may weaken the yield–TON trade-off. Summit originally trains yield and TON separately, so the exact-ratio wrapper must be documented. Pilot all eight fixed-catalyst instances and preselect using a simulator-only rule, not method performance.

### BoTorch DiscBrake

- **Source**: Tanabe and Ishibuchi, “An Easy-to-use Real-world Multi-objective Optimization Problem Suite” (2020), [paper](https://arxiv.org/abs/2009.12867); [BoTorch implementation](https://botorch.readthedocs.io/en/v0.15.0/test_functions.html#botorch.test_functions.multi_objective.DiscBrake).
- **d**: 4: inner radius \(R_i\), outer radius \(R_o\), engaging force \(P\), and number of friction surfaces \(n\).
- **m**: 2: minimize brake mass and stopping time, subject to four constraints.
- **h**: Let \(A=R_o^2-R_i^2\) and \(B=R_o^3-R_i^3\). Use \(h_1=(A,n)\), \(p_1=2\), and \(h_2=(A,B,P,n)\), \(p_2=4\).
- **g**:
  \[
  f_1=4.9{\times}10^{-5}A(n-1),\qquad
  f_2=9.82{\times}10^6\frac{A}{PnB}.
  \]
- **Why the composite structure should pay off here**: The stopping-time map contains both a ratio of radial moments and a reciprocal force/count term. It is a cheap low-\(p\) test of whether exposing physically interpretable geometric moments helps, although the four-dimensional direct function is already simple.
- **Availability**: `pip install botorch`; no data.
- **Cost per evaluation**: About 0.33 ms per scalar Python call in the current environment; batched calls are much faster.
- **Suite**: low-dim.
- **Risk**: The “intermediates” are known analytic transforms of \(x\), not expensive simulator responses. In a real application one would calculate them directly everywhere rather than fit GPs. Treat this as a mechanistic control, not an AI4Mat flagship.

### BoTorch/MORBO WeldedBeam

- **Source**: classical welded-beam design problem; [BoTorch implementation](https://botorch.readthedocs.io/en/v0.15.0/test_functions.html#botorch.test_functions.multi_objective.WeldedBeam); used in the [MORBO benchmark set](https://github.com/facebookresearch/morbo).
- **d**: 4: weld thickness \(x_1\), weld length \(x_2\), beam height \(x_3\), and beam width \(x_4\).
- **m**: 2: minimize fabrication cost and end deflection, with four structural constraints.
- **h**: \(h_1=(V_w,V_b)\), \(p_1=2\), where \(V_w=x_1^2x_2\) and \(V_b=x_3x_4(14+x_2)\); \(h_2=K=x_4x_3^3\), \(p_2=1\), a bending-rigidity proxy.
- **g**:
  \[
  f_1=1.10471V_w+0.04811V_b,\qquad
  f_2=\frac{2.1952}{K}.
  \]
- **Why the composite structure should pay off here**: The reciprocal stiffness map could be noticeably harder for a stationary GP than \(K(x)\). The cost objective is only a linear scalarization, so this isolates a one-objective structural benefit.
- **Availability**: `pip install botorch`; no data. MORBO also vendors its version.
- **Cost per evaluation**: About 0.29 ms per scalar call locally.
- **Suite**: low-dim.
- **Risk**: As with DiscBrake, \(h\) is analytically available from \(x\). The problem is four-dimensional and very smooth, so direct MOBO may learn it too quickly for a meaningful gap.

## High-dimensional candidates

### invrs-gym Ceviche lightweight wavelength demultiplexer

- **Source**: Schubert, “invrs-gym: a toolkit for nanophotonic inverse design research” (2024), [paper](https://arxiv.org/abs/2410.24132), [repository](https://github.com/invrs-io/gym); based on Schubert et al., *ACS Photonics* 2022 and Google’s Ceviche Challenges.
- **d**: 6,400 nominal free density pixels: a \(3.2\,\mu\mathrm m\times3.2\,\mu\mathrm m\) silicon-pattern design region on a 40 nm grid.
- **m**: 2 wavelength-specific routing losses, at 1270 and 1290 nm.
- **h**: For wavelength \(i\),
  \[
  h_i=(\Re S_{11},\Im S_{11},\Re S_{21},\Im S_{21},
       \Re S_{31},\Im S_{31}),\quad p_i=6.
  \]
  These are the complex reflection and two output-port scattering amplitudes returned by the same FDFD solve.
- **g**: First compute \(T_j=(\Re S_{j1})^2+(\Im S_{j1})^2\). At 1270 nm, target \(T_{21}\ge10^{-0.3}\) and \(T_{11},T_{31}\le10^{-2}\); at 1290 nm swap output ports 2 and 3. A globally smooth version of the upstream window loss is
  \[
  g_i=\sum_j\left[
  \operatorname{sp}_\beta(\ell_{ij}-T_j)^2+
  \operatorname{sp}_\beta(T_j-u_{ij})^2\right],
  \quad
  \operatorname{sp}_\beta(z)=\frac{\log(1+e^{\beta z})}{\beta}.
  \]
  This removes the `maximum` kink in invrs-gym’s otherwise smooth orthotope loss.
- **Why the composite structure should pay off here**: Complex S-parameters retain phase and sign information discarded by power and threshold aggregation. The direct objective must learn phase-to-power squaring, port competition, and saturation around a specification window, while each intermediate GP sees a smooth Maxwell response.
- **Availability**: `pip install invrs_gym`; no data files. JAX, `agjax`, and `ceviche_challenges` are dependencies. The repository was archived in October 2025, so pin the last release and its dependency versions. [Installation and archive status](https://github.com/invrs-io/gym).
- **Cost per evaluation**: The paper reports 0.51 s for simulation, loss, and gradient on its workstation; evaluation alone should be no slower. The Ceviche path uses NumPy/autograd rather than the GPU-oriented RCWA backend. [Timing table](https://arxiv.org/pdf/2410.24132).
- **Suite**: high-dim.
- **Risk**: **Strongest candidate, but an MO unscalarization.** Upstream averages the two wavelength losses into one scalar. The lightweight model is two-dimensional and explicitly intended as a fast, less-physical test problem. Acquisition optimization at \(d=6400\), rather than simulation, may dominate runtime.

### invrs-gym Ceviche lightweight beam splitter

- **Source**: same [invrs-gym paper](https://arxiv.org/abs/2410.24132) and [repository](https://github.com/invrs-io/gym), using the released Ceviche beam-splitter model.
- **d**: 4,000 nominal pixels from a \(3.2\,\mu\mathrm m\times2.0\,\mu\mathrm m\) design region at 40 nm. Two reflection symmetries reduce the effective independent count to roughly 1,000.
- **m**: 2 wavelength-specific equal-splitting losses, at 1270 and 1290 nm.
- **h**: Four complex port amplitudes per wavelength,
  \[
  h_i=(\Re S_{11},\Im S_{11},\ldots,\Re S_{41},\Im S_{41}),
  \quad p_i=8.
  \]
- **g**: Set \(T_j=|S_{j1}|^2\) and use the smooth window penalty above with
  \[
  T_{11},T_{41}\le0.01,\qquad
  10^{-0.35}\le T_{21},T_{31}\le1-10^{-0.35}.
  \]
- **Why the composite structure should pay off here**: The raw amplitudes are compact and phase-aware, while \(g\) imposes quadratic power conversion, balance between two outputs, reflection suppression, and wavelength robustness. With \(p=8\), this is the upper edge of your desired intermediate budget but remains far below a spectrum or field representation.
- **Availability**: `pip install invrs_gym`; no data; pin the archived release.
- **Cost per evaluation**: Published simulation+loss+gradient time is 0.50 s on the same workstation.
- **Suite**: high-dim.
- **Risk**: The two objectives may be strongly aligned because they impose the same target at nearby wavelengths. It is scientifically distinct from WDM but not an independent domain, so use it as a replication/control rather than counting both as broad evidence.

### EngiBench Photonics2D wavelength demultiplexer

- **Source**: Felten et al., “EngiBench: A Framework for Data-Driven Engineering Design Research,” NeurIPS 2025, [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0013efa1327c079e73154d4061c3a396-Abstract-Datasets_and_Benchmarks_Track.html), [repository](https://github.com/IDEALLab/EngiBench), [problem documentation](https://engibench.ethz.ch/problems/photonics2d/).
- **d**: 14,400 material-density pixels, \(120\times120\), representing a high/low-permittivity photonic structure.
- **m**: Proposed 2-objective split: coupling power at \(\lambda_1=1.5\,\mu\mathrm m\) into output 1 and at \(\lambda_2=1.3\,\mu\mathrm m\) into output 2.
- **h**: For channel \(i\), let
  \[
  z_i=\sum_r \overline{E_i(r)}\,P_i(r),
  \qquad h_i=(\Re z_i,\Im z_i),\quad p_i=2,
  \]
  where \(E_i\) is the simulated electric field and \(P_i\) the desired output mode.
- **g**:
  \[
  f_i=-\left[(\Re z_i)^2+(\Im z_i)^2\right].
  \]
  This uses coupled power and is everywhere differentiable. EngiBench’s native scalar objective instead multiplies the two amplitude overlaps and subtracts a material-norm penalty.
- **Why the composite structure should pay off here**: Direct power objectives discard complex phase, exactly the kind of information composite modeling can preserve. The two-dimensional intermediate is exceptionally economical relative to the 14,400-dimensional design.
- **Availability**: `pip install engibench` plus the Photonics2D optional dependencies; Ceviche/autograd solver, no dataset required for online simulation. An optional Hugging Face dataset contains optimized designs.
- **Cost per evaluation**: **UNVERIFIED**. Two \(120\times120\) CPU sparse FDFD solves should plausibly take seconds to low tens of seconds, but neither the paper nor current documentation gives a benchmark and I did not install its optional solver stack.
- **Suite**: high-dim.
- **Risk**: This is another wavelength-demultiplexer and therefore overlaps the invrs-gym WDM scientifically. It is also a proposed MO split of a published scalar benchmark. Do not schedule the full \(20\times400\times8\) campaign until a 100-evaluation CPU timing and determinism test passes.

### invrs-gym Bayer color sorter

- **Source**: Schubert 2024, [invrs-gym paper](https://arxiv.org/abs/2410.24132) and [code](https://github.com/invrs-io/gym); based on Zou et al., “Pixel-level Bayer-type colour router based on metasurfaces,” *Nature Communications* 13, 3288 (2022).
- **d**: 40,002 nominal variables: a \(200\times200\) Si\(_3\)N\(_4\) metasurface density grid plus metasurface thickness and focal-plane offset. Diagonal symmetry roughly halves the independent pattern variables.
- **m**: 3: blue-, green-, and red-channel sorting losses at 450, 550, and 650 nm.
- **h**: For each color, the simulator returns transmission into four Bayer pixels under two polarizations: \(p_i=2\times4=8\).
- **g**: Let \(e_i\) be polarization-averaged transmission into the correct pixel, with the two green pixels summed. The published differentiable loss separates exactly as
  \[
  f_i=\operatorname{softplus}\!\left(\frac{0.55-e_i}{0.45}\right),
  \qquad i\in\{B,G,R\}.
  \]
  The upstream scalar loss is \(f_B+f_G+f_R\).
- **Why the composite structure should pay off here**: This is a materials-realistic three-objective split whose intermediate preserves polarization and crosstalk information. The aggregation is less nonlinear than WDM’s complex-amplitude window map, but it combines spatial routing, polarization averaging, channel selection, and saturation.
- **Availability**: `pip install invrs_gym`; no external data or commercial solver; FMMAX/JAX RCWA backend. Repository is archived.
- **Cost per evaluation**: Published loss+gradient time is 5.93 s on a workstation containing an RTX 4090; batched effective time was 0.928 s. **CPU-only time is UNVERIFIED**.
- **Suite**: high-dim.
- **Risk**: It does not yet satisfy your CPU requirement. The default only simulates one wavelength per color, so it is not truly integrating a dense spectrum. Keep it conditional on a CPU timing gate.

## Ranked shortlist

### Low-dimensional

1. **Reizman–Suzuki fixed-catalyst wrapper** — the only low-dimensional chemistry survivor with an actual experimental response and a meaningful nonlinear rate-law ratio.
2. **DiscBrake** — stronger nonlinear \(g\) than WeldedBeam and \(p_i\le4\), but best treated as a mechanistic control.
3. **WeldedBeam** — extremely cheap and already used by MORBO; useful for isolating the reciprocal-stiffness effect, not as materials evidence.

Only the first is a credible scientific test of the stated hypothesis. I would not describe the low-dimensional requirement as fully solved until a second simulator-backed candidate is found or constructed under a preregistered wrapper.

### High-dimensional

1. **invrs-gym lightweight WDM** — best overall match: \(p=6\), complex phase-aware response, strongly nonlinear known map, and approximately 0.5 s reported evaluation path.
2. **invrs-gym lightweight beam splitter** — \(p=8\), same fast CPU solver, and an independent port-balancing transformation; strongest replication benchmark.
3. **EngiBench Photonics2D** — active, published 2025 benchmark with \(p=2\), but admit only after CPU timing verification.

**Bayer sorter** is the better AI4Mat story than EngiBench, but its CPU cost is presently unverified. I would run a timing gate first: install pinned environments, evaluate 100 fixed random designs twice, require bitwise or tight numerical reproducibility, and reject any benchmark exceeding the campaign’s per-evaluation budget.

