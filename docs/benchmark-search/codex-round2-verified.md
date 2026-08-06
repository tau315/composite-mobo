## Bottom line

- **Low-dimensional slot:** use **Reizman–Suzuki case IV with catalyst fixed to P1–L2**. The exact `Y/L` construction passed a 20-seed held-out GP screen strongly: median **69.1% of direct-TON RMSE removed**, 20/20 wins.
- **High-dimensional slot:** **Ceviche lightweight WDM is installable and deterministic today**, but remains conditional. Its map is excellent; its input learnability is unverified, and the measured CPU cost implies about **60 evaluation-hours** for the full 64,000-evaluation campaign.

### Reizman–Suzuki case IV, fixed P1–L2 catalyst

- **Source**: Reizman et al., “Suzuki–Miyaura cross-coupling optimization enabled by automated feedback,” *Reaction Chemistry & Engineering* 1, 658–666 (2016), [paper](https://pubs.rsc.org/en/content/articlelanding/2016/re/c6re00153j); Felton et al., *Summit*, [repository](https://github.com/sustainable-processes/summit). The later Olympus description independently specifies three continuous parameters, eight ligands, and simultaneous maximization of yield and TON. [Olympus enhanced supplement](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6464ae0afb40f6b3eebaab70/original/olympus-enhanced-benchmarking-mixed-parameter-and-multi-objective-optimization-in-chemistry-and-materials-science.pdf)
- **d**: **3 continuous** after fixing the ligand to P1–L2: residence time \(t\in[60,600]\) s, temperature \(T\in[30,110]^\circ\mathrm C\), and catalyst loading \(L\in[0.5,2.5]\) mol%. The published mixed problem also has one eight-level catalyst variable, but fixing it avoids introducing categorical-GP effects into the structural comparison.
- **m**: **2**, both maximized: reaction yield and catalyst turnover number.
- **h**: One unique modeled intermediate, \(h(x)=Y(x)\), \(p=1\): predicted reaction yield. It is reused by both objectives.
- **g**:
  \[
  f_1(Y,x)=Y,\qquad
  f_2(Y,x)=\operatorname{TON}=\frac{Y}{L(x)}.
  \]
  Catalyst loading is an exact chosen coordinate and should enter through `compose(H, X)`, never through another GP. Since \(L\ge0.5\), the ratio is smooth and nonsingular.
- **Why the composite structure should pay off here**: The loading range changes \(1/L\) by a factor of five, creating substantially greater curvature and heteroscedastic effective error in TON than in yield. Composite modeling also fits only one GP instead of separate yield and TON GPs. In my 20-seed check using 50 Latin-hypercube training points, 4,096 held-out Sobol points, and Matérn-5/2 GPs, predicting \(Y\) and dividing by exact \(L\) reduced median TON RMSE from **0.905 to 0.286**, a **69.1% advantage**, with 20/20 wins. This is a regression screen, not a MOBO result.
- **Availability**: Summit includes the four CSV datasets and five-network pretrained ensembles. The Reizman artifacts total about **605 KiB of models plus 13 KiB of data**; one fixed case needs roughly 155 KiB. Summit 0.8.9 is nominally pip-installable, but PyPI restricts it to Python 3.8–<3.11, so installation fails on Python 3.12. A small MIT-licensed forward-only wrapper around the shipped PyTorch weights is the practical route. [Summit on PyPI](https://pypi.org/project/summit/0.8.9/)
- **Cost per evaluation**: My forward-only five-network ensemble wrapper took **0.246 ms median**, 0.345 ms at the 95th percentile, on an Intel i9-14900HX CPU.
- **Suite**: **Low-dimensional**.
- **Risk**: This is a neural-network emulator trained on roughly ninety experimental measurements per case, not a mechanistic simulator. The upstream network was jointly trained on yield and TON; using only its yield output and recomputing TON changes the Summit wrapper, although it restores the physical identity that the independently predicted TON head only approximates. A 20,000-point screen found a real tradeoff for case IV/P1–L2—maximum yield was about 96.7% near \(L=2.46\), while maximum TON was about 148.7 near \(L=0.503\)—but the final benchmark should pin the weights and document this wrapper precisely.

**Verdict:** strong accept. This is the cleanest second low-dimensional scientific benchmark I found.

### invrs-gym Ceviche lightweight wavelength demultiplexer

- **Source**: Schubert, “invrs-gym: a toolkit for nanophotonic inverse design research” (2024), [paper](https://arxiv.org/html/2410.24132v1); [repository](https://github.com/invrs-io/gym). The Ceviche challenges originate from Schubert et al., *ACS Photonics* 9, 2327–2336 (2022). The suite describes the response explicitly as wavelength-dependent scattering parameters and defines transmission windows over their squared magnitudes. [Ceviche section](https://arxiv.org/html/2410.24132v1#S3.SS6)
- **d**: **6,084 free variables**: an 80×80 continuous material-density array for a 3.2×3.2 μm silicon/silica design region at 40 nm resolution, with 316 boundary pixels fixed. Each free pixel lies in \([0,1]\).
- **m**: **2 proposed objectives**, one loss for each wavelength: route 1270 nm light into output port 2 and 1290 nm light into output port 3, while suppressing reflection and leakage. Upstream computes these two per-wavelength losses and then averages them; the MO wrapper simply omits that final mean.
- **h**: For wavelength \(i\),
  \[
  h_i(x)=\left(S_{11}^{(i)},S_{21}^{(i)},S_{31}^{(i)}\right)\in\mathbb C^3,
  \]
  or **\(p_i=6\) real components**. These are physical complex port amplitudes returned by the same FDFD solve; the full two-wavelength response has 12 real values.
- **g**: First convert amplitude to power,
  \[
  \tau_{ij}=\left|S_{j1}^{(i)}\right|^2
           =\Re(S_{j1}^{(i)})^2+\Im(S_{j1}^{(i)})^2.
  \]
  For the desired port, \(d_{ij}=10^{-0.3}-\tau_{ij}\); for reflection and the undesired port, \(d_{ij}=\tau_{ij}-10^{-2}\). The exact upstream per-wavelength loss simplifies on the physical range to
  \[
  f_i=\sum_{j=1}^{3}
  \operatorname{softplus}\!\left(\frac{d_{ij}}{0.01}\right)^2.
  \]
  At 1270 nm the desired port is \(j=2\); at 1290 nm it is \(j=3\). Thus `g` applies amplitude-squaring, tight 1%/50% transmission thresholds, 100× distance scaling, softplus, and quadratic aggregation. See the upstream [loss implementation](https://github.com/invrs-io/gym/blob/main/src/invrs_gym/loss/transmission_loss.py).
- **Why the composite structure should pay off here**: Unlike an integral, this map creates sharp structure: modest changes in smooth complex amplitudes can cross a narrow transmission window and be amplified by the normalized softplus-square loss. The intermediate is compact per objective and directly physical. The unresolved issue is crucial, however: mapping 6,084 pixels to six numbers does **not** prove low intrinsic input dimension. Maxwell smoothing and subwavelength pixels make low effective rank plausible, but resonant topology may make many pixel directions important.
- **Availability**: Verified on **2026-08-05**. `pip install invrs_gym==1.6.2` succeeded in a clean Windows Python 3.12 environment and selected JAX 0.11.0 on CPU; no datasets or proprietary solvers were required. The environment occupied about **0.66 GiB**. PyPI still hosts the 72.4 KiB universal wheel released October 20, 2025. [PyPI](https://pypi.org/project/invrs-gym/) The GitHub repository was archived on October 24, 2025 and is read-only. [Archive notice](https://github.com/invrs-io/gym)
- **Cost per evaluation**: On an Intel i9-14900HX:

  - Same design, serial wavelength solves: **5.63–6.85 s**.
  - Five distinct random designs, serial: **13.1–17.2 s**, median **15.3 s**.
  - Distinct random designs with `max_parallelizm=2`: **3.32–3.42 s**, median **3.38 s**.
  - Repeating the same design produced identical S-parameters, maximum absolute difference **0.0**.

  At 3.38 s, 20×400×8 evaluations require about **60 wall-clock hours** before BO overhead, absent outer parallelism. The paper reports 0.51 s including a gradient on a 28-core Xeon/RTX 4090 workstation, so hardware changes the economics substantially. [Published timing table](https://arxiv.org/html/2410.24132v1#A3)
- **Suite**: **High-dimensional**.
- **Risk**: This is a 2-D photonics proxy, which the paper explicitly says is not fully representative of real integrated-photonic design. Standard ARD GPs at 6,084 dimensions are themselves questionable. More importantly, low intrinsic dimension of \(h(x)\) remains **UNVERIFIED**; a small held-out direct-versus-composite regression gate should precede the full MOBO campaign. The archived package also has broadly unpinned JAX dependencies, so freeze the tested environment. Finally, the MO formulation is a faithful un-averaging of an upstream two-band loss, but it is not an upstream named multi-objective benchmark.

**Verdict:** conditional accept. It fills the high-dimensional slot only after a cheap learnability gate. I would require positive composite advantage on roughly 40–60 pilot evaluations before committing the full campaign.

## Explicit rejections from this round

- **Thermoelectric \(zT=S^2\sigma T/\kappa\)**: the map has exactly the desired product/ratio form, but the open [ThermoElectric](https://github.com/ariahosseini/ThermoElectric) framework I found is a transport-property calculator, not a released deterministic \(m\ge2\) optimization benchmark. Adding power factor, cost, or another objective would create a new benchmark rather than recover an existing one.
- **Shockley–Queisser efficiency**: no released, CPU-cheap multi-objective simulator with a same-evaluation bandgap intermediate was verified. When bandgap is itself the design variable, the entire objective is known and should bypass the GP; spectrum-to-efficiency variants return to the rejected integration regime.
- **Nucleation-rate workflows**: the available [PLUMED/LAMMPS nucleation-rate workflow](https://www.plumed-nest.org/eggs/21/009/) has the right exponential physics but is atomistic, stochastic, not seconds-scale, and not multi-objective.
- **MultiChem VdV1/PK1**: structurally promising—yield is transformed through residence-time ratios and logarithms—but the released [MultiChem repository](https://github.com/adamc1994/MultiChem) is MATLAB-only, deliberately injects random observation noise, has no pip package, and lacks a clear repository-level license. Removing noise and porting the ODEs would amount to maintaining a new benchmark implementation. [Paper](https://www.repository.cam.ac.uk/items/ded75ca3-28a5-404b-8524-f5d2ee63b081)
- **Radiative cooling, Bayer spectral objectives, and similar spectral averages**: reject because the mean/integral smooths the rough optical response.
- **OPF cost/emission sums and other linear aggregations**: reject because `g` is linear and accumulated component errors cannot improve on a direct GP.
- **MatBench, MatBench Discovery, and the 2023–2026 materials-design repositories located this round**: prediction datasets or generative-model workflows, not deterministic same-evaluation scientific simulators exposing a usable intermediate.

## Ranking

### Low-dimensional

1. **Reizman–Suzuki IV / P1–L2** — strongest choice; exact ratio, exact loading coordinate, one GP instead of two, and a +69.1% held-out regression screen.
2. **Summit SNAr** — already validated by you at +46%.
3. **MultiChem VdV1** — good mathematics, but MATLAB/noise/licensing make it a poor paper benchmark today.

### High-dimensional

1. **Ceviche lightweight WDM** — best structural map and verified runnable, conditional on an intrinsic-dimension/held-out screen.
2. **No second candidate verified** — the photonic averaging problems fail the corrected criterion.
3. **No third candidate verified** — thermoelectric, SQ, and nucleation ideas lack a released benchmark satisfying all constraints.

