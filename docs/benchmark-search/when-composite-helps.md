# When composite modeling helps

Working notes backing the paper's "when does composite modeling help" section.
Every number is the **advantage** reported by `diagnose_composite.py`: the
fraction of a direct GP's held-out standardized RMSE that is removed by instead
predicting the objectives through the known map `g`. It needs no BO loop and
tracks the measured paired BO outcomes.

## The criterion

Composite modeling can only win when the intermediate is the **easier**
regression target. `g` being nonlinear is close to necessary but nowhere near
sufficient. Three conditions decide it.

### 1. `g` must create structure, not destroy it

Same intermediate, three different known maps, d=9:

| `g` | advantage |
|---|---:|
| averaging — a weighted mean of `h` | +23% |
| exponential — Arrhenius-style amplification | +53% |
| sharply peaked — Shockley-Queisser shape | +78% |

Maps that average, integrate, or sum **smooth** the objective. `f` becomes
easier to model than `h`, so composite modeling is strictly worse, not neutral.

The linear case is provable rather than empirical. A GP is closed under linear
maps, so modeling `h` and applying a linear `g` is a *constrained* version of
modeling `f` directly — the same predictor class, with `p` component errors
accumulated into the result. **A linear `g` can never help.**

### 2. In high dimension, `h` needs low intrinsic dimension

The sharply-peaked map above, at d=120:

| intermediate | advantage |
|---|---:|
| depends on all `d` inputs | −9.7% |
| depends on an 8-dimensional active subspace | +61.7% |

If the GP cannot learn `h` either, there is nothing to exploit no matter how
favorable `g` is.

### 3. Anything known in closed form must bypass the GP

The largest single lever. DTLZ2 was fitting a GP to `cos(pi*x1/2)`, where `x1`
is a design variable known exactly. Routing it through `g` as an exact input
instead — same `g`, same nonlinearity — moved d=600 from **+3% to +78%**, and
inverted the reading of how advantage scales with dimension: it now *grows*
with `d`, because the exactly-known share of the problem grows while the GP's
remaining job stays a smooth radial term.

Declare `compose(H, X)` to receive the exact designs; see `solvers.composer`.

## Every benchmark, classified

| benchmark | d | p | `g` | advantage |
|---|---:|---:|---|---:|
| DTLZ2 600d | 600 | 1 | product with exact trig | **+77.7%** |
| DTLZ2 6d | 6 | 1 | product with exact trig | **+65.0%** |
| SNAr | 4 | 6 | ratio (E-factor) | **+45.6%** |
| DTLZ2 100d | 100 | 1 | product with exact trig | **+35.0%** |
| nanoparticle RGB | 6 | 6 | ratio of spectral integrals | +18.1% |
| photonic bandgap | 1024 | 48 | extremal (`min` − `max` over k) | +2.5% |
| CORT TG119 | 418 | 6 | dose integrals | +1.0% |
| penicillin | 7 | 26 | **selection** (reads 3 of 26 states) | 0.0% |
| topopt SIMP | 1152 | 48 | `exp` then sum and max | −3.8% |
| RCM46 OPF | 34 | 45 | **linear sum** | −4.2% |
| RCM40 OPF | 34 | 29 | **linear sum** | −10.5% |

## Permanent negative controls

Three benchmarks cannot be rescued by reformulation, and are more useful
presented as confirmations of the theory than as disappointments.

**RCM40 / RCM46.** `g` is a plain sum of components (`f1 = sum(Psp)`). Linear,
so no gain is possible by construction. The measured −10.5% is the accumulated
error of 29 GPs being summed where 2 would have sufficed.

**Penicillin.** `g` selects three of twenty-six simulated states and negates
one — a selection matrix, hence linear. Twenty-six GPs are fitted so that three
can be read off. Exactly 0.0%, as the closure argument predicts.

**Photonic bandgap.** `g` is extremal (`min_k` of one band minus `max_k` of
another), which does create structure — but taken over 48 component GPs, `max`
is dominated by whichever component errs high, so the map amplifies estimation
error alongside signal. Structure-creating is not enough when `p` is large.

## Rejected candidate: integrating a spectrum

A radiative-cooling multilayer film — `h` the reflectance spectrum, `g` the
solar and atmospheric-window integrals — scores **−8% to −13%** at d=8 to 120.
Reflectance oscillates with layer thickness through interference fringes while
its integral is smooth, so the spectrum is the harder target.

This matters because "the intermediate is a spectrum, image, field, or time
series" is the motivating picture in the paper outline. That picture selects for
exactly the wrong `g`. Prefer intermediates that are *few and smooth* with a map
that sharpens them.
