# Task: find scientific MOBO benchmarks with exploitable composite structure

## Context

We are writing an AI4Mat @ NeurIPS 2026 workshop paper asking: **does exploiting
known composite structure improve multi-objective Bayesian optimization (MOBO),
especially in high dimensions?**

Composite structure means objective `i` is

    f_i(x) = g_i(h_i(x)),      x in [0,1]^d

where:
- `h_i(x) in R^{p_i}` is an *intermediate scientific response* observed for free
  whenever the design is evaluated (a spectrum, field, time series, concentration
  profile, set of material properties, ...);
- `g_i` is a **known, cheap, deterministic, differentiable** map from the
  intermediate to the scalar objective.

A *direct* method fits GPs to `f_1..f_m`. A *composite* method fits independent
GPs to the columns of `h`, pushes posterior samples through `g`, and computes the
acquisition on the resulting non-Gaussian objective samples.

## The problem we need solved

Our current 11 benchmarks give a weak, mixed signal. Measured paired results
(20 seeds, hypervolume, composite minus direct):

| benchmark | d | m | p | result |
|---|---|---|---|---|
| DTLZ2 | 6 | 2 | 4 | **+5.3% / +3.6%**, 20/20 wins, p~2e-6 |
| Langermann-Ackley | 6 | 2 | - | **+8.1%**, 20/20 |
| five-Ackley | 6 | 5 | - | +5.1% |
| nanoparticle RGB | 6 | 3 | 6 | +0.2% (noise) |
| Summit SNAr | 4 | 2 | 8 | +0.1% / -0.7% |
| penicillin | 7 | 3 | 26 | **-4.6%** |
| CORT TG119 | 418 | 3 | 6 | **-2.0%** (significantly worse) |
| DTLZ2 | 600 | 2 | 4 | +0.1% |
| RCM40 OPF | 34 | 2 | 29 | +0.8% |

Our working hypothesis for *why*: the gain comes from how much nonlinearity the
known map `g` absorbs. DTLZ2 wins because a single smooth scalar intermediate
`r = sqrt(g(x))` replaces two GPs that would otherwise have to learn
cos/sin-modulated surfaces. Penicillin loses because 26 components means 26 GPs
trained on ~50 evaluations. Note `p/m` alone does NOT predict the outcome:
nanoparticle_rgb has the same `p/m = 2` as DTLZ2 and shows nothing.

## What we need from you

Search the literature and open-source benchmark suites for **candidate
benchmarks that satisfy the structural criterion**, i.e. where:

1. `h` is a smooth, physically meaningful intermediate that is genuinely easier
   to GP-model than `f`;
2. `g` is known, cheap, and carries substantial nonlinearity (integration over a
   spectrum, a norm, a peak/bandgap extraction, a thresholded yield, an
   argmax-free aggregation, a rate-law composition, ...);
3. `p` stays modest (roughly 1-8 components per objective) so each component GP
   is trainable within a few-hundred-evaluation budget;
4. `h` costs nothing extra to observe;
5. m >= 2 objectives, deterministic simulator, and the whole thing is runnable
   on CPU in seconds per evaluation (we need 20 seeds x ~400 evaluations x 8
   methods).

We need **at least 2 low-dimensional (d <= ~20) and at least 2 high-dimensional
(d >= ~50)** candidates. **Materials science is strongly preferred** — the venue
is AI4Mat. Chemistry, photonics, catalysis, alloys, batteries, polymers,
metamaterials, crystal structure, thermoelectrics all count.

Please look at, at minimum:
- BoTorch / Ax built-in multi-objective test problems
- Olympus (Aspuru-Guzik group) benchmark suite
- Summit (Felton et al.) chemistry benchmarks
- MatBench / MatBench Discovery
- the MORBO paper's benchmark set (Daulton et al., UAI 2022)
- HPO-B / LassoBench style suites only if they have real composite structure
- "Bayesian optimization of composite functions" (Astudillo & Frazier, ICML 2019)
  and its follow-ups, for which problems the BOCF literature itself uses
- multi-objective materials-design papers from 2023-2026 that release code

## Output format

Write your final answer as markdown with one section per candidate benchmark:

```
### <name>
- **Source**: paper citation + repo URL
- **d**: input dimension, and what x physically is
- **m**: number of objectives, and what they are
- **h**: the intermediate response, its dimension p, and its physical meaning
- **g**: the known map, written as a formula or precise description
- **Why the composite structure should pay off here**: 2-3 sentences arguing
  from the criterion above
- **Availability**: pip-installable? vendored code? data files needed? size?
- **Cost per evaluation**: rough wall-clock estimate on CPU
- **Suite**: low-dim or high-dim
- **Risk**: what could make this a bad choice
```

Rank the candidates at the end: a shortlist of your top 3 low-dim and top 3
high-dim, with a one-line justification each.

Be concrete and skeptical. Do NOT invent benchmarks or citations — if you are
unsure a benchmark exists or is available, say so explicitly and mark it
UNVERIFIED. A short list of real, verified, runnable benchmarks is far more
useful than a long speculative one.
