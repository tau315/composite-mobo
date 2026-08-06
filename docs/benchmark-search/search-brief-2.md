Important correction to my earlier brief, then a second search round.

## What I got wrong

I told you `g` should "carry substantial nonlinearity (integration over a
spectrum, a norm, a peak extraction, ...)". **The integration examples were
exactly backwards.** I have now measured this.

The criterion is not "is `g` nonlinear". It is: **does `g` make `f` harder to
GP-model than `h`?** Composite modeling can only win when the intermediate is
the *easier* regression target.

Averaging and integrating maps do the opposite. They smooth. `f` ends up easier
to model than `h`, and composite modeling is then strictly worse, not neutral.

Measured, same intermediate, three different `g`, d=9 (advantage = fraction of
direct-GP held-out RMSE removed by predicting through `g`):

| g | advantage |
|---|---|
| averaging (weighted mean of h) | +23% |
| exponential (Arrhenius-style amplification) | +53% |
| sharply peaked (Shockley-Queisser shape) | +78% |

Confirmed on real benchmarks: a radiative-cooling multilayer film where `h` is
the reflectance spectrum and `g` integrates it against solar/atmospheric
weightings scores **-8% to -13%** — the spectrum oscillates with layer thickness
(interference fringes) while its integral is smooth, so the spectrum is the
*harder* target. Same story for an optimal-power-flow benchmark whose `g` is a
plain linear sum of 29 components: **-10.5%**.

Note the linear case is provable, not empirical: a GP is closed under linear
maps, so modeling `h` and summing is a *constrained* version of modeling `f`
directly, with p accumulated component errors. Linear `g` can never help.

## Two further conditions I have since measured

**1. High dimension needs a low intrinsic-dimension intermediate.** The sharply
peaked map above scores +78% at d=9 but **-9.7%** at d=120 when `h` depends on
all inputs. Give the same problem an 8-dimensional active subspace and it
returns to **+62%**. If the GP cannot learn `h` either, there is nothing to
exploit.

**2. Anything known in closed form must bypass the GP entirely.** This turned
out to be the single biggest lever. Our DTLZ2 benchmark was fitting a GP to
`cos(pi*x1/2)` where `x1` is a *design variable* we choose exactly. Routing it
through `g` as an exact input instead of modeling it moved d=600 from **+3% to
+78%** — same `g`, same nonlinearity. Our framework now supports
`compose(H, X)`, so a benchmark can feed exact design coordinates into the
known map.

## What I need now

Same output format as before, same skepticism, same "do not invent benchmarks,
mark anything unverified as UNVERIFIED". Two open slots:

- **1 more low-dimensional (d <= ~20) scientific benchmark.** Materials science
  strongly preferred; chemistry/catalysis/photonics/batteries/polymers count.
  We currently have exactly one credible low-dim scientific candidate (Summit
  SNAr, advantage +46%), and we need a second. Your earlier Reizman-Suzuki
  suggestion is still live — but note its `g = Y/L` is a *ratio*, which is the
  right shape, so please develop it further if it holds up.
- **1 high-dimensional (d >= ~50) scientific benchmark.** Your invrs-gym Ceviche
  WDM pick looks strong under the corrected criterion: complex S-parameters as
  `h`, and `g` squares them to power and applies a threshold window, which
  *creates* structure. Please verify it is actually installable and runnable on
  CPU today given the repo was archived in Oct 2025, and report real timings if
  you can install it.

Prioritize benchmarks that satisfy at least one of:
- `g` is a ratio, exponential, sharply-peaked response, threshold, resonance, or
  otherwise amplifies structure in a smooth `h`;
- a meaningful part of the objective is a known closed-form function of design
  variables, so it can be routed exactly through `g` rather than modeled;
- `h` has low intrinsic dimension even when `d` is large.

Explicitly **reject** anything whose `g` is a mean, a sum, a plain integral over
a rough response, or a linear functional — and say so, so we do not
re-investigate it.

Also worth a look this round, since you have not covered them: physics-based
materials simulators with cheap analytic property-to-performance maps
(thermoelectric zT = S^2*sigma*T/kappa from transport properties; Shockley-Queisser
efficiency from a band gap; nucleation rates from an interfacial energy;
Arrhenius selectivity from activation energies), plus any 2023-2026 AI4Mat /
NeurIPS materials-design papers that release a runnable multi-objective
simulator. Search GitHub directly, not only paper text.
