# Task: run the learnability gate on the Ceviche lightweight WDM

You verified `pip install invrs_gym==1.6.2` works on Python 3.12 and that the
lightweight WDM is deterministic. Now settle the open question you flagged
yourself: **is `h(x)` actually learnable at d=6084?** If a GP cannot predict the
S-parameters any better than it predicts the objectives, the composite structure
is worthless there no matter how good `g` is, and we drop this candidate before
spending ~60 wall-clock hours.

Work inside this repo, on the current branch. Do not modify existing benchmarks.

## What the repo already gives you

`diagnose_composite.py` implements the gate. `diagnose(problem, n_train, n_test)`
fits independent GPs on a Sobol design and returns:

- `direct_rmse` — held-out standardized RMSE of one GP per objective on `f`
- `component_rmse` — same for one GP per component on `h`
- `composite_rmse` — error on `f` when predicting `h` then applying `g`
- `advantage` — `1 - composite_rmse/direct_rmse`, the fraction of direct error removed

Read `diagnose_composite.py` and `benchmark_common.py` (the `BenchmarkProblem`
dataclass) before writing anything.

## The contract you must satisfy

```python
BenchmarkProblem(
    name=..., slug=..., dim=..., num_objectives=2, suite="high",
    evaluate_components=...,   # X (n, d) in [0,1] -> H (n, p), 2-D, finite
    compose=...,               # H -> Y (n, m); may be compose(H, X) if it needs exact inputs
    ideal=..., ref_point=...,  # ref_point must be strictly worse than ideal, elementwise
)
```

Hard requirements:

- Inputs are the unit cube `[0,1]^d`. Map to the Ceviche density array yourself.
- `evaluate_components` returns a real 2-D tensor. Split each complex S-parameter
  into real and imaginary parts, so `p = 6` for two wavelengths x 3 ports.
- `compose` must broadcast over **leading Monte Carlo sample dimensions** of `H`.
  Index the last axis only (`H[..., 0]`), never assume rank. Test it with an
  input of shape `(7, 5, p)` and confirm you get `(7, 5, 2)`.
- `compose(evaluate_components(X)) == evaluate(X)` exactly — `problem.validate()`
  enforces this on a Sobol probe and will raise if it does not hold.
- Everything in `torch.double`.
- The two objectives are the per-wavelength losses; do **not** average them.

There is almost certainly no exact-input opportunity here (the design is the raw
pixel array), so a one-argument `compose(H)` is expected. Say so if you disagree.

## The measurements I want

Put the problem in `tmp/ceviche_wdm.py` (scratch, gitignored) exposing `PROBLEM`,
then run the gate at three training sizes with `n_test=128`:

```
n_train = 64, 128, 256
```

Report, for each, the full dict from `diagnose`: `direct_rmse`,
`component_rmse`, `composite_rmse`, `advantage`.

**`component_rmse` is the actual answer to my question.** A standardized RMSE
near or above 1.0 means the GP is no better than predicting the mean, i.e. `h`
is not learnable at this dimension and the candidate is dead. Well below 1.0
means low effective dimension and the candidate is alive.

Also report:
- wall-clock cost per Ceviche evaluation you actually observe;
- whether GP **fitting** rather than simulation dominates — an ARD kernel with
  6084 lengthscales on a few hundred points may be the real bottleneck, and if
  so say so explicitly, since that would be a problem for the BO campaign too,
  not just for this gate;
- any numerical trouble (non-finite S-parameters, fit failures, ill-conditioning).

## Decision rule, fixed in advance

- `advantage > 20%` at `n_train=256` → accept, we build it properly.
- `advantage` between 0 and 20% → marginal, report and I decide.
- `advantage <= 0` or `component_rmse >= ~0.9` → reject, and say which of the two
  failed.

Do not tune the problem to pass. If it fails, that is a useful result and I want
the honest number. Report the numbers even if they are bad, and do not retry with
different settings to find a better outcome — if you do run extra settings, label
them clearly as exploratory and report the pre-registered ones above too.

Write your final answer as markdown: a results table first, then the timing and
bottleneck notes, then a one-line verdict against the decision rule.
