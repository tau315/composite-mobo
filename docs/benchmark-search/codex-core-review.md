## Verdict

I found no implementation bug explaining the Reizman STCH loss. The composite STCH objective, incumbent, candidate inputs, and trial seeds are consistent. I would treat the loss as a real result of the current surrogate/acquisition construction—not evidence that composite modeling always helps.

The broader framework does have several concrete defects.

## Ranked findings

1. **Definitely wrong — resume validation accepts results from the wrong algorithm.**  
   [_validated_payload](</C:/Users/sahas/Github Repos/composite-mobo/benchmark_common.py:453>) checks that `family` is one of three strings but never checks the expected method/family pairing or that the initial design came from the configured seed. The later recomputation at [lines 520–580](</C:/Users/sahas/Github Repos/composite-mobo/benchmark_common.py:520>) proves only that the stored points and derived values are self-consistent.

   I constructed and passed:

   - an arbitrary non-Sobol trajectory labeled as qLogEHVI;
   - a `direct_qlogehvi` artifact labeled as STCH.

   Therefore “recomputes every derived field” does not establish that the configured solver produced the artifact. This could affect figures if artifacts were mislabeled or produced by different solver code under trusted provenance.

2. **Definitely wrong — launcher provenance is asserted, not verified.**  
   [run_unicorn.sh:92](</C:/Users/sahas/Github Repos/composite-mobo/run_unicorn.sh:92>) optionally checks an operator-supplied archive hash, but never proves that the archive corresponds to `$COMMIT`. It then exports that unverified value as authoritative metadata at [line 138](</C:/Users/sahas/Github Repos/composite-mobo/run_unicorn.sh:138>). A wrong archive can therefore produce artifacts labeled with the requested commit and pass resume validation.

   The launcher also permits an existing `RUN` and extracts over its existing repository at [lines 53 and 99–100](</C:/Users/sahas/Github Repos/composite-mobo/run_unicorn.sh:53>). Reusing a directory can retain files deleted from the new archive or race an existing campaign. There is no fresh-run guard or lock.

   These are potential whole-campaign failures, though I found no evidence they occurred in your 160-task run.

3. **Definitely wrong as a screening gate — `advantage` is unstable and incomplete.**  
   [diagnose_composite.py:52–70](</C:/Users/sahas/Github Repos/composite-mobo/diagnose_composite.py:52>) uses one Sobol split and averages errors across objectives. Running seeds 0–4 produced:

   - Reizman: **+39.3% to +53.6%**
   - SNAr: **−57.0% to +45.6%**

   For SNAr seed 2, one objective improved by 43%, while the other’s standardized error increased by 110%; averaging concealed that failure.

   The ratio at [line 81](</C:/Users/sahas/Github Repos/composite-mobo/diagnose_composite.py:81>) is also degenerate near zero: if both RMSEs are exactly zero it reports +100%; a tiny direct RMSE can produce an arbitrarily large negative score.

   Beyond duplicated components, it is blind to posterior variance, calibration, cross-output dependence, nonlinear Jensen effects, acquisition-relevant tails, and accuracy near the Pareto region. It should report per-objective results across several splits plus a sample-based proper score such as energy score or predictive coverage. Absolute RMSE differences should accompany—or replace—the unstable ratio.

4. **Definitely wrong API coverage — `composer()` does not support every realistic callable.**  
   [_accepts_inputs](</C:/Users/sahas/Github Repos/composite-mobo/solvers.py:228>) treats signature-inspection failure as one-argument and counts only positional parameters.

   Verified failures:

   - `compose(H, *, X)` is classified as one-argument and then fails because `X` is omitted.
   - C-implemented two-argument callables whose signatures are unavailable are treated as one-argument; `torch.add` reproduces this.

   Plain functions, lambdas, ordinary partials, bound methods, callable objects, `*args`, and the current Reizman/SNAr `compose(H, X)` functions work. Thus this did **not** affect the reported campaign.

5. **Definitely wrong — valid paired trials can be silently dropped.**  
   [load_traces](</C:/Users/sahas/Github Repos/composite-mobo/benchmark_common.py:969>) requires not only matching provenance within each direct/composite pair, but identical provenance across every retained trial via `family_metadata`. I constructed two individually valid, internally matched pairs with different commits; only one was returned.

   Local plotting gives no dropped-trial warning. The Slurm aggregate does subsequently require 20 traces and fails loudly, so your stated “20 paired trials” means this did not affect that aggregate.

6. **Lower-impact protocol defect — composite methods repeat expensive simulation calls.**  
   Composite qLogEHVI and STCH evaluate both `evaluate_components(X_initial)` and `evaluate(X_initial)`, even though `evaluate` recomputes those components. High-dimensional STCH and MORBO repeat both calls for adaptive points too. This does not change deterministic hypervolume values, but it undercounts actual simulator calls and biases timing comparisons against composite methods.

## Exact-input and STCH checks

These paths are correct for the current benchmarks:

- `composite_mobo`: BoTorch forwards the exact acquisition candidate `X` alongside every MC sample.
- `_scalarized_runs`: composite utility is `-STCH(compose(samples, candidate_X))`; `best_f` is the maximum of the same utility computed from observed `Y`.
- `_high_dim_scalarized_runs`: same alignment and incumbent convention.
- `_morbo`: `sample[i]` and `cand[i]` are row-aligned. Passing the entire `cand` tensor is correct, not an off-by-broadcast error.
- STCH initial rows are stored once, followed by weight blocks; sequential methods also store initial rows first. `X[:initial]` is correct for both families.
- Trial seeds change by `10_007 * trial`; STCH streams also vary by weight and evaluation count. I found no reused composite-only seed explaining the lower variance.

The Reizman composite posterior is pushed through clipping, an exact-input ratio, and then smooth Tchebycheff. That creates a different, non-Gaussian utility distribution from the direct objective-GP path. The STCH loss is therefore plausible model behavior. The diagnostic’s posterior-mean advantage does not predict that acquisition behavior.

## Verification

`python -m pytest -q`: **45 passed**, nothing failed. Two Torch JIT deprecation warnings were reported, plus an unrelated `requests` dependency-version warning. `bash -n` and ShellCheck also passed; `run_unicorn.sh` is LF-only and `constraint_args` is defined.

No files were modified. Tests ran at HEAD `b7010fb`; HEAD advanced from `82fde7f` during review, but the four target files and tests were unchanged between those commits.