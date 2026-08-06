## Ranked findings

### Critical — `EVALUATIONS=20` does not constrain low-suite STCH to 20 evaluations

[run_unicorn.sh](</C:/Users/sahas/Github Repos/composite-mobo/run_unicorn.sh:160>) passes only `--evaluations`. But [benchmark_common.py](</C:/Users/sahas/Github Repos/composite-mobo/benchmark_common.py:619>) retains the low-suite default `--per-weight=10`.

For Reizman and SNAr:

```text
initial=5, weights=4, per_weight=10
STCH total = 5 + 4×10 = 45
qLogEHVI total = 5 + (20−5) = 20
```

The aggregate reproduces exactly the same configuration, so validation succeeds instead of detecting this. Thus the completed “20-evaluation” campaign’s STCH arms used 45 evaluations. The largest equal STCH allocation not exceeding 20 would be 17 evaluations (`5 + 4×3`).

This directly affects published STCH numbers.

### High — the reference-point invariance claim is false

[benchmark_snar.py](</C:/Users/sahas/Github Repos/composite-mobo/benchmark_snar.py:183>) says tightening the reference cannot change paired comparisons because a shared offset cancels. Changing a reference point is not equivalent to subtracting the initial-design hypervolume.

Recomputing the local 20-trial SNAr artifacts:

| Family | Old ref p / wins | New ref p / wins |
|---|---:|---:|
| qLogEHVI | 0.01923 / 16 | 0.01718 / 14 |
| STCH | 0.92728 / 11 | 1.00000 / 12 |

Mean paired differences also changed. More importantly, qLogEHVI uses the reference inside acquisition construction, so rerunning under the new reference can change the trajectory itself.

### High — `analyze_results.py` accepts stale or incompatible artifacts silently

[_traces](</C:/Users/sahas/Github Repos/composite-mobo/analyze_results.py:33>) reads only stored hypervolume arrays. It does not verify configuration, commit, reference point, shared initial design, or rederive hypervolume.

Concrete result: running it on `paper-lowdim-v2` reports SNAr hypervolume around `5.93`, which came from reference `(2.5, 2.5)`; the current `(1.05, 1.10)` box has maximum possible volume `1.155`. There is no warning.

The cluster aggregate does perform proper validation, so a pristine aggregated directory is safe. The analyzer itself is not safe against stale or mixed results.

### Medium — the SNAr floor is active, not “far below” produced concentrations

[benchmark_snar.py](</C:/Users/sahas/Github Repos/composite-mobo/benchmark_snar.py:28>) and [the clamp](</C:/Users/sahas/Github Repos/composite-mobo/benchmark_snar.py:125>) affect species 0 frequently:

- 65,243 / 262,144 Sobol designs hit the floor: 24.9%.
- 101 / 1,900 local composite evaluations hit it: 5.3%.
- The feasible corner `X=(1,1,1,1)` produces `3.1e-115`, far below `1e-12`.

Because each component GP is independently standardized, `-27.6` does not distort the scaling of the other species. It does create a large censored plateau within the species-0 GP. That may affect acquisition even though species 0 contributes negligibly to the objective at those concentrations.

A physically justified floor is preferable. A smooth `log(c + floor)`-style transform would also avoid the hard plateau.

### Medium — SNAr’s reference is not worse than every attainable point

[benchmark_snar.py](</C:/Users/sahas/Github Repos/composite-mobo/benchmark_snar.py:186>) uses `(1.05, 1.10)`. Differential evolution found:

```text
X = (1, 1, 1, 1)
objectives = (0.997208, 1.694894)
```

So the second coordinate exceeds `1.10`; the strict claim that every attainable point clears the reference is false. The 4,096-point sweep missed a narrow boundary region.

However, this violating corner is badly dominated. A 65,536-point approximate Pareto front had nadir around `(0.762, 0.0183)`, comfortably inside the reference. Therefore the current reference appears adequate for measuring Pareto-relevant hypervolume, even though it does not dominate the entire feasible image.

Reizman is correct: a 1,048,576-point Sobol sweep plus differential evolution found maxima `(0.737198, 0.904102)`, below `(0.75, 0.95)`.

### Medium — `np.allclose` can replace a significant Wilcoxon result with `p=1`

[analyze_results.py](</C:/Users/sahas/Github Repos/composite-mobo/analyze_results.py:48>) treats approximately equal arrays as identical. A synthetic 20-pair example satisfying `np.allclose` produced a real Wilcoxon p-value of `1.91e-6`, while `_compare` would report `1.0`.

Only exactly zero paired differences should trigger the special case. The current local endpoint results do not hit this bug.

## Requested checks that are correct

- The log/exp objective round trip is numerically sound: over 262,144 Sobol points, maximum absolute error was `1.80e-13` and maximum relative error `1.95e-13`.
- That is acceptable numerical noise for reporting objective values, but it does not guarantee an identical deterministic BO trajectory. Pre/post-log composite campaigns are different algorithms because their GP and acquisition distributions changed.
- Concentration posterior samples are log-normal. The objectives are not generally log-normal: STY is affine/clipped log-normal, while E-factor contains sums and ratios of log-normal variables. MC qLogEHVI can handle that induced distribution correctly.
- Restricting STCH reporting to endpoints is conservative and defensible. The literal explanation is inaccurate: [the solver](</C:/Users/sahas/Github Repos/composite-mobo/solvers.py:479>) does store actual execution order. Its prefixes are real budgets, but they depend on arbitrary weight-block ordering and represent incomplete scalarization sweeps.
- Subtracting the same trial-specific initial value is exactly invariant for a paired test:
  `(Cᵢ−Iᵢ)−(Dᵢ−Iᵢ)=Cᵢ−Dᵢ`.
  Empirically, all four local family/benchmark Wilcoxon p-values were bit-for-bit unchanged.
- Percentage deltas are not invariant to that subtraction because the denominator changes. `abs(direct.mean())` is harmless for nonnegative hypervolume, but the result is a relative difference of means, not a mean paired percentage.
- The Wilcoxon pairing works for either 20 or 50 trials. The analyzer does not correct its many anytime tests for multiplicity; those p-values should be called exploratory or adjusted.
- `BENCHMARKS_OVERRIDE` is safe for a single-line whitespace-separated list: `read -a` does not perform glob expansion.
- Worker and aggregate receive the same `TRIALS`/`EVALUATIONS` values.
- `budget_args=()` is safe with empty `EVALUATIONS` under the tested Bash 5.3. Bash before 4.4 had known empty-array/`set -u` differences, so cluster Bash version matters.

## Tests and workspace

`python -m pytest -q`: **49 passed**, no failures, in 35.80 seconds. Two Torch JIT deprecation warnings were reported, plus an external Requests dependency-version warning.

I made no file edits. The worktree began clean; during review an uncommitted one-line change appeared in `run_unicorn.sh`, changing the aggregate message from hardcoded `20` to `args.trials`. I left it untouched.