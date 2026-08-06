| n_train | n_test | d | m | total p | direct_rmse | component_rmse | composite_rmse | advantage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 128 | 6084 | 2 | 12 | 0.990837819 | 0.993560983 | 0.998956008 | -0.00819326 (-0.82%) |
| 128 | 128 | 6084 | 2 | 12 | 1.012730353 | 0.999795807 | 1.001178524 | 0.01140662 (+1.14%) |
| 256 | 128 | 6084 | 2 | 12 | 0.991390559 | 1.001479146 | 0.994316092 | -0.00295094 (-0.30%) |

Here `p=12` overall: each wavelength/objective uses three complex S-parameters, hence `p_i=6` real components after splitting real and imaginary parts.

### Timing and bottlenecks

- 384 unique Ceviche evaluations: mean **3.001 s**, median **2.796 s**, range **2.248–4.842 s** on CPU.
- Total gate wall time: **1,172.3 s (19m 32s)**, of which **1,152.2 s (98.3%)** was simulation.
- Non-simulation overhead, including fitting all direct and component GPs:
  - `n_train=64`: ≤2.66 s
  - `n_train=128`: ≤4.86 s
  - `n_train=256`: ≤12.57 s
- Thus simulation dominated this batched gate. However, at 256 points the combined GP work already exceeds one new three-second simulation, so GP fitting would likely dominate each sequential BO iteration. Each GP does have **6,084 ARD lengthscales**.
- Memoization only avoided recomputing overlapping Sobol points between training sizes; it did not change any designs or model settings.

### Validation and numerical notes

- [`tmp/ceviche_wdm.py`](<C:/Users/sahas/Github Repos/composite-mobo/tmp/ceviche_wdm.py>) exposes `PROBLEM`; no tracked benchmarks were modified.
- `problem.validate()` passed.
- `(7,5,12)` component samples compose to `(7,5,2)`.
- `compose(evaluate_components(X))` and `evaluate(X)` were bitwise equal.
- All tensors were `torch.double`; all S-parameters and objectives were finite.
- No GP fitting failures, ill-conditioning warnings, or numerical exceptions occurred.
- There is no exact-input opportunity: every design coordinate is an unknown pixel density entering the electromagnetic solve.

**Verdict: REJECT — both fixed rejection criteria fail at `n_train=256`: `component_rmse=1.0015 ≥ 0.9` and `advantage=-0.30% ≤ 0`. The S-parameters are effectively no more predictable than their mean.**

