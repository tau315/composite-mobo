The implementation is close, but not an exact Summit reproduction. Two concrete issues remain.

## Findings

1. **Incorrect clipping order** — [reizman_emulator.py:119](</C:/Users/sahas/Github Repos/composite-mobo/reizman_emulator.py:119>)

   Summit unstandardizes and clips **each predictor**, then averages. Your code averages unbounded predictors and clips once at [line 127](</C:/Users/sahas/Github Repos/composite-mobo/reizman_emulator.py:127>).

   On 2,056 P1-L2 designs:

   - Maximum yield difference: `0.2241804984`
   - Mean difference: `0.0005541716`
   - Worst design: `(477.233 s, 92.927 °C, 2.43849 mol%)`
   - Summit yield: `95.46583`
   - Local yield: `95.6900105`
   - Individual predictor range: `[91.6627, 101.12092]`

   Other catalysts expose this more strongly: maximum yield difference reached `5.3242`; TON-head difference reached `3.7710`.

2. **`200` is a ceiling, not the demonstrated supremum** — [benchmark_reizman.py:34](</C:/Users/sahas/Github Repos/composite-mobo/benchmark_reizman.py:34>)

   `100 / 0.5 = 200` combines independent bounds; it does not prove the emulator can produce 100% yield at minimum loading. Three independent differential-evolution runs converged to:

   - Recomputed TON: `149.5951970`
   - Conditions: `445.19098 s`, `103.33217 °C`, `0.5 mol%`
   - Summit yield: `74.7975998`

   Thus `200` is a valid conservative normalization bound, but the “exactly largest value reachable” comment is unsupported and apparently false.

3. **Exact Summit numerical tolerance is not met** — [reizman_emulator.py:81](</C:/Users/sahas/Github Repos/composite-mobo/reizman_emulator.py:81>)

   Excluding clipping-order effects, maximum deviations from Summit were:

   - Yield: `1.4960e-5`
   - TON head: `1.2616e-5`

   This is minor numerical drift from evaluating float32 weights in float64 and using a newer Torch runtime, but it exceeds the requested `~1e-6`.

## Claims verified

- Feature layout is exactly 3 numeric columns followed by 8 one-hot columns.
- Continuous order is exactly `t_res`, `temperature`, `catalyst_loading`.
- Explicit string categories retain domain order. P1-L2 is one-hot index `2`.
- Network is `11 → 512 ReLU → 2`.
- Every predictor has separate input/output scalers and is unstandardized before ensemble aggregation.
- All seven vendored artifacts are byte-identical to Summit commit `1de682d`.
- The released-row head inconsistency is confirmed: maximum `42.9122156`.
- `compose(H, X)` correctly handled `(7,3,5,1)` with `X=(5,3)`, producing `(7,3,5,2)`; recovered TON matched `yield/loading` within `1.42e-14`.
- Other case numbers fail because only case 4 is vendored; other case-4 catalysts map correctly, subject to the clipping defect.
- `weights_only=False` is unnecessary. All five files load successfully with `weights_only=True`; no conversion is needed.

SNAr is correct: old 6 versus new 5 components, species and duplicate product were bit-identical, objectives were bit-identical (`max diff 0.0`) on 259 designs, and MC broadcasting passed.

Full suite: `45 passed`.

I made no repository edits. During review, HEAD advanced externally to `19dad6e`, and `benchmark_reizman.py` acquired an unstaged `REF_POINT` change.

