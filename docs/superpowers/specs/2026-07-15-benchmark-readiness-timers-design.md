# Benchmark Readiness and Timing Design

## Goal

Make the four-method ZDT/DTLZ study scientifically comparable, restartable, and measurable before launching the default 20-trial matrix. Preserve the existing problem formulas, method families, budgets, and plotting intent.

## Run Unit and Outputs

One independently resumable run is one `(problem, trial, method)` tuple. Stable method keys are `standard_qlogehvi`, `composite_qlogehvi`, `objective_gp_stch`, and `composite_stch`. Each run writes one atomic JSON artifact under `results/{problem}/{method}/trial{trial}.json`; an existing valid artifact is skipped. Failed runs write their traceback without stopping other runs.

Artifacts contain arguments, seed, commit and package versions, evaluated `X`, final objectives `Y`, optional components, weights and weight-run IDs, cumulative hypervolume, failure state, and timing data. Plotting and summaries load artifacts from disk instead of depending on in-memory completion.

## Evaluation Accounting

Every recorded point consumes exactly one problem evaluation. Direct methods call `problem.evaluate(X)`. Composite methods call `problem.components(X)` once and derive `Y = problem.compose(C, X)`; they do not call `evaluate` again. Composition equivalence belongs in tests, not the measured optimization loop.

Known component domains are enforced in the outer map: ZDT reconstructs `g >= 1`, and DTLZ2 reconstructs `g >= 0`. This prevents impossible component-posterior samples from creating artificial acquisition improvement while preserving exact composition for observed components.

ZDT3 uses its true componentwise ideal point `(0, -0.7733690123)` for STCH; the other included problems retain `(0, 0)`. Reference points remain unchanged.

## Timing

Each result stores accumulated seconds for:

- initial-design generation;
- initial direct or component evaluation;
- initial composition;
- GP fitting;
- acquisition construction;
- acquisition optimization;
- BO direct or component evaluation;
- BO composition;
- hypervolume calculation;
- end-to-end total.

It also stores one wall-clock duration per recorded observation. Non-applicable phase totals are zero, allowing uniform aggregation across methods. Timers use `time.perf_counter()` and measure synchronization-visible host time.

## Protocol and Comparability

The existing defaults remain: 20 matched trials; five initial points and 40 total evaluations for ZDT1, ZDT3, and DTLZ2; three initial points and 30 total evaluations for ZDT2.

qLogEHVI methods share the same initial design and budget. STCH methods share the same weights, per-weight seeds, initial designs, and budget. STCH output is interleaved by local evaluation index across weight runs before calculating its combined hypervolume trace. Its initial-design boundary is `n_init * n_weights`, not `n_init`.

Primary claims compare direct and composite variants within the same acquisition family. Cross-family qLogEHVI-versus-STCH curves remain descriptive because STCH pools independent preference-weight runs while qLogEHVI maintains one joint history.

## Runner and Failure Handling

Add `--trial` and `--method` selectors so a shell or cluster scheduler can parallelize independent runs. Keep the Python runner sequential; an internal multiprocessing framework is unnecessary. Resume validates required fields and expected evaluation count before skipping an artifact.

Acquisition fallback catches supported candidate-generation failures and rejects every non-finite value, including positive infinity. It raises only when no finite fallback candidate exists.

## Verification

Small automated checks must prove:

1. composite methods call the component oracle once per recorded point;
2. observed composition remains exact and adversarial samples respect known domains;
3. each problem exposes the intended ideal and reference points;
4. all timing keys exist, are finite, and are nonnegative;
5. STCH ordering places all initial points before BO points and preserves final hypervolume;
6. artifact writes, resume, and failure isolation work;
7. each solver completes a small BoTorch smoke run with the expected budget.

The final gate is the full test suite plus one resumed multi-method smoke benchmark.
