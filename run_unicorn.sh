#!/usr/bin/env bash
set -euo pipefail
umask 077

: "${COMMIT:?Set COMMIT to the exact archived git commit}"
: "${RUN:?Set RUN to an NFS-backed run directory}"
: "${REPO_ARCHIVE:?Set REPO_ARCHIVE to the source archive path}"

ENV="${ENV:-$HOME/composite-mobo/env}"
SBATCH="${SBATCH:-sbatch}"
REPO_ARCHIVE_SHA256="${REPO_ARCHIVE_SHA256:-}"
# botorch imports jax, and jaxlib is built with AVX: on a node whose CPU lacks
# it, importing botorch raises before any of our code runs, so the task dies
# without even leaving a failure artifact. Roughly 70 nodes here are unlabelled
# for avx. Set CONSTRAINT= to disable if a future environment does not need it.
CONSTRAINT="${CONSTRAINT-avx}"
readonly COMMIT RUN REPO_ARCHIVE ENV SBATCH REPO_ARCHIVE_SHA256 CONSTRAINT
# Benchmarks to run, as module stems. Each contributes TRIALS x 4 array tasks.
readonly -a BENCHMARKS=(
  benchmark_reizman
  benchmark_snar
)
# Method keys are fixed by `_solver_jobs` in benchmark_common.py: one set per
# suite. `--list-methods` on any benchmark prints the set it will use.
readonly -a LOW_METHODS=(
  direct_qlogehvi
  composite_qlogehvi
  objective_gp_stch
  composite_stch
)
readonly -a HIGH_METHODS=(
  spherical_objective_stch
  spherical_composite_stch
  morbo
  composite_morbo
)
readonly -a HIGH_BENCHMARKS=(
  benchmark_dtlz2_100d
  benchmark_dtlz2_600d
  benchmark_cort_tg119
  benchmark_rcm40
  benchmark_rcm46
)
readonly TRIALS=20
readonly TASK_COUNT=$((${#BENCHMARKS[@]} * TRIALS * 4))

[[ -f "$REPO_ARCHIVE" ]] || { echo "Archive not found: $REPO_ARCHIVE" >&2; exit 2; }
if [[ -n "${MAX_CONCURRENT:-}" && ! "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi

mkdir -p "$RUN"/{repo,logs,metadata,output}
tasks_tmp="$RUN/tasks.tsv.tmp"
: > "$tasks_tmp"
task_id=0
for benchmark in "${BENCHMARKS[@]}"; do
  methods=("${LOW_METHODS[@]}")
  for high in "${HIGH_BENCHMARKS[@]}"; do
    [[ "$benchmark" == "$high" ]] && methods=("${HIGH_METHODS[@]}")
  done
  for ((trial = 0; trial < TRIALS; trial++)); do
    for method in "${methods[@]}"; do
      printf '%d\t%s\t%d\t%s\n' "$task_id" "$benchmark" "$trial" "$method" >> "$tasks_tmp"
      ((task_id += 1))
    done
  done
done
[[ "$task_id" -eq "$TASK_COUNT" ]]
mv "$tasks_tmp" "$RUN/tasks.tsv"

{
  printf 'COMMIT=%q\n' "$COMMIT"
  printf 'RUN=%q\n' "$RUN"
  printf 'REPO_ARCHIVE=%q\n' "$REPO_ARCHIVE"
  printf 'REPO_ARCHIVE_SHA256=%q\n' "$REPO_ARCHIVE_SHA256"
  printf 'ENV=%q\n' "$ENV"
} > "$RUN/run.env"

cat > "$RUN/setup.sbatch" <<'SETUP'
#!/usr/bin/env bash
#SBATCH --job-name=composite-mobo-setup
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --requeue
set -euo pipefail
umask 077

source "$1"
actual_hash=$(sha256sum "$REPO_ARCHIVE" | awk '{print $1}')
expected_hash=${REPO_ARCHIVE_SHA256,,}
if [[ -n "$expected_hash" && "$actual_hash" != "$expected_hash" ]]; then
  echo "Archive SHA-256 mismatch" >&2
  exit 1
fi

mkdir -p "$RUN/repo" "$RUN/metadata"
tar -xzf "$REPO_ARCHIVE" -C "$RUN/repo"
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
if [[ ! -x "$ENV/bin/python" ]]; then
  conda create -y -p "$ENV" python=3.12
fi
conda activate "$ENV"
python -m pip install -r "$RUN/repo/requirements-cluster.txt"
(
  cd "$RUN/repo"
  python -m pytest -q
)
{
  printf 'commit=%s\n' "$COMMIT"
  printf 'archive=%s\n' "$REPO_ARCHIVE"
  printf 'archive_sha256=%s\n' "$actual_hash"
  printf 'supplied_archive_sha256=%s\n' "$REPO_ARCHIVE_SHA256"
} > "$RUN/metadata/provenance.txt"
python -m pip freeze > "$RUN/metadata/pip-freeze.txt"
SETUP

cat > "$RUN/array.sbatch" <<'ARRAY'
#!/usr/bin/env bash
#SBATCH --job-name=composite-mobo
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --requeue
set -euo pipefail
umask 077

source "$1"
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export COMPOSITE_MOBO_COMMIT="$COMMIT"

task=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUN/tasks.tsv")
[[ -n "$task" ]] || { echo "No task for index $SLURM_ARRAY_TASK_ID" >&2; exit 2; }
IFS=$'\t' read -r task_id benchmark trial method <<< "$task"
[[ "$task_id" == "$SLURM_ARRAY_TASK_ID" ]]

cd "$RUN/repo"
python "${benchmark}.py" \
  --trials 20 \
  --trial "$trial" \
  --method "$method" \
  --seed 0 \
  --results-dir "$RUN/output/results"

slug=$(python -c "import $benchmark; print($benchmark.PROBLEM.slug)")
artifact="$RUN/output/results/$slug/$method/trial$trial.json"
python - "$artifact" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
print(f"total runtime: {payload['timing']['total_seconds']:.3f}s")
PY
ARRAY

cat > "$RUN/aggregate.sbatch" <<'AGGREGATE'
#!/usr/bin/env bash
#SBATCH --job-name=composite-mobo-aggregate
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --requeue
set -euo pipefail
umask 077

source "$1"
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"
export COMPOSITE_MOBO_COMMIT="$COMMIT"
cd "$RUN/repo"

python - "$RUN/tasks.tsv" "$RUN/output/results" "$RUN/output/timing_fallback_summary.json" <<'PY'
import importlib
import json
from pathlib import Path
import sys

import benchmark_common

tasks_path, results_dir, summary_path = map(Path, sys.argv[1:])
rows = [line.split("\t") for line in tasks_path.read_text(encoding="utf-8").splitlines()]


def parsed_args(module):
    """Reproduce exactly the settings the array tasks ran with."""

    parser = benchmark_common._argument_parser(module.PROBLEM)
    args = parser.parse_args(
        ["--trials", "20", "--seed", "0", "--results-dir", str(results_dir)]
    )
    if args.iterations is None:
        args.iterations = args.evaluations - args.initial
    if args.per_weight is None:
        args.per_weight = (args.evaluations - args.initial) // args.weights
    return args


modules = {}
bad = []
payloads = []
for expected_index, row in enumerate(rows):
    if len(row) != 4 or row[0] != str(expected_index):
        bad.append(str(expected_index))
        continue
    _, benchmark_name, trial_text, method = row
    trial = int(trial_text)
    if benchmark_name not in modules:
        modules[benchmark_name] = importlib.import_module(benchmark_name)
    module = modules[benchmark_name]
    problem = module.PROBLEM
    args = parsed_args(module)
    path = results_dir / problem.slug / method / f"trial{trial}.json"
    payload = benchmark_common._validated_payload(
        path,
        problem,
        benchmark_common._job_config(problem, args, args.seed + 10_007 * trial),
        allow_foreign_metadata=True,
    )
    if payload is None:
        bad.append(str(expected_index))
    else:
        payloads.append(payload)
if bad:
    print("bad/retry indices: " + ",".join(dict.fromkeys(bad)))
    raise SystemExit(1)
print("bad/retry indices: none")

incompatible = []
for benchmark_name, module in modules.items():
    args = parsed_args(module)
    _, panels = benchmark_common._solver_jobs(module.PROBLEM, args, args.seed)
    expected_labels = {name for _, names in panels for name in names}
    traces = benchmark_common.load_traces(module.PROBLEM, args, panels)
    if set(traces) != expected_labels or any(
        len(trace_list) != args.trials for trace_list in traces.values()
    ):
        incompatible.append(benchmark_name)
if incompatible:
    print("incomplete paired trials: " + ",".join(incompatible))
    raise SystemExit(1)
print("paired trials: 20 per family/benchmark")
summary = {
    "artifacts": len(payloads),
    "timing_totals_seconds": {
        key: sum(payload["timing"][key] for payload in payloads)
        for key in benchmark_common.RESULT_TIMING_KEYS
    },
    "acquisition_fallbacks": {
        "total": sum(payload["acquisition_fallbacks"] for payload in payloads),
        "artifacts": sum(payload["acquisition_fallbacks"] > 0 for payload in payloads),
    },
}
summary_path.write_text(
    json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

while IFS=$'\t' read -r _ benchmark _ _; do
  echo "$benchmark"
done < "$RUN/tasks.tsv" | sort -u | while read -r benchmark; do
  python "${benchmark}.py" \
    --trials 20 \
    --seed 0 \
    --results-dir "$RUN/output/results" \
    --summary-only \
    --output "$RUN/output/hypervolume_$benchmark.png"
done

tar -czf "$RUN/composite-mobo-$COMMIT.tgz" -C "$RUN" \
  output tasks.tsv metadata run.env setup.sbatch array.sbatch aggregate.sbatch job_ids.tsv logs
AGGREGATE

chmod 700 "$RUN/setup.sbatch" "$RUN/array.sbatch" "$RUN/aggregate.sbatch"
array_spec="0-$((TASK_COUNT - 1))"
if [[ -n "${MAX_CONCURRENT:-}" ]]; then
  array_spec+="%$MAX_CONCURRENT"
fi

printf 'stage\tjob_id\n' > "$RUN/job_ids.tsv"
setup_job=$("$SBATCH" --parsable \
  --output="$RUN/logs/setup-%j.out" --error="$RUN/logs/setup-%j.err" \
  "${constraint_args[@]}" "$RUN/setup.sbatch" "$RUN/run.env")
setup_job=${setup_job%%;*}
printf 'setup\t%s\n' "$setup_job" >> "$RUN/job_ids.tsv"
array_job=$("$SBATCH" --parsable --dependency="afterok:$setup_job" \
  --array="$array_spec" \
  --output="$RUN/logs/array-%A_%a.out" --error="$RUN/logs/array-%A_%a.err" \
  "${constraint_args[@]}" "$RUN/array.sbatch" "$RUN/run.env")
array_job=${array_job%%;*}
printf 'array\t%s\n' "$array_job" >> "$RUN/job_ids.tsv"
aggregate_job=$("$SBATCH" --parsable --dependency="afterany:$array_job" \
  --output="$RUN/logs/aggregate-%j.out" --error="$RUN/logs/aggregate-%j.err" \
  "${constraint_args[@]}" "$RUN/aggregate.sbatch" "$RUN/run.env")
aggregate_job=${aggregate_job%%;*}
printf 'aggregate\t%s\n' "$aggregate_job" >> "$RUN/job_ids.tsv"

printf 'run: %s\nsetup job: %s\narray job: %s\naggregate job: %s\narchive: %s\n' \
  "$RUN" "$setup_job" "$array_job" "$aggregate_job" "$RUN/composite-mobo-$COMMIT.tgz"
