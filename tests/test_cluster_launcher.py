from itertools import product
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "run_unicorn.sh"


def _bash():
    if os.name == "nt":
        git_bash = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe"
        if git_bash.exists():
            return str(git_bash)
    return "bash"


def test_launcher_writes_the_240_task_run_contract(tmp_path):
    archive = tmp_path / "repo.tgz"
    archive.write_bytes(b"test archive")
    sbatch_log = tmp_path / "sbatch.log"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "$SBATCH_LOG"
case "$*" in
  *setup.sbatch*) echo '101' ;;
  *array.sbatch*) echo '102' ;;
  *aggregate.sbatch*) echo '103' ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    run = tmp_path / "run"
    env = {
        **os.environ,
        # The launcher reads MAX_CONCURRENT from the environment, and Slurm
        # propagates a submitting shell's variables into the job that runs this
        # suite. Clear it so the test controls the array spec it asserts on.
        "MAX_CONCURRENT": "",
        "COMMIT": "abc1234",
        "RUN": run.as_posix(),
        "REPO_ARCHIVE": archive.as_posix(),
        "SBATCH": fake_sbatch.as_posix(),
        "SBATCH_LOG": sbatch_log.as_posix(),
    }

    completed = subprocess.run(
        [_bash(), LAUNCHER.as_posix()],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    tasks = [line.split("\t") for line in (run / "tasks.tsv").read_text().splitlines()]
    low = (
        "direct_qlogehvi",
        "composite_qlogehvi",
        "objective_gp_stch",
        "composite_stch",
    )
    # RCM46 is a high-suite problem, so it draws the high-dimensional methods.
    high = (
        "spherical_objective_stch",
        "spherical_composite_stch",
        "morbo",
        "composite_morbo",
    )
    expected = [
        [str(index), benchmark, str(trial), method]
        for index, (benchmark, trial, method) in enumerate(
            [
                (benchmark, trial, method)
                for benchmark, methods in (
                    ("benchmark_reizman", low),
                    ("benchmark_snar", low),
                    ("benchmark_rcm46", high),
                )
                for trial, method in product(range(20), methods)
            ]
        )
    ]
    assert tasks == expected
    assert len({tuple(task[1:]) for task in tasks}) == 240
    assert (run / "job_ids.tsv").read_text().splitlines() == [
        "stage\tjob_id",
        "setup\t101",
        "array\t102",
        "aggregate\t103",
    ]
    submissions = sbatch_log.read_text().splitlines()
    assert any("--array=0-239" in line.split() for line in submissions)
    # ... and that a concurrency cap, when asked for, rides on the same flag.
    capped = subprocess.run(
        [_bash(), LAUNCHER.as_posix()],
        cwd=ROOT,
        env={**env, "MAX_CONCURRENT": "100", "RUN": (tmp_path / "run2").as_posix()},
        capture_output=True,
        text=True,
        check=False,
    )
    assert capped.returncode == 0, capped.stdout + capped.stderr
    assert any(
        "--array=0-239%100" in line.split()
        for line in sbatch_log.read_text().splitlines()
    )
    assert any("--dependency=afterok:101" in line for line in submissions)
    # Every stage must be pinned to AVX-capable nodes: jaxlib is built with AVX
    # and botorch imports jax, so a task landing elsewhere dies at import before
    # it can even record a failure. An unset array silently expands to nothing
    # under `set -u`, so assert the flag actually reaches sbatch.
    assert sum("--constraint=avx" in line.split() for line in submissions) == 3
    assert any("--dependency=afterany:102" in line for line in submissions)

    launcher = LAUNCHER.read_text(encoding="utf-8")
    setup = (run / "setup.sbatch").read_text(encoding="utf-8")
    array = (run / "array.sbatch").read_text(encoding="utf-8")
    aggregate = (run / "aggregate.sbatch").read_text(encoding="utf-8")
    for script in (LAUNCHER, run / "setup.sbatch", run / "array.sbatch", run / "aggregate.sbatch"):
        syntax = subprocess.run(
            [_bash(), "-n", script.as_posix()],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr
    for script in (setup, array, aggregate):
        assert 'source "$1"' in script
    assert "set -euo pipefail" in launcher and "umask 077" in launcher
    header = launcher.index("printf 'stage\\tjob_id\\n'")
    assert header < launcher.index('setup_job=$(')
    assert launcher.count('>> "$RUN/job_ids.tsv"') == 3
    assert "#SBATCH --partition=default_partition" in setup
    assert "#SBATCH --cpus-per-task=2" in setup
    assert "#SBATCH --mem=8G" in setup
    assert "#SBATCH --time=01:00:00" in setup
    assert "/share/apps/software/anaconda3/etc/profile.d/conda.sh" in setup
    assert "python=3.12" in setup and "python -m pytest -q" in setup
    assert "rm -rf" not in setup
    assert '${REPO_ARCHIVE_SHA256,,}' in setup
    assert "#SBATCH --cpus-per-task=1" in array
    assert "#SBATCH --mem=4G" in array
    assert "#SBATCH --time=04:00:00" in array
    for name in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
        assert f"export {name}_NUM_THREADS=1" in array
    # Trials now come from the environment rather than being hardcoded, so
    # assert the plumbing rather than a literal count.
    for option in ('--trials "$TRIALS"', "--seed 0"):
        assert option in array
        assert option in aggregate
    assert "--trial " in array and "--method " in array
    assert "COMPOSITE_MOBO_COMMIT" in array
    assert "_validated_payload" in aggregate
    assert "load_traces" in aggregate
    assert "paired trials" in aggregate
    assert "len(trace_list) != args.trials" in aggregate
    assert "--summary-only" in aggregate
    assert "composite-mobo-$COMMIT.tgz" in aggregate
    assert not any(word in launcher.lower() for word in ("api_key", "password", "secret"))


def test_node_constraint_can_be_disabled(tmp_path):
    """CONSTRAINT= opts out, for an environment that does not need the pin."""

    archive = tmp_path / "repo.tgz"
    archive.write_bytes(b"test archive")
    sbatch_log = tmp_path / "sbatch.log"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$SBATCH_LOG"
echo 1
""",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    completed = subprocess.run(
        [_bash(), LAUNCHER.as_posix()],
        cwd=ROOT,
        env={
            **os.environ,
            "CONSTRAINT": "",
            "MAX_CONCURRENT": "",
            "COMMIT": "abc1234",
            "RUN": (tmp_path / "run").as_posix(),
            "REPO_ARCHIVE": archive.as_posix(),
            "SBATCH": fake_sbatch.as_posix(),
            "SBATCH_LOG": sbatch_log.as_posix(),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--constraint" not in sbatch_log.read_text()


def test_cluster_requirements_are_exactly_pinned():
    assert (ROOT / "requirements-cluster.txt").read_text().splitlines() == [
        "numpy==2.3.3",
        "scipy==1.15.2",
        "torch==2.12.0",
        "botorch==0.18.0",
        "gpytorch==1.15.2",
        "matplotlib==3.10.8",
        "pytest==8.4.1",
    ]


def test_trials_budget_and_benchmark_list_are_overridable(tmp_path):
    """A campaign must be pointable at the regime it means to measure."""

    archive = tmp_path / "repo.tgz"
    archive.write_bytes(b"test archive")
    sbatch_log = tmp_path / "sbatch.log"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "$SBATCH_LOG"
echo 1
""",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    run = tmp_path / "run"
    completed = subprocess.run(
        [_bash(), LAUNCHER.as_posix()],
        cwd=ROOT,
        env={
            **os.environ,
            "MAX_CONCURRENT": "",
            "TRIALS": "50",
            "EVALUATIONS": "20",
            "BENCHMARKS_OVERRIDE": "benchmark_reizman benchmark_snar",
            "COMMIT": "abc1234",
            "RUN": run.as_posix(),
            "REPO_ARCHIVE": archive.as_posix(),
            "SBATCH": fake_sbatch.as_posix(),
            "SBATCH_LOG": sbatch_log.as_posix(),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    tasks = [line.split("\t") for line in (run / "tasks.tsv").read_text().splitlines()]
    assert len(tasks) == 2 * 50 * 4
    assert {task[1] for task in tasks} == {"benchmark_reizman", "benchmark_snar"}
    assert any("--array=0-399" in line.split() for line in sbatch_log.read_text().splitlines())

    # Both the worker and the aggregate step must see the same settings, or the
    # aggregate revalidates artifacts against a configuration that never ran.
    run_env = (run / "run.env").read_text(encoding="utf-8")
    assert "TRIALS=50" in run_env and "EVALUATIONS=20" in run_env
    for stage in ("array", "aggregate"):
        script = (run / f"{stage}.sbatch").read_text(encoding="utf-8")
        assert '--evaluations "$EVALUATIONS"' in script
        assert '"${budget_args[@]}"' in script
