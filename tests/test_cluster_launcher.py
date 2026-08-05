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


def test_launcher_writes_the_320_task_run_contract(tmp_path):
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
    expected = [
        [str(index), benchmark, str(trial), method]
        for index, (benchmark, trial, method) in enumerate(
            product(
                (
                    "benchmark_dtlz2",
                    "benchmark_snar",
                    "benchmark_nanoparticle_rgb",
                    "benchmark_penicillin",
                ),
                range(20),
                (
                    "direct_qlogehvi",
                    "composite_qlogehvi",
                    "objective_gp_stch",
                    "composite_stch",
                ),
            )
        )
    ]
    assert tasks == expected
    assert len({tuple(task[1:]) for task in tasks}) == 320
    assert (run / "job_ids.tsv").read_text().splitlines() == [
        "stage\tjob_id",
        "setup\t101",
        "array\t102",
        "aggregate\t103",
    ]
    submissions = sbatch_log.read_text().splitlines()
    assert any("--array=0-319" in line.split() for line in submissions)
    assert any("--dependency=afterok:101" in line for line in submissions)
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
    for option in ("--trials 20", "--seed 0"):
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


def test_cluster_requirements_are_exactly_pinned():
    assert (ROOT / "requirements-cluster.txt").read_text().splitlines() == [
        "numpy==2.3.3",
        "torch==2.12.0",
        "botorch==0.18.0",
        "gpytorch==1.15.2",
        "matplotlib==3.10.8",
        "pytest==8.4.1",
    ]
