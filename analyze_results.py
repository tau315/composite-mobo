"""Paired direct-vs-composite statistics from a completed campaign.

Reports the endpoint comparison every benchmark supports, plus an anytime
comparison for the sequential family only.

    python analyze_results.py results/paper-lowdim/output/results

**The anytime comparison is restricted to qLogEHVI on purpose.** STCH branches
every scalarization weight from one shared initial design and stores the weight
blocks contiguously, so position k in an STCH trace is not the k-th chronological
evaluation: at position 7 the run has finished the initial design and taken two
steps for weight 0 alone, exploring one corner of the front. Reading that as
"performance at a budget of 7" would compare a partial sweep against a genuine
prefix. STCH is compared at its endpoint, where every weight has run.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats
import torch

from benchmark_common import dominated_hypervolume_trace

PAIRS = (
    ("qLogEHVI", "direct_qlogehvi", "composite_qlogehvi", True),
    ("STCH", "objective_gp_stch", "composite_stch", False),
)
ANYTIME_BUDGETS = (7, 10, 15, 20, 30)


def _problem_for(slug: str):
    """The benchmark module whose PROBLEM carries this slug, if it still exists."""

    for path in sorted(Path(".").glob("benchmark_*.py")):
        if path.stem == "benchmark_common":
            continue
        problem = getattr(importlib.import_module(path.stem), "PROBLEM", None)
        if problem is not None and problem.slug == slug:
            return problem
    return None


def _payloads(root: Path, slug: str, method: str, trials: int) -> list[dict]:
    return [
        json.loads((root / slug / method / f"trial{t}.json").read_text())
        for t in range(trials)
    ]


def _traces(root: Path, slug: str, method: str, trials: int) -> np.ndarray:
    return np.array([p["hypervolume"] for p in _payloads(root, slug, method, trials)])


def _check_comparable(root: Path, slug: str, methods: tuple[str, ...], trials: int) -> None:
    """Refuse to compare artifacts that did not come from one campaign.

    The stored hypervolume depends on the reference point, the budget, and the
    commit, none of which appear in the trace itself. Reading a directory that
    mixes them silently produces a plausible-looking table of nonsense -- the
    reason this check exists is that it happened: a stale directory reported
    SNAr hypervolumes near 5.93 under a reference point whose box maximum is
    1.155.
    """

    seen: dict[str, set] = {"config": set(), "commit": set()}
    for method in methods:
        for payload in _payloads(root, slug, method, trials):
            # Seed legitimately varies per trial; everything else must not.
            config = {k: v for k, v in payload["config"].items() if k != "seed"}
            seen["config"].add(json.dumps(config, sort_keys=True))
            seen["commit"].add(payload["metadata"]["git_commit"])
    if len(seen["config"]) > 1:
        raise SystemExit(
            f"{slug}: artifacts disagree on configuration; they are not one campaign"
        )
    if len(seen["commit"]) > 1:
        raise SystemExit(
            f"{slug}: artifacts come from {len(seen['commit'])} different commits"
        )

    # The stored trace depends on the reference point, which the config does not
    # record, so recompute one trace against the current benchmark. This is the
    # check that catches a directory written before the reference was retuned.
    problem = _problem_for(slug)
    if problem is None:
        return
    payload = _payloads(root, slug, methods[0], 1)[0]
    Y = torch.tensor(payload["Y"], dtype=torch.double)
    expected = dominated_hypervolume_trace(Y, problem.ref_point)
    if not np.allclose(expected, payload["hypervolume"], rtol=1e-9, atol=1e-9):
        raise SystemExit(
            f"{slug}: stored hypervolume does not match the current benchmark "
            f"(reference point {problem.ref_point.tolist()}); these artifacts are stale"
        )


def _compare(direct: np.ndarray, composite: np.ndarray) -> dict:
    """Paired statistics at one budget. Wilcoxon is the primary test."""

    delta = 100.0 * (composite.mean() - direct.mean()) / abs(direct.mean())
    # Only exactly-zero paired differences make a Wilcoxon test undefined.
    # Treating merely-close arrays as identical discards real results: a
    # synthetic pair passing np.allclose still had a true p-value of 1.9e-06.
    identical = bool(np.array_equal(direct, composite))
    return {
        "direct": direct.mean(),
        "direct_sem": direct.std(ddof=1) / np.sqrt(len(direct)),
        "composite": composite.mean(),
        "composite_sem": composite.std(ddof=1) / np.sqrt(len(composite)),
        "delta": delta,
        "wins": int((composite > direct).sum()),
        "trials": len(direct),
        "wilcoxon": 1.0 if identical else stats.wilcoxon(composite, direct).pvalue,
    }


def _line(label: str, r: dict) -> str:
    return (
        f"    {label:<16} direct={r['direct']:.5f}+-{r['direct_sem']:.5f}  "
        f"composite={r['composite']:.5f}+-{r['composite_sem']:.5f}  "
        f"delta={r['delta']:+6.2f}%  wins={r['wins']:>2}/{r['trials']}  "
        f"p={r['wilcoxon']:.1e}"
    )


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/paper-lowdim/output/results")
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    for slug in sorted(p.name for p in root.iterdir() if p.is_dir()):
        print(f"=== {slug} ===")
        for family, direct_key, composite_key, sequential in PAIRS:
            _check_comparable(root, slug, (direct_key, composite_key), trials)
            direct = _traces(root, slug, direct_key, trials)
            composite = _traces(root, slug, composite_key, trials)
            budget = direct.shape[1]
            if composite.shape[1] != budget:
                raise SystemExit(
                    f"{slug} {family}: arms used {budget} and {composite.shape[1]} "
                    "evaluations; they are not a matched comparison"
                )
            print(f"  {family} (budget {budget})")
            print(_line("final", _compare(direct[:, -1], composite[:, -1])))
            if not sequential:
                print(
                    "    (no anytime rows: STCH stores weight blocks contiguously, "
                    "so a prefix is not a budget)"
                )
                continue
            for at in ANYTIME_BUDGETS:
                if at >= budget:
                    continue
                print(_line(f"at {at} evals", _compare(direct[:, at - 1], composite[:, at - 1])))


if __name__ == "__main__":
    main()
