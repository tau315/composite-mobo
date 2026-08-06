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

import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

PAIRS = (
    ("qLogEHVI", "direct_qlogehvi", "composite_qlogehvi", True),
    ("STCH", "objective_gp_stch", "composite_stch", False),
)
ANYTIME_BUDGETS = (7, 10, 15, 20, 30)


def _traces(root: Path, slug: str, method: str, trials: int) -> np.ndarray:
    return np.array(
        [
            json.loads((root / slug / method / f"trial{t}.json").read_text())[
                "hypervolume"
            ]
            for t in range(trials)
        ]
    )


def _compare(direct: np.ndarray, composite: np.ndarray) -> dict:
    """Paired statistics at one budget. Wilcoxon is the primary test."""

    delta = 100.0 * (composite.mean() - direct.mean()) / abs(direct.mean())
    identical = np.allclose(direct, composite)
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
            direct = _traces(root, slug, direct_key, trials)
            composite = _traces(root, slug, composite_key, trials)
            budget = direct.shape[1]
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
