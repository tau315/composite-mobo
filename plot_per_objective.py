"""Per-objective convergence for SNAr, in the units a chemist reads.

Hypervolume answers "how good is the front", which is the right headline but
hides which objective the composite model actually helps with. These two figures
plot the best value found so far for each objective separately, in physical
units rather than the normalized minimization form the solver uses.

    python plot_per_objective.py results/final-lowdim/results docs/figures
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import benchmark_snar as snar

SLUG = "summit_snar_2obj_4d"
PAIR = (("Direct qLogEHVI", "direct_qlogehvi"), ("Composite qLogEHVI", "composite_qlogehvi"))
COLORS = {"direct_qlogehvi": "#7B68B5", "composite_qlogehvi": "#C05A2E"}


def _objectives(root: Path, method: str, trials: int) -> np.ndarray:
    return np.array(
        [
            json.loads((root / SLUG / method / f"trial{t}.json").read_text())["Y"]
            for t in range(trials)
        ]
    )


def _figure(root: Path, out: Path, trials: int, index: int, title: str,
            to_physical, ylabel: str, better: str) -> None:
    """One objective, best-so-far, mean over trials with a standard-error band."""

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for label, method in PAIR:
        Y = _objectives(root, method, trials)[..., index]
        # Solvers minimize, so the incumbent is a running minimum; convert to
        # physical units afterwards so the axis reads the way a chemist expects.
        best = np.minimum.accumulate(Y, axis=1)
        physical = to_physical(best)
        evaluations = np.arange(1, physical.shape[1] + 1)
        mean = physical.mean(axis=0)
        sem = physical.std(axis=0, ddof=1) / np.sqrt(physical.shape[0])
        style = ":" if "Composite" in label else "-"
        ax.plot(evaluations, mean, style, color=COLORS[method], linewidth=2.4, label=label)
        ax.fill_between(evaluations, mean - sem, mean + sem,
                        color=COLORS[method], alpha=0.18, linewidth=0)
    ax.axvline(5, color="#666666", linestyle="--", linewidth=1.0, alpha=0.75,
               label="End of shared initial design")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Total function evaluations")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.01)
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.text(0.99, 0.02, better, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#555555")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/final-lowdim/results")
    figures = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/figures")
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    # f1 = 1 - STY/STY_SCALE and f2 = E/E_FACTOR_SCALE, both minimized.
    _figure(
        root, figures / "snar_yield.png", trials, 0,
        "SNAr: best space-time yield found",
        lambda f: (1.0 - f) * snar.STY_SCALE,
        "Space-time yield (kg m$^{-3}$ h$^{-1}$)",
        "higher is better",
    )
    _figure(
        root, figures / "snar_efactor.png", trials, 1,
        "SNAr: best E-factor found",
        lambda f: f * snar.E_FACTOR_SCALE,
        "E-factor (kg waste per kg product)",
        "lower is better",
    )


if __name__ == "__main__":
    main()
