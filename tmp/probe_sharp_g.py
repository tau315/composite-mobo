"""Screening probe for the corrected criterion: smooth h, structure-creating g.

Contrast three known maps over the *same* smooth intermediate, to isolate what
property of g decides whether composite modeling pays:

  averaging   : g is a weighted mean of h            (expected: no gain / loss)
  sharp       : g is a sharply peaked response of h  (expected: large gain)
  exponential : g amplifies h through an exponential (expected: large gain)

The intermediate is an alloy-composition -> band-gap style map (Vegard's law
with bowing), which is genuinely smooth and low-dimensional. The sharp map is
the Shockley-Queisser detailed-balance efficiency, which peaks near 1.34 eV.
"""

import sys

import torch

sys.path.insert(0, ".")
from benchmark_common import BenchmarkProblem
from diagnose_composite import diagnose

JUNCTIONS = 3
GAP_RANGE = (0.5, 2.6)


def band_gaps(X: torch.Tensor, junctions: int) -> torch.Tensor:
    """Smooth composition -> band-gap map: linear mixing plus bowing."""

    X = X.double()
    d = X.shape[-1]
    per = max(d // junctions, 1)
    gaps = []
    for j in range(junctions):
        block = X[..., j * per : (j + 1) * per] if j < junctions - 1 else X[..., (junctions - 1) * per :]
        mean = block.mean(dim=-1)
        # Vegard linear term with a quadratic bowing correction: smooth, monotone-ish.
        low, high = GAP_RANGE
        gaps.append(low + (high - low) * (mean - 0.35 * mean * (1.0 - mean)))
    return torch.stack(gaps, dim=-1)


def shockley_queisser(gap: torch.Tensor) -> torch.Tensor:
    """Sharply peaked detailed-balance efficiency, maximal near 1.34 eV."""

    # Smooth analytic stand-in with the right shape: rises with absorbed
    # photons, falls with thermalization, peaked and narrow.
    absorbed = torch.exp(-((gap - 0.7) / 1.6).clamp_min(0.0) ** 2)
    thermalized = 1.0 - torch.exp(-2.6 * gap.clamp_min(0.0))
    return 0.45 * absorbed * thermalized * torch.exp(-((gap - 1.34) / 0.55) ** 2)


def make_problem(kind: str, dim: int, suite: str) -> BenchmarkProblem:
    def components(X: torch.Tensor) -> torch.Tensor:
        gaps = band_gaps(X, JUNCTIONS)
        # A smooth scarcity/cost proxy travels alongside as one more component.
        cost = X.double().square().mean(dim=-1, keepdim=True)
        return torch.cat((gaps, cost), dim=-1)

    if kind == "averaging":
        def compose(H: torch.Tensor) -> torch.Tensor:
            gaps, cost = H[..., :JUNCTIONS], H[..., JUNCTIONS]
            return torch.stack((-gaps.mean(dim=-1), cost), dim=-1)
    elif kind == "sharp":
        def compose(H: torch.Tensor) -> torch.Tensor:
            gaps, cost = H[..., :JUNCTIONS], H[..., JUNCTIONS]
            efficiency = shockley_queisser(gaps).sum(dim=-1)
            return torch.stack((-efficiency, cost), dim=-1)
    else:
        def compose(H: torch.Tensor) -> torch.Tensor:
            gaps, cost = H[..., :JUNCTIONS], H[..., JUNCTIONS]
            # Arrhenius-style amplification of a smooth intermediate.
            rate = torch.exp(-3.0 * gaps).sum(dim=-1)
            return torch.stack((-rate, cost), dim=-1)

    return BenchmarkProblem(
        name=f"{kind} g, d={dim}",
        slug=f"{kind}_{dim}d",
        dim=dim,
        num_objectives=2,
        suite=suite,
        evaluate_components=components,
        compose=compose,
        ideal=torch.tensor([-10.0, 0.0], dtype=torch.double),
        ref_point=torch.tensor([1.0, 1.5], dtype=torch.double),
    )


print(f"{'g type':<14}{'d':>6}{'p':>4}{'direct':>9}{'composite':>11}{'advantage':>12}")
print("-" * 56)
for kind in ("averaging", "sharp", "exponential"):
    for dim, suite in ((9, "low"), (30, "low"), (120, "high"), (400, "high")):
        problem = make_problem(kind, dim, suite)
        problem.validate()
        r = diagnose(problem)
        print(
            f"{kind:<14}{dim:>6}{r['components']:>4}{r['direct_rmse']:>9.3f}"
            f"{r['composite_rmse']:>11.3f}{r['advantage']:>11.1%}",
            flush=True,
        )
