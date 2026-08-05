"""Why composite advantage vanishes in high dimension, and what restores it.

Two candidate explanations, tested separately:

  budget          : 32 training points is simply too few at d=120, and the
                    high-dimensional suite actually spends ~400 evaluations
  active subspace : the intermediate depends on all d inputs, so it is no more
                    learnable than the objective; real structure-property
                    relations often depend on a few dominant descriptors

Both use the same sharply-peaked known map that scored 78.5% at d=9.
"""

import sys

import torch

sys.path.insert(0, ".")
from benchmark_common import BenchmarkProblem
from diagnose_composite import diagnose
from tmp.probe_sharp_g import shockley_queisser  # noqa: E402

JUNCTIONS = 3
GAP_RANGE = (0.5, 2.6)


def make_problem(dim: int, active: int | None) -> BenchmarkProblem:
    """``active=None`` means the intermediate depends on every input."""

    generator = torch.Generator().manual_seed(20260805)
    if active is not None:
        # A fixed sparse set of dominant descriptors per junction.
        picks = [
            torch.randperm(dim, generator=generator)[:active] for _ in range(JUNCTIONS)
        ]
    else:
        picks = None

    def components(X: torch.Tensor) -> torch.Tensor:
        X = X.double()
        gaps = []
        for j in range(JUNCTIONS):
            block = X[..., picks[j]] if picks is not None else X
            mean = block.mean(dim=-1)
            low, high = GAP_RANGE
            gaps.append(low + (high - low) * (mean - 0.35 * mean * (1.0 - mean)))
        gaps = torch.stack(gaps, dim=-1)
        cost = X.square().mean(dim=-1, keepdim=True)
        return torch.cat((gaps, cost), dim=-1)

    def compose(H: torch.Tensor) -> torch.Tensor:
        gaps, cost = H[..., :JUNCTIONS], H[..., JUNCTIONS]
        return torch.stack((-shockley_queisser(gaps).sum(dim=-1), cost), dim=-1)

    label = "all-inputs" if active is None else f"active={active}"
    return BenchmarkProblem(
        name=f"d={dim} {label}",
        slug=f"probe_{dim}_{label}",
        dim=dim,
        num_objectives=2,
        suite="high",
        evaluate_components=components,
        compose=compose,
        ideal=torch.tensor([-10.0, 0.0], dtype=torch.double),
        ref_point=torch.tensor([1.0, 1.5], dtype=torch.double),
    )


print(f"{'structure':<16}{'d':>6}{'n_train':>9}{'direct':>9}{'composite':>11}{'advantage':>12}")
print("-" * 63)
for active in (None, 8):
    for dim in (120, 400):
        for n_train in (32, 128, 256):
            problem = make_problem(dim, active)
            r = diagnose(problem, n_train=n_train, n_test=200)
            label = "all-inputs" if active is None else f"active={active}"
            print(
                f"{label:<16}{dim:>6}{n_train:>9}{r['direct_rmse']:>9.3f}"
                f"{r['composite_rmse']:>11.3f}{r['advantage']:>11.1%}",
                flush=True,
            )
