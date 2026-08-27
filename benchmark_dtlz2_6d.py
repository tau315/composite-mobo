"""Six-dimensional, two-objective DTLZ2 composition benchmark.

Paper protocol: 5 initial plus 40 adaptive design evaluations (45 total).
The raw response repeats the
distance term so each objective has its own component group, exactly matching
the graph h1=(g, cos(pi*x1/2)), h2=(g, sin(pi*x1/2)).
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark


DIM = 6


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    X = X.double()
    g = (X[..., 1:] - 0.5).square().sum(dim=-1)
    angle = 0.5 * torch.pi * X[..., 0]
    return torch.stack((g, angle.cos(), g, angle.sin()), dim=-1)


def compose(H: torch.Tensor) -> torch.Tensor:
    f1 = (1.0 + H[..., 0]) * H[..., 1]
    f2 = (1.0 + H[..., 2]) * H[..., 3]
    return torch.stack((f1, f2), dim=-1)


PROBLEM = BenchmarkProblem(
    name="DTLZ2 (2 objectives, 6 dimensions, 45 evaluations)",
    slug="dtlz2_2obj_6d",
    dim=DIM,
    num_objectives=2,
    num_components=4,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 1.1, dtype=torch.double),
    exact_max_hypervolume=1.1**2 - torch.pi / 4.0,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
