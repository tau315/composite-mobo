"""Two-objective, six-dimensional DTLZ2 composite benchmark."""

import math

import torch

from benchmark_common import BenchmarkProblem, run_benchmark


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    X = X.double()
    distance = (X[..., 1:] - 0.5).square().sum(dim=-1)
    angle = torch.pi * X[..., 0] / 2.0
    return torch.stack((distance, angle.cos(), distance, angle.sin()), dim=-1)


def compose(H: torch.Tensor) -> torch.Tensor:
    distance_1 = H[..., 0].clamp_min(0)
    angle_1 = H[..., 1].clamp(0.0, 1.0)
    distance_2 = H[..., 2].clamp_min(0)
    angle_2 = H[..., 3].clamp(0.0, 1.0)
    return torch.stack(
        ((1.0 + distance_1) * angle_1, (1.0 + distance_2) * angle_2),
        dim=-1,
    )


PROBLEM = BenchmarkProblem(
    name="DTLZ2 (2 objectives, 6 dimensions)",
    slug="dtlz2_2obj_6d",
    dim=6,
    num_objectives=2,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 2.5, dtype=torch.double),
    exact_max_hypervolume=2.5**2 - math.pi / 4.0,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
