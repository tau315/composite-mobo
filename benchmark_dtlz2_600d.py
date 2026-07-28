"""Two-objective DTLZ2 benchmark in 600 input dimensions."""

import math

import torch

from benchmark_common import BenchmarkProblem, run_benchmark


DIM = 600
REFERENCE_VALUE = 1.0 + 0.25 * (DIM - 1) + 0.1


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Return objective-specific distance and angular intermediates."""

    X = X.double()
    distance = (X[..., 1:] - 0.5).square().sum(dim=-1)
    angle = torch.pi * X[..., 0] / 2.0
    return torch.stack((distance, angle.cos(), distance, angle.sin()), dim=-1)


def compose(H: torch.Tensor) -> torch.Tensor:
    """Apply the known DTLZ2 outer maps."""

    return torch.stack(
        (
            (1.0 + H[..., 0].clamp_min(0.0)) * H[..., 1].clamp(0.0, 1.0),
            (1.0 + H[..., 2].clamp_min(0.0)) * H[..., 3].clamp(0.0, 1.0),
        ),
        dim=-1,
    )


PROBLEM = BenchmarkProblem(
    name="DTLZ2 (2 objectives, 600 dimensions)",
    slug="dtlz2_2obj_600d",
    dim=DIM,
    num_objectives=2,
    suite="high",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), REFERENCE_VALUE, dtype=torch.double),
    exact_max_hypervolume=REFERENCE_VALUE**2 - math.pi / 4.0,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
