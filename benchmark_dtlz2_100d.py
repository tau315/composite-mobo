"""Two-objective DTLZ2 benchmark in 100 input dimensions."""

import math

import torch

from benchmark_common import BenchmarkProblem, run_benchmark


DIM = 100
REFERENCE_VALUE = 1.0 + 0.25 * (DIM - 1) + 0.1


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Model only the radial distance, which is the one unknown quantity.

    The angular coordinate is a design variable, so it is known exactly and is
    routed through ``compose`` rather than given a GP. Modelling ``sqrt`` of the
    distance keeps posterior samples in the valid non-negative domain.
    """

    X = X.double()
    distance = (X[..., 1:] - 0.5).square().sum(dim=-1, keepdim=True)
    return distance.sqrt()


def compose(H: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Apply the known DTLZ2 outer maps using the exact angular coordinate."""

    distance = H[..., 0].square()
    angle = torch.pi * X[..., 0].double() / 2.0
    # Give the exact angle the leading Monte Carlo sample dimensions of the
    # component posterior so the two broadcast together.
    angle = angle + torch.zeros_like(distance)
    radius = 1.0 + distance
    return torch.stack((radius * angle.cos(), radius * angle.sin()), dim=-1)


PROBLEM = BenchmarkProblem(
    name="DTLZ2 (2 objectives, 100 dimensions)",
    slug="dtlz2_2obj_100d",
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
