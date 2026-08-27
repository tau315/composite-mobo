"""Dense 600-D Langermann--Ackley latent-composition benchmark.

Both objectives use the same six dense orthonormal projections of the 600
design variables.  The six values are repeated in objective-specific groups,
giving twelve independently modeled GP outputs and preventing multi-task
transfer between objectives.  Direct solvers must model the oscillatory
objectives; composite solvers apply the known Langermann and Ackley maps to
their respective latent groups.  The objectives deliberately conflict:
Ackley is minimized at the zero latent vector, while every Langermann center
is displaced from zero.

Paper protocol: 20 initial plus 100 adaptive design evaluations (120 total).
"""

from __future__ import annotations

import torch

from benchmark_common import (
    ACKLEY_UPPER_BOUND,
    BenchmarkProblem,
    orthonormal_rows,
    run_benchmark,
)


DIM = 600
LATENT_DIM = 6
LATENT_SCALE = 8.0
PROJECTION = orthonormal_rows(LATENT_DIM, DIM, seed=18_271)
CENTERS = torch.tensor(
    [
        [1.6, -0.9, 0.7, 1.2, -1.4, 0.5],
        [-1.3, 1.5, -0.8, 0.6, 1.0, -1.1],
        [0.8, 1.1, 1.4, -1.5, -0.5, 0.9],
        [-0.7, -1.6, 1.0, 1.4, 0.8, -0.6],
        [1.2, 0.6, -1.5, -0.9, 1.3, 1.0],
    ],
    dtype=torch.double,
)
COEFFICIENTS = torch.tensor([1.0, 2.0, 5.0, 2.0, 3.0], dtype=torch.double)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Two objective-specific copies of six dense latent responses."""

    latent = LATENT_SCALE * ((X.double() - 0.5) @ PROJECTION.T)
    return torch.cat((latent, latent), dim=-1)


def compose(H: torch.Tensor) -> torch.Tensor:
    langermann_latent = H[..., :LATENT_DIM]
    ackley_latent = H[..., LATENT_DIM:]
    centers = CENTERS.to(H)
    coefficients = COEFFICIENTS.to(H)
    distances = (langermann_latent.unsqueeze(-2) - centers).square().sum(dim=-1)
    langermann_reward = (
        coefficients
        * torch.exp(-distances / torch.pi)
        * torch.cos(torch.pi * distances)
    ).sum(dim=-1)
    coefficient_sum = coefficients.sum()
    # Every summand lies in [-c_i, c_i], so this is bounded in [0, 1].
    langermann = (coefficient_sum - langermann_reward) / (2.0 * coefficient_sum)

    mean_square = ackley_latent.square().mean(dim=-1)
    mean_cosine = torch.cos(2.0 * torch.pi * ackley_latent).mean(dim=-1)
    ackley = (
        -20.0 * torch.exp(-0.2 * mean_square.sqrt())
        - torch.exp(mean_cosine)
        + 20.0
        + torch.e
    ) / ACKLEY_UPPER_BOUND
    return torch.stack((langermann, ackley), dim=-1)


PROBLEM = BenchmarkProblem(
    name="Latent Langermann--Ackley (2 objectives, 600 dimensions, 120 evaluations)",
    slug="langermann_ackley_2obj_600d",
    dim=DIM,
    num_objectives=2,
    num_components=2 * LATENT_DIM,
    suite="high",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 1.1, dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
