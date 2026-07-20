"""Three-component Langermann versus two-component Ackley in six dimensions."""

import torch

from benchmark_common import (
    BenchmarkProblem,
    ackley_components,
    compose_ackley,
    compose_langermann,
    orthogonal_matrix,
    run_benchmark,
    transformed_inputs,
)


DIM = 6
LANGERMANN_CENTERS = torch.tensor(
    [
        [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        [0.30, 0.25, 0.20, 0.15, 0.40, 0.35],
        [0.20, 0.35, 0.15, 0.40, 0.25, 0.30],
    ],
    dtype=torch.double,
)
LANGERMANN_COEFFICIENTS = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
ACKLEY_CENTER = torch.full((DIM,), 0.80, dtype=torch.double)
ACKLEY_ROTATION = orthogonal_matrix(DIM, seed=6301)
ACKLEY_SCALE = 8.0


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    X = X.double()
    distances = (X.unsqueeze(-2) - LANGERMANN_CENTERS).square().sum(dim=-1)
    ackley_z = transformed_inputs(X, ACKLEY_CENTER, ACKLEY_ROTATION, ACKLEY_SCALE)
    return torch.cat((distances, ackley_components(ackley_z)), dim=-1)


def compose(H: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            compose_langermann(H[..., 0:3], LANGERMANN_COEFFICIENTS),
            compose_ackley(H[..., 3:5]),
        ),
        dim=-1,
    )


PROBLEM = BenchmarkProblem(
    name="Langermann-3 versus Ackley (2 objectives, 6 dimensions)",
    slug="langermann3_ackley_2obj_6d",
    dim=DIM,
    num_objectives=2,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 2.5, dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
