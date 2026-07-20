"""High-dimensional Ackley-versus-Griewank benchmark in 50 dimensions."""

import torch

from benchmark_common import (
    BenchmarkProblem,
    ackley_components,
    compose_ackley,
    compose_griewank,
    griewank_components,
    griewank_upper_bound,
    orthogonal_matrix,
    run_benchmark,
    transformed_inputs,
)


DIM = 50
ACKLEY_CENTER = torch.full((DIM,), 0.35, dtype=torch.double)
GRIEWANK_CENTER = torch.full((DIM,), 0.65, dtype=torch.double)
ACKLEY_ROTATION = orthogonal_matrix(DIM, seed=6501)
GRIEWANK_ROTATION = orthogonal_matrix(DIM, seed=6502)
ACKLEY_SCALE = 5.0
GRIEWANK_SCALE = 2.5
GRIEWANK_UPPER = griewank_upper_bound(GRIEWANK_CENTER, GRIEWANK_SCALE)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    ackley_z = transformed_inputs(X, ACKLEY_CENTER, ACKLEY_ROTATION, ACKLEY_SCALE)
    griewank_z = transformed_inputs(
        X, GRIEWANK_CENTER, GRIEWANK_ROTATION, GRIEWANK_SCALE
    )
    return torch.cat(
        (ackley_components(ackley_z), griewank_components(griewank_z)), dim=-1
    )


def compose(H: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            compose_ackley(H[..., 0:2]),
            compose_griewank(H[..., 2:4], GRIEWANK_UPPER),
        ),
        dim=-1,
    )


PROBLEM = BenchmarkProblem(
    name="Ackley versus Griewank (2 objectives, 50 dimensions)",
    slug="ackley_griewank_2obj_50d",
    dim=DIM,
    num_objectives=2,
    suite="high",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 2.5, dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
