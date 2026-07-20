"""Five-objective shifted-Ackley benchmark in six input dimensions."""

import torch

from benchmark_common import (
    BenchmarkProblem,
    ackley_components,
    compose_ackley,
    orthogonal_matrix,
    run_benchmark,
    transformed_inputs,
)


DIM = 6
NUM_OBJECTIVES = 5
SCALE = 8.0

# Five vertices of a regular simplex embedded in the first five coordinates.
simplex = torch.eye(NUM_OBJECTIVES, dtype=torch.double)
simplex = simplex - simplex.mean(dim=0, keepdim=True)
simplex = simplex / simplex.norm(dim=-1, keepdim=True)
CENTERS = torch.full((NUM_OBJECTIVES, DIM), 0.5, dtype=torch.double)
CENTERS[:, :NUM_OBJECTIVES] += 0.28 * simplex
ROTATIONS = tuple(
    orthogonal_matrix(DIM, seed=6200 + objective) for objective in range(NUM_OBJECTIVES)
)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    groups = []
    for objective in range(NUM_OBJECTIVES):
        z = transformed_inputs(X, CENTERS[objective], ROTATIONS[objective], SCALE)
        groups.append(ackley_components(z))
    return torch.cat(groups, dim=-1)


def compose(H: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [compose_ackley(H[..., 2 * i : 2 * i + 2]) for i in range(NUM_OBJECTIVES)],
        dim=-1,
    )


PROBLEM = BenchmarkProblem(
    name="Five shifted Ackley objectives (5 objectives, 6 dimensions)",
    slug="five_ackley_5obj_6d",
    dim=DIM,
    num_objectives=NUM_OBJECTIVES,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(NUM_OBJECTIVES, dtype=torch.double),
    ref_point=torch.full((NUM_OBJECTIVES,), 2.5, dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
