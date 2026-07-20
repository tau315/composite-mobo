"""Projected Langermann objectives with 4 and 5 components in 500 dimensions."""

import torch

from benchmark_common import (
    BenchmarkProblem,
    compose_langermann,
    orthonormal_rows,
    run_benchmark,
)


DIM = 500
PROJECTIONS = orthonormal_rows(7, DIM, seed=500_007)
COEFFICIENTS_1 = torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.double)
COEFFICIENTS_2 = torch.tensor([1.0, 1.25, 1.5, 1.75, 2.0], dtype=torch.double)
TARGETS_1 = torch.tensor([-0.65, -0.35, 0.25, 0.55], dtype=torch.double)
TARGETS_2 = torch.tensor([0.65, 0.35, -0.45, 0.05, 0.50], dtype=torch.double)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    projected = (2.0 * X.double() - 1.0) @ PROJECTIONS.T
    objective_1 = projected[..., [0, 1, 2, 3]]
    # Duplicate the two shared physical projections so the two objectives still
    # receive separate, independently fitted component GPs.
    objective_2 = projected[..., [0, 1, 4, 5, 6]]
    return torch.cat((objective_1, objective_2), dim=-1)


def compose(H: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            compose_langermann(H[..., 0:4], COEFFICIENTS_1, TARGETS_1),
            compose_langermann(H[..., 4:9], COEFFICIENTS_2, TARGETS_2),
        ),
        dim=-1,
    )


PROBLEM = BenchmarkProblem(
    name="Projected Langermann (2 objectives, 500 dimensions; 4+5 components)",
    slug="projected_langermann_2obj_500d",
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
