"""High-D scientific benchmark: CORT TG-119 radiotherapy planning.

The 418 variables are beamlet fluences.  Evaluation is a sparse multiplication
by the public precomputed dose-influence matrices followed by dose statistics;
there is no numerical integration.  The final objectives use the target hinge
penalty and quadratic organ-at-risk exposure penalties, making the observed
six dose statistics a meaningful nonlinear composition.

Paper protocol: 20 initial plus 100 adaptive design evaluations (120 total).
Data download and cache
handling live in ``benchmark_cort_tg119.py``.
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark
from benchmark_cort_tg119 import DIM, ORACLE


def compose(H: torch.Tensor) -> torch.Tensor:
    base = ORACLE.compose(H)
    return torch.stack((base[..., 0], base[..., 1].square(), base[..., 2].square()), dim=-1)


PROBLEM = BenchmarkProblem(
    name="CORT TG-119 quadratic dose planning (3 objectives, 418 dimensions)",
    slug="cort_tg119_quadratic_3obj_418d",
    dim=DIM,
    num_objectives=3,
    num_components=6,
    suite="high",
    evaluate_components=ORACLE.evaluate_components,
    compose=compose,
    ideal=torch.zeros(3, dtype=torch.double),
    ref_point=torch.full((3,), 2.5, dtype=torch.double),
    prepare=ORACLE.prepare,
    clear_evaluation_cache=ORACLE.clear_evaluation_cache,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
