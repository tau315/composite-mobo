"""High-dimensional stochastic Shallow Lake management benchmark.

The dynamics and standard constants follow the public 100-decision lake
problem distributed with the Borg multiobjective examples:

    https://gist.github.com/jdherman/d519b3ef81840fc27d96

Each input x_t in [0, 0.1] is an annual anthropogenic phosphorus release over a
100-year planning horizon.  Phosphorus decays, is recycled nonlinearly from
lake sediment, and receives lognormal natural inflow.  The original problem
has four objectives and a reliability constraint.  This file uses the clean
two-objective scientific slice needed here: minimize worst expected phosphorus
and maximize discounted economic utility (implemented as utility loss so all
objectives are minimized).

The published simulator draws new inflow scenarios on every call.  A BO oracle
must compare designs against the same realization, so this benchmark uses 100
fixed common-random-number scenarios from the published lognormal law.  This
makes every objective evaluation deterministic without removing uncertainty
from the modeled lake response.

Composite response (21 intermediates -> 2 objectives):

* h_1..h_20 are normalized maxima of the ensemble-mean phosphorus trajectory
  in consecutive five-year blocks;
* h_21 is normalized discounted utility;
* f_1 = max(h_1, ..., h_20), and f_2 = 1 - h_21.

The outer maximum discards which part of the planning horizon controls water
quality and creates regime changes when the controlling block switches.  The
20-block response retains that temporal information while remaining practical
for methods that fit one GP per component.

Run the paper protocol (10 trials, 20 initial points, 120 total evaluations):

    python benchmark_shallow_lake.py

For an installation smoke test:

    python benchmark_shallow_lake.py --quick

Results are written to
``benchmark_results/shallow_lake_2obj_100d/{benchmark_data.npz,
hypervolume_vs_evaluations.png}`` unless ``--output-dir`` is supplied.
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark


DIM = 100
N_YEARS = DIM
N_SCENARIOS = 100
N_BLOCKS = 20
YEARS_PER_BLOCK = N_YEARS // N_BLOCKS

DECAY_RATE = 0.42
RECYCLING_EXPONENT = 2.0
UTILITY_COEFFICIENT = 0.4
DISCOUNT_FACTOR = 0.98
MAX_ANNUAL_RELEASE = 0.1
NATURAL_INFLOW_LOG_MEAN = -3.52
NATURAL_INFLOW_LOG_STD = 0.105
COMMON_RANDOM_SEED = 20_240_820

# The all-maximum-release design gives a worst expected phosphorus peak of
# about 2.317 for these fixed scenarios.  A 2.4 scale therefore bounds the
# normalized water-quality objective below one throughout the design box.
PHOSPHORUS_SCALE = 2.4

_RANDOM_GENERATOR = torch.Generator().manual_seed(COMMON_RANDOM_SEED)
_STANDARD_NORMAL_INFLOW = torch.randn(
    N_SCENARIOS,
    N_YEARS - 1,
    generator=_RANDOM_GENERATOR,
    dtype=torch.double,
)
NATURAL_INFLOW = torch.exp(
    NATURAL_INFLOW_LOG_MEAN
    + NATURAL_INFLOW_LOG_STD * _STANDARD_NORMAL_INFLOW
)
DISCOUNT_WEIGHTS = DISCOUNT_FACTOR ** torch.arange(N_YEARS, dtype=torch.double)
MAX_DISCOUNTED_UTILITY = float(
    UTILITY_COEFFICIENT * MAX_ANNUAL_RELEASE * DISCOUNT_WEIGHTS.sum()
)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Simulate ensemble lake dynamics and return block peaks plus utility."""

    original_shape = X.shape[:-1]
    flat = X.double().reshape(-1, DIM).clamp(0.0, 1.0)
    releases = MAX_ANNUAL_RELEASE * flat
    inflow = NATURAL_INFLOW.to(flat)

    n_designs = flat.shape[0]
    phosphorus = torch.zeros(
        n_designs, N_SCENARIOS, dtype=flat.dtype, device=flat.device
    )
    mean_trajectory = torch.empty(
        n_designs, N_YEARS, dtype=flat.dtype, device=flat.device
    )
    # Match the published time indexing: P_0 is fixed at zero and release
    # x_{t-1} enters the transition to P_t.  All 100 decisions contribute to
    # utility; the first 99 contribute to the 100 recorded lake states.
    mean_trajectory[:, 0] = 0.0
    for year in range(1, N_YEARS):
        recycling = phosphorus.pow(RECYCLING_EXPONENT) / (
            1.0 + phosphorus.pow(RECYCLING_EXPONENT)
        )
        phosphorus = (
            (1.0 - DECAY_RATE) * phosphorus
            + recycling
            + releases[:, year - 1, None]
            + inflow[None, :, year - 1]
        )
        mean_trajectory[:, year] = phosphorus.mean(dim=-1)

    block_peaks = mean_trajectory.reshape(
        n_designs, N_BLOCKS, YEARS_PER_BLOCK
    ).amax(dim=-1)
    normalized_block_peaks = block_peaks / PHOSPHORUS_SCALE
    normalized_utility = (
        UTILITY_COEFFICIENT
        * (releases * DISCOUNT_WEIGHTS.to(flat)[None, :]).sum(dim=-1)
        / MAX_DISCOUNTED_UTILITY
    )
    components = torch.cat(
        (normalized_block_peaks, normalized_utility[:, None]), dim=-1
    )
    return components.reshape(*original_shape, N_BLOCKS + 1)


def compose(H: torch.Tensor) -> torch.Tensor:
    """Return worst phosphorus peak and discounted-utility loss."""

    worst_phosphorus = H[..., :N_BLOCKS].amax(dim=-1)
    utility_loss = 1.0 - H[..., N_BLOCKS]
    return torch.stack((worst_phosphorus, utility_loss), dim=-1)


PROBLEM = BenchmarkProblem(
    name="Shallow Lake management (2 objectives, 100 dimensions)",
    slug="shallow_lake_2obj_100d",
    dim=DIM,
    num_objectives=2,
    suite="high",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 1.05, dtype=torch.double),
    num_components=N_BLOCKS + 1,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
