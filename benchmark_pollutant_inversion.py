"""Low-dimensional environmental pollutant-source inversion benchmark.

The forward model and calibration constants are the environmental example used
by Astudillo and Frazier for Bayesian optimization of composite functions:

    https://arxiv.org/abs/1906.01537

It predicts pollutant concentration from two equal-mass releases in a narrow
channel.  The first release is at location/time (0, 0); the second is at the
unknown location/time (L, tau).  The four controls are source mass M, diffusion
coefficient D, second-release location L, and second-release time tau.  The
published noiseless target uses (10, 0.07, 1.505, 30.1525), observed at three
locations and four times.

The source problem is single-objective calibration.  This file makes the
scientifically explicit biobjective adaptation needed for the paper:

* objective 1 minimizes mean squared concentration residual over 12 sensors;
* objective 2 minimizes released mass, a source-strength regularizer.

Composite response (13 intermediates -> 2 objectives):

* h_1..h_12 are normalized predicted concentrations at the 12 sensors;
* h_13 is normalized source mass;
* f_1 is the mean squared residual from the fixed target concentrations, and
  f_2 = h_13.

The target subtraction, squaring, and many-to-one averaging erase residual sign
and sensor identity.  That is the nonlinear information loss the composite
methods can avoid.  Both objectives are minimized, and the oracle is
deterministic and closed form.

Run the paper protocol (10 trials, 5 initial points, 45 total evaluations):

    python benchmark_pollutant_inversion.py

For an installation smoke test:

    python benchmark_pollutant_inversion.py --quick

Results are written to
``benchmark_results/pollutant_inversion_2obj_4d/{benchmark_data.npz,
hypervolume_vs_evaluations.png}`` unless ``--output-dir`` is supplied.
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark


DIM = 4
N_OBSERVATIONS = 12

# Bounds are ordered as [mass, diffusion, second-release location,
# second-release time].  Designs supplied by the shared runner are normalized
# to [0, 1]^4 and are mapped into these physical units below.
LOWER_BOUNDS = torch.tensor([7.0, 0.02, 0.01, 30.01], dtype=torch.double)
UPPER_BOUNDS = torch.tensor([13.0, 0.12, 3.0, 30.295], dtype=torch.double)
TRUE_PARAMETERS = torch.tensor(
    [10.0, 0.07, 1.505, 30.1525], dtype=torch.double
)

OBSERVATION_LOCATIONS = torch.tensor(
    [0.0, 1.0, 2.5], dtype=torch.double
).repeat_interleave(4)
OBSERVATION_TIMES = torch.tensor(
    [15.0, 30.0, 45.0, 60.0], dtype=torch.double
).repeat(3)


def _concentration(parameters: torch.Tensor) -> torch.Tensor:
    """Evaluate the two-release one-dimensional diffusion response.

    ``parameters`` has shape [batch, 4] in physical units.  The output has one
    column for each location/time pair in the 3 x 4 observation grid.
    """

    mass, diffusion, location, release_time = parameters.unbind(dim=-1)
    space = OBSERVATION_LOCATIONS.to(parameters)
    time = OBSERVATION_TIMES.to(parameters)

    first_denominator = torch.sqrt(
        4.0 * torch.pi * diffusion[:, None] * time[None, :]
    )
    first_release = mass[:, None] / first_denominator * torch.exp(
        -space[None, :].square()
        / (4.0 * diffusion[:, None] * time[None, :])
    )

    elapsed = time[None, :] - release_time[:, None]
    active = elapsed > 0.0
    safe_elapsed = elapsed.clamp_min(torch.finfo(parameters.dtype).eps)
    second_denominator = torch.sqrt(
        4.0 * torch.pi * diffusion[:, None] * safe_elapsed
    )
    second_release = mass[:, None] / second_denominator * torch.exp(
        -(space[None, :] - location[:, None]).square()
        / (4.0 * diffusion[:, None] * safe_elapsed)
    )
    return first_release + torch.where(
        active, second_release, torch.zeros_like(second_release)
    )


TARGET_CONCENTRATIONS = _concentration(TRUE_PARAMETERS.unsqueeze(0)).squeeze(0)
# Scaling by the largest target concentration keeps the squared-error objective
# on an order-one range without changing its Pareto ordering.
RESIDUAL_SCALE = float(TARGET_CONCENTRATIONS.max())
NORMALIZED_TARGET_CONCENTRATIONS = TARGET_CONCENTRATIONS / RESIDUAL_SCALE


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Return 12 normalized concentrations and normalized source mass."""

    original_shape = X.shape[:-1]
    flat = X.double().reshape(-1, DIM).clamp(0.0, 1.0)
    bounds_lower = LOWER_BOUNDS.to(flat)
    physical = bounds_lower + flat * (UPPER_BOUNDS.to(flat) - bounds_lower)

    predicted = _concentration(physical)
    normalized_concentrations = predicted / RESIDUAL_SCALE
    normalized_mass = flat[:, :1]
    components = torch.cat((normalized_concentrations, normalized_mass), dim=-1)
    return components.reshape(*original_shape, N_OBSERVATIONS + 1)


def compose(H: torch.Tensor) -> torch.Tensor:
    """Reduce sensor concentrations to calibration error and source mass."""

    residuals = (
        H[..., :N_OBSERVATIONS]
        - NORMALIZED_TARGET_CONCENTRATIONS.to(H)
    )
    calibration_error = residuals.square().mean(dim=-1)
    source_mass = H[..., N_OBSERVATIONS]
    return torch.stack((calibration_error, source_mass), dim=-1)


PROBLEM = BenchmarkProblem(
    name="Pollutant-source inversion (2 objectives, 4 dimensions)",
    slug="pollutant_inversion_2obj_4d",
    dim=DIM,
    num_objectives=2,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 1.05, dtype=torch.double),
    num_components=N_OBSERVATIONS + 1,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
