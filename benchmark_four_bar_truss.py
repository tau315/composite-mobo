"""Low-dimensional four-bar truss structural-design benchmark (RE21).

This implements the standard two-objective RE21 definition documented by the
DESDEO optimization framework:

    https://desdeo.readthedocs.io/en/latest/explanation/test_problems/

The four inputs are the cross-sectional areas of the truss members.  Structural
volume and loaded-joint displacement are minimized.  Constants are F=10 kN,
E=200000 kN/cm^2, L=200 cm, and allowable stress sigma=10 kN/cm^2.  The member
area bounds encode the stress restrictions in the published benchmark.

Composite response (8 intermediates -> 2 objectives):

* h_1..h_4 are the four member contributions to structural volume;
* h_5..h_8 are the four signed member contributions to displacement;
* the two objectives sum their respective terms and are affinely normalized
  using exact domain-wide extrema.

This is a deliberately transparent additive scientific composition.  It tests
whether observing member-level mechanics helps BO beyond observing only the two
aggregated engineering objectives.  Both normalized objectives are minimized
and lie in [0, 1] over the design box.

Run the paper protocol (10 trials, 5 initial points, 45 total evaluations):

    python benchmark_four_bar_truss.py

For an installation smoke test:

    python benchmark_four_bar_truss.py --quick

Results are written to
``benchmark_results/four_bar_truss_re21_2obj_4d/{benchmark_data.npz,
hypervolume_vs_evaluations.png}`` unless ``--output-dir`` is supplied.
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark


DIM = 4
FORCE_KN = 10.0
ELASTIC_MODULUS_KN_PER_CM2 = 2.0e5
MEMBER_LENGTH_CM = 200.0
ALLOWABLE_STRESS_KN_PER_CM2 = 10.0
AREA_SCALE = FORCE_KN / ALLOWABLE_STRESS_KN_PER_CM2
SQRT_TWO = 2.0**0.5

LOWER_BOUNDS = AREA_SCALE * torch.tensor(
    [1.0, SQRT_TWO, SQRT_TWO, 1.0], dtype=torch.double
)
UPPER_BOUNDS = AREA_SCALE * torch.tensor(
    [3.0, 3.0, 3.0, 3.0], dtype=torch.double
)


def _volume_terms(areas: torch.Tensor) -> torch.Tensor:
    """Member-level terms in the canonical RE21 volume expression."""

    x1, x2, x3, x4 = areas.unbind(dim=-1)
    return torch.stack(
        (2.0 * x1, SQRT_TWO * x2, x3.sqrt(), x4), dim=-1
    )


def _displacement_terms(areas: torch.Tensor) -> torch.Tensor:
    """Signed member-level terms in the canonical RE21 displacement."""

    x1, x2, x3, x4 = areas.unbind(dim=-1)
    return torch.stack(
        (
            2.0 / x1,
            2.0 * SQRT_TWO / x2,
            -2.0 * SQRT_TWO / x3,
            2.0 / x4,
        ),
        dim=-1,
    )


# Volume is increasing in every area.  Displacement decreases with x1, x2,
# and x4 but increases with x3 because its member contribution is negative.
# These separable monotonicities give exact box extrema rather than sampled
# normalization constants.
VOLUME_MIN = float(
    MEMBER_LENGTH_CM * _volume_terms(LOWER_BOUNDS.unsqueeze(0)).sum()
)
VOLUME_MAX = float(
    MEMBER_LENGTH_CM * _volume_terms(UPPER_BOUNDS.unsqueeze(0)).sum()
)
DISPLACEMENT_FACTOR = FORCE_KN * MEMBER_LENGTH_CM / ELASTIC_MODULUS_KN_PER_CM2

_DISPLACEMENT_MIN_AREAS = torch.tensor(
    [
        UPPER_BOUNDS[0],
        UPPER_BOUNDS[1],
        LOWER_BOUNDS[2],
        UPPER_BOUNDS[3],
    ],
    dtype=torch.double,
)
_DISPLACEMENT_MAX_AREAS = torch.tensor(
    [
        LOWER_BOUNDS[0],
        LOWER_BOUNDS[1],
        UPPER_BOUNDS[2],
        LOWER_BOUNDS[3],
    ],
    dtype=torch.double,
)
DISPLACEMENT_MIN = float(
    DISPLACEMENT_FACTOR
    * _displacement_terms(_DISPLACEMENT_MIN_AREAS.unsqueeze(0)).sum()
)
DISPLACEMENT_MAX = float(
    DISPLACEMENT_FACTOR
    * _displacement_terms(_DISPLACEMENT_MAX_AREAS.unsqueeze(0)).sum()
)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Return four volume and four signed displacement contributions."""

    original_shape = X.shape[:-1]
    flat = X.double().reshape(-1, DIM).clamp(0.0, 1.0)
    lower = LOWER_BOUNDS.to(flat)
    areas = lower + flat * (UPPER_BOUNDS.to(flat) - lower)
    components = torch.cat(
        (_volume_terms(areas), _displacement_terms(areas)), dim=-1
    )
    return components.reshape(*original_shape, 2 * DIM)


def compose(H: torch.Tensor) -> torch.Tensor:
    """Aggregate member mechanics and normalize both objectives to [0, 1]."""

    raw_volume = MEMBER_LENGTH_CM * H[..., :DIM].sum(dim=-1)
    raw_displacement = DISPLACEMENT_FACTOR * H[..., DIM:].sum(dim=-1)
    volume = (raw_volume - VOLUME_MIN) / (VOLUME_MAX - VOLUME_MIN)
    displacement = (raw_displacement - DISPLACEMENT_MIN) / (
        DISPLACEMENT_MAX - DISPLACEMENT_MIN
    )
    return torch.stack((volume, displacement), dim=-1)


PROBLEM = BenchmarkProblem(
    name="Four-bar truss RE21 (2 objectives, 4 dimensions)",
    slug="four_bar_truss_re21_2obj_4d",
    dim=DIM,
    num_objectives=2,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 1.05, dtype=torch.double),
    num_components=2 * DIM,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
