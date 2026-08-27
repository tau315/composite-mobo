"""Symmetric linear-antenna-array design benchmark.

The 100 variables are excitation amplitudes for one half of a symmetric
200-element, half-wavelength-spaced broadside array.  Its signed far-field
array factor is

    A(u) = sum_n I_n cos(pi (n - 1/2) u),  u = sin(theta).

The independently modeled intermediate responses are 32 signed array-factor
samples: eight in the main-beam region and 24 in the sidelobe region.  The
outer map nonlinearly forms (1) peak sampled sidelobe/boresight ratio and (2)
mean normalized main-region power away from boresight, a smooth beamwidth
proxy.  Narrowing the main beam raises sidelobes, giving the standard antenna
design tradeoff.  Positive minimum excitation prevents a degenerate zero
array.

This finite-angle simulator follows the classical symmetric linear-array
factor and the established sidelobe-level/beamwidth objectives described in
https://doi.org/10.1016/j.aeue.2004.05.006 and reviewed in
https://pmc.ncbi.nlm.nih.gov/articles/PMC7570954/.

Run ``python benchmark_antenna_array.py`` for the default 10 independent
trials (20 initial points and 120 total evaluations per trial), or add
``--quick`` for a smoke test.  Results are written beneath
``benchmark_results/antenna_array_2obj_100d/``.
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark


DIM = 100
HALF_ELEMENTS = DIM
MAIN_SAMPLES = torch.linspace(0.0, 0.012, 8, dtype=torch.double)
NEAR_SIDELOBE_SAMPLES = (
    torch.arange(1.0, 13.0, dtype=torch.double) + 0.5
) / HALF_ELEMENTS
FAR_SIDELOBE_SAMPLES = torch.linspace(0.16, 1.0, 12, dtype=torch.double)
SPATIAL_FREQUENCIES = torch.cat(
    (MAIN_SAMPLES, NEAR_SIDELOBE_SAMPLES, FAR_SIDELOBE_SAMPLES)
)
NUM_COMPONENTS = SPATIAL_FREQUENCIES.numel()
ELEMENT_POSITIONS = torch.arange(0.5, HALF_ELEMENTS, dtype=torch.double)
ARRAY_BASIS = torch.cos(
    torch.pi * ELEMENT_POSITIONS.unsqueeze(-1) * SPATIAL_FREQUENCIES.unsqueeze(0)
)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Return normalized signed array factors at all 32 observation angles."""

    original_shape = X.shape[:-1]
    flat = X.double().reshape(-1, DIM).clamp(0.0, 1.0)
    amplitudes = 0.05 + 0.95 * flat
    responses = amplitudes @ ARRAY_BASIS.to(flat) / HALF_ELEMENTS
    return responses.reshape(*original_shape, NUM_COMPONENTS)


def compose(H: torch.Tensor) -> torch.Tensor:
    """Form sidelobe level and main-beam-width proxy from signed fields."""

    # Absolute value is needed because posterior samples of the signed field
    # need not retain the physical positivity of the observed boresight value.
    boresight = H[..., 0].abs().clamp_min(0.02)
    sidelobes = H[..., MAIN_SAMPLES.numel() :].abs() / boresight.unsqueeze(-1)
    sidelobe_level = sidelobes.amax(dim=-1)
    normalized_main_field = H[..., 1 : MAIN_SAMPLES.numel()] / boresight.unsqueeze(-1)
    beamwidth_proxy = normalized_main_field.square().mean(dim=-1)
    return torch.stack((sidelobe_level, beamwidth_proxy), dim=-1)


PROBLEM = BenchmarkProblem(
    name="Symmetric linear antenna array (2 objectives, 100 dimensions)",
    slug="antenna_array_2obj_100d",
    dim=DIM,
    num_objectives=2,
    suite="high",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 0.50, dtype=torch.double),
    num_components=NUM_COMPONENTS,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
