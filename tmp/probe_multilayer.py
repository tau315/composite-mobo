"""Screening probe: radiative-cooling multilayer film via transfer matrices.

x  = layer thicknesses (one per layer, so d scales by stacking layers)
h  = spectral reflectance R(lambda) sampled on a fixed quadrature grid
g  = two known weighted sums of that spectrum (solar reflectance and
     atmospheric-window emittance), i.e. exactly the quadrature of the
     integrals a radiative-cooling paper would report

The intermediate is smooth in the thicknesses and the known map is a pure
linear functional of it, which is the structure composite BO should exploit.
"""

import sys

import torch

sys.path.insert(0, ".")
from benchmark_common import BenchmarkProblem
from diagnose_composite import diagnose

# Alternating low/high index dielectrics, a standard SiO2 / TiO2 style stack.
INDEX_LOW, INDEX_HIGH = 1.45, 2.35
INDEX_AIR, INDEX_SUBSTRATE = 1.0, 1.5
THICKNESS_NM = (20.0, 400.0)

# Eight solar-band and eight atmospheric-window sample points (nanometres).
SOLAR_GRID = torch.linspace(400.0, 2200.0, 8, dtype=torch.double)
WINDOW_GRID = torch.linspace(8000.0, 13000.0, 8, dtype=torch.double)
GRID = torch.cat((SOLAR_GRID, WINDOW_GRID))

# Blackbody-ish solar weighting and a flat atmospheric-window weighting.
SOLAR_WEIGHT = torch.exp(-((SOLAR_GRID - 500.0) / 700.0) ** 2)
SOLAR_WEIGHT = SOLAR_WEIGHT / SOLAR_WEIGHT.sum()
WINDOW_WEIGHT = torch.full((8,), 1.0 / 8.0, dtype=torch.double)


def reflectance(thickness_nm: torch.Tensor, n_layers: int) -> torch.Tensor:
    """Normal-incidence reflectance of the stack at every grid wavelength."""

    indices = torch.tensor(
        [INDEX_LOW if i % 2 == 0 else INDEX_HIGH for i in range(n_layers)],
        dtype=torch.double,
    )
    # Characteristic matrix per layer per wavelength, accumulated along layers.
    batch = thickness_nm.shape[:-1]
    m11 = torch.ones(*batch, len(GRID), dtype=torch.cdouble)
    m12 = torch.zeros(*batch, len(GRID), dtype=torch.cdouble)
    m21 = torch.zeros(*batch, len(GRID), dtype=torch.cdouble)
    m22 = torch.ones(*batch, len(GRID), dtype=torch.cdouble)
    for layer in range(n_layers):
        n = indices[layer]
        delta = 2.0 * torch.pi * n * thickness_nm[..., layer : layer + 1] / GRID
        cos, sin = torch.cos(delta) + 0j, torch.sin(delta) + 0j
        a11, a12 = cos, 1j * sin / n
        a21, a22 = 1j * n * sin, cos
        m11, m12, m21, m22 = (
            m11 * a11 + m12 * a21,
            m11 * a12 + m12 * a22,
            m21 * a11 + m22 * a21,
            m21 * a12 + m22 * a22,
        )
    n0, ns = INDEX_AIR, INDEX_SUBSTRATE
    numerator = n0 * m11 + n0 * ns * m12 - m21 - ns * m22
    denominator = n0 * m11 + n0 * ns * m12 + m21 + ns * m22
    return (numerator / denominator).abs().square().clamp(0.0, 1.0)


def make_problem(n_layers: int, suite: str) -> BenchmarkProblem:
    def components(X: torch.Tensor) -> torch.Tensor:
        low, high = THICKNESS_NM
        return reflectance(low + (high - low) * X.double(), n_layers)

    def compose(H: torch.Tensor) -> torch.Tensor:
        solar = (H[..., :8] * SOLAR_WEIGHT).sum(dim=-1)
        window = (H[..., 8:] * WINDOW_WEIGHT).sum(dim=-1)
        # Minimize: want high solar reflectance and high window emittance.
        return torch.stack((-solar, -(1.0 - window)), dim=-1)

    return BenchmarkProblem(
        name=f"Radiative-cooling multilayer ({n_layers} layers)",
        slug=f"multilayer_2obj_{n_layers}d",
        dim=n_layers,
        num_objectives=2,
        suite=suite,
        evaluate_components=components,
        compose=compose,
        ideal=torch.tensor([-1.0, -1.0], dtype=torch.double),
        ref_point=torch.zeros(2, dtype=torch.double),
    )


for n_layers, suite in ((8, "low"), (16, "low"), (60, "high"), (120, "high")):
    problem = make_problem(n_layers, suite)
    problem.validate()
    report = diagnose(problem)
    print(
        f"multilayer d={n_layers:<5} p={report['components']:<3} "
        f"direct={report['direct_rmse']:.3f} composite={report['composite_rmse']:.3f} "
        f"advantage={report['advantage']:+.1%}",
        flush=True,
    )
