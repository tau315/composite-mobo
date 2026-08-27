"""Multi-illuminant textile color-formulation benchmark.

Six continuous dye concentrations determine an opaque coating's spectral
reflectance through the Kubelka--Munk model.  The independently modeled
intermediate responses are CIE Lab coordinates under a 6504 K daylight
approximation and a 2856 K tungsten (Illuminant-A) approximation.  The two
objectives are daylight color difference and the metameric mismatch: the norm
of the change in the sample-minus-target Lab error between the illuminants.

This is a deterministic scientific simulator, not a fitted experimental data
set.  It is motivated by multi-objective textile recipes that jointly control
color difference and metamerism (https://doi.org/10.1080/15440478.2022.2128145).
The analytic CIE 1931 color-matching curves follow Wyman, Sloan, and Shirley
(2013), https://jcgt.org/published/0002/02/01/.  The unavailable reference
colorant deliberately has a spectrum outside the six-dye recipe family, so a
recipe that matches under one illuminant need not match under the other.

Run ``python benchmark_color_formulation.py`` for the default 10 independent
trials (5 initial points and 45 total evaluations per trial), or add ``--quick``
for a smoke test.  Results are written beneath
``benchmark_results/color_formulation_2obj_6d/``.
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark


DIM = 6
WAVELENGTHS_NM = torch.linspace(400.0, 700.0, 31, dtype=torch.double)
DAYLIGHT_DELTA_E_SCALE = 60.0
METAMERISM_SCALE = 18.0


def _gaussian(center_nm: float, width_nm: float) -> torch.Tensor:
    return torch.exp(-0.5 * ((WAVELENGTHS_NM - center_nm) / width_nm).square())


# Fixed, nonnegative K/S increments for six hypothetical textile dyes.  Broad
# secondary absorption bands keep the spectra realistic enough to exhibit
# illuminant-dependent matches without requiring proprietary dye measurements.
DYE_KS = torch.stack(
    (
        1.75 * _gaussian(610.0, 34.0) + 0.22 * _gaussian(445.0, 52.0),
        1.55 * _gaussian(535.0, 31.0) + 0.18 * _gaussian(650.0, 42.0),
        1.65 * _gaussian(448.0, 28.0) + 0.16 * _gaussian(585.0, 65.0),
        1.20 * _gaussian(495.0, 48.0) + 0.38 * _gaussian(632.0, 27.0),
        1.35 * _gaussian(575.0, 41.0) + 0.24 * _gaussian(420.0, 24.0),
        0.55 + 0.22 * _gaussian(510.0, 115.0),
    )
)
SUBSTRATE_KS = 0.012 + 0.008 * _gaussian(430.0, 75.0)
MAX_DYE_LOADING = 0.85


def _asymmetric_gaussian(
    center_nm: float, left_scale: float, right_scale: float
) -> torch.Tensor:
    scale = torch.where(
        WAVELENGTHS_NM < center_nm,
        torch.tensor(left_scale, dtype=torch.double),
        torch.tensor(right_scale, dtype=torch.double),
    )
    return torch.exp(-0.5 * ((WAVELENGTHS_NM - center_nm) * scale).square())


# Analytic approximations to the CIE 1931 2-degree color-matching functions.
X_BAR = (
    1.056 * _asymmetric_gaussian(599.8, 0.0264, 0.0323)
    + 0.362 * _asymmetric_gaussian(442.0, 0.0624, 0.0374)
    - 0.065 * _asymmetric_gaussian(501.1, 0.0490, 0.0382)
)
Y_BAR = 0.821 * _asymmetric_gaussian(568.8, 0.0213, 0.0247) + 0.286 * (
    _asymmetric_gaussian(530.9, 0.0613, 0.0322)
)
Z_BAR = 1.217 * _asymmetric_gaussian(437.0, 0.0845, 0.0278) + 0.681 * (
    _asymmetric_gaussian(459.0, 0.0385, 0.0725)
)
COLOR_MATCHING = torch.stack((X_BAR, Y_BAR, Z_BAR), dim=-1)


def _blackbody(temperature_kelvin: float) -> torch.Tensor:
    wavelength_m = WAVELENGTHS_NM * 1.0e-9
    second_radiation_constant = 1.438776877e-2
    spectrum = 1.0 / (
        wavelength_m.pow(5)
        * torch.expm1(second_radiation_constant / (wavelength_m * temperature_kelvin))
    )
    return spectrum / spectrum.max()


ILLUMINANTS = torch.stack((_blackbody(6504.0), _blackbody(2856.0)))


def _reflectance(ks: torch.Tensor) -> torch.Tensor:
    """Invert the opaque Kubelka--Munk relation K/S=(1-R)^2/(2R)."""

    return 1.0 + ks - torch.sqrt(ks.square() + 2.0 * ks)


def _xyz_and_white(reflectance: torch.Tensor, illuminant: torch.Tensor):
    weighted_matching = illuminant.unsqueeze(-1) * COLOR_MATCHING.to(reflectance)
    normalization = weighted_matching[:, 1].sum().reciprocal()
    xyz = normalization * torch.einsum(
        "...w,wc->...c", reflectance, weighted_matching
    )
    white = normalization * weighted_matching.sum(dim=0)
    return xyz, white


def _lab(reflectance: torch.Tensor, illuminant: torch.Tensor) -> torch.Tensor:
    xyz, white = _xyz_and_white(reflectance, illuminant.to(reflectance))
    ratio = xyz / white
    delta = 6.0 / 29.0
    transformed = torch.where(
        ratio > delta**3,
        ratio.clamp_min(0.0).pow(1.0 / 3.0),
        ratio / (3.0 * delta**2) + 4.0 / 29.0,
    )
    fx, fy, fz = transformed.unbind(dim=-1)
    return torch.stack(
        (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)),
        dim=-1,
    )


# The target is produced by a reference pigment unavailable in the formulation
# palette.  Its narrow spectral structure creates genuine metameric ambiguity.
TARGET_KS = (
    SUBSTRATE_KS
    + 0.92 * _gaussian(482.0, 19.0)
    + 0.74 * _gaussian(622.0, 25.0)
    + 0.18 * _gaussian(550.0, 72.0)
)
TARGET_REFLECTANCE = _reflectance(TARGET_KS)
TARGET_LAB = torch.cat(
    tuple(_lab(TARGET_REFLECTANCE, illuminant) for illuminant in ILLUMINANTS)
)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Return daylight and tungsten Lab coordinates for each dye recipe."""

    original_shape = X.shape[:-1]
    flat = X.double().reshape(-1, DIM).clamp(0.0, 1.0)
    concentrations = MAX_DYE_LOADING * flat
    ks = SUBSTRATE_KS.to(flat) + concentrations @ DYE_KS.to(flat)
    reflectance = _reflectance(ks)
    components = torch.cat(
        tuple(_lab(reflectance, illuminant) for illuminant in ILLUMINANTS.to(flat)),
        dim=-1,
    )
    return components.reshape(*original_shape, 6)


def compose(H: torch.Tensor) -> torch.Tensor:
    """Map signed Lab components to color error and metameric mismatch."""

    target = TARGET_LAB.to(H)
    daylight_error = H[..., :3] - target[:3]
    tungsten_error = H[..., 3:] - target[3:]
    daylight = torch.linalg.vector_norm(daylight_error, dim=-1)
    metamerism = torch.linalg.vector_norm(
        tungsten_error - daylight_error, dim=-1
    )
    return torch.stack(
        (daylight / DAYLIGHT_DELTA_E_SCALE, metamerism / METAMERISM_SCALE),
        dim=-1,
    )


PROBLEM = BenchmarkProblem(
    name="Multi-illuminant color formulation (2 objectives, 6 dimensions)",
    slug="color_formulation_2obj_6d",
    dim=DIM,
    num_objectives=2,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 1.05, dtype=torch.double),
    num_components=6,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
