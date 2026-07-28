"""Three-objective RGB-selective multilayer nanoparticle benchmark.

The six inputs are the core radius and five shell thicknesses. The simulator
implements the Mie-scattering equations used by Kim et al. over 201 visible
wavelength samples. Each color objective is composed from its own in-band and
out-of-band integrated scattering intensities.
"""

from __future__ import annotations

import numpy as np
from scipy import special
import torch

from benchmark_common import BenchmarkProblem, run_benchmark


DIM = 6
WAVELENGTHS = np.linspace(350.0, 750.0, 201)
BANDS = ((400.0, 500.0), (500.0, 600.0), (600.0, 700.0))
_COMPONENT_CACHE: dict[bytes, np.ndarray] = {}


def _spherical_j(order: int, x: np.ndarray) -> np.ndarray:
    return special.jv(order + 0.5, x) / np.sqrt(x)


def _spherical_y(order: int, x: np.ndarray) -> np.ndarray:
    return special.yv(order + 0.5, x) / np.sqrt(x)


def _spherical_j_derivative(order: int, x: np.ndarray) -> np.ndarray:
    return (
        order * _spherical_j(order - 1, x)
        - (order + 1) * _spherical_j(order + 1, x)
    ) / (2 * order + 1)


def _spherical_y_derivative(order: int, x: np.ndarray) -> np.ndarray:
    return (
        order * _spherical_y(order - 1, x)
        - (order + 1) * _spherical_y(order + 1, x)
    ) / (2 * order + 1)


def _matrix_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply flattened 2x2 matrices stored along the first axis."""

    return np.stack(
        (
            A[0] * B[0] + A[2] * B[1],
            A[1] * B[0] + A[3] * B[1],
            A[0] * B[2] + A[2] * B[3],
            A[1] * B[2] + A[3] * B[3],
        )
    )


def _mie_spectrum(normalized_X: np.ndarray) -> np.ndarray:
    """Return exact normalized scattering spectra for a batch of designs."""

    thicknesses = 30.0 + 40.0 * np.asarray(normalized_X, dtype=np.float64)
    n_designs = len(thicknesses)
    n_wavelengths = len(WAVELENGTHS)
    omega = 2.0 * np.pi / WAVELENGTHS

    permittivity = np.empty((DIM + 1, n_wavelengths), dtype=np.float64)
    titanium_dioxide = 5.913 + 0.2441 / (
        WAVELENGTHS**2 * 1.0e-6 - 0.0803
    )
    for layer in range(DIM):
        permittivity[layer] = 2.04 if layer % 2 == 0 else titanium_dioxide
    permittivity[-1] = 1.77

    radii = np.cumsum(thicknesses, axis=1)
    scattering = np.zeros((n_designs, n_wavelengths), dtype=np.float64)

    for order in range(1, 13):
        for polarization in (1, 2):
            transfer = np.empty(
                (4, n_designs, n_wavelengths), dtype=np.float64
            )
            transfer[0] = 1.0
            transfer[1] = 0.0
            transfer[2] = 0.0
            transfer[3] = 1.0

            for layer in range(DIM):
                radius = radii[:, layer, None]
                x_inner = (
                    omega[None, :]
                    * np.sqrt(permittivity[layer])[None, :]
                    * radius
                )
                x_outer = (
                    omega[None, :]
                    * np.sqrt(permittivity[layer + 1])[None, :]
                    * radius
                )

                j_inner = _spherical_j(order, x_inner)
                y_inner = _spherical_y(order, x_inner)
                j_inner_d = (
                    _spherical_j_derivative(order, x_inner) * x_inner + j_inner
                )
                y_inner_d = (
                    _spherical_y_derivative(order, x_inner) * x_inner + y_inner
                )
                j_outer = _spherical_j(order, x_outer)
                y_outer = _spherical_y(order, x_outer)
                j_outer_d = (
                    _spherical_j_derivative(order, x_outer) * x_outer + j_outer
                )
                y_outer_d = (
                    _spherical_y_derivative(order, x_outer) * x_outer + y_outer
                )

                if polarization == 1:
                    left = np.stack(
                        (y_outer_d, -j_outer_d, -y_outer, j_outer)
                    )
                    right = np.stack(
                        (j_inner, j_inner_d, y_inner, y_inner_d)
                    )
                else:
                    left = np.stack(
                        (
                            permittivity[layer][None, :] * y_outer_d,
                            -permittivity[layer][None, :] * j_outer_d,
                            -y_outer,
                            j_outer,
                        )
                    )
                    right = np.stack(
                        (
                            j_inner,
                            permittivity[layer + 1][None, :] * j_inner_d,
                            y_inner,
                            permittivity[layer + 1][None, :] * y_inner_d,
                        )
                    )
                interface = _matrix_product(left, right)
                transfer = _matrix_product(interface, transfer)

            ratio = transfer[0] / transfer[1]
            reflection = (ratio - 1j) / (ratio + 1j)
            coefficient = (
                (2 * order + 1)
                * np.pi
                / (2.0 * omega**2 * permittivity[-1])
            )
            scattering += coefficient[None, :] * np.abs(1.0 - reflection) ** 2

    area = np.pi * thicknesses.sum(axis=1, keepdims=True) ** 2
    return scattering / area


def _integrated_components(spectra: np.ndarray) -> np.ndarray:
    components: list[np.ndarray] = []
    for lower, upper in BANDS:
        in_band = (WAVELENGTHS >= lower) & (WAVELENGTHS < upper)
        inside = spectra[:, in_band].sum(axis=1)
        outside = spectra[:, ~in_band].sum(axis=1)
        components.extend((inside, outside))
    return np.stack(components, axis=-1)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Evaluate and cache the six color-specific spectral integrals."""

    points = np.asarray(X.detach().double().cpu(), dtype=np.float64)
    result = np.empty((len(points), 6), dtype=np.float64)
    missing_indices: list[int] = []
    missing_points: list[np.ndarray] = []
    keys: list[bytes] = []
    for index, point in enumerate(points):
        key = point.tobytes()
        keys.append(key)
        cached = _COMPONENT_CACHE.get(key)
        if cached is None:
            missing_indices.append(index)
            missing_points.append(point)
        else:
            result[index] = cached

    if missing_points:
        spectra = _mie_spectrum(np.stack(missing_points))
        values = _integrated_components(spectra)
        for index, value in zip(missing_indices, values):
            result[index] = value
            _COMPONENT_CACHE[keys[index]] = value.copy()

    return torch.from_numpy(result).to(dtype=torch.double, device=X.device)


def compose(H: torch.Tensor) -> torch.Tensor:
    """Minimize the fraction of scattering outside each target color band."""

    objectives = []
    for offset in (0, 2, 4):
        inside = H[..., offset].clamp_min(0.0)
        outside = H[..., offset + 1].clamp_min(0.0)
        objectives.append(outside / (inside + outside).clamp_min(1.0e-12))
    return torch.stack(objectives, dim=-1)


PROBLEM = BenchmarkProblem(
    name="RGB-selective multilayer nanoparticle (3 objectives, 6 dimensions)",
    slug="nanoparticle_rgb_3obj_6d",
    dim=DIM,
    num_objectives=3,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(3, dtype=torch.double),
    ref_point=torch.full((3,), 2.5, dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
