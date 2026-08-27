"""Low-D scientific benchmark: discrete-wavelength multilayer scattering.

The six design variables are one core radius and five shell thicknesses.  A
multilayer Mie calculation returns scattering efficiencies at 450, 550, and
650 nm.  There is no wavelength integral.  For each color objective, the
target scattering and mean off-target scattering are exposed as a separate
objective-specific pair.  Repeated measurements are deliberately represented
by separate GP outputs so that no cross-objective transfer confounds the
comparison.  Each objective is their nonlinear off-target fraction.

Paper protocol: 5 initial plus 40 adaptive design evaluations (45 total).
"""

from __future__ import annotations

import numpy as np
import torch

from benchmark_common import BenchmarkProblem, run_benchmark
from benchmark_nanoparticle_rgb import _mie_spectrum


DIM = 6
WAVELENGTHS_NM = np.asarray([450.0, 550.0, 650.0], dtype=np.float64)
_CACHE: dict[bytes, np.ndarray] = {}


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    points = np.asarray(X.detach().double().cpu(), dtype=np.float64)
    values = np.empty((len(points), 3), dtype=np.float64)
    missing_indices: list[int] = []
    missing_points: list[np.ndarray] = []
    keys: list[bytes] = []
    for index, point in enumerate(points):
        key = point.tobytes()
        keys.append(key)
        cached = _CACHE.get(key)
        if cached is None:
            missing_indices.append(index)
            missing_points.append(point)
        else:
            values[index] = cached
    if missing_points:
        calculated = _mie_spectrum(np.stack(missing_points), WAVELENGTHS_NM)
        for index, value in zip(missing_indices, calculated):
            values[index] = value
            _CACHE[keys[index]] = value.copy()
    scattering = torch.from_numpy(values).to(dtype=torch.double, device=X.device)
    total = scattering.sum(dim=-1, keepdim=True)
    off_target = 0.5 * (total - scattering)
    # [target_450, off_450, target_550, off_550, target_650, off_650].
    # Although target/off-target quantities share physical measurements, each
    # objective gets its own modeled columns, matching the paper's
    # objective-specific independent-GP protocol.
    return torch.stack(
        (
            scattering[..., 0],
            off_target[..., 0],
            scattering[..., 1],
            off_target[..., 1],
            scattering[..., 2],
            off_target[..., 2],
        ),
        dim=-1,
    )


def compose(H: torch.Tensor) -> torch.Tensor:
    target = H[..., 0::2].clamp_min(0.0)
    off_target = H[..., 1::2].clamp_min(0.0)
    return off_target / (target + off_target).clamp_min(1.0e-12)


PROBLEM = BenchmarkProblem(
    name="Discrete RGB multilayer nanoparticle (3 objectives, 6 dimensions)",
    slug="nanoparticle_discrete_rgb_3obj_6d",
    dim=DIM,
    num_objectives=3,
    num_components=6,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(3, dtype=torch.double),
    ref_point=torch.full((3,), 1.05, dtype=torch.double),
    clear_evaluation_cache=_CACHE.clear,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
