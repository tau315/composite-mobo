"""RCM46 Optimal Power Flow benchmark: the same real 14-bus system as
RCM40 (benchmark_rcm40.py), but with 4 objectives instead of 2.

From the CEC2021 Real-World Constrained Multi-Objective Optimization suite
(Kumar et al. 2021), case 46 -- identical admittance matrices, load
vectors, and 34-dim design space as case 40, but minimizing fuel cost and
voltage deviation in addition to active/reactive power loss. Two of the
four objectives (fuel cost, voltage deviation) are pure closed-form
functions of the design variables themselves (generator setpoints, bus
voltage magnitudes); the other two need the admittance-matrix solve
(I = YV) that RCM40 also needs.

Confirms a genuine discrepancy with RCM40: RCM40's second objective uses
`imag(V(1)*conj(I(2)))`, an apparent index slip (bus 2's current against
bus 1's voltage). This benchmark's independently-written analogous term
uses the dimensionally-consistent `imag(V(1)*conj(I(1)))` -- the same
formula shape, written correctly, suggesting RCM40's version really is a
typo in the original suite.

Only the admittance solve is a component. Fuel cost and voltage deviation are
exact algebraic functions of the design variables, so they are computed inside
``compose`` from the designs themselves; giving them to a GP would spend a fit
learning a formula we already have, and charge the composite arm that fit's
error on two of its four objectives.
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark
from benchmark_rcm40 import _B, _G, DIM, NUM_BUSES

GEN_BUSES = [1, 2, 5, 7]  # 0-indexed bus numbers {2,3,6,8} (1-indexed)
# Fuel-cost coefficients (CEC2021_func.m case 46: b1, c1 for ng=[1,2,3,6,8];
# bus 1's own coefficients are never used since Pg(1) is fixed at 0, not a
# design variable -- only the last 4 entries, matching GEN_BUSES order).
_FUEL_B = [1.75, 1.0, 3.25, 3.0]
_FUEL_C = [0.0175, 0.0625, 0.00834, 0.025]
_LOWER = torch.tensor([-1.0] * 26 + [0.0] * 8, dtype=torch.double)
_UPPER = torch.tensor([1.0] * 34, dtype=torch.double)


def _native(X: torch.Tensor) -> torch.Tensor:
    """Map the unit design cube onto the suite's own variable bounds."""

    return _LOWER.to(X) + X.double() * (_UPPER - _LOWER).to(X)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Per-bus active and reactive power injection, from the one admittance
    solve (I = YV, S = V conj(I)). Nothing else is simulated."""

    X_native = _native(X).reshape(-1, DIM)
    Y = torch.tensor(_G, dtype=torch.double) + 1j * torch.tensor(_B, dtype=torch.double)
    V = torch.zeros(X_native.shape[0], NUM_BUSES, dtype=torch.complex128)
    V[:, 0] = 1.0
    V[:, 1:14] = torch.complex(X_native[:, 0:13], X_native[:, 13:26])

    S = V * (V @ Y.T).conj()
    return torch.cat([S.real, S.imag], dim=-1)


def compose(H: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """f1 = fuel cost, f2 = active power loss, f3 = reactive power loss,
    f4 = voltage deviation (CEC2021_func.m case 46's own formulas).

    f1 and f4 read the designs directly, so they carry no surrogate error.
    """

    X_native = _native(X)
    Pg = X_native[..., 26:30]
    V_mag = torch.sqrt(X_native[..., 0:13] ** 2 + X_native[..., 13:26] ** 2)

    fuel_b = torch.tensor(_FUEL_B, dtype=H.dtype, device=H.device)
    fuel_c = torch.tensor(_FUEL_C, dtype=H.dtype, device=H.device)
    # Exact, so broadcast onto the component posterior's sample dimensions.
    padding = torch.zeros_like(H[..., 0])
    f1 = (fuel_b * Pg + fuel_c * Pg ** 2).sum(dim=-1) + padding
    f4 = ((1.0 - V_mag) ** 2).sum(dim=-1) + padding
    f2 = H[..., 0:14].sum(dim=-1)
    f3 = H[..., 14:28].sum(dim=-1)
    return torch.stack([f1, f2, f3, f4], dim=-1)


PROBLEM = BenchmarkProblem(
    name="RCM46 Optimal Power Flow (4 objectives, 34 dimensions)",
    slug="rcm46_opf_4obj_34d",
    dim=DIM,
    num_objectives=4,
    suite="high",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.double),
    ref_point=torch.tensor([10.0, 150.0, 400.0, 10.0], dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
