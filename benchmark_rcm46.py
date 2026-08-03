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


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Per-bus active/reactive power injection, plus a pass-through of
    the generator setpoints and bus-voltage magnitudes (not derived from
    any simulation, just relabeled, since `compose` needs them alongside
    the genuinely bus-coupled Psp/Qsp terms and only receives `H`, not
    `X`, in this repo's evaluate_components/compose interface)."""

    X_native = (_LOWER + X.double() * (_UPPER - _LOWER)).reshape(-1, DIM)
    V_r = X_native[:, 0:13]
    V_m = X_native[:, 13:26]
    Y = torch.tensor(_G, dtype=torch.double) + 1j * torch.tensor(_B, dtype=torch.double)
    V = torch.zeros(X_native.shape[0], NUM_BUSES, dtype=torch.complex128)
    V[:, 0] = 1.0
    V[:, 1:14] = torch.complex(V_r, V_m)

    I = V @ Y.T
    S = V * I.conj()
    Psp = S.real
    Qsp = S.imag
    Pg = X_native[:, 26:30]
    V_mag = torch.sqrt(V_r ** 2 + V_m ** 2)  # buses 2-14

    return torch.cat([Psp, Qsp, Pg, V_mag], dim=-1)


def compose(H: torch.Tensor) -> torch.Tensor:
    """f1 = fuel cost, f2 = active power loss, f3 = reactive power loss,
    f4 = voltage deviation (CEC2021_func.m case 46's own formula)."""

    Psp = H[..., 0:14]
    Qsp = H[..., 14:28]
    Pg = H[..., 28:32]
    V_mag = H[..., 32:45]

    fuel_b = torch.tensor(_FUEL_B, dtype=H.dtype, device=H.device)
    fuel_c = torch.tensor(_FUEL_C, dtype=H.dtype, device=H.device)
    f1 = (fuel_b * Pg + fuel_c * Pg ** 2).sum(dim=-1)
    f2 = Psp.sum(dim=-1)
    f3 = Qsp.sum(dim=-1)
    f4 = ((1.0 - V_mag) ** 2).sum(dim=-1)
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
