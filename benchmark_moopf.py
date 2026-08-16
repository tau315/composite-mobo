"""MOOPF-5: 5-objective many-objective Optimal Power Flow.

The same real IEEE 14-bus system and 34-dim design space as RCM40/RCM46
(benchmark_rcm40.py / benchmark_rcm46.py, CEC2021 RWCMOP suite), but with a
FIFTH objective added on top of RCM46's four (fuel cost, active/reactive
power loss, voltage deviation): the Kessel-Glavitsch L-index voltage-
stability margin, the standard static voltage-collapse metric from the
many-objective OPF literature. It needs only the admittance matrix (already
verified for RCM40/46) and the bus voltages -- no fabricated coefficients.

This is a constructed benchmark (assembled from RCM46's verified objectives
plus a standard L-index objective), not a pre-existing CEC2021 case -- the
suite maxes out at RCM46's 4 objectives. It exists to test whether pushing
past that with a fifth genuinely-decoupled objective amplifies composite
modeling's advantage (in this project's own runs it did, dramatically).

The L-index for load bus j: partition the admittance matrix Y into
generator (PV + slack) and load (PQ) blocks, form F = -Y_LL^{-1} Y_LG, and
L_j = |1 - sum_{i in gen} F_ji V_i / V_j|; the system index is max_j L_j
(< 1 = stable, smaller is better). The 9 per-load-bus L_j are exposed as
raw components so the composite GP models a genuinely decoupled per-bus
quantity, reduced by max in `compose`.
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark
from benchmark_rcm40 import _B, _G, DIM, NUM_BUSES
from benchmark_rcm46 import _FUEL_B, _FUEL_C

GEN_BUSES = [1, 2, 5, 7]  # 0-indexed {2,3,6,8}
_GEN_SET = [0] + GEN_BUSES  # slack + generators -> PV/slack buses for the L-index
_LOAD_SET = [b for b in range(NUM_BUSES) if b not in _GEN_SET]  # 9 load (PQ) buses
_LOWER = torch.tensor([-1.0] * 26 + [0.0] * 8, dtype=torch.double)
_UPPER = torch.tensor([1.0] * 34, dtype=torch.double)

_Y = torch.tensor(_G, dtype=torch.double) + 1j * torch.tensor(_B, dtype=torch.double)
_gen_idx = torch.tensor(_GEN_SET)
_load_idx = torch.tensor(_LOAD_SET)
_F_LG = -torch.linalg.solve(_Y[_load_idx][:, _load_idx], _Y[_load_idx][:, _gen_idx])  # (9 x 5)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Per-bus Psp/Qsp injections, pass-through Pg and |V|, plus the 9
    per-load-bus L-index values (see module docstring)."""

    X_native = (_LOWER + X.double() * (_UPPER - _LOWER)).reshape(-1, DIM)
    V_r = X_native[:, 0:13]
    V_m = X_native[:, 13:26]
    V = torch.zeros(X_native.shape[0], NUM_BUSES, dtype=torch.complex128)
    V[:, 0] = 1.0
    V[:, 1:14] = torch.complex(V_r, V_m)

    I = V @ _Y.T
    S = V * I.conj()
    Psp = S.real
    Qsp = S.imag
    Pg = X_native[:, 26:30]
    V_mag = torch.sqrt(V_r ** 2 + V_m ** 2)  # buses 2-14

    V_gen = V[:, _gen_idx]    # (n, 5)
    V_load = V[:, _load_idx]  # (n, 9)
    L_j = torch.abs(1.0 - (V_gen @ _F_LG.T) / V_load)  # (n, 9)

    return torch.cat([Psp, Qsp, Pg, V_mag, L_j], dim=-1)


def compose(H: torch.Tensor) -> torch.Tensor:
    """f1 = fuel cost, f2 = active loss, f3 = reactive loss, f4 = voltage
    deviation (all RCM46's), f5 = max per-load-bus L-index (voltage
    stability)."""

    Psp = H[..., 0:14]
    Qsp = H[..., 14:28]
    Pg = H[..., 28:32]
    V_mag = H[..., 32:45]
    L_j = H[..., 45:54]

    fuel_b = torch.tensor(_FUEL_B, dtype=H.dtype, device=H.device)
    fuel_c = torch.tensor(_FUEL_C, dtype=H.dtype, device=H.device)
    f1 = (fuel_b * Pg + fuel_c * Pg ** 2).sum(dim=-1)
    f2 = Psp.sum(dim=-1)
    f3 = Qsp.sum(dim=-1)
    f4 = ((1.0 - V_mag) ** 2).sum(dim=-1)
    f5 = L_j.amax(dim=-1)
    return torch.stack([f1, f2, f3, f4, f5], dim=-1)


PROBLEM = BenchmarkProblem(
    name="MOOPF 5-objective Optimal Power Flow (5 objectives, 34 dimensions)",
    slug="moopf_opf_5obj_34d",
    dim=DIM,
    num_objectives=5,
    suite="high",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.double),
    ref_point=torch.tensor([10.0, 150.0, 400.0, 10.0, 10.0], dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
