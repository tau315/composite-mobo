"""RCM40 Optimal Power Flow benchmark (real IEEE 14-bus test system).

From the CEC2021 Real-World Constrained Multi-Objective Optimization suite
(Kumar et al. 2021), case 40 -- a real IEEE 14-bus test system (Biswas et al.
2020). The 34 inputs are the real/imaginary bus-voltage components and
active/reactive generator setpoints; the two objectives are total active and
reactive power loss, each a linear combination of a genuinely bus-decoupled
per-bus power-injection vector computed from one admittance-matrix solve
(I = YV, S = V * conj(I)) -- no iterative simulation, just linear algebra,
so this benchmark is cheap to evaluate even at full precision.

All numeric constants (the 14x14 admittance matrices G, B and the per-bus
load vectors P, Q) were extracted directly from the reference MATLAB
implementation (CEC2021_func.m, case 40, and Cal_par.m's xmin40/xmax40) --
https://github.com/P-N-Suganthan/2021-RW-MOP/raw/main/CEC2021-RWCMOP.zip --
not retyped from a paper's typeset equations.
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark

DIM = 34
NUM_BUSES = 14

_G = [[6.025029055768224, -4.999131600798035, 0.0, 0.0, -1.025897454970189, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-4.999131600798035, 9.521323610814779, -1.1350191923073958, -1.686033150614943, -1.7011396670944048, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.1350191923073958, 3.1209949022329564, -1.9859757099255606, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.686033150614943, -1.9859757099255606, 10.512989522036175, -6.840980661495671, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.025897454970189, -1.7011396670944048, 0.0, -6.840980661495671, 9.568017783560265, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 6.5799234074662225, 0.0, 0.0, 0.0, 0.0, -1.9550285631772606, -1.525967440450974, -3.0989274038379877, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.3260550394673585, -3.9020495524474277, 0.0, 0.0, 0.0, -1.424005487019931], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.9020495524474277, 5.782934306147827, -1.8808847537003996, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -1.9550285631772606, 0.0, 0.0, 0.0, -1.8808847537003996, 3.8359133168776602, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -1.525967440450974, 0.0, 0.0, 0.0, 0.0, 0.0, 4.014992027272893, -2.4890245868219187, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -3.0989274038379877, 0.0, 0.0, 0.0, 0.0, 0.0, -2.4890245868219187, 6.724946148466233, -1.1369941578063267], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.424005487019931, 0.0, 0.0, 0.0, -1.1369941578063267, 2.560999644826258]]
_B = [[-19.447070205514382, 15.263086523179553, 0.0, 0.0, 4.234983682334831, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [15.263086523179553, -30.272115398779064, 4.781863151757718, 5.115838325872083, 5.193927397969713, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 4.781863151757718, -9.82238012935164, 5.0688169775939205, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 5.115838325872083, 5.0688169775939205, -38.654171207607796, 21.578553981691588, 0.0, 4.889512660317341, 0.0, 1.8554995578159004, 0.0, 0.0, 0.0, 0.0, 0.0], [4.234983682334831, 5.193927397969713, 0.0, 21.578553981691588, -35.533639456044824, 4.257445335253384, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 4.257445335253384, -17.34073280991911, 0.0, 0.0, 0.0, 0.0, 4.0940743442404415, 3.1759639650294003, 6.102755448193116, 0.0], [0.0, 0.0, 0.0, 4.889512660317341, 0.0, 0.0, -19.549005948264654, 5.676979846721544, 9.09008271975275, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.676979846721544, -5.676979846721544, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.8554995578159004, 0.0, 0.0, 9.09008271975275, 0.0, -24.092506375267877, 10.365394127060915, 0.0, 0.0, 0.0, 3.0290504569306034], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.365394127060915, -14.768337876521436, 4.402943749460521, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 4.0940743442404415, 0.0, 0.0, 0.0, 4.402943749460521, -8.497018093700962, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 3.1759639650294003, 0.0, 0.0, 0.0, 0.0, 0.0, -5.427938591201612, 2.251974626172212, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 6.102755448193116, 0.0, 0.0, 0.0, 0.0, 0.0, 2.251974626172212, -10.66969354947068, 2.314963475105352], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0290504569306034, 0.0, 0.0, 0.0, 2.314963475105352, -5.344013932035955]]
# Design bounds, per Cal_par.m: xmin40 = -1*ones(1,34); xmin40(27:34)=0; xmax40 = +1*ones(1,34).
_LOWER = torch.tensor([-1.0] * 26 + [0.0] * 8, dtype=torch.double)
_UPPER = torch.tensor([1.0] * 34, dtype=torch.double)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Per-bus active/reactive power injection, plus Im(I_2) for the
    second objective's slack-bus term (see CEC2021_func.m case 40)."""

    X_native = (_LOWER + X.double() * (_UPPER - _LOWER)).reshape(-1, DIM)
    n = X_native.shape[0]
    Y = torch.tensor(_G, dtype=torch.double) + 1j * torch.tensor(_B, dtype=torch.double)
    V = torch.zeros(n, NUM_BUSES, dtype=torch.complex128)
    V[:, 0] = 1.0
    V[:, 1:14] = torch.complex(X_native[:, 0:13], X_native[:, 13:26])

    I = V @ Y.T
    S = V * I.conj()
    Psp = S.real
    Qsp = S.imag
    Im_I2 = I[:, 1].imag

    return torch.cat([Psp, Qsp, Im_I2.unsqueeze(-1)], dim=-1)


def compose(H: torch.Tensor) -> torch.Tensor:
    """f1 = total active power loss, f2 = total reactive power loss
    (CEC2021_func.m case 40's own formula, including its apparent
    I(2)/V(1) index slip in f2 -- reproduced exactly as published)."""

    Psp = H[..., 0:14]
    Qsp = H[..., 14:28]
    Im_I2 = H[..., 28]
    f1 = Psp.sum(dim=-1)
    f2 = -Im_I2 + Qsp[..., 1:14].sum(dim=-1)
    return torch.stack([f1, f2], dim=-1)


PROBLEM = BenchmarkProblem(
    name="RCM40 Optimal Power Flow (2 objectives, 34 dimensions)",
    slug="rcm40_opf_2obj_34d",
    dim=DIM,
    num_objectives=2,
    suite="high",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.tensor([0.0, 0.0], dtype=torch.double),
    ref_point=torch.tensor([200.0, 500.0], dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
