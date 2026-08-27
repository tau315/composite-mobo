"""Low-D scientific benchmark: balanced IEEE-14 optimal power flow.

This uses RCM40's official 34-variable voltage/generator representation and
admittance data.  The original problem has 26 equality constraints.  To keep
all four compared solvers on the same bound-constrained multi-objective API,
the active/reactive balance residual is exposed as a third objective, in the
style of the RE engineering suite.  This is therefore a documented
reformulation of RCM40, not the untouched constrained CEC case.

The five observed physical intermediates are total active loss, the first two
moments of voltage magnitude, and active/reactive RMS balance residuals.  The
known outer map reconstructs voltage deviation and squares the RMS residuals.
No iterative power-flow solve or numerical integration occurs--only the fixed
complex admittance matrix product.

Paper protocol: 5 initial plus 40 adaptive design evaluations (45 total).
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark
from benchmark_rcm40 import _B, _G, DIM, NUM_BUSES


_LOWER = torch.tensor([-1.0] * 26 + [0.0] * 8, dtype=torch.double)
_UPPER = torch.ones(DIM, dtype=torch.double)
_YBUS = torch.tensor(_G, dtype=torch.double) + 1j * torch.tensor(
    _B, dtype=torch.double
)
_P_LOAD = torch.tensor(
    [0.0, 0.217, 0.942, 0.478, 0.076, 0.112, 0.0, 0.0,
     0.295, 0.090, 0.035, 0.061, 0.135, 0.149],
    dtype=torch.double,
)
_Q_LOAD = torch.tensor(
    [0.0, 0.127, 0.190, -0.039, 0.016, 0.075, 0.0, 0.0,
     0.166, 0.058, 0.018, 0.016, 0.058, 0.050],
    dtype=torch.double,
)
_GENERATOR_BUSES = torch.tensor([1, 2, 5, 7])
_ACTIVE_LOSS_SCALE = 200.0
_BALANCE_RMS_SCALE = 25.0


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    native = _LOWER.to(X) + X.double() * (_UPPER.to(X) - _LOWER.to(X))
    voltage = torch.zeros(len(native), NUM_BUSES, dtype=torch.complex128, device=X.device)
    voltage[:, 0] = 1.0
    voltage[:, 1:] = torch.complex(native[:, :13], native[:, 13:26])

    current = voltage @ _YBUS.to(X.device).T
    injected = voltage * current.conj()
    p_injected, q_injected = injected.real, injected.imag
    p_generated = torch.zeros_like(p_injected)
    q_generated = torch.zeros_like(q_injected)
    p_generated[:, _GENERATOR_BUSES] = native[:, 26:30]
    q_generated[:, _GENERATOR_BUSES] = native[:, 30:34]
    delta_p = p_injected - p_generated + _P_LOAD.to(X)
    delta_q = q_injected - q_generated + _Q_LOAD.to(X)

    active_loss = (p_injected.sum(dim=-1) / _ACTIVE_LOSS_SCALE).clamp_min(0.0)
    voltage_magnitude = voltage[:, 1:].abs()
    mean_voltage = voltage_magnitude.mean(dim=-1)
    mean_square_voltage = voltage_magnitude.square().mean(dim=-1)
    active_balance_rms = delta_p[:, 1:].square().mean(dim=-1).sqrt() / _BALANCE_RMS_SCALE
    reactive_balance_rms = delta_q[:, 1:].square().mean(dim=-1).sqrt() / _BALANCE_RMS_SCALE
    return torch.stack(
        (
            active_loss,
            mean_voltage,
            mean_square_voltage,
            active_balance_rms,
            reactive_balance_rms,
        ),
        dim=-1,
    )


def compose(H: torch.Tensor) -> torch.Tensor:
    active_loss = H[..., 0]
    voltage_deviation = (1.0 - 2.0 * H[..., 1] + H[..., 2]).clamp_min(0.0)
    balance_error = 0.5 * (H[..., 3].square() + H[..., 4].square())
    return torch.stack((active_loss, voltage_deviation, balance_error), dim=-1)


PROBLEM = BenchmarkProblem(
    name="Balanced IEEE-14 OPF (3 objectives, 34 dimensions)",
    slug="rcm40_balanced_3obj_34d",
    dim=DIM,
    num_objectives=3,
    num_components=5,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(3, dtype=torch.double),
    ref_point=torch.full((3,), 1.1, dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
