"""Penicillin fed-batch fermentation benchmark (Liang et al. 2021).

The 7 inputs are fermentation process controls (culture volume, biomass
concentration, temperature, glucose concentration, substrate feed rate,
substrate feed concentration, H+ concentration). The simulator is a
~2500-step Euler integration of 5 coupled state variables -- penicillin
concentration P, culture volume V, biomass concentration X, glucose
concentration S, and accumulated CO2 -- of which the public benchmark
(botorch.test_functions.multi_objective.Penicillin) only ever exposes the
*final* state as 3 objectives: [-P, CO2, t_stop]. This version additionally
checkpoints the full 5-variable state at K=5 fixed steps along the
integration, exposing a genuinely correlated 26-dim intermediate vector
(5 checkpoints x 5 state variables, plus the stopping time) instead of
only the final 3 numbers.

Ported from https://github.com/HarryQL/TuRBO-Penicillin (credited there as
the origin of botorch's own Penicillin implementation); the reduction
below reproduces that public implementation's own output bit-for-bit
(verified directly: once a design's trajectory goes inactive -- culture
volume exceeds a max, glucose runs out, or the production rate flattens --
its state simply stops updating, so a checkpoint at step 2500 is
guaranteed to equal the true final state regardless of when a given
design actually stopped).
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark

DIM = 7
N_CHECKPOINTS = 5
_BOUNDS = [
    (60.0, 120.0),
    (0.05, 18.0),
    (293.0, 303.0),
    (0.05, 18.0),
    (0.01, 0.5),
    (500.0, 700.0),
    (5.0, 6.5),
]
_LOWER = torch.tensor([b[0] for b in _BOUNDS], dtype=torch.double)
_UPPER = torch.tensor([b[1] for b in _BOUNDS], dtype=torch.double)
_CHECKPOINT_STEPS = torch.linspace(2500 / N_CHECKPOINTS, 2500, N_CHECKPOINTS).round().long()


class _C:
    Y_xs = 0.45
    Y_ps = 0.90
    K_1 = 10 ** (-10)
    K_2 = 7 * 10 ** (-5)
    m_X = 0.014
    alpha_1 = 0.143
    alpha_2 = 4 * 10 ** (-7)
    alpha_3 = 10 ** (-4)
    mu_X = 0.092
    K_X = 0.15
    mu_p = 0.005
    K_p = 0.0002
    K_I = 0.10
    K = 0.04
    k_g = 7.0 * 10**3
    E_g = 5100.0
    k_d = 10.0**33
    E_d = 50000.0
    lambd = 2.5 * 10 ** (-4)
    T_v = 273.0
    T_o = 373.0
    R = 1.9872
    V_max = 180.0


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Checkpointed [P, V, X, S, CO2] state at 5 fixed steps, plus the
    stopping time -- a 26-dim vector."""

    X_native = (_LOWER + X.double() * (_UPPER - _LOWER)).reshape(-1, DIM)
    V, X_bio, T, S, F, s_f, H_ = torch.split(X_native, 1, -1)
    P, CO2 = torch.zeros_like(V), torch.zeros_like(V)
    H = torch.full_like(H_, 10.0).pow(-H_)

    active = torch.ones_like(V).bool()
    t_tensor = torch.full_like(V, 2500)
    checkpoint_set = set(int(s) for s in _CHECKPOINT_STEPS.tolist())
    snapshots = {}

    for t in range(1, 2501):
        if active.sum() == 0:
            break
        F_loss = V[active] * _C.lambd * (torch.exp(5 * ((T[active] - _C.T_o) / (_C.T_v - _C.T_o))) - 1)
        dV_dt = F[active] - F_loss
        mu = (
            (_C.mu_X / (1 + _C.K_1 / H[active] + H[active] / _C.K_2))
            * (S[active] / (_C.K_X * X_bio[active] + S[active]))
            * ((_C.k_g * torch.exp(-_C.E_g / (_C.R * T[active]))) - (_C.k_d * torch.exp(-_C.E_d / (_C.R * T[active]))))
        )
        dX_dt = mu * X_bio[active] - (X_bio[active] / V[active]) * dV_dt
        mu_pp = _C.mu_p * (S[active] / (_C.K_p + S[active] + S[active].pow(2) / _C.K_I))
        dS_dt = (
            -(mu / _C.Y_xs) * X_bio[active]
            - (mu_pp / _C.Y_ps) * X_bio[active]
            - _C.m_X * X_bio[active]
            + F[active] * s_f[active] / V[active]
            - (S[active] / V[active]) * dV_dt
        )
        dP_dt = (mu_pp * X_bio[active]) - _C.K * P[active] - (P[active] / V[active]) * dV_dt
        dCO2_dt = _C.alpha_1 * dX_dt + _C.alpha_2 * X_bio[active] + _C.alpha_3

        P[active] = P[active] + dP_dt
        V[active] = V[active] + dV_dt
        X_bio[active] = X_bio[active] + dX_dt
        S[active] = S[active] + dS_dt
        CO2[active] = CO2[active] + dCO2_dt

        full_dpdt = torch.ones_like(P)
        full_dpdt[active] = dP_dt
        inactive = (V > _C.V_max) + (S < 0) + (full_dpdt < 10e-12)
        t_tensor[inactive] = torch.minimum(t_tensor[inactive], torch.full_like(t_tensor[inactive], t))
        active[inactive] = 0

        if t in checkpoint_set:
            snapshots[t] = (P.clone(), V.clone(), X_bio.clone(), S.clone(), CO2.clone())

    for cp in checkpoint_set:
        if cp not in snapshots:
            snapshots[cp] = (P.clone(), V.clone(), X_bio.clone(), S.clone(), CO2.clone())

    cols = []
    for cp in sorted(checkpoint_set):
        cols.extend(snapshots[cp])
    cols.append(t_tensor)
    return torch.cat(cols, dim=-1)


def compose(H: torch.Tensor) -> torch.Tensor:
    """f1 = -P_final (maximize penicillin -> minimize -P), f2 = CO2_final,
    f3 = t_stop -- exactly the public Penicillin benchmark's own 3
    objectives, read off the last checkpoint (guaranteed to equal the true
    final state, see module docstring)."""

    P_final = H[..., -6]
    CO2_final = H[..., -2]
    t_final = H[..., -1]
    return torch.stack([-P_final, CO2_final, t_final], dim=-1)


PROBLEM = BenchmarkProblem(
    name="Penicillin fed-batch fermentation (3 objectives, 7 dimensions)",
    slug="penicillin_3obj_7d",
    dim=DIM,
    num_objectives=3,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.tensor([-100.0, 0.0, 0.0], dtype=torch.double),
    ref_point=torch.tensor([0.0, 100.0, 2500.0], dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
