#!/usr/bin/env python3
r"""A composite-structure version of the Penicillin fed-batch fermentation
simulator (Liang et al. 2021, ported into botorch as
`botorch.test_functions.multi_objective.Penicillin`).

The underlying simulator is a genuinely correlated, ~2500-step Euler
integration of 5 state variables (penicillin concentration `P`, culture
volume `V`, biomass concentration `X`, glucose concentration `S`, and
accumulated CO2), but botorch's implementation only ever returns the
*final* state `[-P, CO2, t_stop]` -- there's no raw trajectory exposed to
model. This module forks the integrator (copied and modified from
`botorch.test_functions.multi_objective.Penicillin.penicillin_vectorized`,
credited there to https://github.com/HarryQL/TuRBO-Penicillin) to
additionally checkpoint the 5-variable state at `K` fixed absolute step
indices along the integration, giving a genuinely correlated (adjacent
checkpoints are close in time, hence close in value) `5*K + 1`-dim raw
response -- the `+1` is the stopping-time `t_stop`, which isn't part of the
5 checkpointed state variables but is needed for the `time` objective.

Some fermentation runs finish (go "inactive": culture volume exceeds a
max, glucose runs out, or the process rate flattens) before the full 2500
steps. Once a design's trajectory goes inactive, its state variables simply
stop updating -- so a checkpoint taken at any step index is exactly the
state at `min(step, stopping_time)`, a well-defined and correct semantic
whether or not that particular design was still active at that step. The
final checkpoint (step 2500) is therefore *guaranteed* to equal the true
final state for every design, regardless of when it actually stopped --
which is what makes the reduction below exactly reproduce
`Penicillin.penicillin_vectorized`'s own output.
"""
from typing import Tuple

import torch
from torch import Tensor


class _PenicillinConstants:
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
    T_v = 273.0  # Kelvin
    T_o = 373.0
    R = 1.9872  # CAL/(MOL K)
    V_max = 180.0


PENICILLIN_BOUNDS = [
    (60.0, 120.0),
    (0.05, 18.0),
    (293.0, 303.0),
    (0.05, 18.0),
    (0.01, 0.5),
    (500.0, 700.0),
    (5.0, 6.5),
]
PENICILLIN_DIM = 7


def _checkpointed_penicillin_vectorized(
    X_input: Tensor, checkpoint_steps: Tensor
) -> Tensor:
    r"""Fork of `Penicillin.penicillin_vectorized` that additionally
    checkpoints the 5-variable state at `checkpoint_steps`.

    Args:
        X_input: `n x 7`-dim tensor of inputs (same layout as upstream:
            culture volume, biomass concentration, temperature, glucose
            concentration, substrate feed rate, substrate feed
            concentration, H+ concentration).
        checkpoint_steps: `K`-dim `LongTensor` of step indices (1-2500) at
            which to record the state, sorted ascending. The last entry
            should be 2500 to guarantee the final checkpoint equals the
            true final state.

    Returns:
        An `n x (5*K + 1)`-dim tensor: for each of the `K` checkpoints (in
        order), the `[P, V, X, S, CO2]` state at that step, followed by a
        final column with the stopping time `t_stop`.
    """
    c = _PenicillinConstants
    V, X, T, S, F, s_f, H_ = torch.split(X_input, 1, -1)
    P, CO2 = torch.zeros_like(V), torch.zeros_like(V)
    H = torch.full_like(H_, 10.0).pow(-H_)

    active = torch.ones_like(V).bool()
    t_tensor = torch.full_like(V, 2500)

    checkpoint_set = set(int(s) for s in checkpoint_steps.tolist())
    snapshots = {}

    for t in range(1, 2501):
        if active.sum() == 0:
            break
        F_loss = (
            V[active]
            * c.lambd
            * (torch.exp(5 * ((T[active] - c.T_o) / (c.T_v - c.T_o))) - 1)
        )
        dV_dt = F[active] - F_loss
        mu = (
            (c.mu_X / (1 + c.K_1 / H[active] + H[active] / c.K_2))
            * (S[active] / (c.K_X * X[active] + S[active]))
            * (
                (c.k_g * torch.exp(-c.E_g / (c.R * T[active])))
                - (c.k_d * torch.exp(-c.E_d / (c.R * T[active])))
            )
        )
        dX_dt = mu * X[active] - (X[active] / V[active]) * dV_dt
        mu_pp = c.mu_p * (S[active] / (c.K_p + S[active] + S[active].pow(2) / c.K_I))
        dS_dt = (
            -(mu / c.Y_xs) * X[active]
            - (mu_pp / c.Y_ps) * X[active]
            - c.m_X * X[active]
            + F[active] * s_f[active] / V[active]
            - (S[active] / V[active]) * dV_dt
        )
        dP_dt = (mu_pp * X[active]) - c.K * P[active] - (P[active] / V[active]) * dV_dt
        dCO2_dt = c.alpha_1 * dX_dt + c.alpha_2 * X[active] + c.alpha_3

        # UPDATE
        P[active] = P[active] + dP_dt
        V[active] = V[active] + dV_dt
        X[active] = X[active] + dX_dt
        S[active] = S[active] + dS_dt
        CO2[active] = CO2[active] + dCO2_dt

        # Update active indices
        full_dpdt = torch.ones_like(P)
        full_dpdt[active] = dP_dt
        inactive = (V > c.V_max) + (S < 0) + (full_dpdt < 10e-12)
        t_tensor[inactive] = torch.minimum(
            t_tensor[inactive], torch.full_like(t_tensor[inactive], t)
        )
        active[inactive] = 0

        if t in checkpoint_set:
            snapshots[t] = (P.clone(), V.clone(), X.clone(), S.clone(), CO2.clone())

    # Any checkpoint step at or beyond wherever the loop stopped (all designs
    # inactive, or step 2500 reached) never got recorded above -- fill it in
    # with the current (frozen-final) state, which is correct since nothing
    # changes after every design has gone inactive.
    for cp in checkpoint_set:
        if cp not in snapshots:
            snapshots[cp] = (P.clone(), V.clone(), X.clone(), S.clone(), CO2.clone())

    cols = []
    for cp in sorted(checkpoint_set):
        cols.extend(snapshots[cp])
    cols.append(t_tensor)
    return torch.cat(cols, dim=-1)


def get_composite_penicillin_fn(
    n_checkpoints: int = 10, dtype=torch.double, device=None
) -> Tuple[callable, Tensor]:
    r"""Construct the raw-response (checkpointed-trajectory) function and
    bounds for composite Penicillin.

    Args:
        n_checkpoints: number of trajectory checkpoints `K`. The last
            checkpoint is always step 2500 (guarantees the final checkpoint
            equals the true final state); the rest are evenly spaced from
            step 2500/K up to 2500.
        dtype, device: for the returned bounds.

    Returns:
        A tuple `(raw_response, bounds)`:
            raw_response: callable mapping an `n x 7`-dim tensor (raw,
                *unnormalized* problem-space Penicillin inputs) to an
                `n x (5*n_checkpoints + 1)`-dim tensor.
            bounds: `2 x 7`-dim tensor, Penicillin's own raw-space bounds.
    """
    checkpoint_steps = torch.linspace(2500 / n_checkpoints, 2500, n_checkpoints).round().long()

    def raw_response(X_input: Tensor) -> Tensor:
        return _checkpointed_penicillin_vectorized(
            X_input.view(-1, PENICILLIN_DIM).clone(), checkpoint_steps
        ).view(*X_input.shape[:-1], 5 * n_checkpoints + 1)

    bounds = torch.tensor(PENICILLIN_BOUNDS, dtype=dtype, device=device).t()
    return raw_response, bounds


def composite_penicillin_reduction(Y_raw: Tensor) -> Tensor:
    r"""Known reduction: the last checkpoint's `P`/`CO2`, plus the appended
    stopping time, are exactly Penicillin's own 3 objectives.

    Args:
        Y_raw: `... x (5*K + 1)`-dim tensor as produced by `raw_response`
            above (un-negated except for the sign convention already baked
            into `P`, `V`, `X`, `S`, `CO2` themselves, which matches
            upstream `penicillin_vectorized`'s own internal state -- only
            the final `[-P, CO2, t]` stacking applies a negation, exactly
            as upstream does).

    Returns:
        An `... x 3`-dim tensor, exactly `[-P_final, CO2_final, t_stop]` --
        bit-identical to `Penicillin.penicillin_vectorized`'s own return
        value. This is the *minimization*-convention triple, exactly like
        upstream's `evaluate_true(negate=False)`; the composite evalfn
        should therefore use the same `negate=True` `BenchmarkFunction`
        convention as every other non-composite-DTLZ2 evalfn (unlike
        `composite_dtlz2.py`/`composite_dtlz2_curve.py`, which fold the
        negation into their own reductions because of DTLZ2's multiplicative
        `(1+g)` coupling -- no such coupling exists here, so there's nothing
        to fold in).
    """
    P_final = Y_raw[..., -6]
    CO2_final = Y_raw[..., -2]
    t_final = Y_raw[..., -1]
    return torch.stack([-P_final, CO2_final, t_final], dim=-1)
