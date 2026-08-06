"""How much of DTLZ2's composite advantage came from exact input reuse?

The old two-argument contract was ``compose(C, X)``, so a benchmark could route
exact candidate coordinates through the known map and give a GP only the part
that is genuinely unknown. The current one-argument ``compose(H)`` cannot do
that, so coordinates that used to be exact are now modelled.

This measures the same held-out surrogate accuracy as diagnose_composite for
three formulations of DTLZ2.
"""

import sys

import torch

sys.path.insert(0, ".")
from diagnose_composite import _standardized_rmse
from solvers import _independent_gp


def evaluate(X):
    X = X.double()
    distance = (X[..., 1:] - 0.5).square().sum(dim=-1)
    angle = torch.pi * X[..., 0] / 2.0
    return torch.stack(
        ((1.0 + distance) * angle.cos(), (1.0 + distance) * angle.sin()), dim=-1
    )


def run(dim, n_train=32, n_test=128, seed=0):
    X = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed).draw(
        n_train + n_test
    ).double()
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = evaluate(X_train), evaluate(X_test)

    direct = _independent_gp(X_train, Y_train)

    # Current contract: distance and both trig terms are all modelled.
    def four_components(Z):
        distance = (Z[..., 1:] - 0.5).square().sum(dim=-1)
        angle = torch.pi * Z[..., 0] / 2.0
        return torch.stack(
            (distance, angle.cos(), distance, angle.sin()), dim=-1
        )

    modelled = _independent_gp(X_train, four_components(X_train))

    # Old contract: only sqrt(distance) is modelled; angles stay exact.
    def one_component(Z):
        return (Z[..., 1:] - 0.5).square().sum(dim=-1, keepdim=True).sqrt()

    exact = _independent_gp(X_train, one_component(X_train))

    with torch.no_grad():
        direct_pred = direct.posterior(X_test).mean
        H = modelled.posterior(X_test).mean
        four_pred = torch.stack(
            (
                (1.0 + H[..., 0].clamp_min(0)) * H[..., 1].clamp(0, 1),
                (1.0 + H[..., 2].clamp_min(0)) * H[..., 3].clamp(0, 1),
            ),
            dim=-1,
        )
        r = exact.posterior(X_test).mean[..., 0].clamp_min(0)
        angle = torch.pi * X_test[..., 0] / 2.0  # exact, never modelled
        exact_pred = torch.stack(
            ((1.0 + r.square()) * angle.cos(), (1.0 + r.square()) * angle.sin()),
            dim=-1,
        )

    d = _standardized_rmse(direct_pred, Y_test).mean()
    f = _standardized_rmse(four_pred, Y_test).mean()
    e = _standardized_rmse(exact_pred, Y_test).mean()
    return float(d), float(f), float(e)


print(f"{'d':>5}{'direct':>9}{'4-comp':>9}{'1-comp+exact':>14}{'adv(4)':>9}{'adv(exact)':>12}")
print("-" * 58)
for dim in (6, 30, 100, 600):
    d, f, e = run(dim)
    print(
        f"{dim:>5}{d:>9.3f}{f:>9.3f}{e:>14.3f}"
        f"{1 - f / d:>8.1%}{1 - e / d:>12.1%}",
        flush=True,
    )
