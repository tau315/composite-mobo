"""Summit SNAr reaction benchmark with mechanistic intermediates.

The four normalized inputs parameterize residence time, pyrrolidine
equivalents, inlet concentration, and temperature. A vectorized fixed-step
RK4 integration evaluates the published five-species kinetic model.

The intermediates are the five outlet concentrations exactly as the kinetic
model produces them, and the known map forms space-time yield and E-factor from
them. Total flow is a closed-form function of residence time, so it is computed
inside the map rather than modelled.

The concentrations are deliberately left untransformed. A log scale makes the
E-factor's division numerically far better behaved -- it was tried, and it moves
that objective's surrogate advantage from +0.2% to +25.4% -- but it is a
modelling choice layered on top of the benchmark rather than part of it, and
reporting a result that depends on it would be reporting on the choice.
"""

import torch

from benchmark_common import BenchmarkProblem, run_benchmark


DIM = 4
N_RK4_STEPS = 256
INPUT_LOWER = torch.tensor([0.5, 1.0, 0.1, 30.0], dtype=torch.double)
INPUT_UPPER = torch.tensor([2.0, 5.0, 0.5, 120.0], dtype=torch.double)
MOLECULAR_WEIGHTS = torch.tensor(
    [159.09, 71.12, 210.21, 210.21, 261.33], dtype=torch.double
)
PRODUCT_INDEX = 2
REACTOR_VOLUME_ML = 5.0
ETHANOL_DENSITY = 0.789
STY_SCALE = 13_000.0
E_FACTOR_SCALE = 500.0


def _reaction_rates(C: torch.Tensor, temperature_c: torch.Tensor) -> torch.Tensor:
    """Five-species SNAr rates from the Summit benchmark."""

    gas_constant = 8.314 / 1000.0
    reference_temperature = 90.0 + 273.71
    temperature_k = temperature_c + 273.71

    def kinetic_constant(k_ref: float, activation_energy: float) -> torch.Tensor:
        return 0.6 * k_ref * torch.exp(
            -activation_energy
            / gas_constant
            * (1.0 / temperature_k - 1.0 / reference_temperature)
        )

    k_a = kinetic_constant(57.9, 33.3)
    k_b = kinetic_constant(2.70, 35.3)
    k_c = kinetic_constant(0.865, 38.9)
    k_d = kinetic_constant(1.63, 44.8)

    c0, c1, c2, c3, _ = C.unbind(dim=-1)
    primary = c0 * c1
    return torch.stack(
        (
            -(k_a + k_b) * primary,
            -(k_a + k_b) * primary - k_c * c1 * c2 - k_d * c1 * c3,
            k_a * primary - k_c * c1 * c2,
            k_b * primary - k_d * c1 * c3,
            k_c * c1 * c2 + k_d * c1 * c3,
        ),
        dim=-1,
    )


def _outlet_concentrations(
    residence_time: torch.Tensor,
    equivalents: torch.Tensor,
    inlet_concentration: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Integrate each normalized plug-flow trajectory over s in [0, 1]."""

    C = torch.zeros(
        residence_time.shape[0], 5, dtype=torch.double, device=residence_time.device
    )
    C[:, 0] = inlet_concentration
    C[:, 1] = equivalents * inlet_concentration
    step = 1.0 / N_RK4_STEPS

    def derivative(state: torch.Tensor) -> torch.Tensor:
        return residence_time.unsqueeze(-1) * _reaction_rates(state, temperature)

    for _ in range(N_RK4_STEPS):
        k1 = derivative(C)
        k2 = derivative(C + 0.5 * step * k1)
        k3 = derivative(C + 0.5 * step * k2)
        k4 = derivative(C + step * k3)
        C = (C + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0).clamp_min(
            0.0
        )
    return C


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Return the five outlet concentrations, the only simulated quantities.

    Nothing else belongs here. Total flow is fixed by the reactor volume and the
    residence time, both design variables, so it is known in closed form and is
    computed inside ``compose``. Product concentration is used by both
    objectives but is modelled once: giving it a second GP would cost a fit for
    no information and, worse, let a Monte Carlo draw hand the two objectives
    two different product concentrations for the same physical state.

    The values are the raw kinetic-model output, untransformed.
    """

    X = X.double()
    physical = INPUT_LOWER.to(X) + X * (INPUT_UPPER.to(X) - INPUT_LOWER.to(X))
    residence_time, equivalents, inlet_concentration, temperature = physical.T
    return _outlet_concentrations(
        residence_time, equivalents, inlet_concentration, temperature
    )


def compose(H: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Calculate normalized minimization objectives from the intermediates."""

    X = X.double()
    residence_time = (
        INPUT_LOWER.to(X)[0] + X[..., 0] * (INPUT_UPPER.to(X)[0] - INPUT_LOWER.to(X)[0])
    )
    # Exact, so it broadcasts onto the component posterior's sample dimensions.
    total_flow = REACTOR_VOLUME_ML / residence_time.clamp_min(1.0e-8)
    total_flow = total_flow + torch.zeros_like(H[..., 0])

    # A component posterior sample can go negative, and the E-factor divides by
    # the product concentration, so the clamps below are load-bearing rather
    # than defensive.
    outlet = H[..., :5].clamp_min(0.0)
    product = outlet[..., PRODUCT_INDEX]
    sty = (
        6.0e4
        / 1000.0
        * MOLECULAR_WEIGHTS[PRODUCT_INDEX].to(H)
        * product
        * total_flow
        / REACTOR_VOLUME_ML
    )
    sty_objective = 1.0 - (sty / STY_SCALE).clamp(0.0, 1.0)

    product_e = product.clamp_min(1.0e-12)
    weights = MOLECULAR_WEIGHTS.to(H)
    waste_mass = (
        weights[0] * outlet[..., 0]
        + weights[1] * outlet[..., 1]
        + weights[3] * outlet[..., 3]
        + weights[4] * outlet[..., 4]
    )
    numerator = total_flow * ETHANOL_DENSITY + 1.0e-3 * total_flow * waste_mass
    denominator = (
        1.0e-3 * weights[PRODUCT_INDEX] * product_e * total_flow
    ).clamp_min(1.0e-12)
    e_factor = (numerator / denominator).clamp_max(1000.0)
    return torch.stack((sty_objective, e_factor / E_FACTOR_SCALE), dim=-1)


PROBLEM = BenchmarkProblem(
    name="Summit SNAr reaction (2 objectives, 4 dimensions)",
    slug="summit_snar_2obj_4d",
    dim=DIM,
    num_objectives=2,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    # Chosen to bound the Pareto-relevant region rather than the whole feasible
    # image. The previous (2.5, 2.5) credited every method with a large slab no
    # design can reach: 78% of the reported hypervolume was already present after
    # five random points, against 43% here.
    #
    # This does *not* leave paired comparisons unchanged, and an earlier version
    # of this comment wrongly said it did. Moving the reference alters each
    # trial's hypervolume non-uniformly, and qLogEHVI also consumes the reference
    # when it builds its acquisition, so a rerun follows a different trajectory.
    # Results computed under different reference points are not comparable.
    #
    # One corner does exceed it: X=(1,1,1,1) gives (0.997, 1.695). That design is
    # badly dominated -- the approximate Pareto front's nadir is (0.762, 0.018) --
    # so it never contributes to the measured hypervolume, but the reference is a
    # bound on the region of interest, not on every attainable point.
    ref_point=torch.tensor([1.05, 1.10], dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
