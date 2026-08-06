"""Summit SNAr reaction benchmark with mechanistic intermediates.

The four normalized inputs parameterize residence time, pyrrolidine
equivalents, inlet concentration, and temperature. A vectorized fixed-step
RK4 integration evaluates the published five-species kinetic model.

The intermediates are the five outlet log concentrations; the known map raises
them back to concentrations and forms space-time yield and E-factor. E-factor
is a ratio, which is the structure composite modelling exists to exploit, and
the log scale is what makes exploiting it numerically safe.
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
# Concentrations are logged, so a fully consumed species needs a floor. Set it
# at 1 uM, roughly the detection limit of the HPLC/GC monitoring such a reaction
# would use in practice: below this the simulator's value is not a measurable
# quantity anyway.
#
# The floor matters more than it looks. The limiting reagent is consumed to
# ~1e-94 mol/L at long residence times, and a floor of 1e-12 still leaves its
# log at -27.6 against every other component in [-4, 0.6]. A GP fitted to a
# target spanning 28 log units extrapolates wildly, and exponentiating those
# samples inside compose produced concentrations of 3e5 mol/L against a true
# maximum of 1.3 -- which is what drove the composite arm's occasional
# mid-optimization collapses on this benchmark. At 1 uM the same posterior
# reaches 4e2 instead, and no sample lands on the E-factor clamp.
#
# Raising the floor changes the objectives by 2e-7 relative, since a species at
# this concentration contributes ~1e-7 of the E-factor numerator. That is orders
# of magnitude below the precision of any real yield measurement.
CONCENTRATION_FLOOR = 1.0e-6
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
    """Return the five outlet **log** concentrations, the simulated quantities.

    Nothing else belongs here. Total flow is fixed by the reactor volume and the
    residence time, both design variables, so it is known in closed form and is
    computed inside ``compose``. Product concentration is used by both
    objectives but is modelled once: giving it a second GP would cost a fit for
    no information and, worse, let a Monte Carlo draw hand the two objectives
    two different product concentrations for the same physical state.

    Concentrations are modelled on a log scale because the E-factor divides by
    the product concentration, which spans a factor of 27 across the domain. A
    GP fitted to the concentration itself puts posterior mass near and below
    zero in the low-product tail, and the division then amplifies that error
    without bound: across five Sobol splits the E-factor's surrogate advantage
    swung between +47% and -110%. Exponentiating inside ``compose`` makes every
    posterior sample positive by construction and turns the ratio into a
    difference, which leaves the same objectives with a tenfold more stable
    screen (+27.3% +- 4.2% against +20.2% +- 43.4%). This mirrors DTLZ2 modelling
    ``sqrt`` of its radial term to keep samples in the valid domain.
    """

    X = X.double()
    physical = INPUT_LOWER.to(X) + X * (INPUT_UPPER.to(X) - INPUT_LOWER.to(X))
    residence_time, equivalents, inlet_concentration, temperature = physical.T
    outlet = _outlet_concentrations(
        residence_time, equivalents, inlet_concentration, temperature
    )
    return outlet.clamp_min(CONCENTRATION_FLOOR).log()


def compose(H: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Calculate normalized minimization objectives from the intermediates."""

    X = X.double()
    residence_time = (
        INPUT_LOWER.to(X)[0] + X[..., 0] * (INPUT_UPPER.to(X)[0] - INPUT_LOWER.to(X)[0])
    )
    # Exact, so it broadcasts onto the component posterior's sample dimensions.
    total_flow = REACTOR_VOLUME_ML / residence_time.clamp_min(1.0e-8)
    total_flow = total_flow + torch.zeros_like(H[..., 0])

    # Components are log concentrations; exp is positive by construction, so
    # the E-factor denominator can never cross zero.
    outlet = H[..., :5].exp()
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
    # Just past the worst attainable corner. Over a 4096-point Sobol sweep the
    # objectives span [0.18, 1.00] and [0.017, 1.05], so the previous (2.5, 2.5)
    # credited every method with a large constant slab no design can reach: 78%
    # of the reported hypervolume was already present after five random points.
    # Tightening it drops that to 43%, which is what the measurement is actually
    # about. It does not change any paired comparison -- both arms share an
    # initial design, so a common offset cancels in every paired difference --
    # but it stops the headline number from being mostly free volume.
    ref_point=torch.tensor([1.05, 1.10], dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
