"""Summit SNAr reaction benchmark with mechanistic intermediates.

The four normalized inputs parameterize residence time, pyrrolidine
equivalents, inlet concentration, and temperature. A vectorized fixed-step
RK4 integration evaluates the published five-species kinetic model.
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
    """Return objective-specific physical quantities.

    Product concentration and total flow are intentionally repeated in the
    E-factor group. Composite solvers therefore fit each objective's component
    GPs independently even when the underlying physical quantity overlaps.
    """

    X = X.double()
    physical = INPUT_LOWER.to(X) + X * (INPUT_UPPER.to(X) - INPUT_LOWER.to(X))
    residence_time, equivalents, inlet_concentration, temperature = physical.T
    outlet = _outlet_concentrations(
        residence_time, equivalents, inlet_concentration, temperature
    )
    total_flow = REACTOR_VOLUME_ML / residence_time
    sty_group = torch.stack((outlet[:, PRODUCT_INDEX], total_flow), dim=-1)
    e_factor_group = torch.cat((outlet, total_flow.unsqueeze(-1)), dim=-1)
    return torch.cat((sty_group, e_factor_group), dim=-1)


def compose(H: torch.Tensor) -> torch.Tensor:
    """Calculate normalized minimization objectives from the intermediates."""

    product_sty = H[..., 0].clamp_min(0.0)
    flow_sty = H[..., 1].clamp_min(1.0e-8)
    sty = (
        6.0e4
        / 1000.0
        * MOLECULAR_WEIGHTS[PRODUCT_INDEX].to(H)
        * product_sty
        * flow_sty
        / REACTOR_VOLUME_ML
    )
    sty_objective = 1.0 - (sty / STY_SCALE).clamp(0.0, 1.0)

    outlet = H[..., 2:7].clamp_min(0.0)
    flow_e = H[..., 7].clamp_min(1.0e-8)
    product_e = outlet[..., PRODUCT_INDEX].clamp_min(1.0e-12)
    weights = MOLECULAR_WEIGHTS.to(H)
    waste_mass = (
        weights[0] * outlet[..., 0]
        + weights[1] * outlet[..., 1]
        + weights[3] * outlet[..., 3]
        + weights[4] * outlet[..., 4]
    )
    numerator = flow_e * ETHANOL_DENSITY + 1.0e-3 * flow_e * waste_mass
    denominator = (
        1.0e-3 * weights[PRODUCT_INDEX] * product_e * flow_e
    ).clamp_min(1.0e-12)
    e_factor = (numerator / denominator).clamp_max(1000.0)
    return torch.stack(
        (sty_objective, e_factor / E_FACTOR_SCALE),
        dim=-1,
    )


PROBLEM = BenchmarkProblem(
    name="Summit SNAr reaction (2 objectives, 4 dimensions)",
    slug="summit_snar_2obj_4d",
    dim=DIM,
    num_objectives=2,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 2.5, dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
