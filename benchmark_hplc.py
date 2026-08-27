"""Low-dimensional composite HPLC method-development benchmark.

This is a self-contained *screening emulator*, not a claim of new wet-lab
measurements.  It follows the experimental decision variables and response
construction in the open HPLC method-optimisation data/code released by the
Bourne group:

    https://github.com/Bourne-Group/HPLCMethodOptimisationGUI

The three controls are column temperature, gradient duration, and initial
organic-modifier fraction.  Six analytes are propagated through a linear
solvent-strength (LSS) gradient model.  Their reference retention times and
FWHM values are anchored to the first six fitted peaks in the released
``Optimisation1.mat`` archive.  The benchmark is continuous away from the
identity of the critical peak pair, cheap enough for repeated BO trials, and
does not need MATLAB, an HPLC instrument, a network connection, or a separate
data file.

Composite response (12 intermediates -> 2 objectives):

* h_1..h_6 are the six retention times (minutes).
* h_7..h_12 are their fitted peak FWHM values (minutes).
* objective 1 is the last eluting peak time, normalized by the longest run.
* objective 2 is 1 / (1 + R_s,critical), where
  R_s,ij = 2 |t_i-t_j| / (w_i+w_j).

Both objectives are minimized.  The max over retention times and min over all
pairwise resolutions deliberately discard peak-level information; this is
the nonlinear composition that the composite methods are supposed to exploit.

Run the paper protocol (10 trials, 5 initial points, 45 total evaluations):

    python benchmark_hplc.py

For a smoke test:

    python benchmark_hplc.py --quick
"""

from __future__ import annotations

import torch

from benchmark_common import BenchmarkProblem, run_benchmark


DIM = 3
N_ANALYTES = 6
N_TIME_STEPS = 256

# Physical control bounds, in the same order as normalized X.
TEMPERATURE_BOUNDS_C = (30.0, 60.0)
GRADIENT_TIME_BOUNDS_MIN = (1.0, 8.0)
INITIAL_ORGANIC_BOUNDS = (0.05, 0.35)
FINAL_ORGANIC_FRACTION = 0.95
POST_GRADIENT_HOLD_MIN = 2.5
REFERENCE_TEMPERATURE_K = 45.0 + 273.15
REFERENCE_DEAD_TIME_MIN = 0.28
MAX_RUN_TIME_MIN = GRADIENT_TIME_BOUNDS_MIN[1] + POST_GRADIENT_HOLD_MIN

# Peak centers and FWHM values read from the first fitted six-peak table in
# PaperResults/Optimisations123/Optimisation1.mat.  These are observations,
# not objective values: the objectives are reconstructed from the components.
REFERENCE_RETENTION_MIN = torch.tensor(
    [0.33074487, 0.52771131, 0.66676450, 1.01927972, 1.61271133, 3.36639219],
    dtype=torch.double,
)
REFERENCE_FWHM_MIN = torch.tensor(
    [0.04377433, 0.05278751, 0.04938027, 0.05392101, 0.07576851, 0.13765582],
    dtype=torch.double,
)

# LSS model: ln(k_i) = ln(k_w,i) - S_i*phi + thermal_i*(1/T-1/T_ref).
# ln(k_w) was solved so the reference method (45 C, 4.5 min, 20% organic)
# reproduces REFERENCE_RETENTION_MIN under the migration-integral model below.
LOG_KW = torch.tensor(
    [-0.75376595, 1.04810490, 1.67982501, 2.67079929, 3.77940697, 6.25081514],
    dtype=torch.double,
)
SOLVENT_STRENGTH = torch.tensor(
    [4.2, 4.8, 5.3, 5.9, 6.4, 7.0], dtype=torch.double
)
THERMAL_COEFFICIENT_K = torch.tensor(
    [450.0, 700.0, 950.0, 1200.0, 900.0, 1450.0], dtype=torch.double
)


def _physical_controls(X: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Map normalized designs to temperature, gradient time, and organic fraction."""

    X = X.double().reshape(-1, DIM).clamp(0.0, 1.0)
    temperature_c = TEMPERATURE_BOUNDS_C[0] + X[:, 0] * (
        TEMPERATURE_BOUNDS_C[1] - TEMPERATURE_BOUNDS_C[0]
    )
    gradient_time = GRADIENT_TIME_BOUNDS_MIN[0] + X[:, 1] * (
        GRADIENT_TIME_BOUNDS_MIN[1] - GRADIENT_TIME_BOUNDS_MIN[0]
    )
    initial_organic = INITIAL_ORGANIC_BOUNDS[0] + X[:, 2] * (
        INITIAL_ORGANIC_BOUNDS[1] - INITIAL_ORGANIC_BOUNDS[0]
    )
    return temperature_c, gradient_time, initial_organic


def _retention_times(
    temperature_c: torch.Tensor,
    gradient_time: torch.Tensor,
    initial_organic: torch.Tensor,
) -> torch.Tensor:
    """Solve the chromatographic migration integral for all six analytes.

    At time t the fractional column velocity is 1/(1+k_i(t)).  An analyte
    elutes when integral[0,t] 1/(t_0*(1+k_i(s))) ds reaches one.  A fixed grid
    and linear interpolation at the crossing make the oracle deterministic.
    """

    n = temperature_c.numel()
    device = temperature_c.device
    dtype = temperature_c.dtype
    unit_grid = torch.linspace(0.0, 1.0, N_TIME_STEPS, device=device, dtype=dtype)
    run_end = gradient_time + POST_GRADIENT_HOLD_MIN
    time = run_end[:, None] * unit_grid[None, :]
    ramp_fraction = (time / gradient_time[:, None]).clamp(0.0, 1.0)
    organic = initial_organic[:, None] + (
        FINAL_ORGANIC_FRACTION - initial_organic[:, None]
    ) * ramp_fraction

    temperature_k = temperature_c + 273.15
    thermal_shift = THERMAL_COEFFICIENT_K.to(temperature_c)[None, None, :] * (
        1.0 / temperature_k[:, None, None] - 1.0 / REFERENCE_TEMPERATURE_K
    )
    log_k = (
        LOG_KW.to(temperature_c)[None, None, :]
        - SOLVENT_STRENGTH.to(temperature_c)[None, None, :]
        * organic[:, :, None]
        + thermal_shift
    )
    retention_factor = torch.exp(log_k.clamp(-30.0, 30.0))
    dead_time = REFERENCE_DEAD_TIME_MIN * torch.exp(
        -0.006 * (temperature_c - 45.0)
    )
    migration_rate = 1.0 / (
        dead_time[:, None, None] * (1.0 + retention_factor)
    )

    dt = time[:, 1:] - time[:, :-1]
    increments = 0.5 * (
        migration_rate[:, 1:, :] + migration_rate[:, :-1, :]
    ) * dt[:, :, None]
    progress = torch.cat(
        (
            torch.zeros(n, 1, N_ANALYTES, dtype=dtype, device=device),
            increments.cumsum(dim=1),
        ),
        dim=1,
    )

    crossed = progress >= 1.0
    upper = crossed.to(torch.int64).argmax(dim=1)
    never_crossed = ~crossed.any(dim=1)
    upper = torch.where(
        never_crossed,
        torch.full_like(upper, N_TIME_STEPS - 1),
        upper,
    )
    lower = (upper - 1).clamp_min(0)
    p0 = progress.gather(1, lower[:, None, :]).squeeze(1)
    p1 = progress.gather(1, upper[:, None, :]).squeeze(1)
    expanded_time = time[:, :, None].expand(-1, -1, N_ANALYTES)
    t0 = expanded_time.gather(1, lower[:, None, :]).squeeze(1)
    t1 = expanded_time.gather(1, upper[:, None, :]).squeeze(1)
    fraction = ((1.0 - p0) / (p1 - p0).clamp_min(1.0e-12)).clamp(0.0, 1.0)
    return t0 + fraction * (t1 - t0)


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Return six retention times followed by six peak widths."""

    original_shape = X.shape[:-1]
    temperature_c, gradient_time, initial_organic = _physical_controls(X)
    retention = _retention_times(temperature_c, gradient_time, initial_organic)

    # Empirical plate-width scaling around the released reference fit.  It
    # captures broader late peaks, thermal efficiency, and gradient compression.
    widths = REFERENCE_FWHM_MIN.to(retention)[None, :] * (
        retention / REFERENCE_RETENTION_MIN.to(retention)[None, :]
    ).clamp_min(0.05).pow(0.72)
    widths = widths * torch.exp(-0.004 * (temperature_c[:, None] - 45.0))
    widths = widths * (
        1.0 + 0.08 * (4.5 / gradient_time[:, None] - 1.0)
    ).clamp_min(0.5)
    components = torch.cat((retention, widths), dim=-1)
    return components.reshape(*original_shape, 2 * N_ANALYTES)


def compose(H: torch.Tensor) -> torch.Tensor:
    """Reduce peak-level information to runtime and critical-resolution losses."""

    retention = H[..., :N_ANALYTES].clamp(0.20, MAX_RUN_TIME_MIN)
    widths = H[..., N_ANALYTES:].clamp(0.005, 0.50)
    last_peak_loss = retention.amax(dim=-1) / MAX_RUN_TIME_MIN

    time_separation = (retention.unsqueeze(-1) - retention.unsqueeze(-2)).abs()
    width_sum = widths.unsqueeze(-1) + widths.unsqueeze(-2)
    pairwise_resolution = 2.0 * time_separation / width_sum.clamp_min(1.0e-8)
    pair_mask = torch.triu(
        torch.ones(
            N_ANALYTES,
            N_ANALYTES,
            dtype=torch.bool,
            device=H.device,
        ),
        diagonal=1,
    )
    critical_resolution = pairwise_resolution[..., pair_mask].amin(dim=-1)
    unresolved_loss = 1.0 / (1.0 + critical_resolution)
    return torch.stack((last_peak_loss, unresolved_loss), dim=-1)


PROBLEM = BenchmarkProblem(
    name="HPLC gradient method development (2 objectives, 3 dimensions)",
    slug="hplc_gradient_2obj_3d",
    dim=DIM,
    num_objectives=2,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 0.70, dtype=torch.double),
    num_components=2 * N_ANALYTES,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
