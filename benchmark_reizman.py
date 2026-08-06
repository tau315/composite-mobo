"""Reizman-Suzuki cross-coupling: the simplest composite structure we test.

Three continuous reaction conditions, two competing objectives, and a single
simulated intermediate. Catalyst is fixed to P1-L2 so the design space is a
continuous cube; the published benchmark also varies an eight-level ligand.

    h(x) = Y(x)                       reaction yield, the only measured response
    f1   = 1 - Y/100                  maximize yield
    f2   = 1 - (Y/L)/200              maximize turnover number

Turnover number is moles of product per mole of catalyst, i.e. ``Y / L`` where
``L`` is the catalyst loading -- **a design variable we choose exactly**. It
therefore enters through ``compose(H, X)`` rather than being given a second GP.
The identity holds in the source data to the precision the paper reports its
values (ratios of 0.9965 to 1.0009 across the released measurements).

Loading spans 0.5 to 2.5 mol%, so ``1/L`` varies fivefold and turnover number
inherits curvature that yield does not have, while ``L >= 0.5`` keeps the ratio
away from its pole. One GP on a smooth response replaces two GPs on responses of
very different difficulty, which is the whole mechanism this paper studies.

Source: Reizman, Wang, Buchwald & Jensen, React. Chem. Eng. 1, 658-666 (2016).
Emulator weights from Summit (Felton et al.); see ``reizman_emulator``.
"""

import torch

from benchmark_common import BenchmarkProblem, run_benchmark
from reizman_emulator import ReizmanEmulator

EMULATOR = ReizmanEmulator(case=4, catalyst="P1-L2")
DIM = 3
# Normalization constants, chosen as conservative bounds rather than attained
# maxima. The emulator's own reachable extremes are lower -- roughly 96.6% yield,
# and turnover about 149.6 at (445 s, 103 C, 0.5 mol%), where yield is only 75%.
# Naively combining the domain bounds gives 100/0.5 = 200, which no design
# achieves because maximum yield and minimum loading do not coincide. These
# constants only set the objective scale; the comparison is unaffected by them,
# and the reference point below is what bounds the measured hypervolume.
MAX_YIELD = 100.0
MAX_TURNOVER = 200.0
LOADING_INDEX = 2


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    """Reaction yield in percent, the single quantity the emulator predicts."""

    return EMULATOR.yield_percent(X.double()).unsqueeze(-1)


def compose(H: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Yield and turnover number, using the exactly known catalyst loading."""

    reaction_yield = H[..., 0].clamp(0.0, MAX_YIELD)
    lower, upper = EMULATOR.bounds[0][LOADING_INDEX], EMULATOR.bounds[1][LOADING_INDEX]
    loading = lower + X[..., LOADING_INDEX].double() * (upper - lower)
    # Exact, so broadcast it onto the component posterior's sample dimensions.
    loading = loading + torch.zeros_like(reaction_yield)
    turnover = reaction_yield / loading

    return torch.stack(
        (
            1.0 - (reaction_yield / MAX_YIELD).clamp(0.0, 1.0),
            1.0 - (turnover / MAX_TURNOVER).clamp(0.0, 1.0),
        ),
        dim=-1,
    )


# A reference point far outside the attainable range hands every method the same
# large constant slab of hypervolume, which compresses the differences the
# experiment is trying to measure. Over a 512-point Sobol sweep the objectives
# span [0.034, 0.682] and [0.297, 0.891], so this sits just past the worst
# attainable corner: every method still clears it, and none is credited for
# volume no design can reach.
REF_POINT = torch.tensor([0.75, 0.95], dtype=torch.double)


PROBLEM = BenchmarkProblem(
    name="Reizman-Suzuki cross-coupling (2 objectives, 3 dimensions)",
    slug="reizman_suzuki_2obj_3d",
    dim=DIM,
    num_objectives=2,
    suite="low",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=REF_POINT,
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
