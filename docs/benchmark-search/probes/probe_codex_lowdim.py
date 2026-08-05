"""Screen Codex's analytic low-dimensional candidates with the same diagnostic."""

import sys

import torch

sys.path.insert(0, ".")
from benchmark_common import BenchmarkProblem
from diagnose_composite import diagnose

# DiscBrake bounds (Tanabe & Ishibuchi 2020, as implemented in BoTorch).
DISC_BOUNDS = torch.tensor([[55.0, 75.0, 1000.0, 11.0], [80.0, 110.0, 3000.0, 20.0]], dtype=torch.double)
WELD_BOUNDS = torch.tensor([[0.125, 0.1, 0.1, 0.125], [5.0, 10.0, 10.0, 5.0]], dtype=torch.double)


def _unnormalize(X, bounds):
    return bounds[0] + (bounds[1] - bounds[0]) * X.double()


def disc_brake() -> BenchmarkProblem:
    def components(X):
        x = _unnormalize(X, DISC_BOUNDS)
        ri, ro, p, n = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        A = ro.square() - ri.square()
        B = ro.pow(3) - ri.pow(3)
        return torch.stack((A, B, p, n), dim=-1)

    def compose(H):
        A, B, p, n = H[..., 0], H[..., 1], H[..., 2], H[..., 3]
        mass = 4.9e-5 * A * (n - 1.0)
        stopping = 9.82e6 * A / (p * n * B).clamp_min(1e-9)
        return torch.stack((mass, stopping), dim=-1)

    return BenchmarkProblem(
        name="DiscBrake", slug="discbrake", dim=4, num_objectives=2, suite="low",
        evaluate_components=components, compose=compose,
        ideal=torch.zeros(2, dtype=torch.double),
        ref_point=torch.tensor([10.0, 60.0], dtype=torch.double),
    )


def welded_beam() -> BenchmarkProblem:
    def components(X):
        x = _unnormalize(X, WELD_BOUNDS)
        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        weld_volume = x1.square() * x2
        beam_volume = x3 * x4 * (14.0 + x2)
        rigidity = x4 * x3.pow(3)
        return torch.stack((weld_volume, beam_volume, rigidity), dim=-1)

    def compose(H):
        weld_volume, beam_volume, rigidity = H[..., 0], H[..., 1], H[..., 2]
        cost = 1.10471 * weld_volume + 0.04811 * beam_volume
        deflection = 2.1952 / rigidity.clamp_min(1e-9)
        return torch.stack((cost, deflection), dim=-1)

    return BenchmarkProblem(
        name="WeldedBeam", slug="weldedbeam", dim=4, num_objectives=2, suite="low",
        evaluate_components=components, compose=compose,
        ideal=torch.zeros(2, dtype=torch.double),
        ref_point=torch.tensor([400.0, 20.0], dtype=torch.double),
    )


print(f"{'candidate':<16}{'d':>4}{'p':>4}{'direct':>9}{'composite':>11}{'advantage':>12}")
print("-" * 56)
for problem in (disc_brake(), welded_beam()):
    problem.validate()
    r = diagnose(problem)
    print(
        f"{problem.slug:<16}{r['dim']:>4}{r['components']:>4}{r['direct_rmse']:>9.3f}"
        f"{r['composite_rmse']:>11.3f}{r['advantage']:>11.1%}",
        flush=True,
    )
