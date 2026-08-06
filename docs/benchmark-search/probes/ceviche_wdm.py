"""Scratch BenchmarkProblem adapter for invrs-gym's lightweight Ceviche WDM."""

from dataclasses import replace
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from invrs_gym import challenges

from benchmark_common import BenchmarkProblem


_CHALLENGE = challenges.ceviche_lightweight_wdm()
_TEMPLATE = _CHALLENGE.component.init(jax.random.PRNGKey(0))
_FREE = np.flatnonzero(
    ~(np.asarray(_TEMPLATE.fixed_solid) | np.asarray(_TEMPLATE.fixed_void))
)
_CACHE: dict[bytes, torch.Tensor] = {}
EVALUATION_SECONDS: list[float] = []


def evaluate_components(X: torch.Tensor) -> torch.Tensor:
    rows = []
    for x in X.detach().cpu().double().numpy():
        key = x.tobytes()
        if key not in _CACHE:
            density = np.asarray(_TEMPLATE.array).copy().reshape(-1)
            density[_FREE] = x
            params = replace(
                _TEMPLATE,
                array=jnp.asarray(density.reshape(_TEMPLATE.array.shape)),
            )
            started = perf_counter()
            response, _ = _CHALLENGE.component.response(params, max_parallelizm=2)
            s = np.asarray(response.s_parameters).reshape(2, 3)
            EVALUATION_SECONDS.append(perf_counter() - started)
            h = np.stack((s.real, s.imag), axis=-1).reshape(-1)
            if not np.isfinite(h).all():
                raise FloatingPointError("Ceviche returned non-finite S-parameters")
            _CACHE[key] = torch.as_tensor(h, dtype=torch.double)
        rows.append(_CACHE[key])
    return torch.stack(rows)


def compose(H: torch.Tensor) -> torch.Tensor:
    s = H.reshape(*H.shape[:-1], 2, 3, 2)
    transmission = s[..., 0].square() + s[..., 1].square()
    lower = H.new_tensor(((0.0, 0.5011872, 0.0), (0.0, 0.0, 0.5011872)))
    upper = H.new_tensor(((0.01, 1.0, 0.01), (0.01, 0.01, 1.0)))
    lower_distance = torch.where(lower > 0.0, lower - transmission, -1.0)
    upper_distance = torch.where(upper < 1.0, transmission - upper, -1.0)
    return F.softplus(torch.maximum(lower_distance, upper_distance) / 0.01).square().sum(-1)


PROBLEM = BenchmarkProblem(
    name="Ceviche lightweight wavelength demultiplexer",
    slug="ceviche-lightweight-wdm",
    dim=len(_FREE),
    num_objectives=2,
    suite="high",
    evaluate_components=evaluate_components,
    compose=compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 5000.0, dtype=torch.double),
)

