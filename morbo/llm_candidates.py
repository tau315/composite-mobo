#!/usr/bin/env python3
r"""Per-iteration LLM candidate proposals for a single MORBO trust region.

Called once per trust region per BO iteration (not per batch slot) from
`TS_select_batch_MORBO` in `gen.py`. Proposed candidates are concatenated
into the existing Thompson-sampling candidate pool and scored by MORBO's own
hypervolume-improvement / scalarization logic exactly like any other
candidate -- this module only *proposes*, it never scores or selects.

Mirrors MORBO's own `sample_tr_discrete_points_subset_d` (`utils.py`), which
only perturbs a random subset of ~20 dimensions per candidate even at high
dimension (`prob_perturb = min(20/d, 1.0)`): the LLM is prompted the same
way, proposing a *sparse* perturbation (which dimensions to touch, by how
much) rather than a dense d-vector, since verbalizing all d raw dimensions
gives an LLM little to reason about beyond noise at high dimension. Uses the
official `anthropic` SDK with structured output, reading `ANTHROPIC_API_KEY`
from the environment via the SDK's standard credential resolution.
"""
from typing import List, Optional

import torch
from anthropic import Anthropic
from pydantic import BaseModel, Field
from torch import Tensor

MODEL = "claude-opus-4-8"
MAX_DIMS_TO_PERTURB = 20


class Perturbation(BaseModel):
    dimension: int = Field(description="0-indexed dimension to perturb.")
    delta: float = Field(
        description="Signed offset to add to that dimension's current value, in "
        "[-0.5, 0.5] (the search space is normalized to [0, 1] per dimension)."
    )


class CandidateProposal(BaseModel):
    perturbations: List[Perturbation] = Field(
        description=f"Which dimensions to adjust and by how much, at most "
        f"{MAX_DIMS_TO_PERTURB} entries. Unlisted dimensions keep the base point's "
        "value unchanged."
    )
    rationale: str = Field(description="One short sentence on why this candidate.")


class CandidateBatch(BaseModel):
    candidates: List[CandidateProposal]


def propose_candidates(
    tr_center: Tensor,
    tr_bounds: Tensor,
    recent_pareto_points: Optional[Tensor] = None,
    dim_names: Optional[List[str]] = None,
    objective_names: Optional[List[str]] = None,
    n_candidates: int = 4,
    problem_description: str = "",
    client: Optional[Anthropic] = None,
) -> Tensor:
    r"""Propose candidate points for one trust region, one BO iteration.

    Args:
        tr_center: `1 x d`-dim tensor, the TR's current center, normalized
            to `[0, 1]^d`.
        tr_bounds: `2 x d`-dim tensor, the TR's current box in normalized
            `[0, 1]^d` space (as returned by `TrustRegion.get_bounds()`).
        recent_pareto_points: optional `k x M`-dim tensor of recent Pareto
            front objective values, for context on what's already covered.
        dim_names: optional length-`d` list of human-readable dimension
            names; falls back to "dimension i" when not given.
        objective_names: optional names for the `M` objectives, used only
            when describing `recent_pareto_points`.
        n_candidates: number of candidate points to propose.
        problem_description: natural-language description of the problem,
            prepended to the prompt.
        client: an `anthropic.Anthropic` client; constructed from the
            environment (`ANTHROPIC_API_KEY`) if not provided.

    Returns:
        An `n_candidates x d`-dim tensor of candidate points in normalized
        `[0, 1]^d` space, clamped to `tr_bounds`, ready to concatenate into
        the Thompson-sampling candidate pool.
    """
    client = client or Anthropic()
    dim = tr_center.shape[-1]
    dim_names = dim_names or [f"dimension {i}" for i in range(dim)]
    center_list = tr_center.squeeze(0).tolist()

    center_text = "\n".join(
        f"  - {name}: {val:.4f} (box: [{lo:.4f}, {hi:.4f}])"
        for name, val, lo, hi in zip(
            dim_names, center_list, tr_bounds[0].tolist(), tr_bounds[1].tolist()
        )
    )
    pareto_text = ""
    if recent_pareto_points is not None and recent_pareto_points.shape[0] > 0:
        obj_names = objective_names or [
            f"objective {i}" for i in range(recent_pareto_points.shape[-1])
        ]
        rows = "\n".join(
            "  - " + ", ".join(f"{n}={v:.4g}" for n, v in zip(obj_names, row))
            for row in recent_pareto_points.tolist()
        )
        pareto_text = (
            "\nCurrent Pareto frontier (all objectives maximized, higher is "
            f"better):\n{rows}"
        )

    prompt = (
        f"{problem_description}\n\n"
        "A local search region (trust region) is currently centered at, and "
        f"bounded by (all values normalized to [0, 1]):\n{center_text}"
        f"{pareto_text}\n\n"
        f"Propose {n_candidates} candidate points to evaluate near this trust "
        f"region center. For each candidate, list at most {MAX_DIMS_TO_PERTURB} "
        "dimensions to perturb away from the center and by how much -- do not "
        "list dimensions you want left unchanged. Favor proposals that could "
        "expand the current Pareto frontier into an under-covered tradeoff "
        "region, not just ones that look locally promising on a single "
        "objective."
    )

    response = client.messages.parse(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        output_format=CandidateBatch,
    )
    batch = response.parsed_output

    candidates = tr_center.repeat(n_candidates, 1).clone()
    for i, proposal in enumerate(batch.candidates[:n_candidates]):
        for pert in proposal.perturbations[:MAX_DIMS_TO_PERTURB]:
            d = pert.dimension
            if not (0 <= d < dim):
                continue
            candidates[i, d] = candidates[i, d] + pert.delta
    n_returned = min(len(batch.candidates), n_candidates)
    candidates = candidates[:n_returned]
    return torch.clamp(candidates, tr_bounds[0], tr_bounds[1])
