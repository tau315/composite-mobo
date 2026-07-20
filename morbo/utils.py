#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from math import ceil, log
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.utils.sampling import draw_sobol_samples
from gpytorch import settings as gpytorch_settings
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import LinearKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
from torch import Tensor
from torch.nn import Module
from torch.distributions import Normal


def sample_tr_discrete_points(
    X_center: Tensor, length: float, n_discrete_points: int, qmc: bool = False
) -> Tensor:
    r"""Sample points around `X_center` for use in discrete Thompson sampling.

    Sample perturbed points around `X_center` such that the added perturbations
        are sampled from N(0, (length/4)^2) and truncated to be within
        [-length/2, -length/2].

    Args:
        X_center: a `1 x d`-dim tensor containing the center of trust region. `X_center`
            must be normalized to be within `[0, 1]^d`.
        length: edge length of the trust region's hypercube.
        n_discrete_points: number of points to sample for use in discrete TS.
        qmc: boolean indicating whether to use qmc

    Returns:
        Tensor: a `n_discrete_points x d`-dim tensor containing the sampled points.
    """
    d = X_center.shape[1]
    # sample points from N(X_center, (length/4)^2), truncated to be within
    # [X_center-length/2, X_center+length/2].
    # To do this, we sample perturbations from N(0, (length/4)^2) truncated to be
    # within [max(-X_center, -L/2), min(1-X_center, L/2) using the inverse transform
    # and then add these perturbations to X_center.
    sigma = length / 4.0
    if qmc:
        bounds = torch.stack(
            [torch.zeros_like(X_center[0]), torch.ones_like(X_center[0])], dim=0
        )
        u = draw_sobol_samples(bounds=bounds, n=n_discrete_points, q=1).squeeze(1)
    else:
        u = torch.rand(
            (n_discrete_points, d), dtype=X_center.dtype, device=X_center.device
        )
    # compute bounds to sample from
    a = (-X_center).clamp_min(-length / 2.0)
    b = (1 - X_center).clamp_max(length / 2.0)
    # compute z-score of bounds
    alpha = a / sigma
    beta = b / sigma
    normal = Normal(0, 1)
    cdf_alpha = normal.cdf(alpha)
    perturbation = normal.icdf(cdf_alpha + u * (normal.cdf(beta) - cdf_alpha)) * sigma
    X_discrete = X_center + perturbation

    # Clip points that are still outside
    return X_discrete.clamp(0.0, 1.0)


def sample_tr_discrete_points_subset_d(
    best_X: Tensor,
    normalized_tr_bounds: Tensor,
    n_discrete_points: int,
    length: float,
    qmc: bool = False,
    trunc_normal_perturb: bool = False,
    prob_perturb: float = None,
) -> Tensor:
    r"""Sample discrete for TS by perturbing ~20 dims of `best_X`.

    If `trunc_normal_perturb=True`, the perturbed samples are truncated normal
    around `best_X`. Otherwise, these are uniformly distributed in
    `normalize_tr_bounds`.
    """
    assert normalized_tr_bounds.ndim == 2
    d = normalized_tr_bounds.shape[-1]
    if prob_perturb is None:
        # Only perturb a subset of the features
        prob_perturb = min(20.0 / d, 1.0)

    if best_X.shape[0] == 1:
        X_cand = best_X.repeat(n_discrete_points, 1)
    else:
        rand_indices = torch.randint(
            best_X.shape[0], (n_discrete_points,), device=best_X.device
        )
        X_cand = best_X[rand_indices]

    if trunc_normal_perturb:
        pert = sample_tr_discrete_points(
            X_center=X_cand, length=length, n_discrete_points=n_discrete_points, qmc=qmc
        )
        # make sure perturbations are in bounds
        # if X_cand contains pareto points, the perturbed points might not be in the TR
        # TODO: refactor to this into a `project_on_box` helper function T65690436
        pert = torch.min(
            torch.max(pert, normalized_tr_bounds[0]), normalized_tr_bounds[1]
        )
    elif qmc:
        pert = draw_sobol_samples(
            bounds=normalized_tr_bounds, n=n_discrete_points, q=1
        ).squeeze(1)
    else:
        pert = torch.rand(
            n_discrete_points,
            d,
            dtype=normalized_tr_bounds.dtype,
            device=normalized_tr_bounds.device,
        )
        pert = (
            normalized_tr_bounds[1] - normalized_tr_bounds[0]
        ) * pert + normalized_tr_bounds[0]

    # find cases where we are not perturbing any dimensions
    mask = (
        torch.rand(
            n_discrete_points,
            d,
            dtype=normalized_tr_bounds.dtype,
            device=normalized_tr_bounds.device,
        )
        <= prob_perturb
    )
    ind = (~mask).all(dim=-1).nonzero()
    # perturb `n_perturb` of the dimensions
    n_perturb = ceil(d * prob_perturb)
    perturb_mask = torch.zeros(d, dtype=mask.dtype, device=mask.device)
    perturb_mask[:n_perturb].fill_(1)
    for idx in ind:
        mask[idx] = perturb_mask[torch.randperm(d, device=normalized_tr_bounds.device)]
    # Create candidate points
    X_cand[mask] = pert[mask]
    return X_cand


def sample_tr_discrete_points_subset_d_rotated(
    best_X: Tensor,
    X_center: Tensor,
    R: Tensor,
    axis_lengths: Tensor,
    n_discrete_points: int,
    qmc: bool = False,
    prob_perturb: float = None,
) -> Tensor:
    r"""Rotated-frame analogue of `sample_tr_discrete_points_subset_d`.

    Masking/perturbation happens in the TR's rotated coordinate frame
    `w = (x - X_center) @ R` rather than the original coordinates, so that
    "perturb ~20 of `d` directions" bounds simultaneous degrees of freedom
    along the TR's own principal axes -- masking original dims and rotating
    afterward would smear a single "masked-in" raw dimension across every
    principal direction post-rotation, defeating the point of the subset
    perturbation. For `R = I` (the `tr_shape == "isotropic"`/`"ard_box"`
    default rotation) this is the identity change of basis.

    `trunc_normal_perturb` is intentionally not supported here -- callers
    should raise rather than silently falling back to unrotated truncated-
    normal sampling (see `TS_select_batch_MORBO`).

    Args:
        best_X: `n x d`-dim tensor of candidate perturbation centers, in
            original (unrotated) `[0, 1]^d` coordinates.
        X_center: `1 x d`-dim tensor, the TR center in original coordinates.
        R: `d x d`-dim rotation matrix (orthonormal columns = principal axes).
        axis_lengths: `d`-dim tensor, full edge length of the TR along each
            rotated axis (same convention as `length` -- a full width, not a
            half-width).
        n_discrete_points: number of points to sample for use in discrete TS.
        qmc: boolean indicating whether to use qmc.
        prob_perturb: as in `sample_tr_discrete_points_subset_d`.

    Returns:
        Tensor: a `n_discrete_points x d`-dim tensor in original `[0, 1]^d`
            coordinates.
    """
    d = R.shape[-1]
    if prob_perturb is None:
        prob_perturb = min(20.0 / d, 1.0)

    if best_X.shape[0] == 1:
        X_cand = best_X.repeat(n_discrete_points, 1)
    else:
        rand_indices = torch.randint(
            best_X.shape[0], (n_discrete_points,), device=best_X.device
        )
        X_cand = best_X[rand_indices]

    W_cand = (X_cand - X_center) @ R
    half_axis = axis_lengths / 2.0

    if qmc:
        bounds = torch.stack([-half_axis, half_axis], dim=0)
        pert_w = draw_sobol_samples(bounds=bounds, n=n_discrete_points, q=1).squeeze(1)
    else:
        u = torch.rand(
            n_discrete_points, d, dtype=axis_lengths.dtype, device=axis_lengths.device
        )
        pert_w = (2 * half_axis) * u - half_axis

    # find cases where we are not perturbing any dimensions (same scheme as
    # `sample_tr_discrete_points_subset_d`, operating on rotated coordinates)
    mask = (
        torch.rand(
            n_discrete_points, d, dtype=axis_lengths.dtype, device=axis_lengths.device
        )
        <= prob_perturb
    )
    ind = (~mask).all(dim=-1).nonzero()
    n_perturb = ceil(d * prob_perturb)
    perturb_mask = torch.zeros(d, dtype=mask.dtype, device=mask.device)
    perturb_mask[:n_perturb].fill_(1)
    for idx in ind:
        mask[idx] = perturb_mask[torch.randperm(d, device=axis_lengths.device)]
    W_cand[mask] = pert_w[mask]

    X_cand_new = X_center + W_cand @ R.t()
    return X_cand_new.clamp(0.0, 1.0)


def get_tr_center(X: Tensor, f_obj: Tensor) -> Tensor:
    r"""Find the best point in the trust region.

    Args:
        X: a `n x d`-dim tensor of points
        f_obj: a `n`-dim tensor of scalarized objective values. In the noiseless,
            setting these can be (scalarized) observed values. In the noisy setting,
            these can be (scalarized) posterior means.
    Returns:
        Tensor: a `1 x d`-dim tensor containing the trust region center point.
    """
    if f_obj.ndim != 1:
        raise BotorchTensorDimensionError(
            f"f_obj must have 1 dimension, got {f_obj.ndim} dimensions."
        )
    return X[f_obj.argmax()].view(1, -1)


def get_indices_in_hypercube(
    X_center: Tensor, X: Tensor, length: float, eps: float = 1e-10
) -> Tensor:
    r"""Get indices of observed points inside of trust region.

    Args:
        X_center: a `1 x d`-dim tensor containing the trust region center point.
            `X_center` must be normalized to be within `[0, 1]^d`.
        X: `n x d`-dim tensor containing all data points collected by this trust region.
        length: the edge length of the trust region's hypercube.
        eps: absolute tolerance for evaluating equality (necessary on CUDA).

    Returns:
        A `n'`-dim tensor containing the points inside the hypercube.
    """
    return ((X - X_center).abs() - length / 2 <= eps).all(dim=1).nonzero().view(-1)


def get_indices_in_ellipsoid(
    X_center: Tensor, X: Tensor, R: Tensor, axis_lengths: Tensor, eps: float = 1e-10
) -> Tensor:
    r"""Get indices of observed points inside a rotated-box trust region.

    Rotated-frame analogue of `get_indices_in_hypercube`: containment is an
    L-infinity test in the TR's own rotated coordinate frame
    `w = (x - X_center) @ R`, using per-axis `axis_lengths` (full edge
    length, not half-width) instead of a single scalar `length`. For
    `R = I` and uniform `axis_lengths`, this is identical to
    `get_indices_in_hypercube`.

    Args:
        X_center: a `1 x d`-dim tensor containing the trust region center point.
            `X_center` must be normalized to be within `[0, 1]^d`.
        X: `n x d`-dim tensor containing all data points collected by this trust region.
        R: `d x d`-dim rotation matrix (orthonormal columns = principal axes).
        axis_lengths: `d`-dim tensor, full edge length along each rotated axis.
        eps: absolute tolerance for evaluating equality (necessary on CUDA).

    Returns:
        A `n'`-dim tensor containing the indices of points inside the region.
    """
    W = (X - X_center) @ R
    return ((W.abs() - axis_lengths / 2) <= eps).all(dim=1).nonzero().view(-1)


def extract_ard_lengthscale(model: Model, dim: int) -> Optional[Tensor]:
    r"""Extract a `dim`-dim per-input-dimension ARD lengthscale vector from a
    fitted model, geometric-mean-averaged across output dimensions if
    `model` bundles more than one (i.e. is a `ModelListGP`).

    Returns `None` if `model` is a `KroneckerMultiTaskGP` (the ARD-based
    `tr_shape` variants only support the `get_fitted_model`/`ModelListGP`
    path, not the Kronecker joint model) or if any constituent model was fit
    without ARD (`use_ard=False`, a single shared lengthscale rather than one
    per input dimension) -- callers should fall back to the isotropic shape
    in either case.
    """
    if isinstance(model, KroneckerMultiTaskGP):
        return None
    models = [model] if not isinstance(model, ModelListGP) else model.models
    log_ls_per_output = []
    for m in models:
        try:
            ls = m.covar_module.base_kernel.lengthscale
        except AttributeError:
            return None
        if ls is None:  # e.g. LinearKernel: has_lengthscale is False
            return None
        ls = ls.reshape(-1)
        if ls.numel() != dim:
            return None
        log_ls_per_output.append(ls.log())
    # Detach: this is a live GP kernel parameter (requires_grad=True).
    # Callers only ever read it as a fixed number to build a shape from;
    # leaving it attached to the model's autograd graph would leak that
    # graph into whatever downstream tensor (e.g. sampled candidates,
    # buffers) consumes the lengthscale, retaining memory it shouldn't and
    # risking it getting pulled into gradient computations it has no
    # business being part of.
    return torch.stack(log_ls_per_output, dim=0).mean(dim=0).exp().detach()


def compute_ard_box_shape(
    lengthscale: Tensor, length: Tensor, dim: int
) -> Tuple[Tensor, Tensor]:
    r"""Axis-aligned box shape, rescaled per-dimension by GP ARD lengthscales
    (the original TuRBO paper's box-rescaling technique).

    Args:
        lengthscale: a `dim`-dim tensor of (already output-aggregated) ARD
            lengthscales, e.g. from `extract_ard_lengthscale`.
        length: a 0-dim tensor, the TR's current (isotropic) edge length.
        dim: input dimension `d`.

    Returns:
        R: `d x d` identity (no rotation).
        axis_lengths: `d`-dim tensor, geometric-mean-normalized so
            `axis_lengths.prod() ** (1/d) == length` -- same total "volume"
            as the isotropic cube of edge `length`, only shape differs.
    """
    R = torch.eye(dim, device=length.device, dtype=length.dtype)
    log_ls = lengthscale.log()
    weights = (log_ls - log_ls.mean()).exp()  # geometric-mean-normalized, prod == 1
    axis_lengths = length * weights
    return R, axis_lengths


def compute_pca_ellipsoid_shape(
    X: Tensor, X_center: Tensor, length: Tensor, dim: int, eig_floor: float = 1e-8
) -> Tuple[Tensor, Tensor]:
    r"""Rotated-box shape from PCA of the TR's local data about its center.

    Falls back to the isotropic shape (identity `R`, uniform `axis_lengths`)
    if there are fewer than `dim + 1` local points -- PCA is underdetermined
    otherwise.

    Args:
        X: `n x d`-dim tensor, the TR's local accumulated data (normalized
            `[0, 1]^d`).
        X_center: `1 x d`-dim tensor, the TR center. Used as the fixed center
            for the second-moment matrix below (rather than `X`'s own
            empirical mean), keeping the ellipsoid anchored at the same
            point every other TR operation (sampling, containment) is
            already relative to.
        length: a 0-dim tensor, the TR's current (isotropic) edge length.
        dim: input dimension `d`.
        eig_floor: minimum eigenvalue (of the second-moment matrix) before
            taking a square root, guarding near-zero axis widths from a
            rank-deficient/near-degenerate local point cloud.

    Returns:
        R: `d x d` orthonormal rotation (eigenvectors of the local
            second-moment matrix about `X_center`).
        axis_lengths: `d`-dim tensor, geometric-mean-normalized so
            `axis_lengths.prod() ** (1/d) == length`.
    """
    if X.shape[0] < dim + 1:
        return (
            torch.eye(dim, device=length.device, dtype=length.dtype),
            length.expand(dim).clone(),
        )
    delta = X - X_center
    cov = (delta.t() @ delta) / X.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp_min(eig_floor)
    scale = eigvals.sqrt()
    log_scale = scale.log()
    weights = (log_scale - log_scale.mean()).exp()
    axis_lengths = length * weights
    return eigvecs, axis_lengths


def compute_ard_pca_ellipsoid_shape(
    X: Tensor,
    X_center: Tensor,
    lengthscale: Tensor,
    length: Tensor,
    dim: int,
    eig_floor: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    r"""PCA rotation (as in `compute_pca_ellipsoid_shape`) with axis widths
    additionally reweighted by GP ARD lengthscales projected onto each
    principal axis.

    NOTE: this is *not* PCA performed in lengthscale-normalized coordinates
    (`X / lengthscale`) mapped back to the original space -- that
    construction is a mathematical no-op (it recovers exactly the plain-PCA
    covariance regardless of the lengthscales, since undoing the whitening
    exactly cancels it out: `D @ (D^-1 Sigma D^-1) @ D == Sigma` for any
    diagonal `D`). Instead, the PCA rotation `R` is computed exactly as in
    `compute_pca_ellipsoid_shape` (from raw local-data covariance,
    lengthscale-blind), and lengthscales only enter afterward as a per-axis
    reweighting of `R`'s already-fixed principal directions.

    Args:
        X, X_center, length, dim, eig_floor: as in `compute_pca_ellipsoid_shape`.
        lengthscale: a `dim`-dim tensor of (already output-aggregated) ARD
            lengthscales, e.g. from `extract_ard_lengthscale`.

    Returns:
        R: `d x d` orthonormal rotation, identical to
            `compute_pca_ellipsoid_shape`'s (lengthscales do not affect
            orientation, only per-axis width).
        axis_lengths: `d`-dim tensor, geometric-mean-normalized so
            `axis_lengths.prod() ** (1/d) == length`.
    """
    R, axis_lengths_pca = compute_pca_ellipsoid_shape(
        X=X, X_center=X_center, length=length, dim=dim, eig_floor=eig_floor
    )
    # ell_eff_k = || diag(lengthscale) @ R[:, k] ||_2 -- how "wide" the GP
    # thinks the function is along this already-fixed principal direction.
    ell_eff = (lengthscale.unsqueeze(-1) * R).norm(dim=0)
    log_eff = ell_eff.log()
    eff_weights = (log_eff - log_eff.mean()).exp()
    axis_lengths = axis_lengths_pca * eff_weights
    # Re-normalize: the reweighting above shifts the geometric mean away
    # from `length`, so rescale back to `axis_lengths.prod() ** (1/d) == length`.
    log_axis = axis_lengths.log()
    axis_lengths = axis_lengths * (length / log_axis.mean().exp())
    return R, axis_lengths


def compute_labcat_style_shape(
    X: Tensor,
    X_center: Tensor,
    Y_obj: Tensor,
    lengthscale: Tensor,
    length: Tensor,
    dim: int,
    eig_floor: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    r"""LABCAT-style shape (Visser et al. 2023): fitness-weighted PCA computed
    genuinely in lengthscale-whitened coordinates, composed (not undone) with
    the whitening scale -- the opposite construction/order from
    `compute_ard_pca_ellipsoid_shape` (see that function's docstring for the
    no-op this avoids).

    Procedure, following the paper directly: (1) whiten the TR's local data
    by the ARD lengthscale (`X' = (X - X_center) / lengthscale`, LABCAT's
    "length-scale-based rescaling", their Eq. 11-14); (2) compute a
    fitness-weighted covariance of `X'` -- points with better (here: higher,
    since we maximize) objective values get more weight, LABCAT's Eq. 36;
    (3) eigendecompose *that* covariance and keep its eigenvectors directly
    as the rotation `R` -- lengthscale and fitness both shape the rotation,
    unlike our own `ard_pca_ellipsoid`. Axis widths are the same
    lengthscale-proportional widths as `compute_ard_box_shape` (LABCAT draws
    an isotropic box in the fully-transformed, whitened-then-rotated frame,
    which is equivalent to a box whose original-native per-axis widths --
    before rotation -- are proportional to the lengthscale; rotating those
    widths by `R` afterward gives the final rotated-box representation our
    sampling/containment machinery expects).

    Multi-objective adaptation note: LABCAT is single-objective and weights
    by `1 - y'` (normalized loss, lower loss = more weight). We have no
    single scalar to rank by, so we weight by the mean of each point's
    per-objective values, independently min-max normalized across the local
    data (higher = better, since we maximize) -- the natural multi-objective
    analogue of LABCAT's weighting, not a literal reimplementation.

    Falls back to the isotropic shape if there are fewer than `dim + 1`
    local points (weighted PCA is underdetermined otherwise).

    Args:
        X, X_center, length, dim, eig_floor: as in `compute_pca_ellipsoid_shape`.
        Y_obj: `n x m`-dim tensor, `self.objective(self.Y)` for the same
            local points as `X` (row-aligned), i.e. already-selected
            objective values (not raw model outputs).
        lengthscale: a `dim`-dim tensor of (already output-aggregated) ARD
            lengthscales, e.g. from `extract_ard_lengthscale`.

    Returns:
        R: `d x d` orthonormal rotation (eigenvectors of the fitness-weighted
            covariance of the whitened local data).
        axis_lengths: `d`-dim tensor, geometric-mean-normalized so
            `axis_lengths.prod() ** (1/d) == length`.
    """
    if X.shape[0] < dim + 1:
        return (
            torch.eye(dim, device=length.device, dtype=length.dtype),
            length.expand(dim).clone(),
        )
    Xw = (X - X_center) / lengthscale
    y_min = Y_obj.min(dim=0).values
    y_max = Y_obj.max(dim=0).values
    y_range = (y_max - y_min).clamp_min(1e-9)
    y_norm = (Y_obj - y_min) / y_range  # n x m, higher = better, in [0, 1]
    w = y_norm.mean(dim=-1).clamp_min(1e-6)  # n, avoid a zero-weight point
    w_sum = w.sum()
    cov = (Xw * w.unsqueeze(-1)).t() @ Xw / w_sum
    cov = 0.5 * (cov + cov.t())  # symmetrize against floating-point drift
    eigvals, eigvecs = torch.linalg.eigh(cov)
    R = eigvecs
    log_ls = lengthscale.log()
    weights = (log_ls - log_ls.mean()).exp()  # same as compute_ard_box_shape
    axis_lengths = length * weights
    return R, axis_lengths


def compute_cma_ellipsoid_shape(
    elites: Tensor,
    X_center: Tensor,
    prev_center: Optional[Tensor],
    C: Tensor,
    path: Tensor,
    length: Tensor,
    dim: int,
    c_mu: float,
    c1: float,
    c_p: float,
    eig_floor: float = 1e-8,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""CMA-ES-style covariance adaptation for a trust region's shape
    (cf. Wang et al. 2026's AS-SMEA, which maintains each local region as a
    search distribution N(m, sigma^2 C) updated by CMA).

    Two differences from the one-shot PCA variants
    (`compute_pca_ellipsoid_shape`): (1) the covariance `C` is *persistent*
    per-TR state, exponentially smoothed across iterations rather than
    recomputed from scratch -- so one noisy batch can't whipsaw the region's
    orientation; (2) the update is weighted toward *success* (the TR's
    current Pareto-elite points), plus a rank-one evolution-path term
    tracking sustained center movement -- rather than toward wherever
    candidates happened to be sampled, which for PCA partly reflects the
    sampler's own previous shape (a feedback loop this construction avoids).

    Multi-objective adaptation note: classical CMA weights elites by
    fitness rank; a Pareto set has no total order, so elites (the TR's
    current Pareto points, already selected by non-dominated sorting
    upstream) are weighted equally -- the same top-samples-by-nondominated-
    sorting selection AS-SMEA uses.

    Args:
        elites: `k x d`-dim tensor of elite points in normalized `[0,1]^d`
            coordinates (the TR's current Pareto points). May be empty.
        X_center: `1 x d`-dim tensor, current TR center (normalized).
        prev_center: `1 x d`-dim tensor, the TR center at the previous shape
            update, or None on the first update (path term is skipped).
        C: `d x d`-dim tensor, the current persistent covariance state.
        path: `d`-dim tensor, the current evolution path state.
        length: 0-dim tensor, the TR's current (isotropic) edge length --
            used as the step-size sigma normalizing both updates, and as the
            geometric-mean target for `axis_lengths`.
        dim: input dimension `d`.
        c_mu, c1, c_p: CMA learning rates (see `TurboHParams`).
        eig_floor: minimum eigenvalue before sqrt.

    Returns:
        (R, axis_lengths, C_new, path_new): the shape decomposition of the
        updated covariance (same conventions as the other compute_*_shape
        functions -- `axis_lengths.prod()**(1/d) == length`), plus the
        updated persistent state to write back into the TR's buffers.
    """
    sigma = length.clamp_min(1e-12)

    # Evolution path: sustained, direction-consistent center movement.
    if prev_center is not None:
        delta_m = (X_center - prev_center).reshape(-1) / sigma
        path_new = (1 - c_p) * path + (c_p * (2 - c_p)) ** 0.5 * delta_m
    else:
        path_new = path.clone()

    # Rank-mu update from equally-weighted elites (see docstring).
    if elites.numel() > 0 and elites.shape[0] >= 1:
        Y = (elites - X_center) / sigma  # `k x d`
        rank_mu = (Y.t() @ Y) / elites.shape[0]
        C_new = (1 - c_mu - c1) * C + c_mu * rank_mu + c1 * torch.outer(
            path_new, path_new
        )
    else:
        # No elites this iteration: decay toward what we had, path term only.
        C_new = (1 - c1) * C + c1 * torch.outer(path_new, path_new)

    # Symmetrize against floating-point drift before eigh.
    C_new = 0.5 * (C_new + C_new.t())

    eigvals, eigvecs = torch.linalg.eigh(C_new)
    eigvals = eigvals.clamp_min(eig_floor)
    scale = eigvals.sqrt()
    log_scale = scale.log()
    weights = (log_scale - log_scale.mean()).exp()
    axis_lengths = length * weights
    return eigvecs, axis_lengths, C_new, path_new


class HypersphereProjection(InputTransform, Module):
    r"""Bijectively project the (normalized) unit hypercube onto the upper
    hypersphere in `R^{d+1}`, following Doumont et al. 2026 ("We Still Don't
    Understand High-Dimensional Bayesian Optimization"): a linear kernel
    applied to raw hypercube inputs is pathologically boundary-seeking, but
    on the sphere the same kernel becomes a cosine-similarity kernel with
    provable immunity from boundary-seeking behavior.

    The map: center the cube (`x - 0.5`), append a constant bias coordinate
    (0.5, the same scale as the centered coordinates' range), and normalize
    to unit Euclidean norm. Strictly positive bias => upper hemisphere =>
    injective. Applied AFTER the usual Normalize chain, so inputs arriving
    here are in `[0, 1]^d`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = True

    def transform(self, X: Tensor) -> Tensor:
        X_centered = X - 0.5
        bias = torch.full(
            X.shape[:-1] + (1,), 0.5, dtype=X.dtype, device=X.device
        )
        X_aug = torch.cat([X_centered, bias], dim=-1)
        return X_aug / X_aug.norm(dim=-1, keepdim=True)

    def equals(self, other: InputTransform) -> bool:
        return type(self) is type(other)


def get_fitted_model(
    X: Tensor,
    Y: Tensor,
    use_ard: bool,
    max_cholesky_size: int,
    state_dict: Optional[Dict[str, Tensor]] = None,
    input_transform: Optional[InputTransform] = None,
    outcome_transform: Optional[OutcomeTransform] = None,
    fit_gpytorch_options: Optional[Dict[str, Any]] = None,
    use_linear_kernel: bool = False,
    use_dim_scaled_ls_prior: bool = False,
) -> Model:
    r"""Fit one independent SingleTaskGP per output column (bundled as a
    ModelListGP when there is more than one).

    Kernel options (both default off; existing behavior unchanged):
      use_linear_kernel: ScaleKernel(LinearKernel()) over inputs projected
        onto the hypersphere in R^{d+1} (`HypersphereProjection`, chained
        after the usual Normalize transform) -- the linear-bo challenge
        baseline. No lengthscales exist in this mode
        (`extract_ard_lengthscale` returns None; ARD-based tr_shape
        variants fall back to isotropic).
      use_dim_scaled_ls_prior: replace the Matern lengthscale's hard
        Interval(0.05, 4.0) constraint with the dimension-scaled prior of
        Hvarfner et al. 2024, lengthscale ~ LogNormal(sqrt(2) + ln(d)/2,
        sqrt(3)) -- expects lengthscales to grow ~sqrt(d), removing the 4.0
        ceiling that ~99/100 of a fitted d=100 model's lengthscales pin
        against (the direct input to ard_box's region collapse).
    """
    print("Fitting a model")
    use_fast_mvms = True if X.shape[0] > max_cholesky_size else False
    with gpytorch_settings.fast_computations(
        log_prob=use_fast_mvms,
        covar_root_decomposition=use_fast_mvms,
        solves=use_fast_mvms,
    ):
        if use_linear_kernel:
            sphere = HypersphereProjection()
            if input_transform is not None:
                from botorch.models.transforms.input import ChainedInputTransform

                input_transform = ChainedInputTransform(
                    base=input_transform, sphere=sphere
                )
            else:
                input_transform = sphere
        models = []
        for i in range(Y.shape[-1]):
            if use_linear_kernel:
                covar_module = ScaleKernel(LinearKernel())
            elif use_dim_scaled_ls_prior:
                d = X.shape[-1]
                covar_module = ScaleKernel(
                    MaternKernel(
                        nu=2.5,
                        ard_num_dims=d if use_ard else 1,
                        lengthscale_prior=LogNormalPrior(
                            loc=2.0**0.5 + log(d) / 2.0, scale=3.0**0.5
                        ),
                        lengthscale_constraint=GreaterThan(1e-4),
                    ),
                )
            else:
                ard_num_dims = X.shape[-1] if use_ard else 1
                covar_module = ScaleKernel(
                    MaternKernel(
                        nu=2.5,
                        ard_num_dims=ard_num_dims,
                        lengthscale_constraint=Interval(0.05, 4.0),
                    ),
                )
            likelihood = GaussianLikelihood(
                noise_constraint=GreaterThan(1e-6),
                noise_prior=GammaPrior(0.9, 10.0),
            )
            model = SingleTaskGP(
                train_X=X,
                train_Y=Y[:, i : i + 1],
                covar_module=covar_module,
                likelihood=likelihood,
                outcome_transform=outcome_transform.subset_output([i])
                if outcome_transform
                else None,
                input_transform=input_transform,
            )
            models.append(model)

        # TODO: replaced with batched-MO model once MTMVN refactor
        # lands: https://github.com/cornellius-gp/gpytorch/pull/1083
        if Y.shape[-1] > 1:
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
        else:
            model = models[0]
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

        if state_dict is not None:
            model.load_state_dict(state_dict)
        # 50 iterations appears to be a good compromise between fit and overhead.
        if fit_gpytorch_options:
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": fit_gpytorch_options})
        else:
            fit_gpytorch_mll(mll)

    if X.is_cuda:
        print(f"after fitting: {torch.cuda.memory_allocated(X.device) / (1000 ** 3)}")
    return model


def get_fitted_kronecker_model(
    X: Tensor,
    Y: Tensor,
    max_cholesky_size: int,
    input_transform: Optional[InputTransform] = None,
    outcome_transform: Optional[OutcomeTransform] = None,
    fit_gpytorch_options: Optional[Dict[str, Any]] = None,
) -> Model:
    r"""Fit a single Kronecker-structured multi-task GP jointly over all
    output columns, rather than one independent GP per column
    (`get_fitted_model`). Used to test whether modeling *correlation*
    across a composite raw response's dimensions (vs. `get_fitted_model`'s
    decoupled-per-dimension modeling) matters for downstream optimization
    performance -- see `morbo/problems/composite_dtlz2_curve.py`.

    `task_covar_prior=None` and a torch/Adam-based fit
    (`fit_gpytorch_mll_torch`) are used instead of the scipy-based default
    fit (`fit_gpytorch_mll`) -- the latter fails on this model's default
    task-covariance prior (`sample_all_priors` raises "Must provide inverse
    transform to be able to sample from prior").
    """
    print("Fitting a Kronecker multi-task model")
    use_fast_mvms = True if X.shape[0] > max_cholesky_size else False
    with gpytorch_settings.fast_computations(
        log_prob=use_fast_mvms,
        covar_root_decomposition=use_fast_mvms,
        solves=use_fast_mvms,
    ):
        model = KroneckerMultiTaskGP(
            train_X=X,
            train_Y=Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            task_covar_prior=None,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        step_limit = (
            (fit_gpytorch_options or {}).get("maxiter")
        )
        fit_gpytorch_mll_torch(mll, step_limit=step_limit)

    if X.is_cuda:
        print(f"after fitting: {torch.cuda.memory_allocated(X.device) / (1000 ** 3)}")
    return model


def compose(
    reduction: Optional[Callable[[Tensor], Tensor]],
    objective: Callable[[Tensor], Tensor],
) -> Callable[[Tensor], Tensor]:
    r"""Compose a known reduction in front of an objective.

    For composite modeling: `reduction` maps a raw model response to the
    final objectives (e.g. applying a known formula), and `objective` is
    whatever column-selection/scalarization would otherwise be applied
    directly to those objectives. Returns `objective` unchanged if
    `reduction` is None, so this is a no-op for non-composite problems.

    Args:
        reduction: A callable mapping a `... x K`-dim raw response to a
            `... x M`-dim tensor of objectives, or None.
        objective: The objective callable to apply after `reduction`.

    Returns:
        A callable equivalent to `lambda Y: objective(reduction(Y))`.
    """
    if reduction is None:
        return objective
    return lambda Y: objective(reduction(Y))


def coalesce(x1: Optional[Tensor], x2: Optional[Tensor]) -> Optional[Tensor]:
    r"""Helper function the performs a coalesce operation.

    If x1 is not None, it is returned. Otherwise x2 is returned.

    Args:
        x1: a tensor
        x2 a tensor

    Returns:
        A tensor if either of x1 or x2 is not None, otherwise None.
    """
    if x1 is None:
        x1 = x2
    return x1


def decay_function(n: int, n0: int, n_max: int, alpha: float = 1.0) -> float:
    r"""Decay function governed by the used and remaining optimization budget.

    Decay function from:
        Regis R.G., Shoemaker C.A. Combining radial basis function
        surrogates and dynamic coordinate search in high-dimensional
        expensive black-box optimization. Engineering Optimization, 45
        (5) (2013), pp. 529-555

    Args:
        n: number of completed function evaluations
        n0: number of initial function evaluations
        n_max: maximum number of function evaluations (budget)
        alpha: hyperparameter controlling decay

    Returns:
        The probabilty of perturbing a dimension.
    """
    return 1 - alpha * log(n - n0 + 1) / log(n_max - n0 + 1)


def get_constraint_slack_and_feasibility(
    Y: Tensor, constraints: List[Callable[[Tensor], Tensor]]
) -> Tensor:
    r"""Compute feasibility.

    Args:
        Y: A `batch_shape x n x m`-dim tensor of outcomes
        constraints: A list of constraint callables mapping outcomes to the
            constraint slack.

    Returns:
        A `batch_shape x n`-dim boolean tensor indicating whether each example in Y
            is feasible.
    """
    constraint_slack = torch.stack([c(Y) for c in constraints], dim=-1)
    return constraint_slack, (constraint_slack <= 0).all(dim=-1)
