"""
Bayesian Prototype Calibration with Gaussian Process formalism (BPC-GP).

Given a prior on the prototype location (from AGPPI) and a set of support
embeddings with per-sample precision weights (from APM), BPC-GP computes the
*posterior* prototype via a closed-form Gaussian update.

Model
-----
Prior:    μ_c ~ N(μ₀, τ₀⁻¹ I)
Likelihood: z_i | μ_c ~ N(μ_c, α_i⁻¹ I)    for i = 1…K

Posterior (conjugate Gaussian):
    τ_post = τ₀ + Σ_i α_i
    μ_post = (τ₀ · μ₀ + Σ_i α_i · z_i) / τ_post

The posterior variance (uncertainty) is:
    σ²_post = 1 / τ_post

Both posterior mean and variance are returned so that downstream modules
(novelty detection, PIM) can use the uncertainty estimate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPCGPModule(nn.Module):
    """
    Closed-form Bayesian Prototype Calibration.

    Parameters
    ----------
    embedding_dim:
        Dimensionality D of the embedding space.
    prior_var:
        Prior variance σ₀² = 1/τ₀.
    noise_var:
        Observation noise variance (added to α_i⁻¹ for numerical stability).
    use_bpc_gp:
        If ``False``, returns the plain (SQAA-weighted) mean as the prototype
        with variance 1/K (ablation mode).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        prior_var: float = 1.0,
        noise_var: float = 0.01,
        use_bpc_gp: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.noise_var = noise_var
        self.use_bpc_gp = use_bpc_gp

        # Log-parameterised prior precision (learnable)
        if use_bpc_gp:
            self.log_prior_precision = nn.Parameter(
                torch.tensor(1.0 / prior_var).log()
            )
        else:
            self.log_prior_precision = None  # type: ignore[assignment]

    @property
    def prior_precision(self) -> torch.Tensor:
        """Positive prior precision τ₀."""
        return self.log_prior_precision.exp()  # type: ignore[union-attr]

    def forward(
        self,
        support_embeddings: torch.Tensor,
        precision_weights: torch.Tensor,
        prior_mean: Optional[torch.Tensor] = None,
        prior_precision: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the posterior prototype mean and variance.

        Parameters
        ----------
        support_embeddings:
            Shape ``(K, D)`` — L2-normalised support embeddings.
        precision_weights:
            Shape ``(K,)`` — per-sample precision weights α_i > 0.
        prior_mean:
            Shape ``(D,)`` — prior mean μ₀.  If ``None``, uses the zero
            vector (equivalent to no prior mean shift).
        prior_precision:
            Scalar prior precision τ₀.  If ``None``, uses the learnable
            ``self.prior_precision``.

        Returns
        -------
        posterior_mean:
            Shape ``(D,)`` — calibrated prototype (not L2-normalised;
            caller decides whether to normalise).
        posterior_var:
            Scalar — posterior variance = 1 / τ_post.
        """
        K, D = support_embeddings.shape
        device = support_embeddings.device
        dtype = support_embeddings.dtype

        if not self.use_bpc_gp or self.log_prior_precision is None:
            # Ablation: plain weighted mean
            w = precision_weights / precision_weights.sum().clamp(min=1e-8)
            proto = (w.unsqueeze(1) * support_embeddings).sum(dim=0)
            var = torch.tensor(1.0 / K, device=device, dtype=dtype)
            return proto, var

        # Resolve prior parameters
        tau0 = (
            prior_precision
            if prior_precision is not None
            else self.prior_precision.to(device)
        )  # scalar

        mu0 = (
            prior_mean
            if prior_mean is not None
            else torch.zeros(D, device=device, dtype=dtype)
        )  # (D,)

        # Posterior precision
        sum_alpha = precision_weights.sum()  # scalar
        tau_post = tau0 + sum_alpha          # scalar

        # Posterior mean
        weighted_sum = (precision_weights.unsqueeze(1) * support_embeddings).sum(0)  # (D,)
        mu_post = (tau0 * mu0 + weighted_sum) / tau_post.clamp(min=1e-8)             # (D,)

        # Posterior variance (isotropic)
        sigma_sq_post = 1.0 / tau_post.clamp(min=1e-8)

        return mu_post, sigma_sq_post

    def calibrate_class(
        self,
        support_embeddings: torch.Tensor,
        precision_weights: torch.Tensor,
        prior_mean: Optional[torch.Tensor] = None,
        prior_precision: Optional[torch.Tensor] = None,
        normalise_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper: compute posterior prototype and optionally
        L2-normalise the result.

        Parameters
        ----------
        support_embeddings, precision_weights, prior_mean, prior_precision:
            As in ``forward()``.
        normalise_output:
            If ``True``, L2-normalise the posterior mean before returning.

        Returns
        -------
        prototype:
            Shape ``(D,)`` — calibrated (and optionally normalised) prototype.
        uncertainty:
            Scalar posterior variance.
        """
        proto, var = self.forward(
            support_embeddings, precision_weights, prior_mean, prior_precision
        )
        if normalise_output:
            proto = F.normalize(proto, p=2, dim=0)
        return proto, var

    def extra_repr(self) -> str:
        prec = (
            self.log_prior_precision.exp().item()
            if self.log_prior_precision is not None
            else "N/A"
        )
        return (
            f"dim={self.embedding_dim}, use_bpc_gp={self.use_bpc_gp}, "
            f"tau0={prec:.3f}"
        )
