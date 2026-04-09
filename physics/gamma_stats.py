"""
Gamma distribution statistics for SAR imagery.

SAR intensity images (formed by detecting the complex amplitude) follow a
Gamma distribution:

    I ~ Gamma(k, θ),   E[I] = k·θ,   Var[I] = k·θ²

The number of looks L (equivalent looks) is equal to the shape parameter k.
Estimating (k, θ) from an image patch gives us a compact physics-informed
descriptor of local clutter / target properties.

This module provides:
  * Closed-form method-of-moments estimator (fast, differentiable).
  * Newton–Raphson MLE (more accurate; used at training time).
  * A differentiable ``GammaStatisticsModule`` that wraps both estimators
    as a ``nn.Module`` so gradients can flow through the statistics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.special import digamma, polygamma  # available in PyTorch ≥ 1.9


# ---------------------------------------------------------------------------
# Pure functions (can be used outside nn.Module)
# ---------------------------------------------------------------------------

def mom_estimate(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Method-of-moments Gamma estimator.

    Parameters
    ----------
    x:
        Positive intensity values, any shape ``(..., N)`` where N is the
        sample dimension.
    eps:
        Numerical stability floor for variances and denominators.

    Returns
    -------
    k:
        Estimated shape (number of looks), same shape as the batch dims.
    theta:
        Estimated scale, same shape as the batch dims.
    """
    mu = x.mean(dim=-1).clamp(min=eps)
    var = x.var(dim=-1).clamp(min=eps)
    k = (mu ** 2) / var
    theta = var / mu
    return k, theta


def mle_estimate(
    x: torch.Tensor,
    max_iter: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Newton–Raphson MLE for Gamma(k, θ).

    Solves the score equation:

        log(k) - ψ(k) = log(mean(x)) - mean(log(x))   ≡ s

    where ψ is the digamma function.  Initial guess from method-of-moments.

    Parameters
    ----------
    x:
        Positive intensity values, shape ``(..., N)``.
    max_iter:
        Maximum Newton–Raphson iterations.
    eps:
        Numerical floor.

    Returns
    -------
    k, theta:
        MLE estimates, same batch shape as the non-sample dims of *x*.
    """
    x = x.clamp(min=eps)
    mu = x.mean(dim=-1).clamp(min=eps)

    # Sufficient statistic s = log(mean) - mean(log)
    log_mean = mu.log()
    mean_log = x.log().mean(dim=-1)
    s = log_mean - mean_log  # (...,)

    # Initial estimate via method-of-moments
    k, _ = mom_estimate(x, eps=eps)
    k = k.clamp(min=eps)

    # Newton–Raphson:  k_{n+1} = k_n - (log(k_n) - ψ(k_n) - s) / (1/k_n - ψ₁(k_n))
    for _ in range(max_iter):
        f = k.log() - digamma(k) - s          # score
        df = 1.0 / k - polygamma(1, k)        # derivative (using trigamma)
        # Guard against zero derivative
        df = df.clamp(max=-eps)
        k = (k - f / df).clamp(min=eps)

    theta = mu / k
    return k, theta


# ---------------------------------------------------------------------------
# Patch-level batch estimator
# ---------------------------------------------------------------------------

def patch_gamma_stats(
    image: torch.Tensor,
    patch_size: int = 7,
    estimator: str = "mom",
    eps: float = 1e-6,
    max_iter: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate Gamma (k, θ) from local patches via an unfolded sliding window.

    Parameters
    ----------
    image:
        Single-channel SAR intensity image, shape ``(B, 1, H, W)``.
    patch_size:
        Square patch side length (must be odd).
    estimator:
        ``"mom"`` for method-of-moments or ``"mle"`` for Newton–Raphson MLE.
    eps, max_iter:
        Passed to the underlying estimator.

    Returns
    -------
    k_map, theta_map:
        Shape ``(B, 1, H, W)`` — per-pixel Gamma parameter maps estimated
        from the surrounding patch.
    """
    assert patch_size % 2 == 1, "patch_size must be odd"
    B, C, H, W = image.shape
    assert C == 1, "Expected single-channel image"

    pad = patch_size // 2
    x_padded = torch.nn.functional.pad(
        image, (pad, pad, pad, pad), mode="reflect"
    )  # (B, 1, H+2p, W+2p)

    # Unfold into overlapping patches: (B, 1, H, W, patch_size²)
    patches = x_padded.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    # patches: (B, 1, H, W, patch_size, patch_size)
    patches = patches.reshape(B, H, W, -1)  # (B, H, W, P²)
    patches = patches.clamp(min=eps)

    if estimator == "mle":
        k, theta = mle_estimate(patches, max_iter=max_iter, eps=eps)
    else:
        k, theta = mom_estimate(patches, eps=eps)

    # (B, H, W) → (B, 1, H, W)
    k = k.unsqueeze(1)
    theta = theta.unsqueeze(1)
    return k, theta


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------

class GammaStatisticsModule(nn.Module):
    """
    Differentiable Gamma statistics estimator as a ``nn.Module``.

    Accepts a SAR intensity image and returns per-pixel (k, θ) maps.

    Parameters
    ----------
    patch_size:
        Local patch size for estimation.
    estimator:
        ``"mom"`` or ``"mle"``.
    eps, max_iter:
        Numerical options.
    """

    def __init__(
        self,
        patch_size: int = 7,
        estimator: str = "mom",
        eps: float = 1e-6,
        max_iter: int = 20,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.estimator = estimator
        self.eps = eps
        self.max_iter = max_iter

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        image:
            Shape ``(B, 1, H, W)`` — SAR intensity image.

        Returns
        -------
        k_map:
            Shape ``(B, 1, H, W)`` — estimated Gamma shape parameter.
        theta_map:
            Shape ``(B, 1, H, W)`` — estimated Gamma scale parameter.
        """
        return patch_gamma_stats(
            image,
            patch_size=self.patch_size,
            estimator=self.estimator,
            eps=self.eps,
            max_iter=self.max_iter,
        )

    def global_stats(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate a single (k, θ) per image (whole-image statistics).

        Returns
        -------
        k, theta:
            Shape ``(B,)`` each.
        """
        x = image.view(image.size(0), -1).clamp(min=self.eps)
        if self.estimator == "mle":
            return mle_estimate(x, max_iter=self.max_iter, eps=self.eps)
        return mom_estimate(x, eps=self.eps)

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, estimator={self.estimator}"
