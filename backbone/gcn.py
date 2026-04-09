"""
Gamma-Conditioned Normalisation (GCN).

Standard batch/layer normalisation assumes Gaussian-distributed activations.
SAR intensity images follow a Gamma distribution; their feature maps can
inherit non-Gaussian statistics.  GCN adapts the normalisation affine
parameters (scale γ and shift β) based on the *estimated Gamma shape k* of
the current feature map, letting the network learn different normalisation
behaviours for different clutter levels.

Specifically, for a feature map X with estimated shape k:

    X_norm = (X - μ) / (σ + ε)                   [standard normalisation]
    X_out  = γ(k) * X_norm + β(k)                 [Gamma-conditioned affine]

where γ(k) and β(k) are produced by small MLPs that take k as input.

When ``use_gcn=False`` the module degrades to plain LayerNorm (ablation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GammaConditionedNorm(nn.Module):
    """
    Gamma-Conditioned Normalisation applied channel-wise over spatial dims.

    Parameters
    ----------
    num_channels:
        Number of feature channels (C in B×C×H×W).
    cond_hidden:
        Hidden dimension of the small MLPs that produce γ(k) and β(k).
    eps:
        Numerical stability term for the normalisation denominator.
    use_gcn:
        If ``False``, falls back to plain LayerNorm (ablation toggle).
    """

    def __init__(
        self,
        num_channels: int,
        cond_hidden: int = 32,
        eps: float = 1e-5,
        use_gcn: bool = True,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.use_gcn = use_gcn

        # Fallback plain normalisation (also used as base in GCN)
        self.norm = nn.LayerNorm(num_channels)

        if use_gcn:
            # MLP: k (scalar) → γ adjustment (num_channels,)
            self.gamma_net = nn.Sequential(
                nn.Linear(1, cond_hidden),
                nn.SiLU(),
                nn.Linear(cond_hidden, num_channels),
            )
            # MLP: k (scalar) → β adjustment (num_channels,)
            self.beta_net = nn.Sequential(
                nn.Linear(1, cond_hidden),
                nn.SiLU(),
                nn.Linear(cond_hidden, num_channels),
            )
            # Initialise output layers to near-identity
            nn.init.zeros_(self.gamma_net[-1].weight)
            nn.init.ones_(self.gamma_net[-1].bias)
            nn.init.zeros_(self.beta_net[-1].weight)
            nn.init.zeros_(self.beta_net[-1].bias)

    @staticmethod
    def _estimate_gamma_shape(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Fast per-sample Gamma shape estimate using the log-mean estimator.

        For each sample b, collapse spatial + channel dims and compute:

            k ≈ (mean(x))² / Var(x)          [method-of-moments]

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)`` — shape parameter k, clamped to (eps, ∞).
        """
        x_flat = x.reshape(x.size(0), -1).clamp(min=eps)  # (B, C·H·W)
        mu = x_flat.mean(dim=1)                          # (B,)
        var = x_flat.var(dim=1).clamp(min=eps)           # (B,)
        k = (mu ** 2) / var                              # (B,)
        k = k.clamp(min=eps).unsqueeze(1)                # (B, 1)
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise *x* with optional Gamma-conditioned affine transform.

        Parameters
        ----------
        x:
            Feature map of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Normalised feature map, same shape as *x*.
        """
        B, C, H, W = x.shape

        # LayerNorm expects (..., C); permute to (B, H, W, C)
        x_perm = x.permute(0, 2, 3, 1)          # (B, H, W, C)
        x_norm = self.norm(x_perm)               # (B, H, W, C)

        if not self.use_gcn:
            return x_norm.permute(0, 3, 1, 2)   # back to (B, C, H, W)

        # Estimate k from the raw (pre-norm) feature map
        k = self._estimate_gamma_shape(x, self.eps)  # (B, 1)

        # Condition scale and shift on k
        gamma = self.gamma_net(k)  # (B, C)
        beta = self.beta_net(k)    # (B, C)

        # Apply: broadcast over H, W
        gamma = gamma.view(B, 1, 1, C)   # (B, 1, 1, C)
        beta = beta.view(B, 1, 1, C)     # (B, 1, 1, C)

        x_out = gamma * x_norm + beta    # (B, H, W, C)
        return x_out.permute(0, 3, 1, 2)  # (B, C, H, W)

    def extra_repr(self) -> str:
        return f"channels={self.num_channels}, use_gcn={self.use_gcn}"
