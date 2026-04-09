"""
SAR Signal-to-Noise Ratio (SNR) estimation.

In SAR imagery, "signal" is the coherent backscatter from a target and
"noise" is a combination of thermal receiver noise and speckle.  For a
detected (intensity) image the local SNR can be approximated as:

    SNR = (μ_signal² - σ²_noise) / σ²_noise

where μ_signal is the local mean intensity and σ²_noise is estimated from a
background region.  We use the Equivalent Number of Looks (ENL) as a proxy:

    ENL = μ² / σ²   ⟹   true_SNR ≈ ENL - 1

This module provides:
  * Per-pixel local SNR maps via sliding-window statistics.
  * A ``SNREstimationModule`` (nn.Module) that wraps the estimator and
    optionally learns a correction MLP to improve raw estimates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pure function
# ---------------------------------------------------------------------------

def local_snr(
    image: torch.Tensor,
    patch_size: int = 7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Estimate a per-pixel local SNR map using Equivalent Number of Looks.

    SNR ≈ max(ENL - 1, 0),  ENL = μ² / σ²

    Parameters
    ----------
    image:
        Single-channel SAR intensity image, shape ``(B, 1, H, W)``.
    patch_size:
        Square neighbourhood for local statistics (must be odd).
    eps:
        Numerical stability floor.

    Returns
    -------
    torch.Tensor
        Per-pixel SNR map, shape ``(B, 1, H, W)``, values in [0, ∞).
    """
    assert patch_size % 2 == 1, "patch_size must be odd"
    pad = patch_size // 2

    x = image.clamp(min=eps)

    # Local mean via average pooling
    mu = F.avg_pool2d(
        F.pad(x, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=patch_size,
        stride=1,
        padding=0,
    )  # (B, 1, H, W)

    # Local second moment
    x_sq = x ** 2
    mu_sq_raw = F.avg_pool2d(
        F.pad(x_sq, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=patch_size,
        stride=1,
        padding=0,
    )  # (B, 1, H, W)

    # Variance = E[X²] - (E[X])²
    var = (mu_sq_raw - mu ** 2).clamp(min=eps)

    # ENL
    enl = (mu ** 2) / var

    # SNR = ENL - 1, clipped to non-negative
    snr = (enl - 1.0).clamp(min=0.0)
    return snr


def speckle_index(
    image: torch.Tensor,
    patch_size: int = 7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the coefficient of variation (speckle index) = σ / μ.

    For fully developed speckle SI = 1; for a coherent target SI → 0.

    Parameters
    ----------
    image:
        Shape ``(B, 1, H, W)``.

    Returns
    -------
    torch.Tensor
        Speckle index map, shape ``(B, 1, H, W)``.
    """
    assert patch_size % 2 == 1
    pad = patch_size // 2
    x = image.clamp(min=eps)

    mu = F.avg_pool2d(
        F.pad(x, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=patch_size, stride=1, padding=0,
    )
    x_sq = x ** 2
    e_sq = F.avg_pool2d(
        F.pad(x_sq, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=patch_size, stride=1, padding=0,
    )
    var = (e_sq - mu ** 2).clamp(min=eps)
    si = var.sqrt() / mu.clamp(min=eps)
    return si


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------

class SNREstimationModule(nn.Module):
    """
    SAR SNR estimation module with an optional learned correction network.

    The raw sliding-window estimator is inherently noisy.  A small
    convolutional correction network is trained (end-to-end with the
    backbone) to refine the estimate.

    Parameters
    ----------
    patch_size:
        Sliding-window size for local statistics.
    use_snr:
        If ``False``, returns a constant SNR=1 map (ablation mode).
    learned_correction:
        If ``True``, add a 3-layer conv net to refine raw SNR estimates.
    eps:
        Numerical stability floor.
    """

    def __init__(
        self,
        patch_size: int = 7,
        use_snr: bool = True,
        learned_correction: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.use_snr = use_snr
        self.eps = eps

        if use_snr and learned_correction:
            # Input: [raw_snr, speckle_index, raw_image] = 3 channels
            self.correction = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(8, 1, kernel_size=1),
                nn.Softplus(),   # ensure non-negative output
            )
        else:
            self.correction = None  # type: ignore[assignment]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Estimate per-pixel SNR.

        Parameters
        ----------
        image:
            SAR intensity image, shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            SNR map, shape ``(B, 1, H, W)``.
        """
        if not self.use_snr:
            return torch.ones_like(image)

        snr_raw = local_snr(image, patch_size=self.patch_size, eps=self.eps)

        if self.correction is not None:
            si = speckle_index(image, patch_size=self.patch_size, eps=self.eps)
            inp = torch.cat([snr_raw, si, image], dim=1)  # (B, 3, H, W)
            snr_refined = self.correction(inp)             # (B, 1, H, W)
            return snr_refined

        return snr_raw

    def global_snr(self, image: torch.Tensor) -> torch.Tensor:
        """
        Return a single SNR scalar per image (global mean of SNR map).

        Returns
        -------
        torch.Tensor
            Shape ``(B,)``.
        """
        snr_map = self.forward(image)
        return snr_map.mean(dim=(-3, -2, -1))  # (B,)

    def extra_repr(self) -> str:
        return (
            f"patch_size={self.patch_size}, use_snr={self.use_snr}, "
            f"correction={self.correction is not None}"
        )
