"""
Radial-Aware Convolution (RAC) for SRAB.

SAR images exhibit characteristic radial speckle patterns aligned with the
sensor's range and azimuth directions.  A standard square kernel treats all
spatial offsets equally; a radial-aware kernel explicitly decouples the
**angular** and **radial** contributions, letting the network learn different
sensitivities along each radial ring.

Architecture
------------
Given a kernel of spatial size ``K × K`` (K must be odd), each weight is
factored as:

    w(Δx, Δy) = w_base(Δx, Δy) · Σ_r  α_r · φ_r(‖(Δx, Δy)‖)

where φ_r is the r-th radial basis function (Gaussian centred at radius r_r),
and α_r are learnable scalars.  The factorisation is implemented efficiently
by pre-computing a static radial-basis tensor and combining it with the
learnable weights at each forward pass.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialBasisMask(nn.Module):
    """
    Generates a static ``(num_rings, K, K)`` radial-basis-function tensor
    and keeps it as a buffer (no gradient, moved to device automatically).

    Parameters
    ----------
    kernel_size:
        Spatial size of the square kernel (must be odd).
    num_rings:
        Number of Gaussian radial rings.
    sigma_init:
        Initial standard deviation for each Gaussian (in pixel units).
    """

    def __init__(
        self, kernel_size: int = 3, num_rings: int = 8, sigma_init: float = 1.0
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.num_rings = num_rings

        half = kernel_size // 2
        # (K, K) grid of distances from the centre pixel
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32),
            torch.arange(-half, half + 1, dtype=torch.float32),
            indexing="ij",
        )
        dist = torch.sqrt(x ** 2 + y ** 2)  # (K, K)

        # Ring centres uniformly spaced in [0, max_dist]
        max_dist = half * math.sqrt(2)
        centres = torch.linspace(0.0, max_dist, num_rings)  # (num_rings,)

        # Gaussian basis: (num_rings, K, K)
        sigma = torch.tensor(sigma_init)
        basis = torch.exp(
            -0.5 * ((dist.unsqueeze(0) - centres.view(-1, 1, 1)) / sigma) ** 2
        )

        # Normalise each ring so that the mask sums to 1 over spatial dims
        basis = basis / (basis.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        self.register_buffer("basis", basis)  # (R, K, K)

    def forward(self) -> torch.Tensor:
        """Return the ``(num_rings, K, K)`` basis tensor."""
        return self.basis  # type: ignore[return-value]


class RadialAwareConv2d(nn.Module):
    """
    Drop-in replacement for ``nn.Conv2d`` that modulates its kernel with
    learnable radial basis scalings.

    The effective weight tensor is:

        W_eff[out, in, :, :] = W_base[out, in, :, :]
                                * (α[out, in, :] @ basis).squeeze(-3)

    where ``α`` has shape ``(out_channels, in_channels, num_rings)``.

    Parameters
    ----------
    in_channels, out_channels, kernel_size, stride, padding, dilation, groups:
        Same as ``nn.Conv2d``.
    num_rings:
        Number of radial basis functions.
    use_radial:
        If ``False``, the module degrades to a plain ``Conv2d`` (ablation mode).
    bias:
        Whether to add a learnable bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        num_rings: int = 8,
        use_radial: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_radial = use_radial

        # Base kernel weights (same shape as a standard Conv2d)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )

        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias_param = None

        if use_radial:
            # Learnable ring-combination scalars
            self.alpha = nn.Parameter(
                torch.ones(out_channels, in_channels // groups, num_rings) / num_rings
            )
            self.rbm = RadialBasisMask(
                kernel_size=kernel_size, num_rings=num_rings
            )
        else:
            self.alpha = None  # type: ignore[assignment]
            self.rbm = None    # type: ignore[assignment]

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def _compute_effective_weight(self) -> torch.Tensor:
        """
        Compute the radially-modulated effective kernel.

        Returns
        -------
        torch.Tensor
            Shape ``(out_channels, in_channels // groups, K, K)``.
        """
        # basis: (R, K, K) → (1, 1, R, K, K)
        basis = self.rbm()  # (R, K, K)
        basis = basis.unsqueeze(0).unsqueeze(0)

        # alpha: (Cout, Cin, R) → (Cout, Cin, R, 1, 1)
        alpha = self.alpha.unsqueeze(-1).unsqueeze(-1)

        # Weighted sum over rings → (Cout, Cin, K, K)
        radial_mask = (alpha * basis).sum(dim=2)  # (Cout, Cin, K, K)

        return self.weight * radial_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply radially-aware convolution.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, C_out, H', W')``.
        """
        if self.use_radial:
            weight = self._compute_effective_weight()
        else:
            weight = self.weight

        return F.conv2d(
            x,
            weight,
            bias=self.bias_param,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.kernel_size}, radial={self.use_radial}"
        )
