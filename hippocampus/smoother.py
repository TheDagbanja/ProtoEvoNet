"""
DSAR-Weighted Prototype Smoother.

After BPC-GP produces a prototype μ_post, the smoother applies a final
dimension-wise re-weighting using the DSAR reliability scores ρ:

    μ_smooth = ρ ⊙ μ_post + (1 - ρ) ⊙ μ_ref

where μ_ref is a reference prototype (e.g., the class's existing prototype in
memory, or a running average).  Dimensions with low reliability (ρ_d ≈ 0)
are pulled towards the reference; dimensions with high reliability
(ρ_d ≈ 1) retain the freshly computed BPC-GP estimate.

This prevents noisy embedding dimensions from corrupting the stored prototype.

If no reference prototype is available (new class enrolment), the smoother
simply applies ρ as a soft-threshold on the new prototype.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSARWeightedSmoother(nn.Module):
    """
    Smooth a new prototype estimate with DSAR reliability weights.

    Parameters
    ----------
    embedding_dim:
        Embedding space dimensionality.
    use_smoother:
        If ``False``, returns the input prototype unchanged (ablation).
    blend_bias:
        Additive bias to all ρ values before blending.  A positive bias
        makes the smoother more permissive (lean towards the new estimate).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        use_smoother: bool = True,
        blend_bias: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_smoother = use_smoother
        self.blend_bias = blend_bias

        if use_smoother:
            # Learnable global reliability floor (per-dim learnable minimum ρ)
            self.reliability_floor = nn.Parameter(
                torch.zeros(embedding_dim)
            )

    def forward(
        self,
        new_prototype: torch.Tensor,
        dsar_weights: torch.Tensor,
        reference_prototype: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply DSAR-weighted smoothing.

        Parameters
        ----------
        new_prototype:
            Shape ``(D,)`` — freshly computed BPC-GP prototype.
        dsar_weights:
            Shape ``(D,)`` — DSAR reliability in [0, 1].
        reference_prototype:
            Shape ``(D,)`` — existing prototype to blend towards.  If
            ``None``, a zero vector is used (first enrolment).

        Returns
        -------
        torch.Tensor
            Shape ``(D,)`` — smoothed prototype (L2-normalised).
        """
        if not self.use_smoother:
            return F.normalize(new_prototype, p=2, dim=0)

        # Reliability blend coefficient per dimension
        floor = torch.sigmoid(self.reliability_floor)  # (D,) in (0,1)
        rho = (dsar_weights.clamp(0.0, 1.0) + self.blend_bias).clamp(0.0, 1.0)
        rho = torch.max(rho, floor)  # never go below learned floor

        if reference_prototype is None:
            ref = torch.zeros_like(new_prototype)
        else:
            ref = reference_prototype

        smoothed = rho * new_prototype + (1.0 - rho) * ref
        return F.normalize(smoothed, p=2, dim=0)

    def extra_repr(self) -> str:
        return (
            f"dim={self.embedding_dim}, use_smoother={self.use_smoother}, "
            f"blend_bias={self.blend_bias}"
        )
