"""
Adaptive Precision Modulation (APM).

APM is an MLP that takes a concatenation of quality and physics signals and
outputs a per-sample *precision weight* α_i ∈ ℝ₊.  These weights feed
directly into the BPC-GP update:

    posterior_precision = prior_precision + Σ_i α_i
    posterior_mean      = (prior_prec · prior_mean + Σ_i α_i · z_i)
                          / posterior_precision

High α_i means sample i is trusted more and pulls the prototype closer to z_i.

The input to APM combines:
  * SQAA quality score q_i (scalar)
  * Global SNR for the sample (scalar)
  * Gamma shape k_i (scalar) — higher k → more looks → more reliable
  * DSAR reliability vector ρ projected to a scalar summary
  * (Optionally) embedding norm before L2 normalisation
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class APMModule(nn.Module):
    """
    Adaptive Precision Modulation MLP.

    Parameters
    ----------
    embedding_dim:
        Embedding dimensionality D (used only if we accept the embedding
        itself as an additional feature).
    input_dim:
        Dimensionality of the feature vector passed to the MLP.  Default
        is 5 (q, SNR, k, θ, ρ_mean).
    hidden_dim:
        Width of each hidden layer.
    num_layers:
        Depth of the MLP (minimum 2).
    use_apm:
        If ``False``, returns uniform precision = 1.0 (ablation).
    min_precision:
        Lower bound on output precision (prevents degenerate zero weights).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        input_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_apm: bool = True,
        min_precision: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_apm = use_apm
        self.min_precision = min_precision

        if use_apm:
            assert num_layers >= 2, "APM must have at least 2 layers"
            layers: list[nn.Module] = []
            in_features = input_dim
            for _ in range(num_layers - 1):
                layers += [nn.Linear(in_features, hidden_dim), nn.SiLU()]
                in_features = hidden_dim
            layers += [nn.Linear(in_features, 1), nn.Softplus()]
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = None  # type: ignore[assignment]

    def forward(
        self,
        quality_scores: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
        dsar_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-sample precision weights.

        Parameters
        ----------
        quality_scores:
            Shape ``(K,)`` — SQAA quality scores in [0, 1].
        physics:
            Shape ``(K, P)`` — optional physics descriptors.
        dsar_weights:
            Shape ``(D,)`` — DSAR reliability weights.  Summarised to a
            scalar mean before concatenation.

        Returns
        -------
        torch.Tensor
            Shape ``(K,)`` — precision weights α_i ≥ min_precision.
        """
        K = quality_scores.size(0)
        device = quality_scores.device
        dtype = quality_scores.dtype

        if not self.use_apm or self.mlp is None:
            return torch.ones(K, device=device, dtype=dtype)

        # Build feature matrix (K, input_features)
        feat_parts = [quality_scores.unsqueeze(1)]  # (K, 1)

        if physics is not None:
            feat_parts.append(physics.float())  # (K, P)

        if dsar_weights is not None:
            # Summarise DSAR to a scalar: mean reliability
            dsar_mean = dsar_weights.mean().expand(K, 1)  # (K, 1)
            feat_parts.append(dsar_mean)

        feat = torch.cat(feat_parts, dim=1)  # (K, F)

        # Pad or truncate to match MLP input size
        expected_in = self.mlp[0].in_features
        if feat.size(1) < expected_in:
            pad = torch.zeros(K, expected_in - feat.size(1), device=device, dtype=dtype)
            feat = torch.cat([feat, pad], dim=1)
        elif feat.size(1) > expected_in:
            feat = feat[:, :expected_in]

        precision = self.mlp(feat).squeeze(1)  # (K,)
        precision = precision + self.min_precision
        return precision

    def extra_repr(self) -> str:
        return f"use_apm={self.use_apm}, min_precision={self.min_precision}"
