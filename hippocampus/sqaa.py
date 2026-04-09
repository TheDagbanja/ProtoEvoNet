"""
Support Quality Assessment and Aggregation (SQAA).

In few-shot learning the support set may contain low-quality SAR images
(heavy speckle, low-SNR, out-of-focus, obscured targets).  Naively averaging
all support embeddings to form a prototype degrades performance.

SQAA assigns a scalar quality score q ∈ [0, 1] to each support sample by
combining:
  * Embedding confidence: the L2-norm of the pre-normalised embedding
    (higher norm → more distinctive features).
  * Physics quality: the SNR estimate and the Gamma shape parameter k
    (higher k → more looks → cleaner image).
  * A small learned MLP that maps these signals to a quality score.

The scores are used as soft weights when computing the prototype mean.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SQAAModule(nn.Module):
    """
    Support Quality Assessment and Aggregation.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of the L2-normalised support embeddings.
    physics_dim:
        Number of physics descriptor features per sample (k, θ, SNR, SI).
    hidden_dim:
        MLP hidden layer width.
    use_sqaa:
        If ``False``, returns uniform weights = 1/K (ablation).
    """

    # Physics descriptor indices (expected order from caller)
    IDX_K = 0      # Gamma shape
    IDX_THETA = 1  # Gamma scale
    IDX_SNR = 2    # Signal-to-noise ratio
    IDX_SI = 3     # Speckle index

    def __init__(
        self,
        embedding_dim: int = 256,
        physics_dim: int = 4,
        hidden_dim: int = 128,
        use_sqaa: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_sqaa = use_sqaa

        if use_sqaa:
            # Input: [embedding_norm (1), physics_dim] → quality score
            in_dim = 1 + physics_dim
            self.scorer = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.scorer = None  # type: ignore[assignment]

    def forward(
        self,
        embeddings: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
        pre_norm_norms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-sample quality scores for a support set.

        Parameters
        ----------
        embeddings:
            L2-normalised support embeddings, shape ``(K, D)``.
        physics:
            Physics descriptor matrix, shape ``(K, P)``.  Expected columns:
            [k, θ, SNR, speckle_index].
        pre_norm_norms:
            Pre-normalisation L2 norms, shape ``(K,)``.  If ``None``,
            computed from *embeddings* (always 1 for L2-normalised inputs;
            pass the pre-norm norms for a more informative signal).

        Returns
        -------
        torch.Tensor
            Quality weights, shape ``(K,)``, summing to 1 (softmax).
        """
        K = embeddings.size(0)

        if not self.use_sqaa or self.scorer is None:
            return torch.ones(K, device=embeddings.device, dtype=embeddings.dtype) / K

        # Embedding norm feature
        if pre_norm_norms is not None:
            norms = pre_norm_norms.unsqueeze(1)  # (K, 1)
        else:
            norms = torch.ones(K, 1, device=embeddings.device, dtype=embeddings.dtype)

        # Physics features
        if physics is not None:
            feat = torch.cat([norms, physics], dim=1)  # (K, 1+P)
        else:
            feat = norms  # (K, 1)

        # Pad to expected input size
        expected_in = self.scorer[0].in_features
        if feat.size(1) < expected_in:
            pad = torch.zeros(
                K, expected_in - feat.size(1),
                device=feat.device, dtype=feat.dtype
            )
            feat = torch.cat([feat, pad], dim=1)
        elif feat.size(1) > expected_in:
            feat = feat[:, :expected_in]

        raw_scores = self.scorer(feat).squeeze(1)  # (K,)

        # Softmax to obtain weights that sum to 1
        weights = F.softmax(raw_scores, dim=0)  # (K,)
        return weights

    def weighted_mean(
        self,
        embeddings: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
        pre_norm_norms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the quality-weighted mean embedding (prototype candidate).

        Parameters
        ----------
        embeddings:
            Shape ``(K, D)``.
        physics, pre_norm_norms:
            See ``forward()``.

        Returns
        -------
        torch.Tensor
            Shape ``(D,)`` — weighted mean embedding (not L2-normalised here;
            normalisation is applied downstream in BPC-GP or the prototype
            store).
        """
        weights = self.forward(embeddings, physics, pre_norm_norms)  # (K,)
        prototype = (weights.unsqueeze(1) * embeddings).sum(dim=0)   # (D,)
        return prototype

    def extra_repr(self) -> str:
        return f"dim={self.embedding_dim}, use_sqaa={self.use_sqaa}"
