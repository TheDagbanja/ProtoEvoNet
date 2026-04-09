"""
Dimension-wise SAR Reliability (DSAR).

Not all dimensions of the embedding equally reflect physically meaningful
SAR features.  DSAR assigns a reliability weight ρ_d ∈ (0, 1] to each
embedding dimension d based on two complementary signals:

1.  **Variance reliability**: dimensions that are consistently activated
    across a support set carry more useful discriminative information.
    Dimensions dominated by noise tend to have high variance *across classes*
    but low inter-class separation.

2.  **Physics alignment**: dimensions whose activations correlate with
    physically meaningful SAR quantities (k, θ, SNR) are considered more
    reliable.

The DSAR weights are produced by a small learned MLP that takes a
concatenation of:
  * Per-dimension variance statistics of the embedding set
  * Per-dimension Pearson correlation with physical descriptors
and returns ``(D,)`` weights in (0, 1].

These weights can then be used to:
  * Modulate prototype computation (weighted mean)
  * Scale the precision matrix in BPC-GP
  * Smooth embeddings before distance computations
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSARModule(nn.Module):
    """
    Learnable Dimension-wise SAR Reliability scorer.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of the embedding space (D).
    physics_dim:
        Dimensionality of the physics descriptor vector fed alongside the
        embedding statistics.  Set to 0 to disable physics alignment.
    hidden_dim:
        Hidden layer size of the reliability MLP.
    use_dsar:
        If ``False``, returns a uniform weight vector (ablation mode).
    eps:
        Numerical stability floor.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        physics_dim: int = 4,    # [k, theta, SNR, speckle_index]
        hidden_dim: int = 128,
        use_dsar: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_dsar = use_dsar
        self.eps = eps

        if use_dsar:
            # Input: dim-wise variance (D) + dim-wise physics-corr (D) optionally
            in_features = embedding_dim + max(physics_dim, 0)
            self.mlp = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, embedding_dim),
                nn.Sigmoid(),   # outputs ρ ∈ (0, 1)
            )
        else:
            self.mlp = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dim_variance(embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute per-dimension variance across a set of embeddings.

        Parameters
        ----------
        embeddings:
            Shape ``(N, D)`` — a batch / support set of embeddings.

        Returns
        -------
        torch.Tensor
            Shape ``(D,)`` — dimension-wise variance.
        """
        return embeddings.var(dim=0).clamp(min=0.0)  # (D,)

    @staticmethod
    def _dim_physics_corr(
        embeddings: torch.Tensor,
        physics: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute mean absolute Pearson correlation between each embedding
        dimension and each physics descriptor, then average over descriptors.

        Parameters
        ----------
        embeddings:
            Shape ``(N, D)``.
        physics:
            Shape ``(N, P)`` — physics descriptor vectors.

        Returns
        -------
        torch.Tensor
            Shape ``(D,)`` — mean |correlation| per embedding dimension.
        """
        N, D = embeddings.shape
        P = physics.shape[1]

        # Centre
        emb = embeddings - embeddings.mean(dim=0, keepdim=True)   # (N, D)
        phy = physics - physics.mean(dim=0, keepdim=True)          # (N, P)

        # std
        emb_std = emb.std(dim=0).clamp(min=eps)   # (D,)
        phy_std = phy.std(dim=0).clamp(min=eps)    # (P,)

        # Pearson: (N, D)^T (N, P) → (D, P), then normalise
        corr = (emb.T @ phy) / (N * emb_std.unsqueeze(1) * phy_std.unsqueeze(0))
        # Mean |corr| over physics dims
        return corr.abs().mean(dim=1)  # (D,)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embeddings: torch.Tensor,
        physics_descriptors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute DSAR reliability weights for a set of embeddings.

        Parameters
        ----------
        embeddings:
            Shape ``(N, D)`` — support-set or batch embeddings.
        physics_descriptors:
            Shape ``(N, P)`` — optional physics feature vectors aligned with
            *embeddings*.

        Returns
        -------
        torch.Tensor
            Shape ``(D,)`` — per-dimension reliability weights in (0, 1].
        """
        if not self.use_dsar or self.mlp is None:
            return torch.ones(
                self.embedding_dim,
                device=embeddings.device,
                dtype=embeddings.dtype,
            )

        # Dim-wise variance feature
        var_feat = self._dim_variance(embeddings)  # (D,)

        if physics_descriptors is not None and physics_descriptors.numel() > 0:
            corr_feat = self._dim_physics_corr(
                embeddings, physics_descriptors, eps=self.eps
            )  # (D,)
            feat = torch.cat([var_feat, corr_feat], dim=0)  # but we need P dims
            # If physics_dim doesn't match corr_feat, just use what we have
            input_feat = torch.cat([var_feat, corr_feat], dim=0)
        else:
            input_feat = var_feat  # (D,)

        # Pad or truncate to match MLP input size
        expected_in = self.mlp[0].in_features
        if input_feat.shape[0] < expected_in:
            pad = torch.zeros(
                expected_in - input_feat.shape[0],
                device=input_feat.device,
                dtype=input_feat.dtype,
            )
            input_feat = torch.cat([input_feat, pad], dim=0)
        else:
            input_feat = input_feat[:expected_in]

        # MLP expects (1, in_features); squeeze back to (D,)
        weights = self.mlp(input_feat.unsqueeze(0)).squeeze(0)  # (D,)
        return weights

    def weight_embeddings(
        self,
        embeddings: torch.Tensor,
        physics_descriptors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply DSAR weights to a set of embeddings element-wise.

        Parameters
        ----------
        embeddings:
            Shape ``(N, D)`` or ``(D,)``.
        physics_descriptors:
            Shape ``(N, P)``.

        Returns
        -------
        torch.Tensor
            DSAR-weighted embeddings, same shape as input.
        """
        weights = self.forward(embeddings if embeddings.dim() == 2 else embeddings.unsqueeze(0),
                               physics_descriptors)  # (D,)
        return embeddings * weights

    def extra_repr(self) -> str:
        return f"dim={self.embedding_dim}, use_dsar={self.use_dsar}"
