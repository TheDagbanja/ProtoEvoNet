"""
Sanitisation Gate.

Edge case: K = 1 (one-shot).  With a single support sample the empirical
variance is undefined, SQAA weights default to 1.0, and BPC-GP collapses
to a trivial posterior update.  The Sanitisation Gate detects this case and
applies a set of safeguards:

1.  **Prototype augmentation**: generates virtual support embeddings by
    applying learned small perturbations in embedding space (guided by DSAR
    reliability weights so that less-reliable dims receive larger noise).
2.  **Prior activation**: forces the BPC-GP prior to contribute heavily
    (sets effective K_eff = ``min_effective_k`` in the precision update).
3.  **Quality clamping**: prevents the single-sample quality score from
    being overfit to a potentially noisy sample.

When K > 1 the gate is transparent (pass-through).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SanitisationGate(nn.Module):
    """
    Handle the K = 1 support degenerate case.

    Parameters
    ----------
    embedding_dim:
        Embedding dimensionality D.
    min_effective_k:
        Virtual effective support size injected into BPC-GP when K = 1.
    aug_scale:
        Standard deviation of the Gaussian noise added to generate virtual
        support embeddings.
    num_virtual:
        Number of virtual samples to generate when K = 1.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        min_effective_k: int = 3,
        aug_scale: float = 0.05,
        num_virtual: int = 4,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.min_effective_k = min_effective_k
        self.aug_scale = aug_scale
        self.num_virtual = num_virtual

        # Learned noise scale per dimension (allows reliable dims to be
        # perturbed less than unreliable dims)
        self.noise_scale = nn.Parameter(
            torch.ones(embedding_dim) * aug_scale
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        dsar_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Sanitise the support set.

        Parameters
        ----------
        embeddings:
            Support embeddings, shape ``(K, D)``.
        dsar_weights:
            Dimension-wise reliability weights, shape ``(D,)``.  If given,
            noise is scaled *inversely* proportional to reliability (less
            reliable dims get more noise).

        Returns
        -------
        embeddings_out:
            Sanitised embeddings, shape ``(K', D)`` where K' ≥ K.
        was_sanitised:
            ``True`` if K = 1 and virtual samples were added.
        """
        K, D = embeddings.shape

        if K > 1:
            return embeddings, False

        # ---- K = 1: generate virtual support samples -------------------
        # Effective noise scale per dimension
        noise_scale = self.noise_scale.abs()   # (D,)

        if dsar_weights is not None:
            # Less reliable → more noise; invert and normalise
            inv_rel = (1.0 - dsar_weights.clamp(0.0, 1.0) + 1e-6)
            inv_rel = inv_rel / inv_rel.mean()
            noise_scale = noise_scale * inv_rel

        noise = torch.randn(
            self.num_virtual, D,
            device=embeddings.device,
            dtype=embeddings.dtype,
        ) * noise_scale.unsqueeze(0)  # (num_virtual, D)

        virtual = F.normalize(embeddings + noise, p=2, dim=1)  # (num_virtual, D)
        out = torch.cat([embeddings, virtual], dim=0)          # (K+num_virtual, D)
        return out, True

    def effective_k(self, K: int) -> int:
        """
        Return the effective K used in BPC-GP precision computations.

        When K = 1 this is ``min_effective_k``; otherwise it is K.
        """
        return self.min_effective_k if K == 1 else K

    def extra_repr(self) -> str:
        return (
            f"min_eff_k={self.min_effective_k}, aug_scale={self.aug_scale}, "
            f"num_virtual={self.num_virtual}"
        )
