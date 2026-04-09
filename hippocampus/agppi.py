"""
AIS-Guided Prior Prototype Initialisation (AGPPI).

The Automatic Identification System (AIS) broadcast contains vessel metadata:
class, gross tonnage, length, beam, draught, and speed.  When AIS data is
available it provides a strong semantic prior on the prototype location in
embedding space.

AGPPI uses AIS features to:
1.  Compute a prior mean μ₀ for each class prototype in embedding space.
2.  Scale the prior precision τ₀ — classes with high-confidence AIS data
    have a tighter (higher precision) prior, preventing over-fitting to noisy
    support samples.

When AIS data is absent (operational at sea, coastal imaging without AIS),
AGPPI falls back to a class-agnostic learned prior with uniform precision.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# AIS feature encoder
# ---------------------------------------------------------------------------

class AISEncoder(nn.Module):
    """
    Encode a raw AIS feature vector into the embedding space.

    Parameters
    ----------
    ais_dim:
        Raw AIS feature dimensionality.  Expected columns (example):
        [vessel_class_onehot..., log_length, log_beam, log_draft, log_speed,
         confidence_score].  Pass 0 if AIS is never available.
    embedding_dim:
        Target embedding dimension.
    hidden_dim:
        MLP hidden width.
    """

    def __init__(
        self,
        ais_dim: int = 24,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.ais_dim = ais_dim
        self.embedding_dim = embedding_dim

        if ais_dim > 0:
            self.encoder = nn.Sequential(
                nn.Linear(ais_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
        else:
            self.encoder = None  # type: ignore[assignment]

    def forward(self, ais_features: torch.Tensor) -> torch.Tensor:
        """
        Encode AIS features.

        Parameters
        ----------
        ais_features:
            Shape ``(B, ais_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, embedding_dim)``, L2-normalised.
        """
        if self.encoder is None:
            return torch.zeros(
                ais_features.size(0), self.embedding_dim,
                device=ais_features.device, dtype=ais_features.dtype,
            )
        out = self.encoder(ais_features)
        return F.normalize(out, p=2, dim=1)

    def extra_repr(self) -> str:
        return f"ais_dim={self.ais_dim}, embed_dim={self.embedding_dim}"


# ---------------------------------------------------------------------------
# AGPPI module
# ---------------------------------------------------------------------------

class AGPPIModule(nn.Module):
    """
    AIS-Guided Prior Prototype Initialisation.

    Produces a prior mean (μ₀) and prior precision scalar (τ₀) for each
    class, optionally conditioned on AIS metadata.

    Parameters
    ----------
    num_classes:
        Number of known classes.  Can be dynamically extended by calling
        ``add_class()``.
    embedding_dim:
        Embedding space dimensionality.
    ais_dim:
        AIS feature dimension; 0 disables AIS conditioning.
    base_precision:
        Default prior precision τ₀ when AIS is absent.
    agppi_scale:
        Multiplicative scale applied to the AIS-derived precision boost.
    use_agppi:
        If ``False``, returns zero mean and constant precision (ablation).
    """

    def __init__(
        self,
        num_classes: int = 16,
        embedding_dim: int = 256,
        ais_dim: int = 24,
        base_precision: float = 1.0,
        agppi_scale: float = 1.0,
        use_agppi: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.base_precision = base_precision
        self.agppi_scale = agppi_scale
        self.use_agppi = use_agppi

        if use_agppi:
            # Learnable class-conditional prior means (C, D)
            self.class_prior_mean = nn.Embedding(num_classes, embedding_dim)
            nn.init.normal_(self.class_prior_mean.weight, std=0.02)

            # Learnable base precision per class (C,) — log-parameterised
            self.log_class_precision = nn.Parameter(
                torch.zeros(num_classes)
            )

            # AIS encoder
            self.ais_encoder = AISEncoder(
                ais_dim=ais_dim,
                embedding_dim=embedding_dim,
            )

            # AIS confidence → precision boost MLP
            if ais_dim > 0:
                self.precision_boost_net = nn.Sequential(
                    nn.Linear(ais_dim, 64),
                    nn.SiLU(),
                    nn.Linear(64, 1),
                    nn.Softplus(),
                )
            else:
                self.precision_boost_net = None  # type: ignore[assignment]
        else:
            self.class_prior_mean = None         # type: ignore[assignment]
            self.log_class_precision = None      # type: ignore[assignment]
            self.ais_encoder = None              # type: ignore[assignment]
            self.precision_boost_net = None      # type: ignore[assignment]

    def get_prior(
        self,
        class_ids: torch.Tensor,
        ais_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve prior mean and precision for a batch of class IDs.

        Parameters
        ----------
        class_ids:
            Shape ``(N,)`` — class index for each query.
        ais_features:
            Shape ``(N, ais_dim)`` optional AIS data aligned with
            *class_ids*.

        Returns
        -------
        prior_mean:
            Shape ``(N, D)`` — L2-normalised prior mean.
        prior_precision:
            Shape ``(N,)`` — positive prior precision scalar per class.
        """
        N = class_ids.size(0)
        device = class_ids.device
        dtype = torch.float32

        if not self.use_agppi or self.class_prior_mean is None:
            return (
                torch.zeros(N, self.embedding_dim, device=device, dtype=dtype),
                torch.full((N,), self.base_precision, device=device, dtype=dtype),
            )

        # Learned class prior mean
        mu0 = self.class_prior_mean(class_ids)  # (N, D)
        mu0 = F.normalize(mu0, p=2, dim=1)

        # AIS mean correction
        if ais_features is not None and self.ais_encoder is not None:
            ais_emb = self.ais_encoder(ais_features.float())  # (N, D)
            # Blend with learned prior (simple average)
            mu0 = F.normalize(mu0 + ais_emb, p=2, dim=1)

        # Base precision from learnable parameter
        precision = torch.exp(self.log_class_precision[class_ids])  # (N,)
        precision = precision * self.base_precision

        # AIS precision boost
        if (
            ais_features is not None
            and self.precision_boost_net is not None
        ):
            boost = self.precision_boost_net(ais_features.float()).squeeze(1)  # (N,)
            precision = precision + self.agppi_scale * boost

        return mu0, precision

    def add_class(self, n: int = 1) -> None:
        """Dynamically extend to support ``n`` additional classes."""
        if not self.use_agppi or self.class_prior_mean is None:
            return
        old_weight = self.class_prior_mean.weight.data
        new_weight = torch.randn(n, self.embedding_dim, device=old_weight.device) * 0.02
        self.class_prior_mean.weight = nn.Parameter(
            torch.cat([old_weight, new_weight], dim=0)
        )
        old_prec = self.log_class_precision.data
        new_prec = torch.zeros(n, device=old_prec.device)
        self.log_class_precision = nn.Parameter(
            torch.cat([old_prec, new_prec], dim=0)
        )
        self.num_classes += n

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, dim={self.embedding_dim}, "
            f"use_agppi={self.use_agppi}"
        )
