"""
ProtoEvoNet Hippocampal Binding System.

This package provides all components of the Hippocampal Binding System (HBS):
  * SQAA — Support Quality Assessment and Aggregation
  * SanitisationGate — K=1 safeguard
  * AGPPIModule — AIS-Guided Prior Prototype Initialisation
  * APMModule — Adaptive Precision Modulation
  * BPCGPModule — Bayesian Prototype Calibration (closed-form GP update)
  * DSARWeightedSmoother — final reliability-weighted smoothing
  * HippocampalBindingSystem — orchestrator that wires all components
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sqaa import SQAAModule
from .sanitisation_gate import SanitisationGate
from .agppi import AGPPIModule
from .apm import APMModule
from .bpc_gp import BPCGPModule
from .smoother import DSARWeightedSmoother
from utils.config import HippocampusConfig


class HippocampalBindingSystem(nn.Module):
    """
    Orchestrates all HBS components into a single forward pass.

    Given a support set for one class (and optional physics/AIS data),
    produces a calibrated, smoothed prototype vector.

    Parameters
    ----------
    cfg:
        ``HippocampusConfig`` driving all architectural choices.
    num_classes:
        Number of known classes (for AGPPI embedding table).
    ais_dim:
        AIS feature dimension (0 = disabled).
    """

    def __init__(
        self,
        cfg: HippocampusConfig,
        num_classes: int = 16,
        ais_dim: int = 0,
    ) -> None:
        super().__init__()
        D = cfg.embedding_dim

        self.sqaa = SQAAModule(
            embedding_dim=D,
            hidden_dim=cfg.sqaa_hidden,
            use_sqaa=cfg.use_sqaa,
        )
        self.gate = SanitisationGate(embedding_dim=D)
        self.agppi = AGPPIModule(
            num_classes=num_classes,
            embedding_dim=D,
            ais_dim=ais_dim,
            base_precision=1.0 / cfg.bpc_prior_var,
            agppi_scale=cfg.agppi_scale,
            use_agppi=cfg.use_agppi,
        )
        self.apm = APMModule(
            embedding_dim=D,
            hidden_dim=cfg.apm_hidden,
            num_layers=cfg.apm_layers,
            use_apm=cfg.use_apm,
        )
        self.bpc_gp = BPCGPModule(
            embedding_dim=D,
            prior_var=cfg.bpc_prior_var,
            noise_var=cfg.bpc_noise_var,
            use_bpc_gp=cfg.use_bpc_gp,
        )
        self.smoother = DSARWeightedSmoother(
            embedding_dim=D,
            use_smoother=cfg.use_dsar_smoother,
        )

    def bind(
        self,
        support_embeddings: torch.Tensor,
        class_id: Optional[int] = None,
        physics: Optional[torch.Tensor] = None,
        ais_features: Optional[torch.Tensor] = None,
        dsar_weights: Optional[torch.Tensor] = None,
        reference_prototype: Optional[torch.Tensor] = None,
        pre_norm_norms: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full HBS forward pass: sanitise → quality-score → prior → APM → BPC-GP → smooth.

        Parameters
        ----------
        support_embeddings:
            Shape ``(K, D)`` — L2-normalised support embeddings for one class.
        class_id:
            Integer class index (for AGPPI lookup).  If ``None``, AGPPI uses
            a zero prior mean.
        physics:
            Shape ``(K, P)`` — per-sample physics descriptors.
        ais_features:
            Shape ``(P_ais,)`` or ``(1, P_ais)`` — class-level AIS features.
        dsar_weights:
            Shape ``(D,)`` — dimension-wise SAR reliability weights.
        reference_prototype:
            Shape ``(D,)`` — existing prototype for smoother blending.
        pre_norm_norms:
            Shape ``(K,)`` — pre-normalisation embedding norms for SQAA.

        Returns
        -------
        prototype:
            Shape ``(D,)`` — calibrated L2-normalised prototype.
        uncertainty:
            Scalar — posterior variance from BPC-GP.
        """
        # 1. Sanitise (K=1 guard)
        embeddings, was_sanitised = self.gate(support_embeddings, dsar_weights)
        K = embeddings.size(0)

        # 2. SQAA quality scores
        quality_scores = self.sqaa(embeddings, physics, pre_norm_norms)  # (K,)

        # 3. AGPPI prior
        if class_id is not None:
            class_id_t = torch.tensor(
                [class_id], device=embeddings.device, dtype=torch.long
            )
            ais = ais_features.unsqueeze(0) if (
                ais_features is not None and ais_features.dim() == 1
            ) else ais_features
            prior_mean, prior_prec = self.agppi.get_prior(class_id_t, ais)
            prior_mean = prior_mean.squeeze(0)   # (D,)
            prior_prec = prior_prec.squeeze(0)   # scalar
        else:
            prior_mean = None
            prior_prec = None

        # 4. APM precision weights
        alpha = self.apm(quality_scores, physics, dsar_weights)  # (K,)

        # 5. BPC-GP posterior prototype
        prototype, uncertainty = self.bpc_gp.calibrate_class(
            embeddings, alpha, prior_mean, prior_prec, normalise_output=False
        )  # (D,), scalar

        # 6. DSAR smoother
        if dsar_weights is None:
            dsar_weights = torch.ones(
                self.bpc_gp.embedding_dim,
                device=prototype.device,
                dtype=prototype.dtype,
            )
        prototype = self.smoother(prototype, dsar_weights, reference_prototype)
        # smoother returns L2-normalised result

        return prototype, uncertainty

    def forward(
        self,
        support_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for ``bind()`` to make the module callable as a standard nn.Module."""
        return self.bind(support_embeddings, **kwargs)


__all__ = [
    "SQAAModule",
    "SanitisationGate",
    "AGPPIModule",
    "APMModule",
    "BPCGPModule",
    "DSARWeightedSmoother",
    "HippocampalBindingSystem",
]
