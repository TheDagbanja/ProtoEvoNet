"""
Inference Engine for ProtoEvoNet.

The engine implements four operational modes:

Mode 1 — Recognition
    Query → backbone → embedding → nearest-prototype lookup → class label + confidence.

Mode 2 — Novelty Detection
    Query → backbone → embedding → Platt-scaled novelty probability.
    If P(novel) > threshold → "novel"; else proceed as Mode 1.

Mode 3a — Prototype Enrolment
    Support set (new class) → backbone → HBS → OWMS.enrol().
    PIM and TPE are invoked after enrolment.

Mode 3b — Prototype Refinement
    Additional support samples for an existing class → backbone → HBS → TPE.update().
    PIM is re-scanned after the update.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from backbone.srab import SRABBackbone
from physics.gamma_stats import GammaStatisticsModule
from physics.dsar import DSARModule
from physics.snr import SNREstimationModule
from hippocampus import HippocampalBindingSystem
from memory.owms import OnlineWorkingMemoryStore
from memory.pim import PrototypeInterferenceMonitor
from memory.tpe import TemporalPrototypeEvolution
from .novelty import NoveltyDetector, PlattCalibrator
from utils.config import ProtoEvoNetConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference mode enum
# ---------------------------------------------------------------------------

class InferenceMode(Enum):
    RECOGNITION = auto()
    NOVELTY = auto()
    ENROLMENT = auto()
    REFINEMENT = auto()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RecognitionResult:
    """Output from Mode 1 (recognition)."""
    predicted_class_id: int
    predicted_label: str
    confidence: float               # 1 - (normalised distance)
    top_k_ids: List[int]
    top_k_labels: List[str]
    top_k_distances: List[float]
    embedding: torch.Tensor         # (D,)


@dataclass
class NoveltyResult:
    """Output from Mode 2 (novelty detection)."""
    is_novel: bool
    novelty_probability: float
    min_distance: float
    recognition: Optional[RecognitionResult] = None  # if not novel


@dataclass
class EnrolmentResult:
    """Output from Mode 3a (enrolment)."""
    class_id: int
    label: str
    uncertainty: float
    prototype: torch.Tensor         # (D,)
    interference_detected: bool


@dataclass
class RefinementResult:
    """Output from Mode 3b (refinement)."""
    class_id: int
    old_count: int
    new_count: int
    prototype: torch.Tensor         # (D,)
    uncertainty: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ProtoEvoNetInferenceEngine:
    """
    Unified inference engine exposing all four operational modes.

    Parameters
    ----------
    backbone:
        Pre-trained ``SRABBackbone`` (frozen or fine-tunable).
    hbs:
        ``HippocampalBindingSystem`` for prototype computation.
    owms:
        ``OnlineWorkingMemoryStore`` holding current prototypes.
    pim:
        ``PrototypeInterferenceMonitor``.
    tpe:
        ``TemporalPrototypeEvolution``.
    novelty_detector:
        ``NoveltyDetector`` with fitted Platt calibrator.
    gamma_module:
        For computing per-image physics descriptors used by HBS.
    dsar_module:
        DSAR module for reliability weights.
    snr_module:
        SNR estimation module.
    cfg:
        Full system config.
    device:
        Inference device.
    """

    def __init__(
        self,
        backbone: SRABBackbone,
        hbs: HippocampalBindingSystem,
        owms: OnlineWorkingMemoryStore,
        pim: PrototypeInterferenceMonitor,
        tpe: TemporalPrototypeEvolution,
        novelty_detector: NoveltyDetector,
        gamma_module: GammaStatisticsModule,
        dsar_module: DSARModule,
        snr_module: SNREstimationModule,
        cfg: ProtoEvoNetConfig,
    ) -> None:
        self.backbone = backbone
        self.hbs = hbs
        self.owms = owms
        self.pim = pim
        self.tpe = tpe
        self.novelty_detector = novelty_detector
        self.gamma = gamma_module
        self.dsar = dsar_module
        self.snr = snr_module
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # Move learnable modules to device
        self.backbone.to(self.device)
        self.hbs.to(self.device)
        self.gamma.to(self.device)
        self.dsar.to(self.device)
        self.snr.to(self.device)
        self.novelty_detector.calibrator.to(self.device)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of SAR images.

        Parameters
        ----------
        images:
            Shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, D)`` L2-normalised embeddings.
        """
        images = images.to(self.device)
        self.backbone.eval()
        return self.backbone(images)

    @torch.no_grad()
    def _physics_descriptors(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute per-image physics descriptor vectors [k, θ, SNR, SI].

        Returns
        -------
        torch.Tensor
            Shape ``(B, 4)``.
        """
        images = images.to(self.device)
        k, theta = self.gamma.global_stats(images)          # (B,), (B,)
        snr = self.snr.global_snr(images)                   # (B,)

        from physics.snr import speckle_index
        si_map = speckle_index(images)
        si = si_map.mean(dim=(-3, -2, -1))                  # (B,)

        return torch.stack([k, theta, snr, si], dim=1)      # (B, 4)

    def _distance_to_confidence(self, dist: float) -> float:
        """Map Euclidean distance in [0, 2] to confidence in [0, 1]."""
        return float(max(0.0, 1.0 - dist / 2.0))

    # -----------------------------------------------------------------------
    # Mode 1: Recognition
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def recognise(
        self,
        image: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> RecognitionResult:
        """
        Recognise the class of a single SAR image chip.

        Parameters
        ----------
        image:
            Shape ``(1, H, W)`` or ``(1, 1, H, W)``.
        top_k:
            Number of top candidates to return.  Defaults to
            ``cfg.inference.top_k``.

        Returns
        -------
        RecognitionResult
        """
        top_k = top_k or self.cfg.inference.top_k
        if image.dim() == 3:
            image = image.unsqueeze(0)

        emb = self._embed(image).squeeze(0)  # (D,)

        dists, ids, protos = self.owms.nearest(emb, k=top_k)
        # dists: (1, k), ids: list[list[int]], protos: (1, k, D)
        dists = dists.squeeze(0)     # (k,)
        ids_flat = ids[0]            # list[int]

        best_id = ids_flat[0]
        best_entry = self.owms.get(best_id)
        best_label = best_entry.label if best_entry else str(best_id)
        confidence = self._distance_to_confidence(dists[0].item())

        top_labels = [
            (self.owms.get(i).label if self.owms.get(i) else str(i))
            for i in ids_flat
        ]

        return RecognitionResult(
            predicted_class_id=best_id,
            predicted_label=best_label,
            confidence=confidence,
            top_k_ids=ids_flat,
            top_k_labels=top_labels,
            top_k_distances=dists.tolist(),
            embedding=emb.cpu(),
        )

    # -----------------------------------------------------------------------
    # Mode 2: Novelty detection
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def detect_novelty(self, image: torch.Tensor) -> NoveltyResult:
        """
        Decide whether a query image belongs to a known class or is novel.

        If not novel, also returns the recognition result.

        Parameters
        ----------
        image:
            Shape ``(1, H, W)`` or ``(1, 1, H, W)``.

        Returns
        -------
        NoveltyResult
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        emb = self._embed(image).squeeze(0)  # (D,)
        decisions, probs = self.novelty_detector.is_novel(emb, self.owms)

        is_novel = decisions[0].item()
        novel_prob = probs[0].item()
        min_dist = self.novelty_detector.score(emb, self.owms)[0].item()

        recognition = None
        if not is_novel:
            recognition = self.recognise(image)

        return NoveltyResult(
            is_novel=bool(is_novel),
            novelty_probability=novel_prob,
            min_distance=min_dist,
            recognition=recognition,
        )

    # -----------------------------------------------------------------------
    # Mode 3a: Prototype enrolment
    # -----------------------------------------------------------------------

    def enrol(
        self,
        support_images: torch.Tensor,
        label: str,
        class_id: Optional[int] = None,
        ais_features: Optional[torch.Tensor] = None,
    ) -> EnrolmentResult:
        """
        Enrol a new class from a support set of images.

        Parameters
        ----------
        support_images:
            Shape ``(K, 1, H, W)``.
        label:
            Human-readable class name.
        class_id:
            Explicit class ID; auto-assigned if ``None``.
        ais_features:
            Shape ``(P_ais,)`` — AIS metadata for this class.

        Returns
        -------
        EnrolmentResult
        """
        support_images = support_images.to(self.device)
        self.backbone.eval()

        with torch.no_grad():
            embeddings = self._embed(support_images)         # (K, D)
            physics = self._physics_descriptors(support_images)  # (K, 4)

        # Compute DSAR weights from support embeddings
        dsar_weights = self.dsar(embeddings, physics)         # (D,)

        # Run HBS to produce calibrated prototype
        prototype, uncertainty = self.hbs.bind(
            support_embeddings=embeddings,
            class_id=class_id,
            physics=physics,
            ais_features=ais_features,
            dsar_weights=dsar_weights,
        )

        # Enrol in OWMS
        assigned_id = self.owms.enrol(
            prototype=prototype,
            label=label,
            uncertainty=float(uncertainty.detach()),
            count=len(support_images),
            class_id=class_id,
        )

        # Post-enrolment checks
        self.tpe.update(self.owms, assigned_id, prototype, float(uncertainty.detach()))
        interference_pairs = self.pim.scan(self.owms, apply_repulsion=True)

        logger.info(
            "Enrolled class %d ('%s') with %d support images, uncertainty=%.4f",
            assigned_id, label, len(support_images), float(uncertainty.detach()),
        )

        return EnrolmentResult(
            class_id=assigned_id,
            label=label,
            uncertainty=float(uncertainty.detach()),
            prototype=prototype.cpu(),
            interference_detected=len(interference_pairs) > 0,
        )

    # -----------------------------------------------------------------------
    # Mode 3b: Prototype refinement
    # -----------------------------------------------------------------------

    def refine(
        self,
        class_id: int,
        new_support_images: torch.Tensor,
        ais_features: Optional[torch.Tensor] = None,
    ) -> RefinementResult:
        """
        Refine an existing prototype with additional support images.

        Parameters
        ----------
        class_id:
            ID of the class to refine.
        new_support_images:
            Shape ``(K, 1, H, W)``.
        ais_features:
            Shape ``(P_ais,)`` optional AIS data.

        Returns
        -------
        RefinementResult
        """
        if not self.owms.contains(class_id):
            raise KeyError(
                f"Class {class_id} not in OWMS.  Use ``enrol()`` first."
            )

        entry = self.owms.get(class_id)
        old_count = entry.count if entry else 0
        reference = entry.prototype if entry else None

        new_support_images = new_support_images.to(self.device)
        self.backbone.eval()

        with torch.no_grad():
            embeddings = self._embed(new_support_images)         # (K, D)
            physics = self._physics_descriptors(new_support_images)

        dsar_weights = self.dsar(embeddings, physics)

        prototype, uncertainty = self.hbs.bind(
            support_embeddings=embeddings,
            class_id=class_id,
            physics=physics,
            ais_features=ais_features,
            dsar_weights=dsar_weights,
            reference_prototype=reference,
        )

        # TPE update (EMA)
        self.tpe.update(self.owms, class_id, prototype, float(uncertainty))
        self.pim.scan(self.owms, apply_repulsion=True)

        updated_entry = self.owms.get(class_id)
        new_count = updated_entry.count if updated_entry else old_count + 1

        logger.info(
            "Refined class %d: count %d → %d, uncertainty=%.4f",
            class_id, old_count, new_count, float(uncertainty),
        )

        return RefinementResult(
            class_id=class_id,
            old_count=old_count,
            new_count=new_count,
            prototype=(updated_entry.prototype.cpu() if updated_entry else prototype.cpu()),
            uncertainty=float(uncertainty),
        )

    # -----------------------------------------------------------------------
    # Convenience: run in a specified mode
    # -----------------------------------------------------------------------

    def run(
        self,
        mode: InferenceMode,
        image: Optional[torch.Tensor] = None,
        support_images: Optional[torch.Tensor] = None,
        label: str = "",
        class_id: Optional[int] = None,
        ais_features: Optional[torch.Tensor] = None,
    ):
        """
        Dispatch to the correct mode.

        Parameters
        ----------
        mode:
            One of the ``InferenceMode`` enum values.
        image:
            Query image for Modes 1 & 2.
        support_images:
            Support images for Modes 3a & 3b.
        label:
            Class label for Mode 3a.
        class_id:
            Class ID override (Modes 3a & 3b).
        ais_features:
            AIS metadata (Modes 3a & 3b).
        """
        if mode == InferenceMode.RECOGNITION:
            assert image is not None
            return self.recognise(image)
        elif mode == InferenceMode.NOVELTY:
            assert image is not None
            return self.detect_novelty(image)
        elif mode == InferenceMode.ENROLMENT:
            assert support_images is not None
            return self.enrol(support_images, label=label, class_id=class_id,
                              ais_features=ais_features)
        elif mode == InferenceMode.REFINEMENT:
            assert support_images is not None and class_id is not None
            return self.refine(class_id, support_images, ais_features=ais_features)
        else:
            raise ValueError(f"Unknown inference mode: {mode}")
