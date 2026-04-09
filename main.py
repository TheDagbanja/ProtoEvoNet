"""
ProtoEvoNet — Main Entry Point.

This script demonstrates the full end-to-end pipeline:

  1. Build all system components from config.
  2. Phase 1: HRSID contrastive pre-training.
  3. Phase 2: FUSAR-Ship episodic meta-learning.
  4. Inference demo: enrol all FUSAR classes, then run recognition and
     novelty detection on test images.

Usage
-----
Defaults (uses bundled dataset paths from data/):

    python main.py

Custom config file:

    python main.py --config my_config.json

Ablation study (disable GCN and DSAR):

    python main.py --ablate use_gcn=False use_dsar=False

Train Phase 2 only (backbone already pre-trained):

    python main.py --phase2-only --backbone-ckpt checkpoints/phase1/best.pt

"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional, List

from numpy import indices

import torch
import torch.nn.functional as F

# ---- ProtoEvoNet imports -------------------------------------------------
from utils.config import (
    ProtoEvoNetConfig,
    BackboneConfig,
    PhysicsConfig,
    HippocampusConfig,
    MemoryConfig,
    InferenceConfig,
    TrainingConfig,
)
from utils.logging_utils import setup_logging
from utils.checkpoint import load_checkpoint, find_best_checkpoint

from backbone.srab import SRABBackbone

from physics.gamma_stats import GammaStatisticsModule
from physics.dsar import DSARModule
from physics.snr import SNREstimationModule
from physics.fisher_rao import FisherRaoDistance

from hippocampus import HippocampalBindingSystem

from memory.owms import OnlineWorkingMemoryStore
from memory.pim import PrototypeInterferenceMonitor
from memory.tpe import TemporalPrototypeEvolution

from inference.novelty import (
    NoveltyDetector, PlattCalibrator,
    MahalanobisScorer, PCAMahalanobisScorer, KNNScorer,
    PhysicsNoveltyScorer,
)
from inference.engine import (
    ProtoEvoNetInferenceEngine,
    InferenceMode,
    RecognitionResult,
    NoveltyResult,
)

from training.datasets import HRSIDDataset, FUSARShipDataset, MSTARDataset

# ---------------------------------------------------------------------------
# Stable 10-class FUSAR subset.
# DiveVessel (3 images), HighSpeedCraft (7+8), PortTender (6), SAR (5), and
# WingInGrnd (8+8) all have fewer than 10 support images and collapse to ≈12%
# accuracy, which pollutes macro-F1.  We restrict evaluation to the 10 classes
# that have sufficient data for fair few-shot benchmarking.
# ---------------------------------------------------------------------------
FUSAR_10_CLASSES: List[str] = [
    "Cargo", "Dredger", "Fishing", "LawEnforce",
    "Other", "Passenger", "Reserved", "Tanker",
    "Tug", "Unspecified",
]

# Five largest FUSAR classes (each fills the 50-image test cap).
# Used for tighter in-domain recognition experiments where class
# ambiguity is reduced to the core commercial vessel types.
FUSAR_5_CLASSES: List[str] = [
    "Cargo", "Fishing", "Other", "Tanker", "Unspecified",
]

# MSTAR vehicle classes (excludes SLICY which is a calibration reflector,
# not a ground vehicle, and should not be counted as a "novel vehicle" target).
MSTAR_VEHICLE_CLASSES: List[str] = [
    "2S1", "BMP2", "BRDM2", "BTR60", "BTR70",
    "D7", "T62", "T72", "ZIL131", "ZSU_23_4",
]
from training.phase1 import run_phase1
from training.phase2 import run_phase2

from utils.metrics import compute_auroc, fpr_at_tpr, confusion_matrix, per_class_prf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component factory
# ---------------------------------------------------------------------------

def build_system(
    cfg: ProtoEvoNetConfig,
) -> dict:
    """
    Instantiate all ProtoEvoNet components from *cfg*.

    Returns
    -------
    dict with keys:
        backbone, gamma, dsar, snr, fisher_rao,
        hbs, owms, pim, tpe, platt, novelty_detector, engine
    """
    backbone = SRABBackbone(cfg.backbone)

    gamma = GammaStatisticsModule(
        patch_size=cfg.physics.snr_patch_size,
        estimator="mom",
        eps=cfg.physics.gamma_eps,
        max_iter=cfg.physics.gamma_mle_iters,
    )
    dsar = DSARModule(
        embedding_dim=cfg.physics.dsar_dim,
        use_dsar=cfg.physics.use_dsar,
    )
    snr = SNREstimationModule(
        patch_size=cfg.physics.snr_patch_size,
        use_snr=cfg.physics.use_snr,
    )
    fisher_rao = FisherRaoDistance(
        method="approx",
        eps=cfg.physics.fisher_rao_eps,
        use_fisher_rao=cfg.physics.use_fisher_rao,
    )

    hbs = HippocampalBindingSystem(
        cfg=cfg.hippocampus,
        num_classes=16,   # must match training (FUSAR-Ship has 16 classes)
        ais_dim=0,        # no AIS data available in demo
    )

    owms = OnlineWorkingMemoryStore(
        embedding_dim=cfg.backbone.embedding_dim,
        max_classes=cfg.memory.max_classes,
        device=str(torch.device(cfg.device if torch.cuda.is_available() else "cpu")),
    )
    pim = PrototypeInterferenceMonitor(
        threshold=cfg.memory.pim_threshold,
        repulsion_strength=0.01,
        use_pim=cfg.ablation.use_pim,
    )
    tpe = TemporalPrototypeEvolution(
        momentum=cfg.memory.tpe_momentum,
        min_count=cfg.memory.tpe_min_count,
        use_tpe=cfg.ablation.use_tpe,
    )

    platt = PlattCalibrator(
        temperature_init=cfg.inference.platt_temperature,
        bias_init=cfg.inference.platt_bias,
        use_novelty=cfg.ablation.use_novelty,
    )
    novelty_detector = NoveltyDetector(
        calibrator=platt,
        threshold=cfg.inference.novelty_threshold,
    )

    engine = ProtoEvoNetInferenceEngine(
        backbone=backbone,
        hbs=hbs,
        owms=owms,
        pim=pim,
        tpe=tpe,
        novelty_detector=novelty_detector,
        gamma_module=gamma,
        dsar_module=dsar,
        snr_module=snr,
        cfg=cfg,
    )

    return {
        "backbone": backbone,
        "gamma": gamma,
        "dsar": dsar,
        "snr": snr,
        "fisher_rao": fisher_rao,
        "hbs": hbs,
        "owms": owms,
        "pim": pim,
        "tpe": tpe,
        "platt": platt,
        "novelty_detector": novelty_detector,
        "engine": engine,
    }


# ---------------------------------------------------------------------------
# Demo inference
# ---------------------------------------------------------------------------

def _embed_images_batched(
    backbone: torch.nn.Module,
    images: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Embed a list of image tensors in batches. Returns (N, D) tensor."""
    backbone.eval()
    all_emb = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.stack(images[start:start + batch_size]).to(device)
            emb = backbone(batch)
            all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)


def demo_inference(
    engine: ProtoEvoNetInferenceEngine,
    fusar_dataset: FUSARShipDataset,
    n_enrol_per_class: int = 20,
    n_test_max_per_class: int = 50,
    results_dir: str = "logs",
    random_support: bool = False,
) -> dict:
    """
    Enrol all FUSAR-Ship classes and run a full recognition evaluation.

    Produces:
      * Per-class accuracy, precision, recall, F1 logged as a table.
      * Macro-averaged accuracy and F1.
      * Confusion matrix saved to ``results_dir/confusion_matrix.csv``.
      * Per-sample predictions saved to ``results_dir/recognition_results.csv``.

    Parameters
    ----------
    engine:
        Fully built inference engine.
    fusar_dataset:
        FUSAR-Ship dataset.
    n_enrol_per_class:
        Target number of support images per class for enrolment.
        For classes with fewer images the floor(N/2) is used instead.
    n_test_max_per_class:
        Maximum number of query images tested per class.
    results_dir:
        Directory where CSV result files are written.
    """
    os.makedirs(results_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DEMO: Enrolling FUSAR-Ship classes")
    logger.info("=" * 60)

    MIN_ENROL = 5          # skip classes with fewer images than this
    enrolled_class_ids: dict = {}
    test_indices_map: dict = {}

    for cls in fusar_dataset.classes:
        indices = fusar_dataset.get_class_samples(cls)
        n = len(indices)

        # Adaptive enrolment: use requested amount or half the class, whichever
        # is smaller, but require at least MIN_ENROL images to enrol.
        n_enrol = min(n_enrol_per_class, n // 2)
        if n_enrol < MIN_ENROL:
            logger.warning(
                "Class '%s' has only %d images (need ≥ %d for enrolment) — skipping.",
                cls, n, MIN_ENROL * 2,
            )
            continue

        if random_support:
            support_idx = random.sample(indices, n_enrol)
        else:
            support_idx = indices[:n_enrol]
        support_images = torch.stack([fusar_dataset[i][0] for i in support_idx])

        result = engine.enrol(support_images=support_images, label=cls)
        enrolled_class_ids[cls] = result.class_id
        test_indices_map[cls] = indices[n_enrol:][:n_test_max_per_class]

        logger.info(
            "  Enrolled %-20s  id=%3d  n_support=%3d  uncertainty=%.4f  "
            "interference=%s",
            cls, result.class_id, n_enrol, result.uncertainty,
            "YES" if result.interference_detected else "no",
        )

    # ---- Recalibrate novelty detector against the live OWMS ----------------
    logger.info("\nDEMO: Recalibrating novelty detector against live OWMS...")
    engine.novelty_detector.recalibrate_from_owms(
        backbone=engine.backbone,
        owms=engine.owms,
        dataset=fusar_dataset,
        device=str(engine.device),
    )

    # ---- Recognition evaluation --------------------------------------------
    logger.info("\nDEMO: Recognition evaluation")
    logger.info("-" * 60)

    enrolled_classes = [c for c in fusar_dataset.classes if c in enrolled_class_ids]
    n_enrolled = len(enrolled_classes)

    # Build local index (0 … n_enrolled-1) for confusion matrix
    local_idx = {cls: i for i, cls in enumerate(enrolled_classes)}

    all_preds:  List[int] = []
    all_labels: List[int] = []
    csv_rows:   List[dict] = []

    for cls in enrolled_classes:
        for idx in test_indices_map[cls]:
            image, _, _ = fusar_dataset[idx]
            result: RecognitionResult = engine.recognise(image.unsqueeze(0))

            pred_cls  = result.predicted_label
            is_correct = pred_cls == cls

            all_labels.append(local_idx[cls])
            all_preds.append(local_idx.get(pred_cls, -1))

            csv_rows.append({
                "gt_class": cls,
                "pred_class": pred_cls,
                "correct": int(is_correct),
                "confidence": f"{result.confidence:.4f}",
            })

    # ---- Compute metrics ---------------------------------------------------
    preds_t  = torch.tensor(all_preds)
    labels_t = torch.tensor(all_labels)

    overall_acc = (preds_t == labels_t).float().mean().item() * 100.0

    prf = per_class_prf(preds_t, labels_t, n_enrolled)

    # Macro averages (ignore NaN classes)
    valid_f1 = [v["f1"] for v in prf.values()
                if not (isinstance(v["f1"], float) and v["f1"] != v["f1"])]
    macro_f1 = sum(valid_f1) / len(valid_f1) if valid_f1 else float("nan")

    # ---- Per-class table ---------------------------------------------------
    header = f"{'Class':<22} {'N':>5}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
    logger.info("\n%s", header)
    logger.info("-" * len(header))

    for cls in enrolled_classes:
        li   = local_idx[cls]
        mask = labels_t == li
        n_test = mask.sum().item()
        if n_test == 0:
            continue
        cls_acc  = (preds_t[mask] == li).float().mean().item() * 100.0
        p  = prf[li]["precision"] * 100.0
        r  = prf[li]["recall"]    * 100.0
        f1 = prf[li]["f1"]        * 100.0
        logger.info(
            "  %-20s %5d  %6.1f  %6.1f  %6.1f  %6.1f",
            cls, n_test, cls_acc, p, r, f1,
        )

    logger.info("-" * len(header))
    logger.info(
        "  %-20s %5d  %6.1f  %6s  %6s  %6.1f",
        "MACRO", len(all_labels), overall_acc, "", "", macro_f1 * 100.0,
    )

    # ---- Confusion matrix --------------------------------------------------
    cm = confusion_matrix(preds_t, labels_t, n_enrolled)
    cm_path = os.path.join(results_dir, "confusion_matrix.csv")
    with open(cm_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([""] + enrolled_classes)
        for i, cls in enumerate(enrolled_classes):
            writer.writerow([cls] + cm[i].tolist())
    logger.info("\nConfusion matrix saved → %s", cm_path)

    # ---- Per-sample CSV ----------------------------------------------------
    rec_path = os.path.join(results_dir, "recognition_results.csv")
    with open(rec_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["gt_class", "pred_class",
                                                "correct", "confidence"])
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info("Per-sample results saved → %s", rec_path)
    logger.info(
        "\nOverall recognition accuracy: %.2f%%  (%d/%d)  Macro-F1: %.2f%%",
        overall_acc, int(overall_acc * len(all_labels) / 100), len(all_labels),
        macro_f1 * 100.0,
    )

    # ---- Return metrics for programmatic use (e.g. ablation study) --------
    per_class_metrics = {}
    for cls in enrolled_classes:
        li   = local_idx[cls]
        mask = labels_t == li
        if mask.sum().item() == 0:
            continue
        per_class_metrics[cls] = {
            "acc":       (preds_t[mask] == li).float().mean().item() * 100.0,
            "precision": prf[li]["precision"] * 100.0,
            "recall":    prf[li]["recall"]    * 100.0,
            "f1":        prf[li]["f1"]        * 100.0,
        }
    return {
        "overall_acc": overall_acc,
        "macro_f1":    macro_f1 * 100.0,
        "n_correct":   int(overall_acc * len(all_labels) / 100),
        "n_total":     len(all_labels),
        "per_class":   per_class_metrics,
    }


def _compute_physics_batched(
    engine: "ProtoEvoNetInferenceEngine",
    images: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Compute augmented physics descriptor vectors [k, θ, SNR, SI, mean_I].

    The engine returns 4 raw features [k, θ, SNR, SI].  We append
    mean_intensity = k × θ = E[I] as a 5th feature here so that the engine
    module itself does not need modification.  Mean intensity captures the
    absolute brightness of the chip (MSTAR ground vehicles are bright metal
    point scatterers; FUSAR sea-surface targets tend to be dimmer).

    Returns
    -------
    torch.Tensor
        Shape ``(N, 5)``.
    """
    all_phys = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack(images[start:start + batch_size]).to(device)
        with torch.no_grad():
            phys = engine._physics_descriptors(batch)   # (B, 4)
        all_phys.append(phys.cpu())
    raw = torch.cat(all_phys, dim=0)           # (N, 4)
    mean_intensity = (raw[:, 0] * raw[:, 1]).unsqueeze(1)   # k × θ = E[I]
    return torch.cat([raw, mean_intensity], dim=1)           # (N, 5)


def demo_novelty_mstar(
    engine: ProtoEvoNetInferenceEngine,
    fusar_dataset: FUSARShipDataset,
    mstar_dataset: MSTARDataset,
    n_known: int = 200,
    n_novel_per_class: int = 50,
    n_calib_per_class: int = 10,
    n_ref: int = 500,
    mahal_reg: float = 1e-3,
    pca_dim: int = 64,
    knn_k: int = 10,
    energy_temperature: float = 0.1,
    results_dir: str = "logs",
    verbose_novelty: bool = False,
    random_support: bool = False,  # Added missing parameter
) -> None:
    """
    Evaluate novelty detection using MSTAR ground-vehicle images as novel
    queries and held-out FUSAR-Ship images as known queries.

    Default (verbose_novelty=False) reports a clean 6-method comparison table
    plus a 3-row C1→C1b→C1c ablation sub-table.  Pass verbose_novelty=True to
    revert to the full 14-method diagnostic sweep.

    **Main comparison table (6 methods)**
      A1. d₁/d₂  zero-shot    — ratio baseline (motivation for this work)
      B1. Min-distance d₁     — simplest absolute-distance baseline
      B2. Energy score         — standard OOD literature baseline
      B3. KNN (k=10)           — SSD / kNN-distance baseline
      B5. PCA-RMD              — best embedding-space method
      C1c. Physics per-class LDA — our method (BEST)

    **Ablation sub-table (physics scorer evolution)**
      C1 → C1b → C1c

    **Verbose-only methods (--verbose-novelty)**
      A2 (Platt-calibrated d₁/d₂), B4 (PCA-Mahal),
      C2/C2b/C2c (physics+PCA-RMD combos), C3 (three-signal combo)
    """
    os.makedirs(results_dir, exist_ok=True)
    device = engine.device

    logger.info("=" * 68)
    logger.info("DEMO: MSTAR Novelty Detection — Definitive Evaluation")
    logger.info("  MSTAR classes : %s", ", ".join(mstar_dataset.classes))
    logger.info(
        "  Total MSTAR   : %d images  |  Known FUSAR queries: %d",
        len(mstar_dataset), n_known,
    )
    logger.info(
        "  Reference set : %d FUSAR  |  PCA dim: %d  |  KNN k: %d  |  T_energy: %g",
        n_ref, pca_dim, knn_k, energy_temperature,
    )
    logger.info("=" * 68)

    # ================================================================
    # 1. Embed everything
    # ================================================================
    fusar_idx = torch.randperm(len(fusar_dataset))[:n_known].tolist()
    known_imgs = [fusar_dataset[i][0] for i in fusar_idx]
    known_emb  = _embed_images_batched(engine.backbone, known_imgs, device)

    # MSTAR: split calibration / evaluation
    calib_images:      List[torch.Tensor] = []
    eval_images:       List[torch.Tensor] = []
    eval_class_labels: List[str]          = []

    for cls in mstar_dataset.classes:
        cls_idx = mstar_dataset.get_class_samples(cls)
        calib_n = min(n_calib_per_class, len(cls_idx))
        eval_end = calib_n + n_novel_per_class
        for i in cls_idx[:calib_n]:
            calib_images.append(mstar_dataset[i][0])
        for i in cls_idx[calib_n:eval_end]:
            eval_images.append(mstar_dataset[i][0])
            eval_class_labels.append(cls)

    if len(eval_images) == 0:
        logger.error("No MSTAR evaluation images found — check --mstar-root.")
        return

    calib_novel_emb = _embed_images_batched(engine.backbone, calib_images, device)
    eval_novel_emb  = _embed_images_batched(engine.backbone, eval_images,  device)

    # Reference set for density-based methods
    n_ref_actual = min(n_ref, len(fusar_dataset))
    ref_idx  = torch.randperm(len(fusar_dataset))[:n_ref_actual].tolist()
    ref_imgs = [fusar_dataset[i][0] for i in ref_idx]
    ref_emb  = _embed_images_batched(engine.backbone, ref_imgs, device)

    protos, _ = engine.owms.all_prototypes()
    protos    = protos.detach()

    known_emb_dev = known_emb.to(device)
    eval_emb_dev  = eval_novel_emb.to(device)

    all_labels = torch.cat([
        torch.zeros(len(known_emb)),
        torch.ones(len(eval_novel_emb)),
    ])

    # ================================================================
    # Diagnostic: log absolute d₁ values to reveal embedding geometry
    # ================================================================
    with torch.no_grad():
        known_d1 = engine.novelty_detector.min_dist_score(
            known_emb_dev, engine.owms).cpu()
        novel_d1_diag = engine.novelty_detector.min_dist_score(
            eval_emb_dev, engine.owms).cpu()

    logger.info(
        "\n[DIAGNOSTIC — Absolute nearest-prototype distance d₁]"
        "\n  Known (FUSAR) d₁ : mean=%.4f  std=%.4f  min=%.4f  max=%.4f"
        "\n  Novel (MSTAR) d₁ : mean=%.4f  std=%.4f  min=%.4f  max=%.4f"
        "\n  (If means are close, embedding space conflates MSTAR and FUSAR;"
        "\n   energy score and physics scorer are then the best fallback.)",
        known_d1.mean(), known_d1.std(), known_d1.min(), known_d1.max(),
        novel_d1_diag.mean(), novel_d1_diag.std(), novel_d1_diag.min(), novel_d1_diag.max(),
    )

    # ================================================================
    # Family A — d₁/d₂ scorers
    # ================================================================
    with torch.no_grad():
        known_d1d2 = engine.novelty_detector.score(
            known_emb_dev, engine.owms).detach().cpu()
        calib_novel_d1d2 = engine.novelty_detector.score(
            calib_novel_emb.to(device), engine.owms).detach().cpu()
        eval_novel_d1d2 = engine.novelty_detector.score(
            eval_emb_dev, engine.owms).detach().cpu()

    # A1 — zero-shot
    scores_a1 = torch.cat([known_d1d2, eval_novel_d1d2])
    auroc_a1  = compute_auroc(scores_a1, all_labels)
    fpr95_a1  = fpr_at_tpr(scores_a1, all_labels, 0.95)
    logger.info("\n[A1: d₁/d₂  zero-shot]")
    logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.3f  Novel μ=%.3f",
                auroc_a1, fpr95_a1, known_d1d2.mean(), eval_novel_d1d2.mean())

    # A2 — Platt refitted on real MSTAR calib samples (verbose only)
    # AUROC is rank-invariant under Platt, so A2 AUROC == A1 AUROC.
    # Skipped by default to avoid mutating the calibrator state.
    auroc_a2 = auroc_a1
    fpr95_a2 = float("nan")
    if verbose_novelty:
        n_calib_known   = len(calib_images)
        fusar_calib_idx = torch.randperm(len(fusar_dataset))[:n_calib_known].tolist()
        calib_known_emb = _embed_images_batched(
            engine.backbone, [fusar_dataset[i][0] for i in fusar_calib_idx], device)
        with torch.no_grad():
            calib_known_d1d2 = engine.novelty_detector.score(
                calib_known_emb.to(device), engine.owms).detach().cpu()
        engine.novelty_detector.calibrator.fit(
            torch.cat([calib_known_d1d2, calib_novel_d1d2]).to(device),
            torch.cat([torch.zeros(n_calib_known), torch.ones(len(calib_images))]).to(device),
        )
        fpr95_a2 = fpr_at_tpr(
            engine.novelty_detector.calibrator(scores_a1.to(device)).detach().cpu(),
            all_labels, 0.95,
        )
        logger.info("\n[A2: d₁/d₂  Platt-calibrated (FPR@95 improves, AUROC identical)]")
        logger.info("  AUROC: %.4f  FPR@95: %.4f  (Platt T=%.3f  b=%.3f)",
                    auroc_a2, fpr95_a2,
                    engine.novelty_detector.calibrator.temperature.item(),
                    engine.novelty_detector.calibrator.bias.item())

    # ================================================================
    # Family B — Absolute embedding scorers
    # ================================================================

    # B1 — raw min-distance
    scores_b1 = torch.cat([known_d1, novel_d1_diag])
    auroc_b1  = compute_auroc(scores_b1, all_labels)
    fpr95_b1  = fpr_at_tpr(scores_b1, all_labels, 0.95)
    logger.info("\n[B1: Min-distance d₁ (absolute)]")
    logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.4f  Novel μ=%.4f",
                auroc_b1, fpr95_b1, known_d1.mean(), novel_d1_diag.mean())

    # B2 — energy score
    with torch.no_grad():
        known_energy = engine.novelty_detector.energy_score(
            known_emb_dev, engine.owms, temperature=energy_temperature).cpu()
        novel_energy = engine.novelty_detector.energy_score(
            eval_emb_dev, engine.owms, temperature=energy_temperature).cpu()
    scores_b2 = torch.cat([known_energy, novel_energy])
    auroc_b2  = compute_auroc(scores_b2, all_labels)
    fpr95_b2  = fpr_at_tpr(scores_b2, all_labels, 0.95)
    logger.info("\n[B2: Energy score  E(z)=-T·log Σ exp(-d²/T)  T=%g]",
                energy_temperature)
    logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.4f  Novel μ=%.4f",
                auroc_b2, fpr95_b2, known_energy.mean(), novel_energy.mean())

    # B3 — KNN
    knn_scorer = KNNScorer(k=knn_k)
    knn_scorer.fit(reference_emb=ref_emb.to(device))
    known_knn = knn_scorer.score(known_emb_dev).cpu()
    novel_knn = knn_scorer.score(eval_emb_dev).cpu()
    scores_b3 = torch.cat([known_knn, novel_knn])
    auroc_b3  = compute_auroc(scores_b3, all_labels)
    fpr95_b3  = fpr_at_tpr(scores_b3, all_labels, 0.95)
    logger.info("\n[B3: KNN  k=%d  N_ref=%d]", knn_k, n_ref_actual)
    logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.4f  Novel μ=%.4f",
                auroc_b3, fpr95_b3, known_knn.mean(), novel_knn.mean())

    # B4/B5 — PCA-Mahalanobis and PCA-RMD
    pca_dim_actual = min(pca_dim, n_ref_actual - 1, protos.shape[1])
    pca_mahal = PCAMahalanobisScorer(pca_dim=pca_dim_actual, reg_coeff=mahal_reg)
    pca_mahal.fit(class_means=protos, reference_emb=ref_emb)
    known_pca_m = pca_mahal.score(known_emb_dev).cpu()
    novel_pca_m = pca_mahal.score(eval_emb_dev).cpu()
    known_pca_r = pca_mahal.relative_score(known_emb_dev).cpu()
    novel_pca_r = pca_mahal.relative_score(eval_emb_dev).cpu()

    auroc_b4 = compute_auroc(torch.cat([known_pca_m, novel_pca_m]), all_labels)
    fpr95_b4 = fpr_at_tpr(torch.cat([known_pca_m, novel_pca_m]), all_labels, 0.95)
    auroc_b5 = compute_auroc(torch.cat([known_pca_r, novel_pca_r]), all_labels)
    fpr95_b5 = fpr_at_tpr(torch.cat([known_pca_r, novel_pca_r]), all_labels, 0.95)
    if verbose_novelty:
        logger.info("\n[B4: PCA-Mahalanobis  pca_dim=%d]", pca_dim_actual)
        logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.4f  Novel μ=%.4f",
                    auroc_b4, fpr95_b4, known_pca_m.mean(), novel_pca_m.mean())
    logger.info("\n[B5: PCA-RMD  pca_dim=%d]", pca_dim_actual)
    logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.4f  Novel μ=%.4f",
                auroc_b5, fpr95_b5, known_pca_r.mean(), novel_pca_r.mean())

    # ================================================================
    # Family C — Physics scorers
    # ================================================================
    logger.info("\nComputing physics descriptors [k, θ, SNR, SI, mean_intensity]...")
    known_phys = _compute_physics_batched(engine, known_imgs, device)
    eval_phys  = _compute_physics_batched(engine, eval_images, device)
    ref_phys   = _compute_physics_batched(engine, ref_imgs, device)

    logger.info(
        "  FUSAR ref physics  k=%.3f  SNR=%.3f  mean_I=%.4f",
        ref_phys[:, 0].mean().item(), ref_phys[:, 2].mean().item(),
        ref_phys[:, 4].mean().item(),
    )
    logger.info(
        "  MSTAR eval physics k=%.3f  SNR=%.3f  mean_I=%.4f",
        eval_phys[:, 0].mean().item(), eval_phys[:, 2].mean().item(),
        eval_phys[:, 4].mean().item(),
    )

    phys_scorer = PhysicsNoveltyScorer()
    phys_scorer.fit(ref_phys)   # global Gaussian (fallback for C1)

    known_phys_s = phys_scorer.score(known_phys).cpu()
    novel_phys_s = phys_scorer.score(eval_phys).cpu()
    scores_c1    = torch.cat([known_phys_s, novel_phys_s])
    auroc_c1     = compute_auroc(scores_c1, all_labels)
    fpr95_c1     = fpr_at_tpr(scores_c1, all_labels, 0.95)
    logger.info("\n[C1: Physics Gaussian — global  (k, θ, SNR, SI)]")
    logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.4f  Novel μ=%.4f",
                auroc_c1, fpr95_c1, known_phys_s.mean().item(), novel_phys_s.mean().item())
    logger.info(
        "  NOTE: Known μ > Novel μ means within-FUSAR physics variance inflates "
        "FUSAR scores.\n  Per-class model (C1b) corrects this."
    )

    # C1b — Per-class physics: one Gaussian per enrolled FUSAR class
    # Each class's own physics mean/std is estimated from its support images,
    # so a FUSAR Cargo ship is measured against the Cargo physics mean (not
    # the global mean across all 12 ship types).
    logger.info("\nBuilding per-class physics model from enrolled FUSAR support sets...")
    class_physics_list: List[torch.Tensor] = []
    enrolled_labels: List[str] = []
    protos_all, proto_ids = engine.owms.all_prototypes()
    for cid in proto_ids:
        entry = engine.owms.get(cid)
        if entry is None:
            continue
        cls_name = entry.label
        cls_indices = fusar_dataset.get_class_samples(cls_name)
        # Use up to 30 images per class for physics estimation
        n_phys_samples = min(30, len(cls_indices))
        if n_phys_samples == 0:
            # No images found — use a dummy single-element list so fallback triggers
            class_physics_list.append(ref_phys[:1])
        else:
            cls_imgs = [fusar_dataset[i][0] for i in cls_indices[:n_phys_samples]]
            cls_phys = _compute_physics_batched(engine, cls_imgs, device)
            class_physics_list.append(cls_phys)
        enrolled_labels.append(cls_name)
        logger.info(
            "  %-20s  n=%2d  k=%.3f  SNR=%.3f  mean_I=%.4f",
            cls_name, len(class_physics_list[-1]),
            class_physics_list[-1][:, 0].mean().item(),
            class_physics_list[-1][:, 2].mean().item(),
            class_physics_list[-1][:, 4].mean().item(),
        )

    phys_scorer.fit_per_class(class_physics_list)

    known_phys_pc = phys_scorer.min_class_score(known_phys).cpu()
    novel_phys_pc = phys_scorer.min_class_score(eval_phys).cpu()
    scores_c1b    = torch.cat([known_phys_pc, novel_phys_pc])
    auroc_c1b     = compute_auroc(scores_c1b, all_labels)
    fpr95_c1b     = fpr_at_tpr(scores_c1b, all_labels, 0.95)
    logger.info("\n[C1b: Physics Gaussian — per-class diagonal  (k, θ, SNR, SI)]")
    logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.4f  Novel μ=%.4f",
                auroc_c1b, fpr95_c1b,
                known_phys_pc.mean().item(), novel_phys_pc.mean().item())

    # C1c — per-class LDA: shared pooled full covariance (360 samples for Σ)
    # vs per-class diagonal from 30 samples.  Pooled Σ is 12× more stable and
    # captures feature correlations (k↔SNR↔SI) that the diagonal misses.
    phys_scorer.fit_per_class_pooled(class_physics_list, reg=0.01)
    known_phys_lda = phys_scorer.min_class_score_pooled(known_phys).cpu()
    novel_phys_lda = phys_scorer.min_class_score_pooled(eval_phys).cpu()
    scores_c1c     = torch.cat([known_phys_lda, novel_phys_lda])
    auroc_c1c      = compute_auroc(scores_c1c, all_labels)
    fpr95_c1c      = fpr_at_tpr(scores_c1c, all_labels, 0.95)
    logger.info("\n[C1c: Physics LDA — per-class pooled-cov  (k, θ, SNR, SI)]")
    logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.4f  Novel μ=%.4f",
                auroc_c1c, fpr95_c1c,
                known_phys_lda.mean().item(), novel_phys_lda.mean().item())

    # ----------------------------------------------------------------
    # Pool-normalisation helper (used by verbose combined methods)
    # ----------------------------------------------------------------
    def _pool_zscore(
        t_known: torch.Tensor,
        t_novel: torch.Tensor,
    ):
        combined = torch.cat([t_known, t_novel])
        mu    = combined.mean()
        sigma = combined.std().clamp(min=1e-8)
        return (t_known - mu) / sigma, (t_novel - mu) / sigma

    # C2 / C2b / C2c / C3 — combined signals (verbose-only)
    # These combinations did not improve over C1c alone; they are retained
    # for diagnostic purposes but excluded from the default output.
    if verbose_novelty:
        k_phys_z,  n_phys_z  = _pool_zscore(known_phys_s, novel_phys_s)
        k_pcar_z,  n_pcar_z  = _pool_zscore(known_pca_r,  novel_pca_r)

        known_c2 = k_phys_z + k_pcar_z
        novel_c2 = n_phys_z + n_pcar_z
        scores_c2 = torch.cat([known_c2, novel_c2])
        auroc_c2  = compute_auroc(scores_c2, all_labels)
        fpr95_c2  = fpr_at_tpr(scores_c2, all_labels, 0.95)
        logger.info("\n[C2: Physics(global) + PCA-RMD  (pool-z-score combined)]")
        logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.4f  Novel μ=%.4f",
                    auroc_c2, fpr95_c2, known_c2.mean().item(), novel_c2.mean().item())

        k_phys_pc_z, n_phys_pc_z = _pool_zscore(known_phys_pc, novel_phys_pc)
        known_c2b  = k_phys_pc_z + k_pcar_z
        novel_c2b  = n_phys_pc_z + n_pcar_z
        scores_c2b = torch.cat([known_c2b, novel_c2b])
        auroc_c2b  = compute_auroc(scores_c2b, all_labels)
        fpr95_c2b  = fpr_at_tpr(scores_c2b, all_labels, 0.95)
        logger.info("\n[C2b: Physics(per-class diag) + PCA-RMD  (pool-z-score)]")
        logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.4f  Novel μ=%.4f",
                    auroc_c2b, fpr95_c2b, known_c2b.mean().item(), novel_c2b.mean().item())

        k_phys_lda_z, n_phys_lda_z = _pool_zscore(known_phys_lda, novel_phys_lda)
        known_c2c  = k_phys_lda_z + k_pcar_z
        novel_c2c  = n_phys_lda_z + n_pcar_z
        scores_c2c = torch.cat([known_c2c, novel_c2c])
        auroc_c2c  = compute_auroc(scores_c2c, all_labels)
        fpr95_c2c  = fpr_at_tpr(scores_c2c, all_labels, 0.95)
        logger.info("\n[C2c: Physics(per-class LDA) + PCA-RMD  (pool-z-score)]")
        logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.4f  Novel μ=%.4f",
                    auroc_c2c, fpr95_c2c, known_c2c.mean().item(), novel_c2c.mean().item())

        k_d1_z, n_d1_z = _pool_zscore(known_d1, novel_d1_diag)
        known_c3 = k_phys_z + k_pcar_z + k_d1_z
        novel_c3 = n_phys_z + n_pcar_z + n_d1_z
        scores_c3 = torch.cat([known_c3, novel_c3])
        auroc_c3  = compute_auroc(scores_c3, all_labels)
        fpr95_c3  = fpr_at_tpr(scores_c3, all_labels, 0.95)
        logger.info("\n[C3: Physics + PCA-RMD + Min-dist  (three independent signals)]")
        logger.info("  AUROC: %.4f  FPR@95: %.4f  |  Known μ=%.4f  Novel μ=%.4f",
                    auroc_c3, fpr95_c3, known_c3.mean().item(), novel_c3.mean().item())
        logger.info(
            "  Signal AUROCs → Physics(global)=%.4f  Physics(pc-diag)=%.4f  "
            "Physics(pc-LDA)=%.4f  PCA-RMD=%.4f  MinDist=%.4f",
            auroc_c1, auroc_c1b, auroc_c1c, auroc_b5, auroc_b1,
        )
    else:
        auroc_c2 = auroc_c2b = auroc_c2c = auroc_c3 = float("nan")
        fpr95_c2 = fpr95_c2b = fpr95_c2c = fpr95_c3 = float("nan")
        known_c2 = novel_c2 = known_c2b = novel_c2b = None
        known_c2c = novel_c2c = known_c3 = novel_c3 = None

    # ================================================================
    # Best-method per-class detection rate
    # C1c is always computed and is consistently the strongest method.
    # In verbose mode the dict includes all 14; in default mode it holds
    # the 6 reported methods so the auto-selector still works correctly.
    # ================================================================
    auroc_dict = {
        "a1_d1d2_zs":   auroc_a1,
        "b1_min_dist":  auroc_b1,
        "b2_energy":    auroc_b2,
        "b3_knn":       auroc_b3,
        "b5_pca_r":     auroc_b5,
        "c1c_phys_lda": auroc_c1c,
    }
    novel_scores_map = {
        "a1_d1d2_zs":   eval_novel_d1d2,
        "b1_min_dist":  novel_d1_diag,
        "b2_energy":    novel_energy,
        "b3_knn":       novel_knn,
        "b5_pca_r":     novel_pca_r,
        "c1c_phys_lda": novel_phys_lda,
    }
    known_scores_map = {
        "a1_d1d2_zs":   known_d1d2,
        "b1_min_dist":  known_d1,
        "b2_energy":    known_energy,
        "b3_knn":       known_knn,
        "b5_pca_r":     known_pca_r,
        "c1c_phys_lda": known_phys_lda,
    }
    if verbose_novelty:
        auroc_dict.update({
            "a2_d1d2_cal":  auroc_a2,
            "b4_pca_m":     auroc_b4,
            "c1_physics":   auroc_c1,
            "c1b_phys_pc":  auroc_c1b,
            "c2_combined":  auroc_c2,
            "c2b_phys_pc":  auroc_c2b,
            "c2c_phys_lda": auroc_c2c,
            "c3_combined":  auroc_c3,
        })
        novel_scores_map.update({
            "a2_d1d2_cal":  eval_novel_d1d2,
            "b4_pca_m":     novel_pca_m,
            "c1_physics":   novel_phys_s,
            "c1b_phys_pc":  novel_phys_pc,
            "c2_combined":  novel_c2,
            "c2b_phys_pc":  novel_c2b,
            "c2c_phys_lda": novel_c2c,
            "c3_combined":  novel_c3,
        })
        known_scores_map.update({
            "a2_d1d2_cal":  known_d1d2,
            "b4_pca_m":     known_pca_m,
            "c1_physics":   known_phys_s,
            "c1b_phys_pc":  known_phys_pc,
            "c2_combined":  known_c2,
            "c2b_phys_pc":  known_c2b,
            "c2c_phys_lda": known_c2c,
            "c3_combined":  known_c3,
        })

    best_method       = max(auroc_dict, key=auroc_dict.get)
    best_novel_scores = novel_scores_map[best_method]
    best_known_scores = known_scores_map[best_method]

    sorted_novel = best_novel_scores.sort().values
    thr_idx      = max(int(0.05 * len(sorted_novel)) - 1, 0)
    thr_95tpr    = sorted_novel[thr_idx].item()

    logger.info(
        "\nPer-MSTAR-class detection rate  [best: %s  AUROC=%.4f  thr@95%%TPR=%.4f]:",
        best_method, auroc_dict[best_method], thr_95tpr,
    )
    logger.info("  %-20s  %6s  %8s  %8s", "Class", "N", "Det.Rate", "Score μ")

    offset = 0
    per_class_rows = []
    for cls in mstar_dataset.classes:
        cls_count = eval_class_labels.count(cls)
        if cls_count == 0:
            continue
        cls_s    = best_novel_scores[offset:offset + cls_count]
        det_rate = (cls_s > thr_95tpr).float().mean().item()
        logger.info("  %-20s  %6d  %8.3f  %8.3f",
                    cls, cls_count, det_rate, cls_s.mean().item())
        per_class_rows.append({
            "mstar_class":         cls,
            "n_images":            cls_count,
            "detection_rate_best": f"{det_rate:.4f}",
            "score_mean_best":     f"{cls_s.mean().item():.4f}",
            "best_method":         best_method,
        })
        offset += cls_count

    # ================================================================
    # Save CSVs
    # ================================================================
    novelty_path = os.path.join(results_dir, "novelty_scores.csv")
    with open(novelty_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "source", "class",
            "score_d1d2", "score_min_dist", "score_energy",
            "score_knn", "score_pca_rmd",
            "score_c1_global", "score_c1b_pc_diag", "score_c1c_pc_lda",
            "is_novel_gt",
        ])
        for i in range(len(known_d1d2)):
            writer.writerow([
                "FUSAR", "known",
                f"{known_d1d2[i].item():.6f}",
                f"{known_d1[i].item():.6f}",
                f"{known_energy[i].item():.6f}",
                f"{known_knn[i].item():.6f}",
                f"{known_pca_r[i].item():.6f}",
                f"{known_phys_s[i].item():.6f}",
                f"{known_phys_pc[i].item():.6f}",
                f"{known_phys_lda[i].item():.6f}",
                0,
            ])
        for i, cls in enumerate(eval_class_labels):
            writer.writerow([
                "MSTAR", cls,
                f"{eval_novel_d1d2[i].item():.6f}",
                f"{novel_d1_diag[i].item():.6f}",
                f"{novel_energy[i].item():.6f}",
                f"{novel_knn[i].item():.6f}",
                f"{novel_pca_r[i].item():.6f}",
                f"{novel_phys_s[i].item():.6f}",
                f"{novel_phys_pc[i].item():.6f}",
                f"{novel_phys_lda[i].item():.6f}",
                1,
            ])
    logger.info("\nNovelty scores saved → %s", novelty_path)

    summary_path = os.path.join(results_dir, "novelty_summary.csv")
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "mstar_class", "n_images",
            "detection_rate_best", "score_mean_best", "best_method",
        ])
        writer.writeheader()
        writer.writerows(per_class_rows)
    logger.info("Novelty summary saved → %s", summary_path)

    # ================================================================
    # Final comparison tables
    # ================================================================
    sep = "=" * 64
    thin = "-" * 55

    # ---- Main 6-method comparison table (always printed) --------
    logger.info("\n%s", sep)
    logger.info("NOVELTY DETECTION — METHOD COMPARISON  (FUSAR known vs MSTAR novel)")
    logger.info("  %-36s  %8s  %10s", "Method", "AUROC", "FPR@95TPR")
    logger.info("  %s", thin)
    main_results = [
        ("d₁/d₂ ratio  (baseline)",              auroc_a1,  fpr95_a1,  "a1_d1d2_zs"),
        ("Min-distance d₁",                      auroc_b1,  fpr95_b1,  "b1_min_dist"),
        ("Energy score",                          auroc_b2,  fpr95_b2,  "b2_energy"),
        (f"KNN (k={knn_k})",                     auroc_b3,  fpr95_b3,  "b3_knn"),
        (f"PCA-RMD  (d={pca_dim_actual})",       auroc_b5,  fpr95_b5,  "b5_pca_r"),
        ("Physics per-class LDA  (ours)",        auroc_c1c, fpr95_c1c, "c1c_phys_lda"),
    ]
    for name, auroc, fpr, key in main_results:
        marker = "  ◄ BEST" if key == best_method else ""
        logger.info("  %-36s  %8.4f  %10.4f%s", name, auroc, fpr, marker)
    logger.info("  %s", thin)
    logger.info("  Target: AUROC → 1.0,  FPR@95 → 0.0")
    logger.info("%s", sep)

    # ---- Physics ablation sub-table (always printed) ------------
    logger.info("\n%s", sep)
    logger.info("PHYSICS SCORER ABLATION  (C1 → C1b → C1c)")
    logger.info("  %-36s  %8s  %10s", "Variant", "AUROC", "FPR@95TPR")
    logger.info("  %s", thin)
    ablation_results = [
        ("C1:  Global Gaussian",                 auroc_c1,  fpr95_c1),
        ("C1b: Per-class diagonal Gaussian",     auroc_c1b, fpr95_c1b),
        ("C1c: Per-class LDA  (ours)",           auroc_c1c, fpr95_c1c),
    ]
    for name, auroc, fpr in ablation_results:
        logger.info("  %-36s  %8.4f  %10.4f", name, auroc, fpr)
    logger.info("  %s", thin)
    logger.info("  Each row adds one modelling improvement over the previous.")
    logger.info("%s", sep)

    # ---- Full 14-method table (verbose only) --------------------
    if verbose_novelty:
        logger.info("\n%s", sep)
        logger.info("VERBOSE: FULL 14-METHOD SWEEP")
        logger.info("  %-36s  %8s  %10s", "Method", "AUROC", "FPR@95TPR")
        logger.info("  %s", thin)
        all_results = [
            ("A1: d₁/d₂  zero-shot",                 auroc_a1,  fpr95_a1,  "a1_d1d2_zs"),
            ("A2: d₁/d₂  Platt-calibrated",          auroc_a2,  fpr95_a2,  "a2_d1d2_cal"),
            ("B1: Min-distance d₁",                  auroc_b1,  fpr95_b1,  "b1_min_dist"),
            ("B2: Energy score",                      auroc_b2,  fpr95_b2,  "b2_energy"),
            (f"B3: KNN (k={knn_k})",                 auroc_b3,  fpr95_b3,  "b3_knn"),
            (f"B4: PCA-Mahal (d={pca_dim_actual})",  auroc_b4,  fpr95_b4,  "b4_pca_m"),
            (f"B5: PCA-RMD  (d={pca_dim_actual})",   auroc_b5,  fpr95_b5,  "b5_pca_r"),
            ("C1:  Physics global",                   auroc_c1,  fpr95_c1,  "c1_physics"),
            ("C1b: Physics per-class diagonal",       auroc_c1b, fpr95_c1b, "c1b_phys_pc"),
            ("C1c: Physics per-class LDA",            auroc_c1c, fpr95_c1c, "c1c_phys_lda"),
            ("C2:  Physics(global)+PCA-RMD",          auroc_c2,  fpr95_c2,  "c2_combined"),
            ("C2b: Physics(per-class diag)+PCA-RMD",  auroc_c2b, fpr95_c2b, "c2b_phys_pc"),
            ("C2c: Physics(per-class LDA)+PCA-RMD",   auroc_c2c, fpr95_c2c, "c2c_phys_lda"),
            ("C3:  Physics+PCA-RMD+d₁",               auroc_c3,  fpr95_c3,  "c3_combined"),
        ]
        for name, auroc, fpr, key in all_results:
            marker = "  ◄ BEST" if key == best_method else ""
            logger.info("  %-36s  %8.4f  %10.4f%s", name, auroc, fpr, marker)
        logger.info("  %s", thin)
        logger.info("  NOTE: A1 = A2 AUROC by construction (Platt is rank-invariant).")
        logger.info("%s", sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProtoEvoNet training and inference pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file.  If omitted, uses default config.",
    )
    parser.add_argument(
        "--data-root", type=str, default="data",
        help="Root data directory containing hrsid/ and fusar_ship/ sub-folders.",
    )
    parser.add_argument(
        "--phase1-only", action="store_true",
        help="Run Phase 1 training only.",
    )
    parser.add_argument(
        "--phase2-only", action="store_true",
        help="Run Phase 2 training only (requires Phase 1 checkpoint).",
    )
    parser.add_argument(
        "--demo-only", action="store_true",
        help="Skip training and run inference demo only.",
    )
    parser.add_argument(
        "--backbone-ckpt", type=str, default=None,
        help="Path to backbone checkpoint for Phase 2 / demo.",
    )
    parser.add_argument(
        "--system-ckpt", type=str, default=None,
        help="Path to full system checkpoint for demo / refinement.",
    )
    parser.add_argument(
        "--ablate", nargs="*", metavar="FLAG=VALUE",
        help="Ablation flags, e.g. --ablate use_gcn=False use_dsar=False",
    )
    parser.add_argument(
        "--mstar-root", type=str, default=None,
        help="Path to MSTAR dataset root for novelty detection evaluation.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda / cpu).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    # ---- Low-shot evaluation ------------------------------------------------
    parser.add_argument(
        "--low-shot-eval", action="store_true",
        help="Run low-shot enrollment evaluation (1/5/10/20-shot, 5 seeds).",
    )
    parser.add_argument(
        "--n-shot", type=int, default=None, nargs="+",
        metavar="N",
        help=(
            "Number(s) of support images per class for --low-shot-eval.  "
            "Accepts one or more values, e.g. --n-shot 1 5 10 20.  "
            "Defaults to [1, 5, 10, 20] when --low-shot-eval is set."
        ),
    )
    parser.add_argument(
        "--dataset", type=str, default="fusar",
        choices=["fusar", "opensarship", "mstar"],
        help="Dataset to use for --low-shot-eval.",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of random seeds to average over in --low-shot-eval.",
    )
    parser.add_argument(
        "--opensarship-root", type=str, default=None,
        help="Path to OpenSARShip 2.0 data root (used with --dataset opensarship).",
    )
    parser.add_argument(
        "--verbose-novelty", action="store_true",
        help=(
            "Print the full 14-method novelty sweep instead of the default "
            "6-method comparison + physics ablation sub-table."
        ),
    )
    parser.add_argument(
        "--fusar-n-classes", type=int, default=10, choices=[5, 10],
        help=(
            "Number of FUSAR classes used for the in-domain recognition demo. "
            "5 = FUSAR_5_CLASSES (Cargo/Fishing/Other/Tanker/Unspecified). "
            "10 = FUSAR_10_CLASSES (default, all stable classes)."
        ),
    )
    parser.add_argument(
        "--n-enrol", type=int, default=20,
        help=(
            "Number of support images per class used for prototype enrolment "
            "in the in-domain FUSAR demo (default: 20)."
        ),
    )
    parser.add_argument(
        "--random-support", action="store_true",
        help="Randomly sample support images each run (for multi‑seed evaluation)."
    )
    return parser.parse_args()


def run_low_shot_eval(
    backbone: torch.nn.Module,
    dataset,
    dataset_name: str,
    n_shot_list: List[int],
    n_seeds: int,
    device: torch.device,
    results_dir: str = "results",
    batch_size: int = 64,
) -> None:
    """
    Low-shot prototype-enrollment evaluation.

    For each n_shot in *n_shot_list* and each seed in range(*n_seeds*):
      1. Sample n_shot images per class as support.
      2. Build prototype = L2-normalised mean of support embeddings.
      3. Evaluate remaining images as queries (argmax cosine similarity).
      4. Compute per-class accuracy and macro-F1.

    Results are averaged over seeds and saved as:
      ``{results_dir}/low_shot_{dataset_name}_n{n}.csv``

    Parameters
    ----------
    backbone:
        Frozen SRAB backbone (eval mode, already on *device*).
    dataset:
        A dataset returning ``(image_tensor, label_int, class_name_str)`` triples.
        Must have a ``classes`` attribute and ``get_class_samples(cls)`` method.
    dataset_name:
        Short identifier used in log messages and filenames.
    n_shot_list:
        List of k-shot values to evaluate.
    n_seeds:
        Number of random seeds to average over.
    device:
        Torch device for inference.
    results_dir:
        Directory where CSV files are written.
    batch_size:
        Images per forward pass.
    """
    import numpy as np
    try:
        from sklearn.metrics import f1_score as sklearn_f1
        _HAS_SKLEARN = True
    except ImportError:
        _HAS_SKLEARN = False

    os.makedirs(results_dir, exist_ok=True)

    backbone.eval()

    # --- Embed the full dataset once to avoid repeated forward passes -------
    logger.info(
        "Low-shot eval [%s]: embedding %d images...", dataset_name, len(dataset),
    )
    all_emb: List[torch.Tensor] = []
    all_lbl: List[int] = []
    all_cls: List[str] = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch_imgs  = []
            batch_lbls  = []
            batch_names = []
            for i in range(start, min(start + batch_size, len(dataset))):
                img, lbl, cls = dataset[i]
                batch_imgs.append(img)
                batch_lbls.append(int(lbl))
                batch_names.append(cls)
            imgs_t = torch.stack(batch_imgs).to(device)
            embs   = backbone(imgs_t)                 # (B, D)
            embs   = F.normalize(embs, p=2, dim=1)
            all_emb.append(embs.cpu())
            all_lbl.extend(batch_lbls)
            all_cls.extend(batch_names)

    embeddings = torch.cat(all_emb, dim=0)   # (N, D)
    labels_arr = torch.tensor(all_lbl)        # (N,)

    # Build class → indices mapping
    class_to_indices: dict = {}
    for cls in dataset.classes:
        class_to_indices[cls] = [
            i for i, c in enumerate(all_cls) if c == cls
        ]

    classes = dataset.classes
    n_classes = len(classes)
    cls_to_local = {c: i for i, c in enumerate(classes)}

    logger.info(
        "Low-shot eval [%s]: %d classes, %d total images",
        dataset_name, n_classes, len(dataset),
    )

    all_summary_rows: List[dict] = []

    for n_shot in n_shot_list:
        seed_accs: List[float] = []
        seed_f1s:  List[float] = []
        per_class_acc_seeds: dict = {c: [] for c in classes}

        for seed_idx in range(n_seeds):
            rng = random.Random(seed_idx * 1000 + n_shot)

            # ---- Prototype construction -----------------------------------
            prototypes: List[torch.Tensor] = []
            support_mask = torch.zeros(len(dataset), dtype=torch.bool)

            for cls in classes:
                cls_idx = class_to_indices[cls]
                if len(cls_idx) < n_shot + 1:
                    # Not enough images: use all but one for support
                    support_n = max(1, len(cls_idx) - 1)
                else:
                    support_n = n_shot
                chosen = rng.sample(cls_idx, support_n)
                support_mask[chosen] = True
                proto = embeddings[chosen].mean(dim=0)
                prototypes.append(F.normalize(proto, p=2, dim=0))

            proto_matrix = torch.stack(prototypes, dim=0)  # (C, D)

            # ---- Query evaluation ----------------------------------------
            query_mask = ~support_mask
            query_emb  = embeddings[query_mask]          # (Q, D)
            query_lbl  = labels_arr[query_mask]          # (Q,)

            if query_emb.shape[0] == 0:
                logger.warning(
                    "n_shot=%d seed=%d: no query images remain — skipping.",
                    n_shot, seed_idx,
                )
                continue

            # Cosine similarity → argmax = nearest prototype
            sim   = query_emb @ proto_matrix.t()         # (Q, C)
            preds = sim.argmax(dim=1)                    # (Q,)

            # Map integer labels to local indices (labels may not be 0..C-1)
            local_lbl = torch.tensor(
                [cls_to_local.get(classes[l.item()], l.item())
                 if l.item() < n_classes else -1
                 for l in query_lbl],
                dtype=torch.long,
            )

            acc = (preds == local_lbl).float().mean().item() * 100.0
            seed_accs.append(acc)

            if _HAS_SKLEARN:
                valid_mask = local_lbl >= 0
                if valid_mask.sum() > 0:
                    f1 = sklearn_f1(
                        local_lbl[valid_mask].numpy(),
                        preds[valid_mask].numpy(),
                        average="macro",
                        zero_division=0,
                    ) * 100.0
                    seed_f1s.append(f1)

            # Per-class accuracy
            for cls in classes:
                li   = cls_to_local[cls]
                mask = local_lbl == li
                if mask.sum() > 0:
                    cls_acc = (preds[mask] == li).float().mean().item() * 100.0
                    per_class_acc_seeds[cls].append(cls_acc)

        if not seed_accs:
            logger.warning("n_shot=%d: no valid seeds — skipping.", n_shot)
            continue

        mean_acc = float(np.mean(seed_accs))
        std_acc  = float(np.std(seed_accs))
        mean_f1  = float(np.mean(seed_f1s))  if seed_f1s  else float("nan")
        std_f1   = float(np.std(seed_f1s))   if seed_f1s  else float("nan")

        logger.info(
            "\nLow-shot [%s]  n_shot=%2d (%d seeds): "
            "Acc=%.2f±%.2f%%  Macro-F1=%.2f±%.2f%%",
            dataset_name, n_shot, n_seeds,
            mean_acc, std_acc, mean_f1, std_f1,
        )

        per_class_means = {
            c: float(np.mean(v)) if v else float("nan")
            for c, v in per_class_acc_seeds.items()
        }
        logger.info(
            "  %-22s  %6s", "Class", "Acc%",
        )
        for cls, cls_acc in sorted(per_class_means.items()):
            logger.info("  %-22s  %6.1f", cls, cls_acc)

        # ---- Save CSV -------------------------------------------------------
        csv_path = os.path.join(
            results_dir, f"low_shot_{dataset_name}_n{n_shot}.csv"
        )
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "dataset", "n_shot", "n_seeds",
                "mean_acc", "std_acc",
                "mean_f1",  "std_f1",
            ] + [f"acc_{c}" for c in classes])
            writer.writerow([
                dataset_name, n_shot, n_seeds,
                f"{mean_acc:.4f}", f"{std_acc:.4f}",
                f"{mean_f1:.4f}",  f"{std_f1:.4f}",
            ] + [f"{per_class_means.get(c, float('nan')):.4f}" for c in classes])
        logger.info("  Saved → %s", csv_path)

        all_summary_rows.append({
            "dataset":  dataset_name,
            "n_shot":   n_shot,
            "n_seeds":  n_seeds,
            "mean_acc": mean_acc,
            "std_acc":  std_acc,
            "mean_f1":  mean_f1,
            "std_f1":   std_f1,
        })

    # ---- Consolidated summary CSV ----------------------------------------
    if all_summary_rows:
        summary_path = os.path.join(results_dir, f"low_shot_{dataset_name}_summary.csv")
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_summary_rows)
        logger.info("Low-shot summary saved → %s", summary_path)

        # Pretty-print summary table
        logger.info("\n%s", "=" * 58)
        logger.info("LOW-SHOT EVALUATION SUMMARY  [%s]", dataset_name.upper())
        logger.info("  %6s  %12s  %12s", "n_shot", "Acc (mean±std)", "F1 (mean±std)")
        logger.info("  %s", "-" * 46)
        for row in all_summary_rows:
            logger.info(
                "  %6d  %6.2f ± %5.2f  %6.2f ± %5.2f",
                row["n_shot"],
                row["mean_acc"], row["std_acc"],
                row["mean_f1"],  row["std_f1"],
            )
        logger.info("%s", "=" * 58)


def apply_ablation_overrides(cfg: ProtoEvoNetConfig, overrides: list) -> None:
    """Parse 'key=value' ablation overrides and set them on cfg.ablation."""
    for override in overrides:
        if "=" not in override:
            logger.warning("Ignoring malformed ablation flag: '%s'", override)
            continue
        key, val = override.split("=", 1)
        val_bool = val.strip().lower() not in {"false", "0", "no"}
        if hasattr(cfg.ablation, key):
            setattr(cfg.ablation, key, val_bool)
            logger.info("Ablation: %s = %s", key, val_bool)
        else:
            logger.warning("Unknown ablation flag '%s'", key)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Entry point for the ProtoEvoNet pipeline."""
    args = parse_args()

    # ---- Config ---------------------------------------------------------
    if args.config and Path(args.config).exists():
        cfg = ProtoEvoNetConfig.load(args.config)
    else:
        cfg = ProtoEvoNetConfig()

    if args.device:
        cfg.device = args.device
    if args.ablate:
        apply_ablation_overrides(cfg, args.ablate)
    cfg.apply_ablation()

    cfg.training.seed = args.seed
    set_seed(args.seed)

    # ---- Logging --------------------------------------------------------
    setup_logging(cfg.training.log_dir)
    logger.info("ProtoEvoNet starting. Device=%s", cfg.device)
    logger.info("Config: %s", cfg)

    # ---- Dataset paths --------------------------------------------------
    data_root = Path(args.data_root)
    hrsid_root = str(data_root / "hrsid")
    fusar_root = str(data_root / "fusar_ship")

    # ---- Build system ---------------------------------------------------
    system = build_system(cfg)
    backbone: SRABBackbone = system["backbone"]
    gamma: GammaStatisticsModule = system["gamma"]
    dsar: DSARModule = system["dsar"]
    snr: SNREstimationModule = system["snr"]
    hbs: HippocampalBindingSystem = system["hbs"]
    owms: OnlineWorkingMemoryStore = system["owms"]
    pim: PrototypeInterferenceMonitor = system["pim"]
    tpe: TemporalPrototypeEvolution = system["tpe"]
    platt: PlattCalibrator = system["platt"]
    novelty_detector: NoveltyDetector = system["novelty_detector"]
    engine: ProtoEvoNetInferenceEngine = system["engine"]

    # ---- Optionally restore backbone checkpoint -------------------------
    if args.backbone_ckpt and Path(args.backbone_ckpt).exists():
        load_checkpoint(
            args.backbone_ckpt,
            state_dict_targets={"backbone": backbone},
            strict=False,
        )
        logger.info("Loaded backbone from %s", args.backbone_ckpt)

    if args.system_ckpt and Path(args.system_ckpt).exists():
        load_checkpoint(
            args.system_ckpt,
            state_dict_targets={
                "backbone": backbone,
                "hbs": hbs,
                "dsar": dsar,
                "platt": platt,
            },
            strict=False,
        )
        logger.info("Loaded system state from %s", args.system_ckpt)

    # ---- Training -------------------------------------------------------
    if not args.demo_only:
        if not args.phase2_only:
            if Path(hrsid_root).exists():
                logger.info("Starting Phase 1 (HRSID contrastive pre-training)...")
                backbone = run_phase1(
                    backbone=backbone,
                    gamma_module=gamma,
                    dsar_module=dsar,
                    snr_module=snr,
                    cfg=cfg,
                    hrsid_root=hrsid_root,
                )
            else:
                logger.warning(
                    "HRSID dataset not found at '%s'. Skipping Phase 1.", hrsid_root
                )

        if not args.phase1_only:
            if Path(fusar_root).exists():
                logger.info("Starting Phase 2 (FUSAR-Ship episodic meta-learning)...")
                run_phase2(
                    backbone=backbone,
                    hbs=hbs,
                    gamma_module=gamma,
                    dsar_module=dsar,
                    snr_module=snr,
                    novelty_detector=novelty_detector,
                    cfg=cfg,
                    fusar_root=fusar_root,
                    fine_tune_backbone=True,
                )
            else:
                logger.warning(
                    "FUSAR-Ship dataset not found at '%s'. Skipping Phase 2.", fusar_root
                )

    # ---- Inference demo -------------------------------------------------
    if Path(fusar_root).exists():
        logger.info("Running inference demo...")
        # Select the class list based on --fusar-n-classes (5 or 10).
        _class_list = FUSAR_5_CLASSES if args.fusar_n_classes == 5 else FUSAR_10_CLASSES
        fusar_10_on_disk = [
            c for c in _class_list
            if (Path(fusar_root) / c).is_dir()
        ]
        logger.info(
            "FUSAR demo: %d-class subset %s",
            len(fusar_10_on_disk), fusar_10_on_disk,
        )
        fusar_dataset = FUSARShipDataset(
            root=fusar_root,
            image_size=(128, 128),
            classes=fusar_10_on_disk,
        )
        # Rebuild engine with potentially updated engine state
        engine._owms = owms  # noqa: SLF001 — update reference
        demo_inference(
            engine=engine,
            fusar_dataset=fusar_dataset,
            n_enrol_per_class=args.n_enrol,
            n_test_max_per_class=50,
            results_dir=cfg.training.log_dir,
            random_support=args.random_support,
        )
    else:
        logger.info("No FUSAR-Ship data found — skipping demo.")

    # ---- MSTAR novelty evaluation ---------------------------------------
    mstar_root = args.mstar_root or str(Path(args.data_root) / "mstar")
    if Path(mstar_root).exists():
        logger.info("Running MSTAR novelty evaluation...")
        # Load all MSTAR images then filter to the 10 vehicle classes,
        # excluding SLICY (a calibration reflector, not a vehicle target).
        mstar_dataset_full = MSTARDataset(root=mstar_root, image_size=(128, 128))
        vehicle_classes_on_disk = [
            c for c in MSTAR_VEHICLE_CLASSES
            if c in mstar_dataset_full.classes
        ]
        if vehicle_classes_on_disk:
            from training.datasets import _load_sar_image as _lsi  # noqa: F401
            # Re-filter samples to vehicle classes only
            mstar_dataset = MSTARDataset.__new__(MSTARDataset)
            mstar_dataset.root       = mstar_dataset_full.root
            mstar_dataset.image_size = mstar_dataset_full.image_size
            mstar_dataset.transform  = mstar_dataset_full.transform
            mstar_dataset.samples    = [
                s for s in mstar_dataset_full.samples
                if s["class_name"] in vehicle_classes_on_disk
            ]
            mstar_dataset.classes      = vehicle_classes_on_disk
            mstar_dataset.class_to_idx = {
                c: i for i, c in enumerate(vehicle_classes_on_disk)
            }
            for s in mstar_dataset.samples:
                s["label"] = mstar_dataset.class_to_idx[s["class_name"]]
            logger.info(
                "MSTAR: using %d vehicle classes (%d images) — SLICY excluded.",
                len(vehicle_classes_on_disk), len(mstar_dataset),
            )
        else:
            mstar_dataset = mstar_dataset_full
        if len(mstar_dataset) > 0:
            demo_novelty_mstar(
                engine=engine,
                fusar_dataset=fusar_dataset if Path(fusar_root).exists() else None,
                mstar_dataset=mstar_dataset,
                n_known=200,
                n_novel_per_class=50,
                n_ref=500,
                pca_dim=64,
                knn_k=10,
                energy_temperature=0.1,
                results_dir=cfg.training.log_dir,
                verbose_novelty=args.verbose_novelty,
            )
        else:
            logger.warning("MSTAR dataset at '%s' loaded 0 images — check structure.",
                           mstar_root)
    else:
        logger.info("No MSTAR data found at '%s' — skipping novelty evaluation.", mstar_root)

    # ---- Low-shot evaluation ------------------------------------------------
    if args.low_shot_eval:
        n_shot_list = args.n_shot if args.n_shot else [1, 5, 10, 20]
        n_seeds     = args.n_seeds
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        device_t = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        if args.dataset == "fusar":
            if not Path(fusar_root).exists():
                logger.error("FUSAR root '%s' not found; skipping low-shot eval.", fusar_root)
            else:
                fusar_10_on_disk = [
                    c for c in FUSAR_10_CLASSES if (Path(fusar_root) / c).is_dir()
                ]
                ls_dataset = FUSARShipDataset(
                    root=fusar_root, image_size=(128, 128), classes=fusar_10_on_disk,
                )
                run_low_shot_eval(
                    backbone=backbone,
                    dataset=ls_dataset,
                    dataset_name="fusar",
                    n_shot_list=n_shot_list,
                    n_seeds=n_seeds,
                    device=device_t,
                    results_dir=results_dir,
                )

        elif args.dataset == "mstar":
            mstar_root_ls = args.mstar_root or str(Path(args.data_root) / "mstar")
            if not Path(mstar_root_ls).exists():
                logger.error("MSTAR root '%s' not found; skipping.", mstar_root_ls)
            else:
                mstar_ds_full = MSTARDataset(root=mstar_root_ls, image_size=(128, 128))
                vehicle_cls_on_disk = [
                    c for c in MSTAR_VEHICLE_CLASSES if c in mstar_ds_full.classes
                ]
                if not vehicle_cls_on_disk:
                    logger.error("No MSTAR vehicle classes found; skipping.")
                else:
                    mstar_ls = MSTARDataset.__new__(MSTARDataset)
                    mstar_ls.root       = mstar_ds_full.root
                    mstar_ls.image_size = mstar_ds_full.image_size
                    mstar_ls.transform  = mstar_ds_full.transform
                    mstar_ls.samples    = [
                        s for s in mstar_ds_full.samples
                        if s["class_name"] in vehicle_cls_on_disk
                    ]
                    mstar_ls.classes      = vehicle_cls_on_disk
                    mstar_ls.class_to_idx = {
                        c: i for i, c in enumerate(vehicle_cls_on_disk)
                    }
                    for s in mstar_ls.samples:
                        s["label"] = mstar_ls.class_to_idx[s["class_name"]]
                    run_low_shot_eval(
                        backbone=backbone,
                        dataset=mstar_ls,
                        dataset_name="mstar",
                        n_shot_list=n_shot_list,
                        n_seeds=n_seeds,
                        device=device_t,
                        results_dir=results_dir,
                    )

        elif args.dataset == "opensarship":
            opensarship_root = (
                args.opensarship_root
                or str(Path(args.data_root) / "open_sar" / "OpenSARShip2")
            )
            try:
                from prepare_opensarship import OpenSARShipDataset
                splits_csv = Path("data") / "opensarship_splits.csv"
                if splits_csv.exists():
                    oss_dataset = OpenSARShipDataset(
                        splits_csv=str(splits_csv),
                        data_root=opensarship_root,
                        split="train",
                        polarization="VV",
                        image_size=(128, 128),
                    )
                else:
                    logger.warning(
                        "OpenSARShip splits CSV not found at %s. "
                        "Run prepare_opensarship.py first.", splits_csv,
                    )
                    oss_dataset = None

                if oss_dataset is not None and len(oss_dataset) > 0:
                    run_low_shot_eval(
                        backbone=backbone,
                        dataset=oss_dataset,
                        dataset_name="opensarship",
                        n_shot_list=n_shot_list,
                        n_seeds=n_seeds,
                        device=device_t,
                        results_dir=results_dir,
                    )
                else:
                    logger.error("OpenSARShip dataset is empty — skipping.")
            except ImportError:
                logger.error("prepare_opensarship.py not found — run it first.")

    # ---- Save final config ----------------------------------------------
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.training.checkpoint_dir, "final_config.json"))
    logger.info("Config saved to %s/final_config.json", cfg.training.checkpoint_dir)
    logger.info("ProtoEvoNet pipeline complete.")


if __name__ == "__main__":
    main()
