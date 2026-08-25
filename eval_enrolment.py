#!/usr/bin/env python3
"""
eval_enrolment.py

True incremental enrolment evaluation for OpenSARShip and MSTAR.

Uses the full ProtoEvoNet inference engine (engine.enrol) to:
  1. Enrol new classes from a target dataset (support split)
  2. Recognise query images (query split) using the evolved prototypes

This tests the core claim: the model can evolve new prototypes after deployment
and use them for open-world recognition, without any retraining.

Usage:
    # Enrol OpenSARShip known classes and evaluate
    python eval_enrolment.py --dataset opensarship \\
        --system-ckpt checkpoints/phase2/best.pt \\
        --data-root data --device cuda

    # Enrol MSTAR vehicle classes and evaluate
    python eval_enrolment.py --dataset mstar \\
        --system-ckpt checkpoints/phase2/best.pt \\
        --data-root data --device cuda

    # Also test novelty detection on OpenSARShip novel classes (if needed)
    python eval_enrolment.py --dataset opensarship --test-novelty \\
        --system-ckpt checkpoints/phase2/best.pt
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

# ---- ProtoEvoNet imports ----
from utils.config import ProtoEvoNetConfig
from utils.logging_utils import setup_logging
from utils.checkpoint import load_checkpoint
from backbone.srab import SRABBackbone
from physics.gamma_stats import GammaStatisticsModule
from physics.dsar import DSARModule
from physics.snr import SNREstimationModule
from hippocampus import HippocampalBindingSystem
from memory.owms import OnlineWorkingMemoryStore
from memory.pim import PrototypeInterferenceMonitor
from memory.tpe import TemporalPrototypeEvolution
from inference.novelty import NoveltyDetector, PlattCalibrator
from inference.engine import ProtoEvoNetInferenceEngine, RecognitionResult
from training.datasets import FUSARShipDataset, MSTARDataset
from prepare_opensarship import OpenSARShipDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build system (copied from main.py)
# ---------------------------------------------------------------------------

def build_system(cfg: ProtoEvoNetConfig):
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
    hbs = HippocampalBindingSystem(
        cfg=cfg.hippocampus,
        num_classes=16,   # as trained (FUSAR has 16 classes)
        ais_dim=0,
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
    novelty_detector = NoveltyDetector(calibrator=platt, threshold=cfg.inference.novelty_threshold)

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
        "hbs": hbs,
        "owms": owms,
        "engine": engine,
    }


def load_system_with_shape_filtering(
    checkpoint_path: str,
    cfg: ProtoEvoNetConfig,
    device: torch.device,
):
    """Build system, then load checkpoint with shape‑filtering for HBS."""
    system = build_system(cfg)
    backbone = system["backbone"]
    hbs = system["hbs"]
    dsar = system["engine"].dsar
    platt = system["engine"].novelty_detector.calibrator

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "backbone" in checkpoint:
        backbone.load_state_dict(checkpoint["backbone"], strict=False)
        logger.info("Loaded backbone from checkpoint")

    if "hbs" in checkpoint:
        hbs_state = checkpoint["hbs"]
        filtered = {}
        for name, param in hbs_state.items():
            if name in hbs.state_dict():
                target_shape = hbs.state_dict()[name].shape
                if param.shape == target_shape:
                    filtered[name] = param
                else:
                    logger.warning(
                        "Skipping HBS key '%s' (checkpoint shape %s vs current %s)",
                        name, param.shape, target_shape
                    )
        missing, unexpected = hbs.load_state_dict(filtered, strict=False)
        logger.info(
            "Loaded HBS with shape‑filtering: missing=%d, unexpected=%d",
            len(missing), len(unexpected)
        )

    if "dsar" in checkpoint:
        dsar.load_state_dict(checkpoint["dsar"], strict=False)
    if "platt" in checkpoint:
        platt.load_state_dict(checkpoint["platt"], strict=False)

    backbone.to(device)
    hbs.to(device)
    dsar.to(device)
    platt.to(device)
    system["engine"].device = device
    return system


# ---------------------------------------------------------------------------
# Enrolment + recognition evaluation
# ---------------------------------------------------------------------------

def evaluate_enrolment(
    engine: ProtoEvoNetInferenceEngine,
    dataset,  # must have .classes, .get_class_samples, __getitem__
    dataset_name: str,
    n_enrol_per_class: int = 10,
    results_dir: str = "results",
):
    """
    Enrol each class using engine.enrol() on its support images,
    then recognise query images.
    """
    os.makedirs(results_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Enrolment evaluation on {dataset_name}")
    logger.info(f"Using {n_enrol_per_class} support images per class")
    logger.info("=" * 60)

    enrolled_classes = []
    all_preds = []
    all_labels = []
    per_class_metrics = {}

    for cls in dataset.classes:
        indices = dataset.get_class_samples(cls)
        if len(indices) < n_enrol_per_class + 1:
            logger.warning(
                "Class '%s' has only %d images – need at least %d+1, skipping",
                cls, len(indices), n_enrol_per_class
            )
            continue

        # Enrol with first n_enrol_per_class images
        support_idx = indices[:n_enrol_per_class]
        support_images = torch.stack([dataset[i][0] for i in support_idx])
        result = engine.enrol(support_images, label=cls)
        enrolled_classes.append(cls)
        logger.info(
            "Enrolled class %s (id=%d) with %d images, uncertainty=%.4f",
            cls, result.class_id, n_enrol_per_class, result.uncertainty
        )

        # Evaluate on remaining images
        query_idx = indices[n_enrol_per_class:]
        for idx in query_idx:
            image, label_int, _ = dataset[idx]
            rec: RecognitionResult = engine.recognise(image.unsqueeze(0))
            all_preds.append(rec.predicted_label)
            all_labels.append(cls)

    # Compute per‑class accuracy
    n_correct = 0
    n_total = len(all_labels)
    for cls in enrolled_classes:
        cls_mask = [gt == cls for gt in all_labels]
        if not any(cls_mask):
            continue
        cls_preds = [p for i, p in enumerate(all_preds) if all_labels[i] == cls]
        cls_acc = sum(p == cls for p in cls_preds) / len(cls_preds) * 100.0
        per_class_metrics[cls] = cls_acc

    overall_acc = sum(p == gt for p, gt in zip(all_preds, all_labels)) / n_total * 100.0

    # Log results
    logger.info("\nRecognition results after enrolment:")
    logger.info("  %-20s %8s", "Class", "Acc%")
    for cls, acc in per_class_metrics.items():
        logger.info("  %-20s %8.1f", cls, acc)
    logger.info("  %-20s %8.1f", "OVERALL", overall_acc)

    # Save CSV
    csv_path = Path(results_dir) / f"enrolment_{dataset_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "accuracy_%", "n_support", "n_test"])
        for cls, acc in per_class_metrics.items():
            n_support = n_enrol_per_class
            n_test = sum(1 for gt in all_labels if gt == cls)
            writer.writerow([cls, f"{acc:.2f}", n_support, n_test])
    logger.info("Results saved to %s", csv_path)

    return overall_acc, per_class_metrics


# ---------------------------------------------------------------------------
# MSTAR wrapper that provides the expected interface
# ---------------------------------------------------------------------------

class MSTAREnrolmentDataset:
    """
    Wrapper around MSTARDataset to provide:
      - .classes
      - .get_class_samples(cls) -> list of indices in this wrapper
      - __getitem__(idx) -> (image, label, class_name) by delegating to original dataset
    """
    def __init__(self, mstar_dataset, class_filter=None):
        self.original = mstar_dataset
        # Build list of (original_index, class_name, original_label)
        self._items = []
        for idx in range(len(self.original)):
            img, orig_label, cls = self.original[idx]  # original returns (image, label, class_name)
            if class_filter is None or cls in class_filter:
                self._items.append((idx, cls, orig_label))
        # Determine classes present after filtering
        self.classes = sorted(set(cls for _, cls, _ in self._items))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # Rebuild items with mapped label (0..C-1)
        self._items = [(idx, self.class_to_idx[cls], cls) for (idx, cls, _) in self._items]
        # Build per-class indices mapping
        self._class_to_indices = {c: [] for c in self.classes}
        for i, (_, _, cls) in enumerate(self._items):
            self._class_to_indices[cls].append(i)

    def get_class_samples(self, cls):
        return self._class_to_indices.get(cls, [])

    def __getitem__(self, idx):
        orig_idx, label, class_name = self._items[idx]
        img, _, _ = self.original[orig_idx]   # get image (ignore original label and class_name)
        return img, label, class_name

    def __len__(self):
        return len(self._items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="True incremental enrolment evaluation")
    parser.add_argument("--dataset", choices=["opensarship", "mstar"], required=True)
    parser.add_argument("--system-ckpt", type=str, required=True,
                        help="Path to checkpoint (phase2 best.pt)")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-shot", type=int, default=10,
                        help="Number of support images per class for enrolment")
    parser.add_argument("--test-novelty", action="store_true",
                        help="Also test novelty detection on novel classes (OpenSARShip only)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load config and system
    cfg = ProtoEvoNetConfig()
    cfg.device = str(device)
    system = load_system_with_shape_filtering(args.system_ckpt, cfg, device)
    engine = system["engine"]

    if args.dataset == "opensarship":
        splits_csv = Path(args.data_root) / "opensarship_splits.csv"
        if not splits_csv.exists():
            logger.error("OpenSARShip splits CSV not found. Run prepare_opensarship.py first.")
            sys.exit(1)
        dataset = OpenSARShipDataset(
            csv_path=str(splits_csv),
            split="all",
            image_size=(128, 128),
            polarization="VV",
            normalize=True,
            stats_csv=str(Path(args.data_root) / "opensarship_stats_VV.csv")
        )
        logger.info(f"OpenSARShip dataset: {len(dataset)} images, classes={dataset.classes}")
        evaluate_enrolment(engine, dataset, "opensarship", n_enrol_per_class=args.n_shot)

        if args.test_novelty:
            novel_csv = Path(args.data_root) / "opensarship_novel_classes.csv"
            if novel_csv.exists():
                novel_dataset = OpenSARShipDataset(
                    csv_path=str(novel_csv),
                    split="all",
                    image_size=(128, 128),
                    polarization="VV",
                    normalize=True,
                )
                logger.info(f"Testing novelty on {len(novel_dataset)} novel images")
                novel_correct = 0
                for i in range(len(novel_dataset)):
                    img, _, _ = novel_dataset[i]
                    result = engine.detect_novelty(img.unsqueeze(0))
                    if result.is_novel:
                        novel_correct += 1
                novel_acc = novel_correct / len(novel_dataset) * 100.0
                logger.info("Novelty detection accuracy on novel classes: %.2f%%", novel_acc)
            else:
                logger.warning("Novel classes CSV not found, skipping novelty test")

    elif args.dataset == "mstar":
        mstar_root = Path(args.data_root) / "mstar"
        if not mstar_root.exists():
            logger.error("MSTAR root not found at %s", mstar_root)
            sys.exit(1)

        from main import MSTAR_VEHICLE_CLASSES
        full_dataset = MSTARDataset(root=str(mstar_root), image_size=(128, 128))
        # Filter to only the 10 vehicle classes
        vehicle_classes_present = [c for c in MSTAR_VEHICLE_CLASSES if c in full_dataset.classes]
        if not vehicle_classes_present:
            logger.error("No MSTAR vehicle classes found. Available: %s", full_dataset.classes)
            sys.exit(1)

        dataset = MSTAREnrolmentDataset(full_dataset, class_filter=vehicle_classes_present)
        logger.info(f"MSTAR dataset: {len(dataset)} images, classes={dataset.classes}")
        evaluate_enrolment(engine, dataset, "mstar", n_enrol_per_class=args.n_shot)
def evaluate_enrolment_with_samples(
    engine: ProtoEvoNetInferenceEngine,
    dataset,
    dataset_name: str,
    n_enrol_per_class: int = 10,
    results_dir: str = "results",
):
    """
    Enrol each class using engine.enrol(), then recognise all query images.
    Returns (overall_accuracy, per_class_accuracy_dict, list_of_samples).
    Each sample is (image_tensor, true_label, predicted_label, confidence).
    """
    os.makedirs(results_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Enrolment evaluation (with samples) on {dataset_name}")
    logger.info(f"Support per class: {n_enrol_per_class} images")
    logger.info("=" * 60)

    enrolled_classes = []
    all_preds = []
    all_labels = []
    samples = []   # (image, true_label, pred_label, confidence)

    for cls in dataset.classes:
        indices = dataset.get_class_samples(cls)
        if len(indices) < n_enrol_per_class + 1:
            logger.warning(
                "Class '%s' has only %d images – need at least %d+1, skipping",
                cls, len(indices), n_enrol_per_class
            )
            continue

        support_idx = indices[:n_enrol_per_class]
        query_idx = indices[n_enrol_per_class:]

        support_images = torch.stack([dataset[i][0] for i in support_idx])
        result = engine.enrol(support_images, label=cls)
        enrolled_classes.append(cls)
        logger.info("Enrolled class %s (id=%d) with %d images, uncertainty=%.4f",
                    cls, result.class_id, n_enrol_per_class, result.uncertainty)

        for idx in query_idx:
            image, _, _ = dataset[idx]
            rec = engine.recognise(image.unsqueeze(0))
            pred = rec.predicted_label
            conf = rec.confidence
            all_preds.append(pred)
            all_labels.append(cls)
            samples.append((image.cpu(), cls, pred, conf))

    # Compute overall accuracy
    correct = sum(p == gt for p, gt in zip(all_preds, all_labels))
    total = len(all_labels)
    overall_acc = correct / total * 100.0

    # Per‑class accuracy
    per_class_acc = {}
    for cls in enrolled_classes:
        mask = [gt == cls for gt in all_labels]
        if not any(mask):
            continue
        cls_correct = sum(p == cls for p, gt in zip(all_preds, all_labels) if gt == cls)
        cls_total = sum(mask)
        per_class_acc[cls] = cls_correct / cls_total * 100.0

    logger.info("Overall accuracy: %.2f%% (%d/%d)", overall_acc, correct, total)
    return overall_acc, per_class_acc, samples

if __name__ == "__main__":
    main()
