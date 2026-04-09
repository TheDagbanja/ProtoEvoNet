#!/usr/bin/env python3
"""
eval_enrolment_improved.py

Incremental enrolment with data augmentation and prototype fine‑tuning.

Usage:
    python eval_enrolment_improved.py --dataset opensarship \\
        --system-ckpt checkpoints/phase2/best.pt --augment --fine-tune
"""

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

# ---- ProtoEvoNet imports ----
from utils.config import ProtoEvoNetConfig
from utils.logging_utils import setup_logging
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
from training.datasets import MSTARDataset
from prepare_opensarship import OpenSARShipDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build system (same as before)
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
        num_classes=16,
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
        logger.info("Loaded HBS with shape‑filtering: missing=%d, unexpected=%d",
                    len(missing), len(unexpected))

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
# Data augmentation functions
# ---------------------------------------------------------------------------

def add_speckle_noise(image: torch.Tensor, mean: float = 1.0, variance: float = 0.1) -> torch.Tensor:
    """Add multiplicative Gamma speckle noise."""
    gamma = torch.distributions.Gamma(1/variance, 1/variance)
    noise = gamma.sample(image.shape).to(image.device)
    return image * noise


def augment_image(image: torch.Tensor) -> List[torch.Tensor]:
    """Generate augmented versions of a single image."""
    aug_list = [image]  # original
    # rotation (90, 180, 270)
    for k in [1, 2, 3]:
        aug_list.append(torch.rot90(image, k, dims=(-2, -1)))
    # horizontal flip
    aug_list.append(torch.flip(image, dims=(-1,)))
    # speckle noise (two variants)
    aug_list.append(add_speckle_noise(image, variance=0.05))
    aug_list.append(add_speckle_noise(image, variance=0.15))
    return aug_list


def augment_support_set(images: torch.Tensor) -> torch.Tensor:
    """Augment a batch of support images (K, C, H, W). Returns (K * n_aug, C, H, W)."""
    augmented = []
    for i in range(images.shape[0]):
        augmented.extend(augment_image(images[i]))
    return torch.stack(augmented)


# ---------------------------------------------------------------------------
# Prototype fine‑tuning
# ---------------------------------------------------------------------------

def fine_tune_prototype(
    engine: ProtoEvoNetInferenceEngine,
    class_id: int,
    support_embeddings: torch.Tensor,
    val_embeddings: Optional[torch.Tensor] = None,
    steps: int = 5,
    lr: float = 0.01,
) -> None:
    """
    Refine the prototype for a given class using gradient descent.

    Uses either a validation set (if provided) or the support embeddings themselves
    with a simple cosine similarity maximisation loss.

    Parameters
    ----------
    engine:
        Inference engine containing OWMS and HBS.
    class_id:
        Class ID already enrolled.
    support_embeddings:
        Embeddings of support images (already used for enrolment).
    val_embeddings:
        Optional held‑out embeddings for validation (to avoid overfitting).
    steps:
        Number of gradient steps.
    lr:
        Learning rate for prototype update.
    """
    entry = engine.owms.get(class_id)
    if entry is None:
        logger.warning("Class %d not found in OWMS, cannot fine‑tune.", class_id)
        return

    prototype = entry.prototype.clone().detach().requires_grad_(True)
    if val_embeddings is None:
        # Use support embeddings, maximise cosine similarity
        target = support_embeddings
    else:
        target = val_embeddings

    optimizer = optim.SGD([prototype], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        # Cosine similarity between prototype and target embeddings
        sim = F.cosine_similarity(prototype.unsqueeze(0), target, dim=1)  # (N,)
        loss = -sim.mean()  # maximise similarity
        loss.backward()
        optimizer.step()
        # Re‑normalise prototype
        prototype.data = F.normalize(prototype.data, p=2, dim=0)

    # Update OWMS with refined prototype
    entry.prototype = prototype.detach()
    # Also update the HBS's internal state? Not needed for recognition, OWMS is source of truth.
    logger.info("Fine‑tuned prototype for class %d, final similarity = %.4f",
                class_id, sim.mean().item())


# ---------------------------------------------------------------------------
# Enrolment evaluation with improvements
# ---------------------------------------------------------------------------

def evaluate_enrolment(
    engine: ProtoEvoNetInferenceEngine,
    dataset,
    dataset_name: str,
    n_enrol_per_class: int = 10,
    augment: bool = False,
    fine_tune: bool = False,
    results_dir: str = "results",
):
    os.makedirs(results_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Enrolment evaluation on {dataset_name}")
    logger.info(f"Support per class: {n_enrol_per_class} images")
    if augment:
        logger.info("Data augmentation ENABLED (each support image → 6 variants)")
    if fine_tune:
        logger.info("Prototype fine‑tuning ENABLED (5 steps)")
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

        # Split into support (first n_enrol) and test (rest)
        support_idx = indices[:n_enrol_per_class]
        query_idx = indices[n_enrol_per_class:]

        # Optionally hold out one support image for validation (if fine‑tuning)
        if fine_tune and len(support_idx) > 1:
            val_idx = [support_idx[-1]]
            support_idx = support_idx[:-1]
        else:
            val_idx = []

        # Load support images
        support_images = torch.stack([dataset[i][0] for i in support_idx])

        # Augment support set if requested
        if augment:
            support_images = augment_support_set(support_images)
            logger.debug("Augmented support set size for %s: %d", cls, support_images.shape[0])

        # Enrol
        result = engine.enrol(support_images, label=cls)
        enrolled_classes.append(cls)
        logger.info(
            "Enrolled class %s (id=%d) with %d images (augmented=%d), uncertainty=%.4f",
            cls, result.class_id, len(support_idx), support_images.shape[0] - len(support_idx),
            result.uncertainty
        )

        # Fine‑tune prototype if requested
        if fine_tune:
            # Compute embeddings of support (original, not augmented) and validation images
            with torch.no_grad():
                support_embs = engine.backbone(torch.stack([dataset[i][0] for i in support_idx]).to(engine.device))
                support_embs = F.normalize(support_embs, p=2, dim=1)
                if val_idx:
                    val_embs = engine.backbone(torch.stack([dataset[i][0] for i in val_idx]).to(engine.device))
                    val_embs = F.normalize(val_embs, p=2, dim=1)
                else:
                    val_embs = None
            fine_tune_prototype(engine, result.class_id, support_embs, val_embs, steps=5, lr=0.01)

        # Evaluate on test set
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

    logger.info("\nRecognition results after enrolment:")
    logger.info("  %-20s %8s", "Class", "Acc%")
    for cls, acc in per_class_metrics.items():
        logger.info("  %-20s %8.1f", cls, acc)
    logger.info("  %-20s %8.1f", "OVERALL", overall_acc)

    # Save CSV
    csv_path = Path(results_dir) / f"enrolment_{dataset_name}_{n_enrol_per_class}shot.csv"
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
# MSTAR wrapper (same as before)
# ---------------------------------------------------------------------------

class MSTAREnrolmentDataset:
    def __init__(self, mstar_dataset, class_filter=None):
        self.original = mstar_dataset
        self._items = []
        for idx in range(len(self.original)):
            img, orig_label, cls = self.original[idx]
            if class_filter is None or cls in class_filter:
                self._items.append((idx, cls, orig_label))
        self.classes = sorted(set(cls for _, cls, _ in self._items))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self._items = [(idx, self.class_to_idx[cls], cls) for (idx, cls, _) in self._items]
        self._class_to_indices = {c: [] for c in self.classes}
        for i, (_, _, cls) in enumerate(self._items):
            self._class_to_indices[cls].append(i)

    def get_class_samples(self, cls):
        return self._class_to_indices.get(cls, [])

    def __getitem__(self, idx):
        orig_idx, label, class_name = self._items[idx]
        img, _, _ = self.original[orig_idx]
        return img, label, class_name

    def __len__(self):
        return len(self._items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Incremental enrolment with augmentation and fine‑tuning")
    parser.add_argument("--dataset", choices=["opensarship", "mstar"], required=True)
    parser.add_argument("--system-ckpt", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-shot", type=int, default=10)
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation during enrolment")
    parser.add_argument("--fine-tune", action="store_true", help="Apply prototype fine‑tuning after enrolment")
    parser.add_argument("--test-novelty", action="store_true", help="Also test novelty on OpenSARShip novel classes")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
        evaluate_enrolment(
            engine, dataset, "opensarship",
            n_enrol_per_class=args.n_shot,
            augment=args.augment,
            fine_tune=args.fine_tune,
        )
        if args.test_novelty:
            novel_csv = Path(args.data_root) / "opensarship_novel_classes.csv"
            if novel_csv.exists():
                novel_dataset = OpenSARShipDataset(
                    csv_path=str(novel_csv), split="all", image_size=(128, 128),
                    polarization="VV", normalize=True,
                )
                logger.info(f"Testing novelty on {len(novel_dataset)} novel images")
                correct = 0
                for i in range(len(novel_dataset)):
                    img, _, _ = novel_dataset[i]
                    result = engine.detect_novelty(img.unsqueeze(0))
                    if result.is_novel:
                        correct += 1
                logger.info("Novelty detection accuracy: %.2f%%", correct / len(novel_dataset) * 100)
            else:
                logger.warning("Novel classes CSV not found")

    elif args.dataset == "mstar":
        mstar_root = Path(args.data_root) / "mstar"
        if not mstar_root.exists():
            logger.error("MSTAR root not found at %s", mstar_root)
            sys.exit(1)
        from main import MSTAR_VEHICLE_CLASSES
        full_dataset = MSTARDataset(root=str(mstar_root), image_size=(128, 128))
        vehicle_classes_present = [c for c in MSTAR_VEHICLE_CLASSES if c in full_dataset.classes]
        if not vehicle_classes_present:
            logger.error("No MSTAR vehicle classes found.")
            sys.exit(1)
        dataset = MSTAREnrolmentDataset(full_dataset, class_filter=vehicle_classes_present)
        logger.info(f"MSTAR dataset: {len(dataset)} images, classes={dataset.classes}")
        evaluate_enrolment(
            engine, dataset, "mstar",
            n_enrol_per_class=args.n_shot,
            augment=args.augment,
            fine_tune=args.fine_tune,
        )


if __name__ == "__main__":
    main()