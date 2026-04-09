"""
Phase 1 Training: HRSID contrastive pre-training.

Objectives
----------
1. **SimCLR-style instance contrastive learning**: two randomly augmented
   views of the same image are treated as a positive pair.  This avoids the
   InfoNCE collapse that occurs when all batch samples share the same class
   label (e.g., every HRSID chip is a ship).
2. **AGCP physics-guided hard negatives**: pairs that are physically similar
   (small Fisher–Rao distance between Gamma params) but differ in embedding
   space receive an extra repulsion penalty.
3. Backbone is **frozen** at the end of Phase 1 for Phase 2.

Why SimCLR pairs?
-----------------
HRSID is a ship-chip dataset — nearly every image contains a vessel.  With
single-class batches the classic InfoNCE reduces to
    loss = -(log_sum_positives - log_sum_all) = 0
because every non-self pair is positive.  Instance-level pairing sidesteps
this by making the positive signal about *which image*, not *which class*.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from backbone.srab import SRABBackbone
from physics.gamma_stats import GammaStatisticsModule
from physics.dsar import DSARModule
from physics.snr import SNREstimationModule
from .datasets import HRSIDDataset
from .losses import AGCPLoss
from utils.config import ProtoEvoNetConfig
from utils.checkpoint import CheckpointManager
from utils.logging_utils import AverageMeter, MetricLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAR-appropriate augmentation
# ---------------------------------------------------------------------------

def sar_augment(images: torch.Tensor) -> torch.Tensor:
    """
    Apply randomised augmentations suitable for SAR intensity images.

    Operations (each applied independently per image):
      * Random horizontal flip
      * Random vertical flip
      * Random 0 / 90 / 180 / 270-degree rotation
      * Multiplicative speckle noise  (SAR noise is multiplicative)
      * Random brightness scaling in [0.8, 1.2]

    Parameters
    ----------
    images:
        Shape ``(B, 1, H, W)``, float32 in [0, 1].

    Returns
    -------
    torch.Tensor
        Augmented batch, same shape and device as *images*.
    """
    B, C, H, W = images.shape
    out = images.clone()

    for i in range(B):
        img = out[i]  # (1, H, W)

        # Horizontal flip
        if random.random() > 0.5:
            img = img.flip(-1)

        # Vertical flip
        if random.random() > 0.5:
            img = img.flip(-2)

        # 90-degree rotation (0/1/2/3 quarter turns)
        k = random.randint(0, 3)
        if k:
            img = torch.rot90(img, k, dims=(-2, -1))

        # Multiplicative speckle noise: I_noisy = I * exp(N(0, σ²))
        sigma = random.uniform(0.0, 0.15)
        noise = torch.randn_like(img) * sigma
        img = img * torch.exp(noise)

        # Random brightness scale
        scale = random.uniform(0.8, 1.2)
        img = img * scale

        # Clamp to valid range
        img = img.clamp(0.0, 1.0)
        out[i] = img

    return out


# ---------------------------------------------------------------------------
# Optimiser / scheduler builders
# ---------------------------------------------------------------------------

def build_optimizer_phase1(
    backbone: SRABBackbone,
    gamma: GammaStatisticsModule,
    dsar: DSARModule,
    snr: SNREstimationModule,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build a single AdamW optimiser over all Phase 1 parameters."""
    params = (
        list(backbone.parameters())
        + list(gamma.parameters())
        + list(dsar.parameters())
        + list(snr.parameters())
    )
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_scheduler_phase1(
    optimizer: torch.optim.Optimizer,
    cfg: ProtoEvoNetConfig,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build LR scheduler for Phase 1."""
    if cfg.training.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.phase1_epochs
        )
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.training.step_lr_step,
        gamma=cfg.training.step_lr_gamma,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_phase1(
    backbone: SRABBackbone,
    gamma_module: GammaStatisticsModule,
    dsar_module: DSARModule,
    snr_module: SNREstimationModule,
    cfg: ProtoEvoNetConfig,
    hrsid_root: str,
    resume_from: Optional[str] = None,
) -> SRABBackbone:
    """
    Execute Phase 1 training.

    Each optimiser step:
      1. Load a batch of B images.
      2. Create two independently augmented views → combined batch of 2B images.
      3. Encode with backbone → (2B, D) embeddings.
      4. Build instance labels [0…B-1, 0…B-1] so that view₁[i] and view₂[i]
         are the positive pair for anchor i.
      5. Compute AGCP InfoNCE loss (now guaranteed non-zero).

    Parameters
    ----------
    backbone:
        SRAB backbone to train.
    gamma_module, dsar_module, snr_module:
        Physics modules trained jointly with the backbone.
    cfg:
        Full system config.
    hrsid_root:
        Path to HRSID dataset root (``data/hrsid/``).
    resume_from:
        Path to a Phase 1 checkpoint to resume from.

    Returns
    -------
    SRABBackbone
        The trained (and frozen) backbone.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_cfg = cfg.training

    # ---- Dataset --------------------------------------------------------
    full_dataset = HRSIDDataset(
        root=hrsid_root,
        split="train",
        image_size=(128, 128),
    )
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.phase1_batch_size,
        shuffle=True,
        num_workers=train_cfg.phase1_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.phase1_batch_size,
        shuffle=False,
        num_workers=train_cfg.phase1_num_workers,
    )

    logger.info(
        "Phase 1 dataset: %d train / %d val images",
        train_size, val_size,
    )

    # ---- Model setup ---------------------------------------------------
    backbone = backbone.to(device)
    gamma_module = gamma_module.to(device)
    dsar_module = dsar_module.to(device)
    snr_module = snr_module.to(device)

    optimizer = build_optimizer_phase1(
        backbone, gamma_module, dsar_module, snr_module,
        lr=train_cfg.phase1_lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = build_scheduler_phase1(optimizer, cfg)
    loss_fn = AGCPLoss(
        temperature=train_cfg.agcp_temperature,
        hard_negative_weight=0.5,
        margin=train_cfg.agcp_margin,
        use_fisher_rao=cfg.ablation.use_fisher_rao,
    ).to(device)

    ckpt_manager = CheckpointManager(
        directory=train_cfg.checkpoint_dir + "/phase1",
        metric="val_loss",
        higher_is_better=False,
    )

    start_epoch = 0
    if resume_from:
        from utils.checkpoint import load_checkpoint
        payload = load_checkpoint(
            resume_from,
            state_dict_targets={
                "backbone": backbone,
                "optimizer": optimizer,
            },
            strict=False,
            device=str(device),
        )
        start_epoch = payload.get("epoch", 0) + 1
        logger.info("Resumed Phase 1 from epoch %d", start_epoch)

    with MetricLogger(train_cfg.log_dir, run_name="phase1") as ml:
        for epoch in range(start_epoch, train_cfg.phase1_epochs):
            # ---- Train loop ------------------------------------------
            backbone.train()
            gamma_module.train()
            dsar_module.train()
            snr_module.train()

            loss_meter = AverageMeter("loss")

            for step, (images, _labels, _ids) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)  # (B, 1, H, W)
                B = images.size(0)

                # Two independently augmented views
                view1 = sar_augment(images)   # (B, 1, H, W)
                view2 = sar_augment(images)   # (B, 1, H, W)
                both = torch.cat([view1, view2], dim=0)  # (2B, 1, H, W)

                # Encode
                embeddings = backbone(both)  # (2B, D)

                # Instance-level labels: view1[i] and view2[i] are the same instance
                instance_labels = torch.cat([
                    torch.arange(B, device=device),
                    torch.arange(B, device=device),
                ])  # (2B,)

                # Physics descriptors on view1 only (no grad needed)
                with torch.no_grad():
                    k, theta = gamma_module.global_stats(both)  # (2B,)

                # AGCP loss with instance-level positive pairs
                loss, info = loss_fn(embeddings, instance_labels, k, theta)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters())
                    + list(gamma_module.parameters()),
                    max_norm=train_cfg.grad_clip,
                )
                optimizer.step()

                loss_meter.update(loss.item(), n=B)

                if step % train_cfg.log_interval == 0:
                    info["epoch"] = epoch
                    info["step"] = step
                    ml.log(info)
                    logger.debug(
                        "Phase1 [%d/%d] step=%d  %s",
                        epoch, train_cfg.phase1_epochs, step, loss_meter,
                    )

            scheduler.step()

            # ---- Validation -------------------------------------------
            if epoch % train_cfg.eval_interval == 0:
                val_loss = _validate_phase1(
                    backbone, gamma_module, loss_fn, val_loader, device
                )
                ml.log({"val_loss": val_loss, "epoch": epoch})
                logger.info(
                    "Phase1 epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch, train_cfg.phase1_epochs,
                    loss_meter.avg, val_loss,
                )

                ckpt_manager.save(
                    epoch=epoch,
                    state_dicts={
                        "backbone": backbone.state_dict(),
                        "gamma": gamma_module.state_dict(),
                        "dsar": dsar_module.state_dict(),
                        "snr": snr_module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    metrics={"val_loss": val_loss},
                )

    # Freeze backbone for Phase 2
    for param in backbone.parameters():
        param.requires_grad_(False)
    backbone.eval()
    logger.info("Phase 1 complete — backbone frozen.")
    return backbone


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate_phase1(
    backbone: SRABBackbone,
    gamma_module: GammaStatisticsModule,
    loss_fn: AGCPLoss,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Run one validation pass and return average SimCLR contrastive loss."""
    backbone.eval()
    gamma_module.eval()
    meter = AverageMeter("val_loss")
    for images, _labels, _ids in loader:
        images = images.to(device)
        B = images.size(0)
        view1 = sar_augment(images)
        view2 = sar_augment(images)
        both = torch.cat([view1, view2], dim=0)
        emb = backbone(both)
        k, theta = gamma_module.global_stats(both)
        instance_labels = torch.cat([
            torch.arange(B, device=device),
            torch.arange(B, device=device),
        ])
        loss, _ = loss_fn(emb, instance_labels, k, theta)
        meter.update(loss.item(), n=B)
    return meter.avg
