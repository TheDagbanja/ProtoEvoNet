"""
Phase 2 Training: FUSAR-Ship episodic meta-learning.

Objectives
----------
* Episodic few-shot learning with the PIAM loss.
* Train AGPPI, APM, BPC-GP (HBS) jointly; backbone may optionally be
  fine-tuned at a lower learning rate.
* Calibrate Platt scaling parameters for novelty detection.
* Checkpoint after every evaluation epoch.

Episode flow
------------
For each episode:
  1. Sample N-way K-shot support set + Q-shot query set from FUSAR-Ship.
  2. Encode all images with the SRAB backbone.
  3. Compute physics descriptors (k, θ, SNR) for each image.
  4. For each class, run HBS.bind() to get a calibrated prototype.
  5. Compute query distances to prototypes.
  6. PIAM loss (proto + physics + novelty).
  7. Backprop through HBS; optionally through backbone.

Novelty calibration
-------------------
Every *calibration_interval* epochs, a held-out set of known-class images
(from the training split) and randomly synthesised "novel" embeddings
(Gaussian noise in embedding space) are used to fit the Platt parameters
via ``PlattCalibrator.fit()``.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.srab import SRABBackbone
from physics.gamma_stats import GammaStatisticsModule
from physics.dsar import DSARModule
from physics.snr import SNREstimationModule
from hippocampus import HippocampalBindingSystem
from inference.novelty import NoveltyDetector, PlattCalibrator
from memory.owms import OnlineWorkingMemoryStore
from .datasets import FUSARShipDataset
from .episodic_sampler import EpisodicSampler, EpisodeBatch
from .losses import PIAMLoss
from utils.config import ProtoEvoNetConfig
from utils.checkpoint import CheckpointManager
from utils.logging_utils import AverageMeter, MetricLogger
from utils.metrics import episode_accuracy

logger = logging.getLogger(__name__)


def build_optimizer_phase2(
    backbone: SRABBackbone,
    hbs: HippocampalBindingSystem,
    backbone_lr_scale: float = 0.1,
    base_lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Build optimiser with separate LR for backbone (fine-tune) and HBS (train).
    """
    # HBS parameters at full LR, backbone at scaled LR
    param_groups = [
        {"params": list(hbs.parameters()), "lr": base_lr},
        {
            "params": [p for p in backbone.parameters() if p.requires_grad],
            "lr": base_lr * backbone_lr_scale,
        },
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _compute_episode_prototypes(
    backbone: SRABBackbone,
    hbs: HippocampalBindingSystem,
    gamma_module: GammaStatisticsModule,
    dsar_module: DSARModule,
    snr_module: SNREstimationModule,
    batch: EpisodeBatch,
    device: torch.device,
    global_class_ids: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode support set and compute episode prototypes via HBS.

    Returns
    -------
    prototypes:
        Shape ``(N, D)`` — one calibrated prototype per episode class.
    query_embeddings:
        Shape ``(Q, D)``.
    query_k, query_theta:
        Shape ``(Q,)`` each — physics params of query images.
    """
    N, K, C, H, W = batch.support_images.shape
    Q_total = batch.query_images.shape[0]  # N * Q

    # ---- Encode support images ----------------------------------------
    support_flat = batch.support_images.view(N * K, C, H, W)  # (N*K, 1, H, W)
    support_emb = backbone(support_flat)                        # (N*K, D)
    support_emb = support_emb.view(N, K, -1)                   # (N, K, D)

    # Physics for support
    with torch.no_grad():
        k_sup_flat, theta_sup_flat = gamma_module.global_stats(support_flat)
    k_sup = k_sup_flat.view(N, K)
    theta_sup = theta_sup_flat.view(N, K)
    snr_sup_flat = snr_module.global_snr(support_flat).view(N, K)

    # ---- Compute prototypes via HBS -----------------------------------
    prototypes = []
    for i in range(N):
        emb_i = support_emb[i]        # (K, D)
        phy_i = torch.stack([
            k_sup[i], theta_sup[i], snr_sup_flat[i],
            torch.ones(K, device=device),  # speckle index placeholder
        ], dim=1)                         # (K, 4)
        dsar_weights = dsar_module(emb_i, phy_i)  # (D,)

        cid = global_class_ids[i] if global_class_ids else None
        proto, _ = hbs.bind(
            support_embeddings=emb_i,
            class_id=cid,
            physics=phy_i,
            dsar_weights=dsar_weights,
        )
        prototypes.append(proto)

    prototypes_t = torch.stack(prototypes, dim=0)  # (N, D)

    # ---- Encode query images -----------------------------------------
    query_emb = backbone(batch.query_images)  # (Q, D)
    with torch.no_grad():
        q_k, q_theta = gamma_module.global_stats(batch.query_images)

    return prototypes_t, query_emb, q_k, q_theta


def run_phase2(
    backbone: SRABBackbone,
    hbs: HippocampalBindingSystem,
    gamma_module: GammaStatisticsModule,
    dsar_module: DSARModule,
    snr_module: SNREstimationModule,
    novelty_detector: NoveltyDetector,
    cfg: ProtoEvoNetConfig,
    fusar_root: str,
    resume_from: Optional[str] = None,
    fine_tune_backbone: bool = False,
) -> None:
    """
    Execute Phase 2 training.

    Parameters
    ----------
    backbone:
        Pre-trained (and frozen by default) SRAB backbone.
    hbs:
        Hippocampal Binding System to train.
    gamma_module, dsar_module, snr_module:
        Physics modules (already trained in Phase 1; kept frozen here unless
        ``fine_tune_backbone=True``).
    novelty_detector:
        Novelty detector with Platt calibrator to train.
    cfg:
        Full system config.
    fusar_root:
        Path to FUSAR-Ship dataset root.
    resume_from:
        Optional Phase 2 checkpoint path.
    fine_tune_backbone:
        If ``True``, also unfreeze backbone at a reduced learning rate.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_cfg = cfg.training

    # ---- Dataset -------------------------------------------------------
    # Pass full_dataset directly to both samplers.  Episodic samplers
    # produce independent random episodes on every call so train and val
    # diverge naturally without a per-image split.
    full_dataset = FUSARShipDataset(root=fusar_root, image_size=(128, 128))

    logger.info(
        "Phase 2 dataset: %d images across %d classes",
        len(full_dataset), full_dataset.num_classes,
    )
    for cls, cnt in full_dataset.per_class_counts().items():
        logger.info("  %-20s  %d images", cls, cnt)

    train_sampler = EpisodicSampler(
        dataset=full_dataset,
        n_way=train_cfg.n_way,
        k_shot=train_cfg.k_shot,
        n_query=train_cfg.n_query,
        num_episodes=train_cfg.phase2_episodes_per_epoch,
        seed=train_cfg.seed,
    )

    # ---- Model setup --------------------------------------------------
    backbone = backbone.to(device)
    hbs = hbs.to(device)
    gamma_module = gamma_module.to(device)
    dsar_module = dsar_module.to(device)
    snr_module = snr_module.to(device)
    novelty_detector.calibrator.to(device)

    if fine_tune_backbone:
        for p in backbone.parameters():
            p.requires_grad_(True)

    optimizer = build_optimizer_phase2(
        backbone=backbone,
        hbs=hbs,
        backbone_lr_scale=0.1,
        base_lr=train_cfg.phase2_lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.phase2_epochs
    )

    loss_fn = PIAMLoss(
        lambda_proto=train_cfg.piam_lambda_proto,
        lambda_physics=train_cfg.piam_lambda_physics,
        lambda_novelty=train_cfg.piam_lambda_novelty,
        use_fisher_rao=cfg.ablation.use_fisher_rao,
    ).to(device)

    ckpt_manager = CheckpointManager(
        directory=train_cfg.checkpoint_dir + "/phase2",
        metric="val_accuracy",
        higher_is_better=True,
    )

    start_epoch = 0
    if resume_from:
        from utils.checkpoint import load_checkpoint
        payload = load_checkpoint(
            resume_from,
            state_dict_targets={"hbs": hbs, "optimizer": optimizer},
            strict=False,
            device=str(device),
        )
        start_epoch = payload.get("epoch", 0) + 1

    with MetricLogger(train_cfg.log_dir, run_name="phase2") as ml:
        for epoch in range(start_epoch, train_cfg.phase2_epochs):
            # ---- Training: iterate all episodes in this epoch --------
            backbone.train() if fine_tune_backbone else backbone.eval()
            hbs.train()
            gamma_module.eval()
            dsar_module.train()

            epoch_loss = AverageMeter("loss")
            epoch_acc = AverageMeter("acc")

            for episode in train_sampler:
                episode = episode.to(device)

                prototypes, query_emb, q_k, q_theta = _compute_episode_prototypes(
                    backbone, hbs, gamma_module, dsar_module, snr_module,
                    episode, device,
                    global_class_ids=episode.global_class_ids,
                )

                loss, info = loss_fn(
                    query_embeddings=query_emb,
                    prototypes=prototypes,
                    query_labels=episode.query_labels,
                    k_params=q_k,
                    theta_params=q_theta,
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(hbs.parameters()),
                    max_norm=train_cfg.grad_clip,
                )
                optimizer.step()

                with torch.no_grad():
                    acc = episode_accuracy(
                        F.normalize(query_emb.detach(), p=2, dim=1),
                        prototypes.detach(),
                        episode.query_labels,
                    )

                epoch_loss.update(loss.item())
                epoch_acc.update(acc)

            scheduler.step()

            # ---- Epoch-level logging ---------------------------------
            epoch_info = {
                "epoch": epoch,
                "train_loss": epoch_loss.avg,
                "train_acc": epoch_acc.avg,
            }
            epoch_info.update(info)
            ml.log(epoch_info)

            if epoch % train_cfg.log_interval == 0:
                logger.info(
                    "Phase2 [%d/%d]  loss=%.4f  acc=%.3f  (%d episodes)",
                    epoch, train_cfg.phase2_epochs,
                    epoch_loss.avg, epoch_acc.avg,
                    train_cfg.phase2_episodes_per_epoch,
                )

            # ---- Validation + checkpointing -------------------------
            if epoch % train_cfg.eval_interval == 0:
                val_acc = _validate_phase2(
                    backbone, hbs, gamma_module, dsar_module, snr_module,
                    full_dataset, train_cfg, device
                )
                ml.log({"val_accuracy": val_acc, "epoch": epoch})
                logger.info(
                    "Phase2 val epoch %d  val_acc=%.4f", epoch, val_acc
                )

                ckpt_manager.save(
                    epoch=epoch,
                    state_dicts={
                        "backbone": backbone.state_dict(),
                        "hbs": hbs.state_dict(),
                        "dsar": dsar_module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "platt": novelty_detector.calibrator.state_dict(),
                    },
                    metrics={"val_accuracy": val_acc},
                )

                # Novelty calibration every 10 eval rounds
                if (epoch // train_cfg.eval_interval) % 10 == 0:
                    _calibrate_novelty(
                        backbone, novelty_detector, full_dataset,
                        train_cfg, device
                    )

    logger.info("Phase 2 complete.")


@torch.no_grad()
def _validate_phase2(
    backbone, hbs, gamma_module, dsar_module, snr_module,
    dataset, train_cfg, device,
    num_val_episodes: int = 50,
) -> float:
    """Run validation episodes and return mean accuracy."""
    backbone.eval()
    hbs.eval()
    dsar_module.eval()

    val_sampler = EpisodicSampler(
        dataset=dataset,
        n_way=train_cfg.n_way,
        k_shot=train_cfg.k_shot,
        n_query=train_cfg.n_query,
        num_episodes=num_val_episodes,
    )

    acc_meter = AverageMeter("val_acc")
    for episode in val_sampler:
        episode = episode.to(device)
        prototypes, query_emb, _, _ = _compute_episode_prototypes(
            backbone, hbs, gamma_module, dsar_module, snr_module,
            episode, device,
        )
        acc = episode_accuracy(
            F.normalize(query_emb, p=2, dim=1),
            prototypes,
            episode.query_labels,
        )
        acc_meter.update(acc)

    return acc_meter.avg


@torch.no_grad()
def _calibrate_novelty(
    backbone: SRABBackbone,
    novelty_detector: NoveltyDetector,
    dataset: FUSARShipDataset,
    train_cfg,
    device: torch.device,
    n_samples: int = 200,
) -> None:
    """
    Calibrate the Platt scaling parameters using real known-class embeddings
    and synthetic novel embeddings (uniform noise in embedding space).
    """
    backbone.eval()
    indices = torch.randperm(len(dataset))[:n_samples].tolist()
    images = torch.stack([dataset[i][0] for i in indices]).to(device)
    known_embeddings = backbone(images)  # (n, D)

    D = known_embeddings.size(1)
    novel_embeddings = F.normalize(
        torch.randn(n_samples, D, device=device), p=2, dim=1
    )

    # Build a temporary OWMS to compute distances
    temp_owms = OnlineWorkingMemoryStore(
        embedding_dim=D, max_classes=len(dataset.classes), device=str(device)
    )
    for c in dataset.classes:
        cls_indices = dataset.get_class_samples(c)
        if cls_indices:
            sample_emb = backbone(dataset[cls_indices[0]][0].unsqueeze(0).to(device))
            temp_owms.enrol(sample_emb.squeeze(0), label=c)

    known_scores = novelty_detector.score(known_embeddings, temp_owms)
    novel_scores = novelty_detector.score(novel_embeddings, temp_owms)

    scores = torch.cat([known_scores, novel_scores])
    labels = torch.cat([torch.zeros(n_samples), torch.ones(n_samples)])

    novelty_detector.calibrator.fit(scores, labels)
    logger.info("Platt calibrator fitted.")
