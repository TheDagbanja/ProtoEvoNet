"""
ProtoEvoNet configuration system.

All hyperparameters, architectural choices, and ablation toggles live here.
Sub-configs are plain Python dataclasses so they are JSON-serialisable and
easy to override from the command line or a sweep.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class BackboneConfig:
    """SRAB backbone hyperparameters."""
    in_channels: int = 1          # SAR images are single-channel intensity
    base_channels: int = 64       # feature map width at the first stage
    num_stages: int = 4           # number of down-sampling stages
    num_radial_rings: int = 8     # radial basis functions per conv kernel
    embedding_dim: int = 256      # output embedding dimension
    num_scales: int = 4           # FPN levels fed to cross-scale attention
    cross_scale_heads: int = 8    # multi-head attention heads
    dropout: float = 0.1          # dropout in attention
    use_gcn: bool = True          # Gamma-Conditioned Normalisation
    use_radial_conv: bool = True  # RadialAwareConv2d in each ConvBNAct block
    use_cross_scale_attention: bool = True  # CrossScaleAttention over FPN levels


@dataclass
class PhysicsConfig:
    """Physics-layer hyperparameters."""
    dsar_dim: int = 256           # must equal BackboneConfig.embedding_dim
    gamma_eps: float = 1e-6       # numerical stability for Gamma MLE
    snr_patch_size: int = 7       # local patch side length for SNR estimation
    fisher_rao_eps: float = 1e-8  # numerical stability in Fisher–Rao metric
    gamma_mle_iters: int = 20     # Newton–Raphson iterations for Gamma MLE
    use_dsar: bool = True
    use_fisher_rao: bool = True
    use_snr: bool = True


@dataclass
class HippocampusConfig:
    """Hippocampal Binding System hyperparameters."""
    embedding_dim: int = 256
    sqaa_hidden: int = 128        # hidden dim of SQAA MLP
    apm_hidden: int = 128         # hidden dim of APM MLP
    apm_layers: int = 2           # depth of APM MLP
    agppi_scale: float = 1.0     # base prior scaling factor
    bpc_prior_var: float = 1.0   # prior variance σ₀² for BPC-GP
    bpc_noise_var: float = 0.01  # observation noise variance
    use_sqaa: bool = True
    use_agppi: bool = True
    use_apm: bool = True
    use_bpc_gp: bool = True
    use_dsar_smoother: bool = True


@dataclass
class MemoryConfig:
    """OWMS / PIM / TPE hyperparameters."""
    max_classes: int = 200        # maximum prototype slots
    embedding_dim: int = 256
    pim_threshold: float = 0.85  # cosine-similarity threshold for interference
    tpe_momentum: float = 0.9    # EMA momentum for prototype evolution
    tpe_min_count: int = 5       # minimum observations before TPE kicks in


@dataclass
class InferenceConfig:
    """Inference engine hyperparameters."""
    novelty_threshold: float = 0.5  # Platt-scaled probability cut-off
    platt_temperature: float = 1.0  # Platt scaling temperature (learnable)
    platt_bias: float = 0.0         # Platt scaling bias (learnable)
    top_k: int = 5                  # top-K classes returned in recognition mode


@dataclass
class TrainingConfig:
    """Training-loop hyperparameters for both phases."""
    # ---- Phase 1 (HRSID – contrastive pre-training) ----------------------
    phase1_epochs: int = 50
    phase1_lr: float = 1e-3
    phase1_batch_size: int = 32
    phase1_num_workers: int = 4
    agcp_temperature: float = 0.07  # InfoNCE temperature
    agcp_margin: float = 0.5        # margin for physics-guided negatives

    # ---- Phase 2 (FUSAR-Ship – episodic meta-learning) -------------------
    phase2_epochs: int = 150
    phase2_episodes_per_epoch: int = 600  # episodes per training epoch
    phase2_lr: float = 3e-4
    phase2_batch_size: int = 1      # one episode per optimiser step
    phase2_num_workers: int = 4
    n_way: int = 5
    k_shot: int = 5
    n_query: int = 15
    piam_lambda_physics: float = 0.1
    piam_lambda_proto: float = 1.0
    piam_lambda_novelty: float = 0.1

    # ---- Shared ----------------------------------------------------------
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    warmup_epochs: int = 5
    scheduler: str = "cosine"       # "cosine" | "step"
    step_lr_gamma: float = 0.5
    step_lr_step: int = 20
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    log_interval: int = 1
    eval_interval: int = 5
    seed: int = 42


@dataclass
class AblationConfig:
    """
    Binary toggles for every major module.

    Set a flag to False to ablate that component.  Call
    ``ProtoEvoNetConfig.apply_ablation()`` after creating the config so the
    flags propagate to the relevant sub-configs.
    """
    use_radial_conv: bool = True
    use_cross_scale_attention: bool = True
    use_gcn: bool = True
    use_dsar: bool = True
    use_fisher_rao: bool = True
    use_snr: bool = True
    use_sqaa: bool = True
    use_agppi: bool = True
    use_apm: bool = True
    use_bpc_gp: bool = True
    use_dsar_smoother: bool = True
    use_pim: bool = True
    use_tpe: bool = True
    use_novelty: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class ProtoEvoNetConfig:
    """
    Master configuration for ProtoEvoNet.

    Usage::

        cfg = ProtoEvoNetConfig()
        cfg.ablation.use_gcn = False   # ablate GCN
        cfg.apply_ablation()           # propagate to sub-configs
        cfg.save("run_config.json")

        cfg2 = ProtoEvoNetConfig.load("run_config.json")
    """
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    hippocampus: HippocampusConfig = field(default_factory=HippocampusConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    device: str = "cuda"

    def apply_ablation(self) -> None:
        """Propagate ablation flags into the relevant sub-configs."""
        self.backbone.use_gcn = self.ablation.use_gcn
        self.backbone.use_radial_conv = self.ablation.use_radial_conv
        self.backbone.use_cross_scale_attention = self.ablation.use_cross_scale_attention
        self.physics.use_dsar = self.ablation.use_dsar
        self.physics.use_fisher_rao = self.ablation.use_fisher_rao
        self.physics.use_snr = self.ablation.use_snr
        self.hippocampus.use_sqaa = self.ablation.use_sqaa
        self.hippocampus.use_agppi = self.ablation.use_agppi
        self.hippocampus.use_apm = self.ablation.use_apm
        self.hippocampus.use_bpc_gp = self.ablation.use_bpc_gp
        self.hippocampus.use_dsar_smoother = self.ablation.use_dsar_smoother

    def save(self, path: str) -> None:
        """Serialise to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(dataclasses.asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProtoEvoNetConfig":
        """Deserialise from JSON.

        Unknown keys in the JSON (e.g. from an older config format) are
        silently ignored so that checkpoints saved before a field was added
        still load without errors.
        """
        def _filter(dc_cls, raw_dict: dict) -> dict:
            """Return only the keys that dc_cls accepts."""
            known = {f.name for f in dataclasses.fields(dc_cls)}
            return {k: v for k, v in raw_dict.items() if k in known}

        with open(path) as fh:
            raw = json.load(fh)
        return cls(
            backbone=BackboneConfig(**_filter(BackboneConfig, raw.get("backbone", {}))),
            physics=PhysicsConfig(**_filter(PhysicsConfig, raw.get("physics", {}))),
            hippocampus=HippocampusConfig(**_filter(HippocampusConfig, raw.get("hippocampus", {}))),
            memory=MemoryConfig(**_filter(MemoryConfig, raw.get("memory", {}))),
            inference=InferenceConfig(**_filter(InferenceConfig, raw.get("inference", {}))),
            training=TrainingConfig(**_filter(TrainingConfig, raw.get("training", {}))),
            ablation=AblationConfig(**_filter(AblationConfig, raw.get("ablation", {}))),
            device=raw.get("device", "cuda"),
        )
