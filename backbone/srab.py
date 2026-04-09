"""
SAR-Radial-Aware Backbone (SRAB).

SRAB integrates:
  * RadialAwareConv2d at every convolutional layer
  * GammaConditionedNorm after each stage
  * A Feature Pyramid Network (FPN) producing multi-scale maps
  * CrossScaleAttention (CSA) to aggregate cross-scale information
  * Global average pooling + L2 normalisation to produce the final embedding

Input:   (B, in_channels, H, W)  — single-channel SAR intensity image
Output:  (B, embedding_dim)      — L2-normalised embedding
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .radial_conv import RadialAwareConv2d
from .gcn import GammaConditionedNorm
from .cross_scale_attention import CrossScaleAttention
from utils.config import BackboneConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """
    RadialAwareConv2d → GCN → SiLU block.

    This is the atomic building block for every stage of SRAB.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_rings: int = 8,
        use_radial: bool = True,
        use_gcn: bool = True,
    ) -> None:
        super().__init__()
        self.conv = RadialAwareConv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_rings=num_rings,
            use_radial=use_radial,
            bias=False,
        )
        self.norm = GammaConditionedNorm(out_ch, use_gcn=use_gcn)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SRABStage(nn.Module):
    """
    One encoder stage: two ConvBNAct blocks + optional strided down-sampling.

    Parameters
    ----------
    in_ch, out_ch:
        Input / output channel counts.
    stride:
        If > 1, the first conv down-samples the spatial resolution.
    num_rings, use_radial, use_gcn:
        Forwarded to ConvBNAct.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 2,
        num_rings: int = 8,
        use_radial: bool = True,
        use_gcn: bool = True,
    ) -> None:
        super().__init__()
        self.block1 = ConvBNAct(
            in_ch, out_ch,
            stride=stride, padding=1,
            num_rings=num_rings, use_radial=use_radial, use_gcn=use_gcn,
        )
        self.block2 = ConvBNAct(
            out_ch, out_ch,
            stride=1, padding=1,
            num_rings=num_rings, use_radial=use_radial, use_gcn=use_gcn,
        )
        # Residual projection if channel count changes
        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x)) + self.residual(x)


# ---------------------------------------------------------------------------
# FPN neck
# ---------------------------------------------------------------------------

class FPNNeck(nn.Module):
    """
    Lightweight top-down Feature Pyramid Network.

    Takes the list of stage outputs (coarsest → finest) and produces a
    feature pyramid of the same length, each level at ``fpn_channels``.
    """

    def __init__(self, in_channels_list: List[int], fpn_channels: int = 256) -> None:
        super().__init__()
        # Lateral 1×1 convs
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, fpn_channels, kernel_size=1) for c in in_channels_list]
        )
        # 3×3 smoothing convs
        self.outputs = nn.ModuleList(
            [
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        features:
            Stage outputs in coarsest-to-finest order.

        Returns
        -------
        list of torch.Tensor
            FPN pyramid, coarsest-to-finest, each ``(B, fpn_channels, Hᵢ, Wᵢ)``.
        """
        # Lateral projections
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]

        # Top-down merging (start from the coarsest level)
        for i in range(len(laterals) - 1):
            laterals[i + 1] = laterals[i + 1] + F.interpolate(
                laterals[i],
                size=laterals[i + 1].shape[-2:],
                mode="nearest",
            )

        # Smoothing
        out = [smooth(lat) for smooth, lat in zip(self.outputs, laterals)]
        return out


# ---------------------------------------------------------------------------
# Main SRAB backbone
# ---------------------------------------------------------------------------

class SRABBackbone(nn.Module):
    """
    SAR-Radial-Aware Backbone.

    Processes a single-channel SAR intensity image and outputs an
    L2-normalised embedding vector of dimension ``cfg.embedding_dim``.

    Parameters
    ----------
    cfg:
        ``BackboneConfig`` instance controlling all architectural choices.

    Attributes
    ----------
    embedding_dim:
        Output embedding dimension (mirrors ``cfg.embedding_dim``).
    """

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = cfg.embedding_dim

        # ---- Stem --------------------------------------------------------
        self.stem = ConvBNAct(
            cfg.in_channels, cfg.base_channels,
            kernel_size=7, stride=2, padding=3,
            num_rings=cfg.num_radial_rings,
            use_radial=cfg.use_radial_conv,
            use_gcn=cfg.use_gcn,
        )

        # ---- Encoder stages ---------------------------------------------
        stage_channels = [
            cfg.base_channels * (2 ** i)
            for i in range(cfg.num_stages)
        ]  # e.g. [64, 128, 256, 512]

        self.stages = nn.ModuleList()
        in_ch = cfg.base_channels
        for i, out_ch in enumerate(stage_channels):
            stride = 1 if i == 0 else 2
            self.stages.append(
                SRABStage(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    stride=stride,
                    num_rings=cfg.num_radial_rings,
                    use_radial=cfg.use_radial_conv,
                    use_gcn=cfg.use_gcn,
                )
            )
            in_ch = out_ch

        # ---- FPN neck ---------------------------------------------------
        self.fpn = FPNNeck(stage_channels, fpn_channels=cfg.embedding_dim)

        # ---- Cross-scale attention --------------------------------------
        self.csa = CrossScaleAttention(
            in_channels_list=[cfg.embedding_dim] * cfg.num_stages,
            embed_dim=cfg.embedding_dim,
            num_heads=cfg.cross_scale_heads,
            dropout=cfg.dropout,
            use_cross_scale=cfg.use_cross_scale_attention,
        )

        # ---- Head: GAP → projection → L2 normalisation -----------------
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of SAR images.

        Parameters
        ----------
        x:
            SAR image tensor of shape ``(B, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            L2-normalised embedding of shape ``(B, embedding_dim)``.
        """
        # Stem
        x = self.stem(x)  # (B, base_ch, H/2, W/2)

        # Encoder
        stage_feats: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            stage_feats.append(x)
        # stage_feats: list of (B, Cᵢ, Hᵢ, Wᵢ), coarsest → finest

        # FPN
        fpn_feats = self.fpn(stage_feats)

        # Cross-scale attention — takes FPN output, returns (B, D, H₀, W₀)
        csa_out = self.csa(fpn_feats)

        # Head
        embedding = self.head(csa_out)  # (B, D)

        # L2 normalise
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def extract_multi_scale(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Like ``forward`` but also returns the intermediate FPN feature maps.

        Useful for the Physics layer which operates on raw feature maps.

        Returns
        -------
        embedding:
            ``(B, D)`` L2-normalised embedding.
        fpn_feats:
            List of ``(B, D, Hᵢ, Wᵢ)`` FPN feature maps.
        """
        x = self.stem(x)
        stage_feats = []
        for stage in self.stages:
            x = stage(x)
            stage_feats.append(x)

        fpn_feats = self.fpn(stage_feats)
        csa_out = self.csa(fpn_feats)
        embedding = F.normalize(self.head(csa_out), p=2, dim=1)
        return embedding, fpn_feats
