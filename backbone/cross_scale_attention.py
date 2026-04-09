"""
Cross-Scale Attention (CSA) module for SRAB.

SAR targets occupy a wide range of spatial scales: a full vessel may span
hundreds of pixels while its superstructure is only a few pixels across.
CSA processes a feature pyramid (from small/high-level to large/low-level
feature maps), projects all levels to a common embedding dimension, and then
runs multi-head attention *across* levels so that each spatial position can
attend to its counterparts at every other scale.

Architecture
------------
1.  Each FPN level is projected to ``embed_dim`` with a 1×1 conv.
2.  All levels are resized to the largest spatial resolution (using bilinear
    interpolation) and concatenated along the sequence dimension.
3.  Multi-head self-attention is applied over the combined token sequence.
4.  The attended tokens are split back into levels and summed with the
    projected features (residual connection).
5.  The finest-level output is returned as the cross-scale feature map.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention over a Feature Pyramid Network output.

    Parameters
    ----------
    in_channels_list:
        Number of channels at each FPN level, ordered coarsest → finest
        (or any order — the module handles heterogeneous channel counts).
    embed_dim:
        Common embedding dimension after the level projections.
    num_heads:
        Number of attention heads.
    dropout:
        Attention dropout probability.
    use_cross_scale:
        If ``False``, simply project and return the finest-level features
        (ablation mode — disables cross-scale attention).
    """

    def __init__(
        self,
        in_channels_list: List[int],
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cross_scale: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = len(in_channels_list)
        self.use_cross_scale = use_cross_scale

        # 1×1 projections: one per FPN level
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.SiLU(),
                )
                for c in in_channels_list
            ]
        )

        if use_cross_scale:
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

        # Level-position embedding so attention knows which level each token
        # came from (not which spatial position — spatial is handled by the
        # projection conv).
        if use_cross_scale:
            self.level_embed = nn.Parameter(
                torch.randn(self.num_levels, 1, embed_dim) * 0.02
            )

    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        feature_maps:
            List of feature tensors, one per FPN level.  The list must have
            the same length as ``in_channels_list``.  Shapes may differ.

        Returns
        -------
        torch.Tensor
            Cross-scale feature map with shape ``(B, embed_dim, H₀, W₀)``
            where H₀, W₀ are the spatial dimensions of the *last* (finest)
            element in *feature_maps*.
        """
        assert len(feature_maps) == self.num_levels

        # Project each level
        projected: List[torch.Tensor] = [
            proj(fm) for proj, fm in zip(self.projections, feature_maps)
        ]  # each: (B, embed_dim, Hᵢ, Wᵢ)

        if not self.use_cross_scale:
            # Ablation: return the last (finest) projected level as-is
            return projected[-1]

        # Target spatial size = spatial dims of the finest (last) level
        H_target, W_target = projected[-1].shape[-2:]
        B = projected[0].size(0)

        # Resize all levels to target and flatten spatial → tokens
        tokens_per_level: List[torch.Tensor] = []
        for lvl_idx, feat in enumerate(projected):
            if feat.shape[-2:] != (H_target, W_target):
                feat = F.interpolate(
                    feat,
                    size=(H_target, W_target),
                    mode="bilinear",
                    align_corners=False,
                )
            # (B, D, H, W) → (B, H*W, D)
            tokens = feat.flatten(2).permute(0, 2, 1)

            # Add level positional embedding: (1, 1, D) broadcast over (B, H*W, D)
            level_emb = self.level_embed[lvl_idx]  # (1, D)
            tokens = tokens + level_emb.unsqueeze(0)  # (B, H*W, D)
            tokens_per_level.append(tokens)

        # Concatenate all levels along the sequence dimension
        # (B, num_levels * H * W, D)
        all_tokens = torch.cat(tokens_per_level, dim=1)

        # Multi-head self-attention with residual + LayerNorm
        attn_out, _ = self.attn(all_tokens, all_tokens, all_tokens)
        all_tokens = self.norm1(all_tokens + attn_out)

        # FFN with residual
        all_tokens = self.norm2(all_tokens + self.ffn(all_tokens))

        # Take the tokens belonging to the finest level (last chunk)
        seq_len_per_level = H_target * W_target
        finest_tokens = all_tokens[:, -seq_len_per_level:, :]  # (B, H*W, D)

        # Reshape back to spatial map
        out = finest_tokens.permute(0, 2, 1).view(B, self.embed_dim, H_target, W_target)
        return out

    def extra_repr(self) -> str:
        return (
            f"levels={self.num_levels}, embed_dim={self.embed_dim}, "
            f"use_csa={self.use_cross_scale}"
        )
