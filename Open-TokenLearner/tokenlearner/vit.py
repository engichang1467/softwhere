"""ViT/ViViT backbone with optional TokenLearner insertion.

The TokenLearner module is inserted at depth ``tokenlearner_loc`` in one of
two configurations:

* ``use_fuser=False`` — Figure 3a in the paper. A single TokenLearner
  shrinks ``N`` patch tokens to ``S`` learned tokens; subsequent layers
  operate on the reduced set.
* ``use_fuser=True`` — Figure 3b. From ``tokenlearner_loc`` onward, each
  block is ``TokenLearner -> Transformer -> TokenFuser`` with a residual
  back to the pre-module features.

For video, ``num_frames`` and ``tubelet_t`` configure a 3D tubelet patch
embedding (ViViT-style) and ``temporal_dimensions`` (``T = num_frames /
tubelet_t``) tells the encoder how to fold per-frame for TokenLearner.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .encoder import EncoderMod, EncoderModFuser


class PatchEmbed(nn.Module):
    """2D patch embedding via a single strided ``Conv2d``."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})."
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, N, D]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class TubeletEmbed(nn.Module):
    """3D tubelet embedding for video, via ``Conv3d`` (ViViT-style)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_t: int = 2,
        num_frames: int = 32,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})."
            )
        if num_frames % tubelet_t != 0:
            raise ValueError(
                f"num_frames ({num_frames}) must be divisible by tubelet_t ({tubelet_t})."
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_t = tubelet_t
        self.num_frames = num_frames
        self.grid_size = img_size // patch_size
        self.grid_t = num_frames // tubelet_t
        self.num_patches_per_frame = self.grid_size * self.grid_size
        self.num_patches = self.num_patches_per_frame * self.grid_t
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_t, patch_size, patch_size),
            stride=(tubelet_t, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] -> [B, T'*H'*W', D]
        x = self.proj(x)  # [B, D, T', H', W']
        b, d, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, d).contiguous()
        return x


class TokenLearnerViT(nn.Module):
    """Vision Transformer with an optional TokenLearner module.

    Setting ``tokenlearner_loc >= depth`` disables the TokenLearner path
    and yields a vanilla ViT (useful as a baseline / sanity check).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        num_tokens: int = 8,
        tokenlearner_loc: int = 6,
        use_v11: bool = True,
        use_fuser: bool = False,
        bottleneck_dim: int = 64,
        num_frames: int = 1,
        tubelet_t: int = 1,
        temporal_dimensions: int = 1,
        classifier: str = "gap",
        representation_size: Optional[int] = None,
        init_head_bias: float = -10.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if tokenlearner_loc < 0 or tokenlearner_loc > depth:
            raise ValueError(
                f"tokenlearner_loc must be in [0, depth]={depth}, got {tokenlearner_loc}."
            )
        if classifier not in ("gap", "gmp", "gsp"):
            raise ValueError(
                f"classifier must be one of 'gap', 'gmp', 'gsp', got {classifier!r}."
            )

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.temporal_dimensions = temporal_dimensions
        self.classifier = classifier
        self.init_head_bias = init_head_bias
        self.is_video = num_frames > 1 or tubelet_t > 1

        if self.is_video:
            self.patch_embed: nn.Module = TubeletEmbed(
                img_size=img_size,
                patch_size=patch_size,
                tubelet_t=tubelet_t,
                num_frames=num_frames,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            num_patches = self.patch_embed.num_patches
            expected_t = num_frames // tubelet_t
            if temporal_dimensions != expected_t:
                raise ValueError(
                    f"temporal_dimensions ({temporal_dimensions}) must equal "
                    f"num_frames/tubelet_t ({expected_t})."
                )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(drop_rate)

        EncoderCls = EncoderModFuser if use_fuser else EncoderMod
        self.encoder = EncoderCls(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_tokens=num_tokens,
            loc=tokenlearner_loc,
            use_v11=use_v11,
            bottleneck_dim=bottleneck_dim,
            temporal_dimensions=temporal_dimensions,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Optional pre-logits projection + tanh (matches the Scenic reference's
        # representation_size head). If None, classify straight off the pooled
        # features.
        if representation_size is not None:
            self.pre_logits: nn.Module = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh(),
            )
            head_in = representation_size
        else:
            self.pre_logits = nn.Identity()
            head_in = embed_dim

        self.head = nn.Linear(head_in, num_classes) if num_classes > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Restore the zero-init for TokenFuser's token_mix that we just clobbered.
        for m in self.modules():
            if hasattr(m, "token_mix") and isinstance(m.token_mix, nn.Linear):
                nn.init.zeros_(m.token_mix.weight)
                nn.init.zeros_(m.token_mix.bias)
        # Zero-init the classifier head weight (Scenic output_projection kernel)
        # and set its bias to init_head_bias (-10.0 by default), so a sigmoid
        # head starts at near-zero probability and the initial loss is small.
        if isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, self.init_head_bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.patch_embed(x)  # [B, N, D]
        f = f + self.pos_embed
        f = self.pos_drop(f)
        f = self.encoder(f)
        return self.norm(f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.forward_features(x)
        # Pool over tokens: gap=mean, gmp=max, gsp=sum (matches Scenic).
        if self.classifier == "gap":
            f = f.mean(dim=1)
        elif self.classifier == "gmp":
            f = f.max(dim=1).values
        else:  # "gsp"
            f = f.sum(dim=1)
        f = self.pre_logits(f)
        return self.head(f)
