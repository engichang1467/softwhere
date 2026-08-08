"""TokenLearner / TokenLearnerV11 / TokenFuser modules.

PyTorch port of the Scenic (JAX/Flax) reference implementation:
https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner

Tensor conventions
------------------
The Flax reference uses channel-last NHWC tensors. The PyTorch ports here
accept either ``[B, H, W, C]`` (channel-last) or the flattened
``[B, H*W, C]`` form so they slot into ViT-style pipelines without a
transpose at the call site. The conv-based v1.0 module transposes to
``[B, C, H, W]`` internally because ``nn.Conv2d`` is channel-first.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Accept ``[B, H, W, C]`` or square ``[B, H*W, C]``; return ``[B, H, W, C]``."""
    if x.dim() == 4:
        return x
    if x.dim() == 3:
        b, hw, c = x.shape
        side = int(math.isqrt(hw))
        if side * side != hw:
            raise ValueError(
                f"TokenLearner v1.0 expects a square spatial grid when given a "
                f"flattened input, got H*W={hw}."
            )
        return x.reshape(b, side, side, c)
    raise ValueError(f"Expected 3D or 4D input, got shape {tuple(x.shape)}.")


def _to_bnc(x: torch.Tensor) -> torch.Tensor:
    """Accept ``[B, H, W, C]`` or ``[B, N, C]``; return ``[B, N, C]``."""
    if x.dim() == 3:
        return x
    if x.dim() == 4:
        b, h, w, c = x.shape
        return x.reshape(b, h * w, c)
    raise ValueError(f"Expected 3D or 4D input, got shape {tuple(x.shape)}.")


class TokenLearner(nn.Module):
    """TokenLearner v1.0 (conv + sigmoid + pool).

    Four ``3x3`` convolutions with ``num_tokens`` output channels (no bias)
    produce ``S`` spatial attention maps. After a sigmoid gate, each map is
    broadcast against the input features and pooled over the spatial axes to
    yield one token. LayerNorm is applied to the input first.
    """

    def __init__(
        self,
        in_channels: int,
        num_tokens: int,
        use_sum_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.use_sum_pooling = use_sum_pooling

        self.norm = nn.LayerNorm(in_channels)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, num_tokens, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1, bias=False),
            ]
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x = _to_bhwc(x)
        b, h, w, c = x.shape

        attn = self.norm(x).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        for i, conv in enumerate(self.convs):
            attn = conv(attn)
            if i < len(self.convs) - 1:
                attn = F.gelu(attn, approximate="tanh")
        attn = torch.sigmoid(attn)  # [B, S, H, W]

        out = torch.einsum("bshw,bhwc->bsc", attn, x)
        if not self.use_sum_pooling:
            out = out / (h * w)
        if return_attn:
            # attn: [B, S, H, W] spatial attention maps (one per learned token).
            return out, attn
        return out


class TokenLearnerV11(nn.Module):
    """TokenLearner v1.1 (MLP + softmax + weighted sum).

    A two-layer MLP (``in_channels -> bottleneck_dim -> num_tokens``) with
    GeLU produces per-position logits. Softmax over the spatial axis turns
    them into attention weights, and an einsum collapses space.
    """

    def __init__(
        self,
        in_channels: int,
        num_tokens: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens

        self.norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, num_tokens)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x_flat = _to_bnc(x)  # [B, HW, C]

        # MlpBlock: Dense -> gelu -> dropout -> Dense -> dropout (matches Scenic).
        h = self.drop(F.gelu(self.fc1(self.norm(x_flat)), approximate="tanh"))
        attn = self.drop(self.fc2(h))  # [B, HW, S]
        attn = attn.transpose(-1, -2)  # [B, S, HW]
        attn = F.softmax(attn, dim=-1)  # softmax over spatial positions

        out = torch.einsum("bsi,bic->bsc", attn, x_flat)
        if return_attn:
            # attn: [B, S, HW] softmax-over-space maps; reshape to [B,S,H,W] at
            # the consumer (this module keeps its native flattened layout).
            return out, attn
        return out


class TokenFuser(nn.Module):
    """TokenFuser: remap learned tokens back to the original spatial grid.

    Given transformer outputs ``Y`` of shape ``[B, S, C]`` and the
    pre-TokenLearner features ``original`` of shape ``[B, H*W, C]`` (or
    ``[B, H, W, C]``):

    1. Mix across tokens with a zero-initialized ``Linear(S, S)`` on the
       token axis.
    2. Build a mask ``M = sigmoid(MLP(LN(original)))`` of shape
       ``[B, H*W, S]``.
    3. ``out = einsum('bsc,bhs->bhc', Y_mixed, M) + original``.

    The token-axis linear is zero-initialized so the fuser starts as a
    near-identity contribution on top of the residual, matching the
    Scenic reference.
    """

    def __init__(
        self,
        in_channels: int,
        num_tokens: int,
        bottleneck_dim: int = 64,
        use_normalization: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.use_normalization = use_normalization

        if use_normalization:
            self.norm_in = nn.LayerNorm(in_channels)
            self.norm_mid = nn.LayerNorm(in_channels)
        else:
            self.norm_in = nn.Identity()
            self.norm_mid = nn.Identity()

        # Token-axis mixing layer (operates on the S dimension).
        # Zero-initialized so the residual dominates at the start of training.
        self.token_mix = nn.Linear(num_tokens, num_tokens)
        nn.init.zeros_(self.token_mix.weight)
        nn.init.zeros_(self.token_mix.bias)

        # Mask MLP from original features to per-position token weights.
        self.norm_mask = nn.LayerNorm(in_channels)
        self.mask_fc1 = nn.Linear(in_channels, bottleneck_dim)
        self.mask_fc2 = nn.Linear(bottleneck_dim, num_tokens)

        # Dropout on the fused output (matches the Scenic TokenFuser).
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, y: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        # y: [B, S, C]; original: [B, H, W, C] or [B, HW, C].
        orig = _to_bnc(original)

        x = self.norm_in(y)
        x = x.transpose(-1, -2)           # [B, C, S]
        x = self.token_mix(x)             # [B, C, S]
        x = x.transpose(-1, -2)           # [B, S, C]
        x = self.norm_mid(x)

        m = self.mask_fc2(F.gelu(self.mask_fc1(self.norm_mask(orig)), approximate="tanh"))  # [B, HW, S]
        m = torch.sigmoid(m)

        out = torch.einsum("bsc,bhs->bhc", x, m)  # [B, HW, C]
        out = self.drop(out)
        return out + orig
