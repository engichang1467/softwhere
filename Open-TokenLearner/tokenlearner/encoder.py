"""Transformer encoder variants that plug a TokenLearner into a ViT stack.

Two integration patterns from the paper:

* :class:`EncoderMod` — Figure 3a. Run the first ``loc`` blocks on the
  full ``N`` patch tokens, drop down to ``S`` learned tokens via one
  TokenLearner, and run the remaining ``depth - loc`` blocks on the
  reduced set.
* :class:`EncoderModFuser` — Figure 3b. Same prefix, but from ``loc``
  onward every layer is ``TokenLearner -> Transformer -> TokenFuser``
  with a residual onto the pre-module features.

For video, the encoder receives joint space-time tokens shaped
``[B, T*HW, C]``. The TokenLearner runs per frame (it is folded as
``[B*T, HW, C]`` -> ``[B*T, S, C]`` and unfolded to ``[B, T*S, C]``),
which matches the recipe in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modules import TokenFuser, TokenLearner, TokenLearnerV11


class DropPath(nn.Module):
    """Stochastic depth: drop whole residual branches per sample (Scenic uses
    ``stochastic_depth=0.1`` with a per-layer linearly-scaled rate)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # One Bernoulli mask per sample, broadcast over the remaining dims.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x / keep * mask


class Attention(nn.Module):
    """Multi-head self-attention (no relative pos bias, no masking)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _make_tokenlearner(
    use_v11: bool,
    embed_dim: int,
    num_tokens: int,
    bottleneck_dim: int,
    drop: float = 0.0,
) -> nn.Module:
    if use_v11:
        return TokenLearnerV11(embed_dim, num_tokens, bottleneck_dim=bottleneck_dim, dropout=drop)
    return TokenLearner(embed_dim, num_tokens, use_sum_pooling=True)


class EncoderMod(nn.Module):
    """TokenLearner inserted once at depth ``loc`` (Figure 3a)."""

    def __init__(
        self,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_tokens: int,
        loc: int,
        use_v11: bool = True,
        bottleneck_dim: int = 64,
        temporal_dimensions: int = 1,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.loc = loc
        self.depth = depth
        self.temporal_dimensions = temporal_dimensions
        self.has_tokenlearner = loc < depth

        # Stochastic depth rate scales linearly with block index across the stack.
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks_pre = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_dim, drop=drop, attn_drop=attn_drop, drop_path=dpr[i]) for i in range(loc)]
        )
        self.blocks_post = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_dim, drop=drop, attn_drop=attn_drop, drop_path=dpr[loc + j])
                for j in range(depth - loc)
            ]
        )
        if self.has_tokenlearner:
            self.tokenlearner = _make_tokenlearner(use_v11, embed_dim, num_tokens, bottleneck_dim, drop=drop)
        else:
            self.tokenlearner = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks_pre:
            x = blk(x)

        if self.has_tokenlearner:
            x = _per_frame_tokenlearner(x, self.tokenlearner, self.temporal_dimensions)

        for blk in self.blocks_post:
            x = blk(x)
        return x


class EncoderModFuser(nn.Module):
    """``TokenLearner -> Transformer -> TokenFuser`` from ``loc`` onward (Figure 3b)."""

    def __init__(
        self,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_tokens: int,
        loc: int,
        use_v11: bool = True,
        bottleneck_dim: int = 64,
        temporal_dimensions: int = 1,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.loc = loc
        self.depth = depth
        self.temporal_dimensions = temporal_dimensions
        self.has_tokenlearner = loc < depth

        # Stochastic depth rate scales linearly with block index across the stack.
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks_pre = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_dim, drop=drop, attn_drop=attn_drop, drop_path=dpr[i]) for i in range(loc)]
        )

        n_post = depth - loc
        if self.has_tokenlearner:
            self.tl_modules = nn.ModuleList(
                [_make_tokenlearner(use_v11, embed_dim, num_tokens, bottleneck_dim, drop=drop) for _ in range(n_post)]
            )
            self.blocks_post = nn.ModuleList(
                [
                    Block(embed_dim, num_heads, mlp_dim, drop=drop, attn_drop=attn_drop, drop_path=dpr[loc + j])
                    for j in range(n_post)
                ]
            )
            self.fusers = nn.ModuleList(
                [
                    TokenFuser(embed_dim, num_tokens, bottleneck_dim=bottleneck_dim, dropout=drop)
                    for _ in range(n_post)
                ]
            )
        else:
            self.tl_modules = nn.ModuleList()
            self.blocks_post = nn.ModuleList()
            self.fusers = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks_pre:
            x = blk(x)

        for tl, blk, fuser in zip(self.tl_modules, self.blocks_post, self.fusers):
            original = x  # [B, N, C], with N = T*HW for video
            tokens = _per_frame_tokenlearner(original, tl, self.temporal_dimensions)
            tokens = blk(tokens)
            x = _per_frame_fuser(tokens, original, fuser, self.temporal_dimensions)
        return x


def _per_frame_tokenlearner(
    x: torch.Tensor, module: nn.Module, temporal_dimensions: int
) -> torch.Tensor:
    """Run TokenLearner per frame: ``[B, T*HW, C] -> [B, T*S, C]``.

    For images (``temporal_dimensions == 1``) this is a no-op fold.
    """
    if temporal_dimensions <= 1:
        return module(x)

    b, n, c = x.shape
    t = temporal_dimensions
    if n % t != 0:
        raise ValueError(
            f"Token count {n} is not divisible by temporal_dimensions {t}; "
            "did the patch embed shape change?"
        )
    hw = n // t

    folded = x.reshape(b * t, hw, c)
    tokens = module(folded)  # [B*T, S, C]
    s = tokens.shape[1]
    return tokens.reshape(b, t * s, c)


def _per_frame_fuser(
    tokens: torch.Tensor,
    original: torch.Tensor,
    fuser: TokenFuser,
    temporal_dimensions: int,
) -> torch.Tensor:
    """Run TokenFuser per frame: ``([B, T*S, C], [B, T*HW, C]) -> [B, T*HW, C]``."""
    if temporal_dimensions <= 1:
        return fuser(tokens, original)

    b, ts, c = tokens.shape
    t = temporal_dimensions
    if ts % t != 0:
        raise ValueError(
            f"Learned token count {ts} is not divisible by temporal_dimensions {t}."
        )
    s = ts // t
    n = original.shape[1]
    hw = n // t

    tokens_r = tokens.reshape(b * t, s, c)
    orig_r = original.reshape(b * t, hw, c)
    fused = fuser(tokens_r, orig_r)  # [B*T, HW, C]
    return fused.reshape(b, t * hw, c)
