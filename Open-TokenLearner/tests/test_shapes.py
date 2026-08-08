"""Shape and gradient checks for the TokenLearner port.

These mirror the math section of the README: every module is exercised at
its documented input shapes, the output shape is asserted, and gradients
are checked to flow back to the input.
"""

from __future__ import annotations

import math

import pytest
import torch

from tokenlearner import (
    EncoderMod,
    EncoderModFuser,
    TokenFuser,
    TokenLearner,
    TokenLearnerV11,
    TokenLearnerViT,
)


# ---------------------------------------------------------------------------
# Module-level shape checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_sum_pooling", [True, False])
def test_tokenlearner_v10_shape(use_sum_pooling: bool) -> None:
    x = torch.randn(2, 14, 14, 768)
    tl = TokenLearner(in_channels=768, num_tokens=8, use_sum_pooling=use_sum_pooling)
    out = tl(x)
    assert out.shape == (2, 8, 768)


def test_tokenlearner_v10_accepts_flat_input() -> None:
    x = torch.randn(2, 196, 768)
    tl = TokenLearner(in_channels=768, num_tokens=8)
    out = tl(x)
    assert out.shape == (2, 8, 768)


def test_tokenlearner_v10_mean_pooling_normalizes() -> None:
    """Mean pooling should equal sum / (H*W)."""
    torch.manual_seed(0)
    x = torch.randn(2, 7, 7, 32)
    sum_tl = TokenLearner(in_channels=32, num_tokens=4, use_sum_pooling=True)
    mean_tl = TokenLearner(in_channels=32, num_tokens=4, use_sum_pooling=False)
    mean_tl.load_state_dict(sum_tl.state_dict())
    out_sum = sum_tl(x)
    out_mean = mean_tl(x)
    assert torch.allclose(out_sum / (7 * 7), out_mean, atol=1e-6)


def test_tokenlearner_v10_grad() -> None:
    x = torch.randn(2, 14, 14, 768, requires_grad=True)
    tl = TokenLearner(in_channels=768, num_tokens=8)
    tl(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_tokenlearner_v10_return_attn() -> None:
    """return_attn yields the [B,S,H,W] maps; the pooled output is unchanged."""
    torch.manual_seed(0)
    x = torch.randn(2, 14, 14, 768)
    tl = TokenLearner(in_channels=768, num_tokens=8)
    out_only = tl(x)
    out, attn = tl(x, return_attn=True)
    assert attn.shape == (2, 8, 14, 14)
    assert torch.equal(out, out_only)  # byte-identical, no behavior drift


def test_tokenlearner_v11_shape_flat() -> None:
    x = torch.randn(2, 196, 768)
    tl = TokenLearnerV11(in_channels=768, num_tokens=8, bottleneck_dim=64)
    out = tl(x)
    assert out.shape == (2, 8, 768)


def test_tokenlearner_v11_shape_bhwc() -> None:
    x = torch.randn(2, 14, 14, 768)
    tl = TokenLearnerV11(in_channels=768, num_tokens=16, bottleneck_dim=128)
    out = tl(x)
    assert out.shape == (2, 16, 768)


def test_tokenlearner_v11_softmax_is_normalized() -> None:
    """The attention weights are a softmax over space, so they sum to 1."""
    torch.manual_seed(0)
    x = torch.randn(2, 196, 768)
    tl = TokenLearnerV11(in_channels=768, num_tokens=8, bottleneck_dim=64)

    # Reproduce the internal attention computation and check it sums to 1.
    y = tl.norm(x)
    attn = tl.fc2(torch.nn.functional.gelu(tl.fc1(y))).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_tokenlearner_v11_grad() -> None:
    x = torch.randn(2, 196, 768, requires_grad=True)
    tl = TokenLearnerV11(in_channels=768, num_tokens=8)
    tl(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_tokenlearner_v11_return_attn() -> None:
    """return_attn yields the [B,S,HW] softmax maps; pooled output is unchanged."""
    torch.manual_seed(0)
    x = torch.randn(2, 196, 768)
    tl = TokenLearnerV11(in_channels=768, num_tokens=8, bottleneck_dim=64)
    out_only = tl(x)
    out, attn = tl(x, return_attn=True)
    assert attn.shape == (2, 8, 196)
    # maps are a softmax over the spatial axis -> each map sums to 1.
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert torch.equal(out, out_only)


def test_tokenfuser_shape() -> None:
    y = torch.randn(2, 8, 768)
    orig = torch.randn(2, 14, 14, 768)
    fuser = TokenFuser(in_channels=768, num_tokens=8, bottleneck_dim=64)
    out = fuser(y, orig)
    assert out.shape == (2, 196, 768)


def test_tokenfuser_zero_init_is_identity_on_residual() -> None:
    """At init, token_mix has zero weights+bias, so the fuser only contributes the residual.

    LayerNorm(y) is fed into a Linear with zero weights and zero bias, giving
    zero. The einsum is therefore zero and the output equals ``original``.
    """
    torch.manual_seed(0)
    y = torch.randn(2, 8, 768)
    orig = torch.randn(2, 196, 768)
    fuser = TokenFuser(in_channels=768, num_tokens=8, bottleneck_dim=64)
    out = fuser(y, orig)
    assert torch.allclose(out, orig, atol=1e-5)


def test_tokenfuser_grad() -> None:
    y = torch.randn(2, 8, 768, requires_grad=True)
    orig = torch.randn(2, 196, 768, requires_grad=True)
    fuser = TokenFuser(in_channels=768, num_tokens=8)
    fuser(y, orig).sum().backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert orig.grad is not None and torch.isfinite(orig.grad).all()


# ---------------------------------------------------------------------------
# Encoder-level shape checks
# ---------------------------------------------------------------------------


def test_encoder_mod_reduces_token_count() -> None:
    """EncoderMod should drop from N tokens to S tokens once it hits ``loc``."""
    enc = EncoderMod(
        depth=4, embed_dim=128, num_heads=4, mlp_dim=256,
        num_tokens=8, loc=2, use_v11=True,
    )
    x = torch.randn(2, 196, 128)
    out = enc(x)
    assert out.shape == (2, 8, 128)


def test_encoder_mod_fuser_preserves_spatial_dim() -> None:
    """EncoderModFuser should keep the original token count via the fuser path."""
    enc = EncoderModFuser(
        depth=4, embed_dim=128, num_heads=4, mlp_dim=256,
        num_tokens=8, loc=2, use_v11=True,
    )
    x = torch.randn(2, 196, 128)
    out = enc(x)
    assert out.shape == (2, 196, 128)


def test_encoder_mod_loc_equals_depth_is_vanilla() -> None:
    """When loc == depth, the encoder is a plain stack of transformer blocks."""
    enc = EncoderMod(
        depth=3, embed_dim=64, num_heads=4, mlp_dim=128,
        num_tokens=8, loc=3, use_v11=True,
    )
    x = torch.randn(2, 49, 64)
    out = enc(x)
    assert out.shape == (2, 49, 64)


# ---------------------------------------------------------------------------
# ViT-level shape checks (image)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_v11", [True, False])
@pytest.mark.parametrize("use_fuser", [True, False])
def test_vit_image_forward(use_v11: bool, use_fuser: bool) -> None:
    model = TokenLearnerViT(
        img_size=64, patch_size=16, in_chans=3, num_classes=100,
        embed_dim=96, depth=4, num_heads=4, mlp_dim=192,
        num_tokens=8, tokenlearner_loc=2,
        use_v11=use_v11, use_fuser=use_fuser, bottleneck_dim=32,
    )
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 100)


def test_vit_image_backward() -> None:
    model = TokenLearnerViT(
        img_size=64, patch_size=16, in_chans=3, num_classes=10,
        embed_dim=96, depth=4, num_heads=4, mlp_dim=192,
        num_tokens=8, tokenlearner_loc=2, use_v11=True,
    )
    x = torch.randn(2, 3, 64, 64)
    loss = model(x).sum()
    loss.backward()
    # Every learnable parameter should have a finite gradient.
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


# ---------------------------------------------------------------------------
# ViT-level shape checks (video)
# ---------------------------------------------------------------------------


def test_vit_video_forward() -> None:
    model = TokenLearnerViT(
        img_size=64, patch_size=16, in_chans=3, num_classes=50,
        embed_dim=96, depth=4, num_heads=4, mlp_dim=192,
        num_tokens=4, tokenlearner_loc=2,
        num_frames=8, tubelet_t=2, temporal_dimensions=4,
        use_v11=True,
    )
    x = torch.randn(2, 3, 8, 64, 64)
    out = model(x)
    assert out.shape == (2, 50)


def test_vit_video_fuser_forward() -> None:
    model = TokenLearnerViT(
        img_size=64, patch_size=16, in_chans=3, num_classes=50,
        embed_dim=96, depth=4, num_heads=4, mlp_dim=192,
        num_tokens=4, tokenlearner_loc=2,
        num_frames=8, tubelet_t=2, temporal_dimensions=4,
        use_v11=True, use_fuser=True,
    )
    x = torch.randn(2, 3, 8, 64, 64)
    out = model(x)
    assert out.shape == (2, 50)


def test_vit_video_temporal_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        TokenLearnerViT(
            img_size=64, patch_size=16, in_chans=3, num_classes=10,
            embed_dim=96, depth=2, num_heads=4, mlp_dim=128,
            num_tokens=4, tokenlearner_loc=1,
            num_frames=8, tubelet_t=2, temporal_dimensions=8,  # mismatch
        )
