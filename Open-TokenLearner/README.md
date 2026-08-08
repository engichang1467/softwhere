# Open TokenLearner (PyTorch)

A PyTorch reimplementation of **TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?** (Ryoo et al., IEEE TPAMI / NeurIPS 2021, [arXiv:2106.11297](https://arxiv.org/abs/2106.11297)).

TokenLearner is a small learnable module that replaces fixed patch tokenization with a handful of adaptively mined tokens. Instead of pushing all `H*W` (or `H*W*T`) patch tokens through every transformer layer, it learns to produce `S` tokens (typically 8 or 16) from an intermediate feature map. Because attention is quadratic in token count, every layer after the module gets dramatically cheaper while accuracy holds or improves.

The original code is written in JAX/Flax as part of [google-research/scenic](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner). This repo ports the modules and the ViT/ViViT integration to PyTorch.

## What this repo covers

- `TokenLearner` (v1.0): the conv-based spatial attention module from the paper experiments. Four `3x3` conv layers with GeLU, sigmoid gating, sum (or mean) pooling.
- `TokenLearnerV11` (v1.1): the MLP-based variant. A two-layer MLP with GeLU produces the attention weights, softmax normalizes over space, and tokens are formed by a weighted sum (einsum). The authors report this version works better in general, especially when inserted early.
- `TokenFuser`: mixes information across the learned tokens with a token-axis linear layer, then remaps them back to the original `H*W` spatial resolution so downstream layers can keep the original shape. Used with a residual connection.
- ViT integration in two flavors:
  - **TokenLearner only** (Figure 3a in the paper): insert the module once, after layer `tokenlearner_loc`, then continue with standard transformer blocks on the reduced token set.
  - **TokenLearner + TokenFuser** (Figure 3b): from `tokenlearner_loc` onward, every layer becomes `TokenLearner -> Transformer -> TokenFuser` with a residual back to the pre-module features.
- Video support through a `temporal_dimensions` flag. The module runs per frame, producing `S` tokens per frame for `S*T` tokens total, which a joint space-time transformer then attends over.

## Module math

Let `X` be an input feature map of shape `[B, H, W, C]` (or `[B, H*W, C]`).

**TokenLearner v1.0.** For each of `S` tokens, a spatial attention map `alpha_i(X)` of shape `[H, W]` is computed by a stack of conv layers, gated with a sigmoid, broadcast against `X`, and pooled:

```
z_i = pool( X (.) sigmoid(alpha_i(X)) )      shape [B, C]
Z   = [z_1, ..., z_S]                          shape [B, S, C]
```

The reference uses four `3x3` convs with `num_tokens` output channels (no bias), GeLU between the first three, then sum pooling over the spatial axis. LayerNorm is applied to the input first.

**TokenLearner v1.1.** Spatial attention comes from an MLP (`bottleneck_dim` hidden units, GeLU, `num_tokens` output), softmax over the spatial axis, then a weighted sum:

```
A = softmax( MLP(LayerNorm(X)) , axis=space )   shape [B, S, H*W]
Z = einsum('bsi,bic->bsc', A, X)                shape [B, S, C]
```

**TokenFuser.** Given transformer outputs `Y` of shape `[B, S, C]` and the pre-module features of shape `[B, H*W, C]`:

1. Mix across tokens: transpose to `[B, C, S]`, apply a `Dense(S)` (zero-initialized) over the token axis, transpose back. This is the token-wise linear layer connected to the MLP-Mixer observation in the paper.
2. Remap to space: compute a mask `M = sigmoid(MLP(LayerNorm(original)))` of shape `[B, H*W, S]`, then `out = einsum('bsc,bhs->bhc', Y_mixed, M)` to get `[B, H*W, C]`.
3. Add the residual `original`.

## Installation

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Core requirements: `torch>=2.0`, `einops`, `numpy`. Add `timm` if you want to borrow a ViT backbone rather than building one from scratch.

## Training data

The training script auto-detects two image-classification layouts:

- ImageFolder directories such as `data/imagenette2-320/train/<class>` and `data/imagenette2-320/val/<class>`.
- Hugging Face parquet shards such as `data/imagenet-1k/data/train-*.parquet` and `data/imagenet-1k/data/validation-*.parquet`.
- Use the script `parquet2jpeg.py` to convert parquet files into jpeg formatted iamges.

For ImageNet-1k via Hugging Face, first install the full requirements and download the gated dataset:

```bash
uv pip install -r requirements.txt
sh scripts/download_imagenet1k.sh
python parquet2jpeg.py
python train.py --config configs/vit_b16_imagenet.yaml --output run/imagenet-1k
```

## Quick start

```python
import torch
from tokenlearner import TokenLearnerV11, TokenLearner, TokenFuser

x = torch.randn(2, 14, 14, 768)          # [B, H, W, C] intermediate features

# v1.1 (MLP + softmax), recommended default
tl = TokenLearnerV11(in_channels=768, num_tokens=8, bottleneck_dim=64)
tokens = tl(x)                            # [2, 8, 768]

# v1.0 (conv + sigmoid), matches the paper's main experiments
tl_v1 = TokenLearner(in_channels=768, num_tokens=8, use_sum_pooling=True)
tokens_v1 = tl_v1(x)                      # [2, 8, 768]

# fuse processed tokens back to spatial resolution
fuser = TokenFuser(in_channels=768, num_tokens=8, bottleneck_dim=64)
restored = fuser(tokens, original=x)      # [2, 196, 768], residual already added
```

### Inside a ViT

```python
from tokenlearner import TokenLearnerViT

model = TokenLearnerViT(
    img_size=224, patch_size=16, in_chans=3, num_classes=1000,
    embed_dim=768, depth=12, num_heads=12, mlp_dim=3072,
    num_tokens=8,            # S
    tokenlearner_loc=6,      # insert after layer 6 (mid-network)
    use_v11=True,
    use_fuser=False,         # True for the TokenLearner+TokenFuser architecture
)
logits = model(torch.randn(2, 3, 224, 224))   # [2, 1000]
```

### Video (ViViT-style)

```python
model = TokenLearnerViT(
    img_size=224, patch_size=16, tubelet_t=2, num_frames=32,
    embed_dim=1024, depth=24, num_heads=16, mlp_dim=4096,
    num_tokens=16, tokenlearner_loc=12, use_v11=True,
    temporal_dimensions=16,  # number of temporal positions; learns S tokens per frame -> S*T total
)
logits = model(torch.randn(2, 3, 32, 224, 224))
```

## Where to place the module

The paper's ablation (Figure 4, Table 1) is the main design guide:

- Inserting at the middle (`loc = depth/2`) gives roughly the same accuracy as the base ViT while cutting FLOPS by about half.
- Inserting later (after ~3/4 of the network) can beat the base model and still runs faster.
- Inserting very early (after layer 2 or 3) in a large model can make it cheaper than a small base model while scoring higher (Table 4): ViT-L/16 with `16-TL at 3` beat ViT-B/16 at roughly half the runtime.
- `S = 8` and `S = 16` are the recommended defaults. v1.1 tends to win when the module sits early.

## Reference numbers to target

These come from the paper and are the targets a faithful reimplementation should approach (JFT pre-training, ImageNet fine-tune unless noted). Exact reproduction depends on access to JFT-300M, which is internal to Google; expect to validate on ImageNet-1k or ImageNet-21k pre-training instead and treat these as directional.

| Model | GFLOPS | ImageNet Top-1 |
|---|---|---|
| ViT B/16 (base) | 55.6 | 84.73 |
| TokenLearner B/16 | 28.7 | 83.65 |
| TokenLearner B/16 (21 layers) | 47.1 | 85.21 |
| 16-TokenLearner B/16 (21 layers) | 47.7 | 85.45 |
| ViT L/16 (base) | 363.1 | 87.35 |
| L/16, 16-TL at 12 | 178.1 | 87.68 |
| TL L/8 (24+11) | — | 88.87 (ReaL 91.05) |

Video (Kinetics-400, ViViT-L backbone): base ViViT-L/16 scores 83.4 Top-1 at 1446 GFLOPS/view; `TokenLearner 16at12 + L/16` matches it at 766 GFLOPS; `TokenLearner 16at18 + L/10` reaches 85.4. On Charades the bottleneck-transformer variant reports 66.3 mAP, and on AViD 53.8 accuracy.

## Notes on the port

- The original is Flax with `[B, H, W, C]` channel-last tensors. PyTorch convs are channel-first, so the conv-based v1.0 module transposes to `[B, C, H, W]` internally and back. v1.1 operates on flattened `[B, H*W, C]` and needs no transpose.
- The TokenFuser token-axis linear layer is zero-initialized in the reference (`kernel_init=zeros`), which makes the fuser start as a near-identity contribution on top of the residual. Keep this init to match training dynamics.
- `temporal_dimensions` controls the per-frame reshape: features are folded to `[B*T, H*W, C]` before the module and unfolded to `[B, T*S, C]` after, so the joint transformer sees `S*T` tokens. For images set it to 1.
- v1.0 uses sum pooling by default; when sum pooling is used the TokenFuser keeps its LayerNorms (`use_normalization=True`) to control activation scale.
- LayerNorm placement and the GeLU activation follow the reference exactly; small numerical differences from JAX are expected.

## Citation

```bibtex
@article{ryoo2021tokenlearner,
  title={TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?},
  author={Ryoo, Michael S. and Piergiovanni, AJ and Arnab, Anurag and Dehghani, Mostafa and Angelova, Anelia},
  journal={arXiv preprint arXiv:2106.11297},
  year={2021}
}
```

## License

The original Scenic code is Apache 2.0. This reimplementation follows the same license. JFT-300M is not publicly available; pre-train on a public dataset (ImageNet-21k, LAION subsets, etc.) to reproduce results.
