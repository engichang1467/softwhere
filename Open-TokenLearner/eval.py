"""Evaluation entry point: top-1 accuracy on the val split.

Usage:
    python eval.py --config configs/vit_b16_imagenet.yaml --checkpoint runs/b16/last.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import wandb

from train import build_loaders, build_model, evaluate, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", default=None, help="wandb run name (defaults to checkpoint stem)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    _, val_loader = build_loaders(cfg)

    w = cfg.get("wandb", {}) or {}
    run_name = args.run_name or f"eval-{Path(args.checkpoint).stem}"
    with wandb.init(
        project=w.get("project", "opentokenlearner"),
        entity=w.get("entity"),
        name=run_name,
        tags=(w.get("tags") or []) + ["eval"],
        mode=os.environ.get("WANDB_MODE", w.get("mode", "online")),
        config={**cfg, "eval_checkpoint": args.checkpoint, "train_run_id": ckpt.get("wandb_run_id")},
        job_type="eval",
    ) as run:
        acc = evaluate(model, val_loader, device)
        print(f"val_top1 {acc:.4f}")
        wandb.log({"val/top1": acc})
        run.summary["val/top1"] = acc


if __name__ == "__main__":
    main()
