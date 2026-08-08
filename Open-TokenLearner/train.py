"""Minimal training entry point.

Loads a YAML config, builds a :class:`TokenLearnerViT`, and trains on an
``ImageFolder``-style dataset with AdamW + cosine schedule and label
smoothing. Kept small on purpose — the model code is the interesting part;
this script is here so you can sanity-check the model end to end and
swap in a serious recipe (timm, FFCV, DeepSpeed, etc.) at your leisure.

Usage:
    python train.py --config configs/vit_b16_imagenet.yaml --output runs/b16
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader

from tokenlearner import TokenLearnerViT


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs for reproducible runs (Scenic uses
    ``rng_seed=42``). cuDNN autotuning is left on for throughput."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """Per-worker RNG seeding so DataLoader augmentation is reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(cfg: dict[str, Any]) -> TokenLearnerViT:
    m = dict(cfg["model"])
    m.pop("name", None)
    return TokenLearnerViT(**m)


def build_loaders(cfg: dict[str, Any], seed: int = 42) -> tuple[DataLoader, DataLoader]:
    """ImageNet-style ImageFolder loaders. Override for other datasets."""
    from torchvision import datasets, transforms

    img_size = cfg["model"]["img_size"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    root = data_cfg["root"]
    train_ds = datasets.ImageFolder(os.path.join(root, data_cfg["train_split"]), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(root, data_cfg["val_split"]), val_tf)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=loader_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )
    return train_loader, val_loader


def lr_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    end_lr: float,
    schedule: str,
) -> float:
    """Learning rate with linear warmup then linear (default) or cosine decay.

    The Scenic TokenLearner recipe uses ``constant * linear_warmup *
    linear_decay`` (linear decay to ``end_learning_rate``), so ``linear`` is the
    aligned default; ``cosine`` is kept as an option.
    """
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "linear":
        return (base_lr - end_lr) * (1.0 - progress) + end_lr
    if schedule == "cosine":
        return end_lr + 0.5 * (base_lr - end_lr) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown lr_schedule {schedule!r}; expected 'linear' or 'cosine'.")


def compute_loss(
    loss_type: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    label_smoothing: float,
) -> torch.Tensor:
    """Training loss.

    ``sigmoid`` reproduces the Scenic recipe: sigmoid cross-entropy on one-hot
    targets, summed over classes and averaged over the batch, with no label
    smoothing. ``softmax`` is standard cross-entropy with optional smoothing.
    """
    if loss_type == "sigmoid":
        onehot = F.one_hot(targets, num_classes).to(logits.dtype)
        per_class = F.binary_cross_entropy_with_logits(logits, onehot, reduction="none")
        return per_class.sum(dim=-1).mean()
    if loss_type == "softmax":
        return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss {loss_type!r}; expected 'sigmoid' or 'softmax'.")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=-1) == targets).sum().item()
        total += targets.size(0)
    return correct / max(1, total)


def init_wandb(cfg: dict[str, Any], run_name: str, resume_run_id: str | None) -> wandb.sdk.wandb_run.Run:
    """Initialize a wandb run from the ``wandb:`` config section.

    ``WANDB_MODE`` env var overrides the config's ``mode`` field. When the run
    is resumed from a checkpoint that carried a ``wandb_run_id``, we resume it.
    """
    w = cfg.get("wandb", {}) or {}
    return wandb.init(
        project=w.get("project", "opentokenlearner"),
        entity=w.get("entity"),
        name=run_name,
        tags=w.get("tags"),
        mode=os.environ.get("WANDB_MODE", w.get("mode", "online")),
        config=cfg,
        job_type="train",
        id=resume_run_id,
        resume="allow" if resume_run_id else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="runs/exp")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--run-name", default=None, help="wandb run name (defaults to output dir basename)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    train_loader, val_loader = build_loaders(cfg, seed=seed)

    train_cfg = cfg["train"]
    wandb_cfg = cfg.get("wandb", {}) or {}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["base_lr"],
        betas=tuple(train_cfg["betas"]),
        weight_decay=train_cfg["weight_decay"],
    )
    # Loss + schedule (Scenic recipe: sigmoid loss, linear-decay schedule).
    loss_type = train_cfg.get("loss", "sigmoid")
    label_smoothing = train_cfg.get("label_smoothing") or 0.0
    lr_schedule = train_cfg.get("lr_schedule", "linear")
    num_classes = cfg["model"]["num_classes"]
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["mixed_precision"] and device.type == "cuda")

    # Gradient accumulation emulates a larger effective batch (Scenic uses bs
    # 4096): effective batch = batch_size * grad_accum_steps. Scale base_lr with
    # the effective batch accordingly.
    accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    # Step-based eval/checkpoint cadence in optimizer updates (Scenic uses
    # log_eval_steps / checkpoint_steps). 0/None falls back to per-epoch.
    eval_steps = int(train_cfg.get("eval_steps") or 0)
    checkpoint_steps = int(train_cfg.get("checkpoint_steps") or 0)

    epochs = train_cfg["epochs"]
    # A "step" is one optimizer update, i.e. accum_steps micro-batches.
    steps_per_epoch = len(train_loader) // accum_steps
    total_steps = epochs * steps_per_epoch
    # Warmup: prefer an explicit step count, else a fraction of total training
    # (preserves the Scenic warmup shape across batch sizes), else warmup_epochs.
    if train_cfg.get("warmup_steps") is not None:
        warmup_steps = int(train_cfg["warmup_steps"])
    elif train_cfg.get("warmup_fraction") is not None:
        warmup_steps = int(train_cfg["warmup_fraction"] * total_steps)
    else:
        warmup_steps = int(train_cfg.get("warmup_epochs", 0)) * steps_per_epoch

    start_epoch = 0
    resume_step = None
    resume_run_id = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        resume_step = ckpt.get("global_step")
        resume_run_id = ckpt.get("wandb_run_id")

    run_name = args.run_name or out_dir.name
    log_freq = int(wandb_cfg.get("log_freq", 50))
    log_model = bool(wandb_cfg.get("log_model", False))

    with init_wandb(cfg, run_name, resume_run_id) as run:
        if wandb_cfg.get("watch_model"):
            wandb.watch(model, log="all", log_freq=max(log_freq, 100))

        best_acc = 0.0
        last_acc = 0.0
        lr = train_cfg["base_lr"]
        global_step = resume_step if resume_step is not None else start_epoch * steps_per_epoch

        def evaluate_and_log(step: int, epoch: int) -> float:
            nonlocal best_acc, last_acc
            last_acc = evaluate(model, val_loader, device)
            model.train()
            wandb.log({"val/top1": last_acc, "epoch": epoch}, step=step)
            if last_acc > best_acc:
                best_acc = last_acc
                run.summary["val/best_top1"] = best_acc
                run.summary["val/best_epoch"] = epoch
            return last_acc

        def save_checkpoint(epoch: int, step: int) -> None:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": step,
                    "cfg": cfg,
                    "wandb_run_id": run.id,
                },
                out_dir / "last.pt",
            )

        for epoch in range(start_epoch, epochs):
            model.train()
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            micro = 0
            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    logits = model(images)
                    # Divide by accum_steps so accumulated grads average to the
                    # mean loss over the effective batch.
                    loss = compute_loss(loss_type, logits, targets, num_classes, label_smoothing) / accum_steps

                scaler.scale(loss).backward()
                running_loss += loss.item()
                micro += 1
                if micro % accum_steps != 0:
                    continue

                # One optimizer update per accum_steps micro-batches.
                lr = lr_at_step(
                    global_step, total_steps, warmup_steps,
                    train_cfg["base_lr"], train_cfg["min_lr"], lr_schedule,
                )
                for g in optimizer.param_groups:
                    g["lr"] = lr

                if train_cfg.get("grad_clip_norm"):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if global_step % log_freq == 0:
                    wandb.log(
                        {"train/loss": running_loss, "train/lr": lr, "epoch": epoch},
                        step=global_step,
                    )
                running_loss = 0.0
                global_step += 1

                if eval_steps and global_step % eval_steps == 0:
                    acc = evaluate_and_log(global_step, epoch)
                    print(f"step {global_step:7d}  lr {lr:.2e}  val_top1 {acc:.4f}")
                if checkpoint_steps and global_step % checkpoint_steps == 0:
                    save_checkpoint(epoch, global_step)

            epoch_time = time.time() - t0
            # Fall back to per-epoch eval/checkpoint when step cadence is off.
            acc = last_acc if eval_steps else evaluate_and_log(global_step, epoch)
            print(
                f"epoch {epoch:3d}  lr {lr:.2e}  val_top1 {acc:.4f}  time {epoch_time:.1f}s"
            )
            wandb.log({"epoch": epoch, "epoch_time_s": epoch_time}, step=global_step)
            if not checkpoint_steps:
                save_checkpoint(epoch, global_step)

        # Final eval + checkpoint so the last state is always captured.
        evaluate_and_log(global_step, epochs - 1)
        save_checkpoint(epochs - 1, global_step)
        acc = last_acc

        if log_model:
            art = wandb.Artifact(
                f"{run_name}-checkpoint",
                type="model",
                metadata={"epoch": epochs - 1, "val_top1": acc, "val_best_top1": best_acc},
            )
            art.add_file(str(out_dir / "last.pt"))
            run.log_artifact(art, aliases=["latest"])


if __name__ == "__main__":
    main()
