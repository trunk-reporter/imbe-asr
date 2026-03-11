#!/usr/bin/env python3
"""Train Conformer-CTC on decoded IMBE raw parameters (170-dim).

The "middle path": deterministic decode of u[0..7] -> f0, L, spectral
amplitudes, V/UV flags, harmonic mask. Full information, no DCT smoothing.

Usage:
    python -m src.train \
        --pairs-dir data/pairs \
        --librispeech-dir data/LibriSpeech/train-clean-100 \
        --epochs 50 --batch-size 32 --lr 3e-4
"""

import argparse
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import IMBEDataset, collate_fn, get_speaker_split, RAW_PARAM_DIM
from .model import ConformerCTC
from .tokenizer import VOCAB_SIZE
from .eval import compute_wer_cer, decode_batch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def train_epoch(model, loader, optimizer, scheduler, device, accum_steps=1):
    model.train()
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    total_loss = 0.0
    n_batches = 0
    t0 = time.time()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        log_probs, output_lengths = model(features, input_lengths)
        log_probs_t = log_probs.transpose(0, 1)

        loss = ctc_loss_fn(log_probs_t, targets, output_lengths, target_lengths)
        loss = loss / accum_steps
        loss.backward()

        total_loss += loss.item() * accum_steps
        n_batches += 1

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        if n_batches % 50 == 0:
            elapsed = time.time() - t0
            avg = total_loss / n_batches
            rate = n_batches / elapsed
            print("  [%4d/%d] loss=%.4f %.1f batch/s" %
                  (n_batches, len(loader), avg, rate), flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    total_loss = 0.0
    n_batches = 0
    all_refs = []
    all_hyps = []

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        log_probs, output_lengths = model(features, input_lengths)
        log_probs_t = log_probs.transpose(0, 1)
        loss = ctc_loss_fn(log_probs_t, targets, output_lengths, target_lengths)

        total_loss += loss.item()
        n_batches += 1

        refs, hyps = decode_batch(log_probs, output_lengths,
                                  targets, target_lengths)
        all_refs.extend(refs)
        all_hyps.extend(hyps)

    avg_loss = total_loss / max(n_batches, 1)
    wer, cer = compute_wer_cer(all_refs, all_hyps)
    return avg_loss, wer, cer


def main():
    parser = argparse.ArgumentParser(
        description="IMBE raw params (170-dim) CTC training")
    parser.add_argument("--pairs-dir", default="data/pairs")
    parser.add_argument("--librispeech-dir",
                        default="data/LibriSpeech/train-clean-100")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--conv-kernel", type=int, default=31)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--subsample", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="imbe-asr")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    # Data split (by speaker)
    train_speakers, val_speakers = get_speaker_split(
        args.pairs_dir, args.val_fraction)
    print("Speakers: %d train, %d val" %
          (len(train_speakers), len(val_speakers)))

    # Datasets
    print("Loading training data (170-dim raw IMBE params)...")
    train_ds = IMBEDataset(
        args.pairs_dir, args.librispeech_dir,
        speaker_ids=train_speakers, normalize=True,
    )
    stats = train_ds.get_stats()

    print("Loading validation data...")
    val_ds = IMBEDataset(
        args.pairs_dir, args.librispeech_dir,
        speaker_ids=val_speakers, normalize=True, stats=stats,
    )
    print("Samples: %d train, %d val" % (len(train_ds), len(val_ds)))

    # Save normalization stats
    np.savez(os.path.join(args.output, "stats.npz"),
             mean=stats[0], std=stats[1])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    # Model
    model = ConformerCTC(
        input_dim=RAW_PARAM_DIM,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        conv_kernel=args.conv_kernel,
        vocab_size=VOCAB_SIZE,
        dropout=args.dropout,
        subsample=args.subsample,
    ).to(device)

    n_params = model.count_parameters()
    print("Model: %s parameters (%.1fM)" % (f"{n_params:,}", n_params / 1e6))
    print("Input: %d-dim raw IMBE params -> Linear -> %d-dim" %
          (RAW_PARAM_DIM, args.d_model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), weight_decay=0.01)

    total_steps = args.epochs * (len(train_loader) // args.accum_steps)
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_wer = float("inf")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_wer" in ckpt:
            best_wer = ckpt["best_wer"]
        print("Resumed from epoch %d" % start_epoch)

    # W&B
    if args.wandb and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or "raw-params-170dim",
            config={
                "phase": "pretrain",
                "input": "raw_params",
                "input_dim": RAW_PARAM_DIM,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "d_ff": args.d_ff,
                "conv_kernel": args.conv_kernel,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "accum_steps": args.accum_steps,
                "effective_batch": args.batch_size * args.accum_steps,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "epochs": args.epochs,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "n_params": n_params,
            },
        )
        wandb.watch(model, log_freq=100)

    print("\nTraining: %d epochs, batch=%d, accum=%d, lr=%s" %
          (args.epochs, args.batch_size, args.accum_steps, args.lr))
    print("Batches per epoch: %d\n" % len(train_loader))

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 device, accum_steps=args.accum_steps)
        val_loss, val_wer, val_cer = validate(model, val_loader, device)
        dt = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        improved = " *" if val_wer < best_wer else ""
        print("Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
              "WER=%.1f%%  CER=%.1f%%  lr=%.2e  time=%.0fs%s" %
              (epoch + 1, args.epochs, train_loss, val_loss,
               val_wer, val_cer, lr, dt, improved))

        if args.wandb and HAS_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "wer": val_wer,
                "cer": val_cer,
                "lr": lr,
                "epoch_time": dt,
                "best_wer": min(best_wer, val_wer),
            })

        if val_wer < best_wer:
            best_wer = val_wer
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "best_wer": best_wer,
                "config": {
                    "input_dim": RAW_PARAM_DIM,
                    "d_model": args.d_model,
                    "n_layers": args.n_layers,
                    "n_heads": args.n_heads,
                    "d_ff": args.d_ff,
                    "conv_kernel": args.conv_kernel,
                    "vocab_size": VOCAB_SIZE,
                    "dropout": args.dropout,
                    "subsample": args.subsample,
                    "input_type": "raw_params",
                },
            }, os.path.join(args.output, "best.pth"))

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "best_wer": best_wer,
            }, os.path.join(args.output, "epoch_%d.pth" % (epoch + 1)))
            print("  Saved checkpoint epoch_%d.pth" % (epoch + 1))

    print("\nDone. Best WER: %.1f%%" % best_wer)

    if args.wandb and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
