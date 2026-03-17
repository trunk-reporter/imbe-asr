#!/usr/bin/env python3
"""Train Conformer-CTC on decoded IMBE raw parameters (170-dim).

The "middle path": deterministic decode of u[0..7] -> f0, L, spectral
amplitudes, V/UV flags, harmonic mask. Full information, no DCT smoothing.

Usage:
    # Single-GPU (backward compatible):
    python -m src.train \
        --pairs-dir data/pairs \
        --librispeech-dir data/LibriSpeech/train-clean-100 \
        --epochs 50 --batch-size 32 --lr 3e-4

    # Multi-source:
    python -m src.train \
        --data-config configs/data_expanded.yaml \
        --epochs 30 --batch-size 32 --lr 2e-4

    # Multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=2 -m src.train \
        --data-config configs/data_expanded.yaml \
        --epochs 30 --batch-size 32 --lr 2e-4
"""

import argparse
import math
import os
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import IMBEDataset, collate_fn, get_speaker_split, RAW_PARAM_DIM
from .model import ConformerCTC
from .tokenizer import VOCAB_SIZE
from .eval import compute_wer_cer, decode_batch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def s3_upload(local_path, s3_uri, endpoint=None):
    """Upload a file to S3 in the background."""
    cmd = ["aws", "s3", "cp", local_path, s3_uri]
    if endpoint:
        cmd += ["--endpoint-url", endpoint]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main():
    return get_rank() == 0


def train_epoch(model, loader, optimizer, scheduler, device, accum_steps=1,
                use_wandb=False, use_amp=False):
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

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=use_amp):
            log_probs, output_lengths = model(features, input_lengths)
            log_probs_t = log_probs.transpose(0, 1)
            loss = ctc_loss_fn(log_probs_t, targets, output_lengths,
                               target_lengths)

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

            # Log to W&B every optimizer step
            if use_wandb and is_main() and (step + 1) % (accum_steps * 50) == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({
                    "train/loss": total_loss / n_batches,
                    "train/batch_loss": loss.item() * accum_steps,
                    "train/lr": lr,
                    "train/batch_rate": n_batches / (time.time() - t0),
                })

        if is_main() and n_batches % 50 == 0:
            elapsed = time.time() - t0
            avg = total_loss / n_batches
            rate = n_batches / elapsed
            print("  [%4d/%d] loss=%.4f %.1f batch/s" %
                  (n_batches, len(loader), avg, rate), flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, use_amp=False):
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

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=use_amp):
            log_probs, output_lengths = model(features, input_lengths)
            log_probs_t = log_probs.transpose(0, 1)
            loss = ctc_loss_fn(log_probs_t, targets, output_lengths,
                               target_lengths)

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
    parser.add_argument("--data-config", default=None,
                        help="YAML config for multi-source data (overrides "
                             "--pairs-dir/--librispeech-dir)")
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
    parser.add_argument("--amp", action="store_true",
                        help="Mixed precision training (bf16)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--mmap-dir", default=None,
                        help="Dir with packed .features.bin/.meta.pkl "
                             "(from scripts/pack_dataset.py)")
    parser.add_argument("--s3-sync", default=None,
                        help="S3 URI to sync checkpoints to "
                             "(e.g. s3://bucket/imbe-asr/checkpoints)")
    parser.add_argument("--s3-endpoint", default=None,
                        help="S3 endpoint URL (for OVH/non-AWS)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="imbe-asr")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    # ----- DDP setup ----- #
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)

    rank = get_rank()
    world_size = get_world_size()

    os.makedirs(args.output, exist_ok=True)

    # Datasets -- multi-source or single-source
    if is_main():
        print("Loading training data (170-dim raw IMBE params)...")
    if args.mmap_dir:
        from .dataset_unified import MmapIMBEDataset, load_data_config

        bin_path = os.path.join(args.mmap_dir, "all.features.bin")
        val_frac = 0.05
        if args.data_config:
            cfg = load_data_config(args.data_config)
            val_frac = cfg.get("val_fraction", 0.05)

        train_ds = MmapIMBEDataset(
            bin_path, split="train", val_fraction=val_frac, normalize=True)
        stats = train_ds.get_stats()

        if is_main():
            print("Loading validation data...")
        val_ds = MmapIMBEDataset(
            bin_path, split="val", val_fraction=val_frac,
            normalize=True, stats=stats)
    elif args.data_config:
        from .dataset_unified import UnifiedIMBEDataset

        train_ds = UnifiedIMBEDataset(
            args.data_config, split="train", normalize=True)
        stats = train_ds.get_stats()

        if is_main():
            print("Loading validation data...")
        val_ds = UnifiedIMBEDataset(
            args.data_config, split="val", normalize=True, stats=stats)
    else:
        # Single-source backward-compatible path
        train_speakers, val_speakers = get_speaker_split(
            args.pairs_dir, args.val_fraction)
        if is_main():
            print("Speakers: %d train, %d val" %
                  (len(train_speakers), len(val_speakers)))

        train_ds = IMBEDataset(
            args.pairs_dir, args.librispeech_dir,
            speaker_ids=train_speakers, normalize=True,
        )
        stats = train_ds.get_stats()

        if is_main():
            print("Loading validation data...")
        val_ds = IMBEDataset(
            args.pairs_dir, args.librispeech_dir,
            speaker_ids=val_speakers, normalize=True, stats=stats,
        )

    if is_main():
        print("Samples: %d train, %d val" % (len(train_ds), len(val_ds)))

    # Save normalization stats (rank 0 only)
    if is_main():
        np.savez(os.path.join(args.output, "stats.npz"),
                 mean=stats[0], std=stats[1])

    # DataLoaders -- use DistributedSampler for DDP
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler,
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

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # For counting params / saving, use the unwrapped module
    raw_model = model.module if distributed else model

    n_params = raw_model.count_parameters()
    if is_main():
        print("Model: %s parameters (%.1fM)" %
              (f"{n_params:,}", n_params / 1e6))
        print("Input: %d-dim raw IMBE params -> Linear -> %d-dim" %
              (RAW_PARAM_DIM, args.d_model))
        if distributed:
            print("DDP: %d GPUs" % world_size)

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
        raw_model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_wer" in ckpt:
            best_wer = ckpt["best_wer"]
        if is_main():
            print("Resumed from epoch %d" % start_epoch)

    # W&B (rank 0 only)
    if args.wandb and HAS_WANDB and is_main():
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
                "effective_batch": args.batch_size * args.accum_steps * world_size,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "epochs": args.epochs,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "n_params": n_params,
                "world_size": world_size,
            },
        )
        wandb.watch(raw_model, log_freq=100)

    if is_main():
        print("\nTraining: %d epochs, batch=%d, accum=%d, lr=%s%s" %
              (args.epochs, args.batch_size, args.accum_steps, args.lr,
               ", amp=bf16" if args.amp else ""))
        print("Batches per epoch: %d\n" % len(train_loader))

    for epoch in range(start_epoch, args.epochs):
        # Set epoch on DistributedSampler for proper shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        use_wandb = args.wandb and HAS_WANDB and is_main()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 device, accum_steps=args.accum_steps,
                                 use_wandb=use_wandb, use_amp=args.amp)
        val_loss, val_wer, val_cer = validate(model, val_loader, device,
                                              use_amp=args.amp)
        dt = time.time() - t0

        # Only log / checkpoint on rank 0
        if is_main():
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

            # Save checkpoint every epoch (for preemption safety)
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "best_wer": min(best_wer, val_wer),
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
            }

            if val_wer < best_wer:
                best_wer = val_wer
                best_path = os.path.join(args.output, "best.pth")
                torch.save(ckpt_data, best_path)
                if args.s3_sync:
                    s3_upload(best_path,
                              args.s3_sync + "/best.pth",
                              args.s3_endpoint)

            # Save latest every epoch + periodic named checkpoints
            latest_path = os.path.join(args.output, "latest.pth")
            torch.save(ckpt_data, latest_path)
            if args.s3_sync:
                s3_upload(latest_path,
                          args.s3_sync + "/latest.pth",
                          args.s3_endpoint)

            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                epoch_path = os.path.join(
                    args.output, "epoch_%d.pth" % (epoch + 1))
                torch.save(ckpt_data, epoch_path)
                print("  Saved checkpoint epoch_%d.pth" % (epoch + 1))
                if args.s3_sync:
                    s3_upload(epoch_path,
                              args.s3_sync + "/epoch_%d.pth" % (epoch + 1),
                              args.s3_endpoint)

    if is_main():
        print("\nDone. Best WER: %.1f%%" % best_wer)

    if args.wandb and HAS_WANDB and is_main():
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
