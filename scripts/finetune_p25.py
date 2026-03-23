#!/usr/bin/env python3
"""Fine-tune base IMBE-ASR model on P25 pseudo-labeled data.

Mixes P25 data with base LibriSpeech data to prevent catastrophic forgetting.
Uses lower LR and fewer epochs than base training.

Usage:
    python3 scripts/finetune_p25.py \
        --checkpoint checkpoints/eddie_512d/best.pth \
        --p25-dir data/p25_labeled \
        --base-mmap data/packed_eddie \
        --output checkpoints/p25_finetuned

    # Multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=2 scripts/finetune_p25.py \
        --checkpoint checkpoints/sarah_1024d/best.pth \
        --p25-dir data/p25_labeled \
        --base-mmap data/packed \
        --output checkpoints/p25_finetuned_1024d --amp
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import collate_fn, RAW_PARAM_DIM
from src.dataset_p25 import P25Dataset
from src.dataset_unified import MmapIMBEDataset
from src.model import ConformerCTC
from src.tokenizer import VOCAB_SIZE
from src.eval import compute_wer_cer, decode_batch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main():
    return dist.get_rank() == 0 if is_distributed() else True


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def main():
    parser = argparse.ArgumentParser(description="Fine-tune IMBE-ASR on P25 data")
    parser.add_argument("--checkpoint", required=True, help="Base model checkpoint")
    parser.add_argument("--p25-dir", default="data/p25_labeled",
                        help="P25 pseudo-labeled NPZ directory")
    parser.add_argument("--base-mmap", default=None,
                        help="Base training data mmap dir (for mixing)")
    parser.add_argument("--base-fraction", type=float, default=0.3,
                        help="Fraction of base data to mix in (default: 0.3)")
    parser.add_argument("--output", default="checkpoints/p25_finetuned")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Fine-tuning LR (lower than base training)")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name", default="p25-finetune")
    args = parser.parse_args()

    # DDP setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output, exist_ok=True)

    # Load base model
    if is_main():
        print("Loading checkpoint: %s" % args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = ConformerCTC(
        input_dim=cfg["input_dim"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"], conv_kernel=cfg["conv_kernel"],
        vocab_size=cfg["vocab_size"], dropout=cfg.get("dropout", 0.1),
        subsample=cfg.get("subsample", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    n_params = sum(p.numel() for p in model.parameters())
    if is_main():
        print("Model: %s parameters (%.1fM)" % (f"{n_params:,}", n_params / 1e6))
        print("Base WER: %.1f%%" % ckpt.get("best_wer", -1))

    # Load stats from base checkpoint
    stats_path = os.path.join(os.path.dirname(args.checkpoint), "stats.npz")
    s = np.load(stats_path)
    base_stats = (s["mean"], s["std"])
    if is_main():
        print("Using base normalization stats from %s" % stats_path)

    # P25 dataset
    if is_main():
        print("\nLoading P25 data from %s..." % args.p25_dir)
    p25_train = P25Dataset(
        args.p25_dir, normalize=True, stats=base_stats,
        split="train", val_fraction=0.05)
    p25_val = P25Dataset(
        args.p25_dir, normalize=True, stats=base_stats,
        split="val", val_fraction=0.05)
    if is_main():
        print("P25: %d train, %d val" % (len(p25_train), len(p25_val)))

    # Mix with base data if provided
    if args.base_mmap:
        if is_main():
            print("Loading base data from %s (%.0f%% mix)..." %
                  (args.base_mmap, args.base_fraction * 100))
        bin_path = os.path.join(args.base_mmap, "all.features.bin")
        base_train = MmapIMBEDataset(
            bin_path, split="train", val_fraction=0.05,
            normalize=True, stats=base_stats,
            data_fraction=args.base_fraction)
        base_val = MmapIMBEDataset(
            bin_path, split="val", val_fraction=0.05,
            normalize=True, stats=base_stats)
        if is_main():
            print("Base: %d train (%.0f%% subset), %d val" %
                  (len(base_train), args.base_fraction * 100, len(base_val)))

        train_ds = ConcatDataset([p25_train, base_train])
        val_ds = ConcatDataset([p25_val, base_val])
        if is_main():
            print("Combined: %d train, %d val" % (len(train_ds), len(val_ds)))
    else:
        train_ds = p25_train
        val_ds = p25_val

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
        val_sampler = DistributedSampler(
            val_ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn, num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0)

    if distributed:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if distributed else model

    if is_main() and distributed:
        print("DDP: %d GPUs" % get_world_size())

    # Optimizer — lower LR for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)

    total_steps = args.epochs * (len(train_loader) // args.accum_steps)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # W&B
    if args.wandb and HAS_WANDB and is_main():
        wandb.init(project="imbe-asr", name=args.wandb_name, config={
            "task": "p25_finetune",
            "base_checkpoint": args.checkpoint,
            "base_wer": ckpt.get("best_wer", -1),
            "p25_samples": len(p25_train),
            "base_fraction": args.base_fraction,
            "lr": args.lr,
            "epochs": args.epochs,
        })

    # Save base stats
    if is_main():
        np.savez(os.path.join(args.output, "stats.npz"),
                 mean=base_stats[0], std=base_stats[1])

    best_wer = float("inf")

    if is_main():
        print("\nFine-tuning: %d epochs, batch=%d, accum=%d, lr=%.1e, amp=%s" %
              (args.epochs, args.batch_size, args.accum_steps, args.lr,
               "bf16" if args.amp else "off"))
        print("Batches per epoch: %d\n" % len(train_loader))

    for epoch in range(args.epochs):
        t0 = time.time()
        if distributed:
            train_sampler.set_epoch(epoch)

        # ---- Train ----
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.amp):
                log_probs, output_lengths = model(features, input_lengths)
                log_probs_t = log_probs.transpose(0, 1)
                loss = ctc_loss_fn(log_probs_t, targets, output_lengths,
                                   target_lengths)

            (loss / args.accum_steps).backward()
            total_loss += loss.item()
            n_batches += 1

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if n_batches % 100 == 0 and is_main():
                avg = total_loss / n_batches
                rate = n_batches / (time.time() - t0)
                print("  [%4d/%d] loss=%.4f %.1f batch/s" %
                      (n_batches, len(train_loader), avg, rate), flush=True)

        train_loss = total_loss / max(n_batches, 1)

        # ---- Validate ----
        model.eval()
        val_loss_total = 0.0
        val_n = 0
        all_refs = []
        all_hyps = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)
                input_lengths = batch["input_lengths"].to(device)
                target_lengths = batch["target_lengths"].to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=args.amp):
                    log_probs, output_lengths = model(features, input_lengths)
                    log_probs_t = log_probs.transpose(0, 1)
                    loss = ctc_loss_fn(log_probs_t, targets, output_lengths,
                                       target_lengths)

                val_loss_total += loss.item()
                val_n += 1

                refs, hyps = decode_batch(
                    log_probs, output_lengths, targets, target_lengths)
                all_refs.extend(refs)
                all_hyps.extend(hyps)

        val_loss = val_loss_total / max(val_n, 1)
        wer, cer = compute_wer_cer(all_refs, all_hyps)
        dt = time.time() - t0

        improved = " *" if wer < best_wer else ""

        if is_main():
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "config": cfg,
                "best_wer": min(best_wer, wer),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            if wer < best_wer:
                best_wer = wer
                torch.save(ckpt_data, os.path.join(args.output, "best.pth"))

            torch.save(ckpt_data, os.path.join(args.output, "latest.pth"))

            print("Epoch %2d/%d  train=%.4f  val=%.4f  WER=%.1f%%  CER=%.1f%%  "
                  "lr=%.2e  time=%.0fs%s" %
                  (epoch + 1, args.epochs, train_loss, val_loss, wer, cer,
                   optimizer.param_groups[0]["lr"], dt, improved))

            if args.wandb and HAS_WANDB:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "wer": wer,
                    "cer": cer,
                    "best_wer": best_wer,
                    "lr": optimizer.param_groups[0]["lr"],
                })
        else:
            if wer < best_wer:
                best_wer = wer

    if distributed:
        dist.destroy_process_group()

    if is_main():
        print("\nDone. Best WER: %.1f%%" % best_wer)


if __name__ == "__main__":
    main()
