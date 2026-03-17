#!/usr/bin/env python3
"""W&B sweep agent for IMBE-ASR hyperparameter search.

Each trial trains for a few epochs on the mmap dataset and reports WER.
Bayesian optimization picks the next config based on prior results.

Usage (after creating sweep on wandb):
    wandb agent <sweep_id>

Or via the launch script:
    bash scripts/vast_sweep.sh
"""

import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb

# Add project root to path so src.* imports work
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import collate_fn, RAW_PARAM_DIM
from src.dataset_unified import MmapIMBEDataset
from src.model import ConformerCTC
from src.tokenizer import VOCAB_SIZE
from src.eval import compute_wer_cer, decode_batch

# Fixed params -- same across all trials
MMAP_DIR = os.environ.get("MMAP_DIR", "data/packed")
EPOCHS = int(os.environ.get("SWEEP_EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("SWEEP_BATCH_SIZE", "16"))
ACCUM_STEPS = int(os.environ.get("SWEEP_ACCUM_STEPS", "4"))
VAL_FRACTION = 0.05
N_HEADS = 8
CONV_KERNEL = 31
WORKERS = int(os.environ.get("SWEEP_WORKERS", "4"))
DATA_FRACTION = float(os.environ.get("SWEEP_DATA_FRACTION", "1.0"))


def train_one():
    """Single sweep trial: build model, train N epochs, report WER."""
    run = wandb.init()
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_model = cfg.d_model
    n_layers = cfg.n_layers
    d_ff = d_model * cfg.ff_mult
    lr = cfg.lr
    dropout = cfg.dropout
    warmup_steps = cfg.warmup_steps

    n_params_est = d_model * 170 + n_layers * (4 * d_model**2 + 2 * d_model * d_ff) + d_model * VOCAB_SIZE
    print("Trial: d=%d, L=%d, ff=%d, lr=%.1e, drop=%.2f, warm=%d (~%.1fM params)" %
          (d_model, n_layers, d_ff, lr, dropout, warmup_steps, n_params_est / 1e6))

    # Dataset (shared across trials via mmap -- no copy overhead)
    bin_path = os.path.join(MMAP_DIR, "all.features.bin")
    train_ds = MmapIMBEDataset(
        bin_path, split="train", val_fraction=VAL_FRACTION, normalize=True,
        data_fraction=DATA_FRACTION)
    stats = train_ds.get_stats()
    val_ds = MmapIMBEDataset(
        bin_path, split="val", val_fraction=VAL_FRACTION,
        normalize=True, stats=stats)

    frac_str = " (%.0f%% subset)" % (DATA_FRACTION * 100) if DATA_FRACTION < 1.0 else ""
    print("Samples: %d train, %d val%s" % (len(train_ds), len(val_ds), frac_str))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=WORKERS,
        pin_memory=True, drop_last=True,
        persistent_workers=WORKERS > 0)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=WORKERS > 0)

    # Model
    model = ConformerCTC(
        input_dim=RAW_PARAM_DIM,
        d_model=d_model,
        n_heads=N_HEADS,
        d_ff=d_ff,
        n_layers=n_layers,
        conv_kernel=CONV_KERNEL,
        vocab_size=VOCAB_SIZE,
        dropout=dropout,
    ).to(device)

    n_params = model.count_parameters()
    print("Model: %s parameters (%.1fM)" % (f"{n_params:,}", n_params / 1e6))
    wandb.log({"n_params": n_params})

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)

    total_steps = EPOCHS * (len(train_loader) // ACCUM_STEPS)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    best_wer = float("inf")

    for epoch in range(EPOCHS):
        t0 = time.time()

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

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                log_probs, output_lengths = model(features, input_lengths)
                log_probs_t = log_probs.transpose(0, 1)
                loss = ctc_loss_fn(log_probs_t, targets, output_lengths,
                                   target_lengths)

            (loss / ACCUM_STEPS).backward()
            total_loss += loss.item()
            n_batches += 1

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if n_batches % 200 == 0:
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

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
        best_wer = min(best_wer, wer)

        print("Epoch %d/%d  train=%.4f  val=%.4f  WER=%.1f%%  CER=%.1f%%  "
              "time=%.0fs%s" %
              (epoch + 1, EPOCHS, train_loss, val_loss, wer, cer, dt, improved))

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "wer": wer,
            "cer": cer,
            "best_wer": best_wer,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": dt,
        })

    print("Trial done. Best WER: %.1f%%" % best_wer)
    run.finish()


if __name__ == "__main__":
    train_one()
