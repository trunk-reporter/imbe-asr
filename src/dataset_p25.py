"""Dataset for real P25 TAP data with raw_params + transcripts.

Loads NPZ files from prepare_p25_tap.py output:
  raw_params (N, 170) -- decoded IMBE parameters
  transcript -- Qwen3-ASR transcription string
  tgid, src_id -- metadata
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import encode


RAW_PARAM_DIM = 170


class P25Dataset(Dataset):
    """Loads (raw_params, transcript) pairs from P25 TAP NPZ files.

    Args:
        data_dir: Directory with flat NPZ files from prepare_p25_tap
        normalize: If True, standardize features per-dimension
        stats: Tuple of (mean, std). If None and normalize=True,
               computed from this dataset.
        min_frames: Skip utterances shorter than this
        max_frames: Skip utterances longer than this
        split: 'train' or 'val' -- splits by talkgroup hash
        val_fraction: Fraction of talkgroups to hold out for val
    """

    def __init__(self, data_dir, normalize=True, stats=None,
                 min_frames=10, max_frames=2000,
                 split=None, val_fraction=0.1, seed=42):
        self.data_dir = Path(data_dir)
        self.normalize = normalize

        # Find all NPZ files
        all_files = sorted(self.data_dir.glob("*.npz"))

        # Split by talkgroup (from filename: tgid-timestamp-Nframes.npz)
        if split is not None:
            tgids = set()
            for f in all_files:
                parts = f.stem.split("-")
                if len(parts) >= 1:
                    tgids.add(parts[0])
            tgids = sorted(tgids)
            rng = np.random.RandomState(seed)
            rng.shuffle(tgids)
            n_val = max(1, int(len(tgids) * val_fraction))
            val_tgids = set(tgids[:n_val])
            train_tgids = set(tgids[n_val:])
            keep_tgids = val_tgids if split == "val" else train_tgids
        else:
            keep_tgids = None

        self.samples = []
        for npz_path in all_files:
            if keep_tgids is not None:
                tgid = npz_path.stem.split("-")[0]
                if tgid not in keep_tgids:
                    continue

            try:
                d = np.load(str(npz_path), allow_pickle=True)
                if "raw_params" not in d or "transcript" not in d:
                    continue
                rp = d["raw_params"]
                if rp.shape[1] != RAW_PARAM_DIM:
                    continue
                if rp.shape[0] < min_frames or rp.shape[0] > max_frames:
                    continue

                text = str(d["transcript"]).strip().upper()
                tokens = encode(text)
                if len(tokens) < 2:
                    continue

                self.samples.append((str(npz_path), tokens))
            except Exception:
                continue

        if normalize:
            if stats is not None:
                self.mean, self.std = stats
            else:
                self.mean, self.std = self._compute_stats()

    def _compute_stats(self, max_samples=2000):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(self.samples),
                             min(max_samples, len(self.samples)), replace=False)
        all_feats = []
        for i in indices:
            d = np.load(self.samples[i][0])
            all_feats.append(d["raw_params"])
        all_feats = np.concatenate(all_feats, axis=0)
        mean = all_feats.mean(axis=0).astype(np.float32)
        std = all_feats.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-6)
        return mean, std

    def get_stats(self):
        return (self.mean, self.std)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, tokens = self.samples[idx]
        d = np.load(npz_path)
        feats = d["raw_params"].astype(np.float32)

        if self.normalize:
            feats = (feats - self.mean) / self.std

        return {
            "features": torch.from_numpy(feats),
            "tokens": torch.tensor(tokens, dtype=torch.long),
        }
