"""Dataset for LibriSpeech IMBE training data (170-dim raw params).

Loads precomputed raw_params from NPZ files:
  [0]       f0 (fundamental frequency)
  [1]       L  (number of harmonics)
  [2:58]    sa[0..55] spectral amplitudes (zero-padded)
  [58:114]  v_uv[0..55] voiced/unvoiced flags (zero-padded)
  [114:170] mask[0..55] binary harmonic validity (1=real, 0=pad)

The mask disambiguates zero-energy harmonics from padding -- critical
since IMBE preserves per-band spectral granularity.
"""

from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .tokenizer import encode


RAW_PARAM_DIM = 170

# IMBE encoder introduces a 2-frame analysis delay: encoded frame N describes
# audio at frame N-2. Training data encoded with imbe_encode (LibriSpeech,
# TEDLIUM, GigaSpeech) needs the first 2 frames trimmed so features align
# with transcripts. Real P25 TAP data (captured from radios) is NOT affected.
ENCODER_DELAY_FRAMES = 2


class IMBEDataset(Dataset):
    """Loads (raw_params, transcript) pairs from LibriSpeech NPZ files.

    Args:
        pairs_dir: Directory with speaker/chapter/utterance NPZ structure
        librispeech_dir: LibriSpeech transcripts root
        speaker_ids: Set of speaker IDs to include (for train/val split)
        normalize: If True, standardize features per-dimension
        stats: Tuple of (mean, std). If None and normalize=True,
               computed from this dataset.
        min_frames: Skip utterances shorter than this
        max_frames: Skip utterances longer than this
    """

    def __init__(self, pairs_dir, librispeech_dir, speaker_ids=None,
                 normalize=True, stats=None,
                 min_frames=10, max_frames=2000):
        self.pairs_dir = Path(pairs_dir)
        self.normalize = normalize

        # Load transcripts from LibriSpeech .trans.txt
        ls_dir = Path(librispeech_dir)
        transcripts = {}
        for trans_file in ls_dir.glob("*/*/*.trans.txt"):
            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        transcripts[parts[0]] = parts[1].upper()

        # Find NPZ files with raw_params
        self.samples = []
        for speaker_dir in sorted(self.pairs_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name
            if speaker_ids is not None and speaker_id not in speaker_ids:
                continue

            for npz_path in sorted(speaker_dir.glob("*/*.npz")):
                utt_id = npz_path.stem
                if utt_id not in transcripts:
                    continue

                text = transcripts[utt_id]
                tokens = encode(text)
                if len(tokens) == 0:
                    continue

                try:
                    d = np.load(str(npz_path), mmap_mode='r')
                    if 'raw_params' not in d:
                        continue
                    n_frames = d['raw_params'].shape[0]
                    if n_frames < min_frames or n_frames > max_frames:
                        continue
                except Exception:
                    continue

                self.samples.append((str(npz_path), tokens))

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

        # Compensate for IMBE encoder delay in software-encoded data
        if feats.shape[0] > ENCODER_DELAY_FRAMES:
            feats = feats[ENCODER_DELAY_FRAMES:]

        if self.normalize:
            feats = (feats - self.mean) / self.std

        return {
            "features": torch.from_numpy(feats),
            "tokens": torch.tensor(tokens, dtype=torch.long),
        }


def collate_fn(batch):
    """Pad variable-length sequences for CTC training.

    Returns:
        features: (B, T_max, D) padded features
        targets: (sum(L_i),) concatenated targets (CTC format)
        input_lengths: (B,) actual frame counts
        target_lengths: (B,) actual token counts
    """
    features = [s["features"] for s in batch]
    tokens = [s["tokens"] for s in batch]

    input_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in tokens], dtype=torch.long)

    features_padded = pad_sequence(features, batch_first=True)
    targets = torch.cat(tokens)

    return {
        "features": features_padded,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
    }


def get_speaker_split(pairs_dir, val_fraction=0.1, seed=42):
    """Split speakers into train/val sets.

    Returns:
        train_speakers: set of speaker ID strings
        val_speakers: set of speaker ID strings
    """
    pairs_dir = Path(pairs_dir)
    speakers = sorted([d.name for d in pairs_dir.iterdir() if d.is_dir()])
    rng = np.random.RandomState(seed)
    rng.shuffle(speakers)
    n_val = max(1, int(len(speakers) * val_fraction))
    val_speakers = set(speakers[:n_val])
    train_speakers = set(speakers[n_val:])
    return train_speakers, val_speakers
