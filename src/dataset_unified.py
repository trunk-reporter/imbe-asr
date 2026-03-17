"""Unified multi-source dataset for IMBE-ASR training.

Loads (raw_params, transcript) pairs from multiple data sources with
different transcript formats. Supports:
  - LibriSpeech-style: transcripts from .trans.txt files
  - Embedded: transcripts stored in NPZ files (TEDLIUM, GigaSpeech, P25)
  - Memory-mapped: pre-packed binary format (MmapIMBEDataset)

Splits by group key (speaker, talk, show) across all sources to prevent
data leakage.

Config YAML format:
    sources:
      - pairs_dir: data/pairs
        transcript_source: librispeech
        librispeech_dir: data/LibriSpeech/train-clean-100
      - pairs_dir: data/pairs_tedlium
        transcript_source: embedded
    val_fraction: 0.05
    min_frames: 10
    max_frames: 2000
"""

import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import encode

RAW_PARAM_DIM = 170

# IMBE encoder introduces a 2-frame analysis delay: encoded frame N describes
# audio at frame N-2. Software-encoded training data needs the first 2 frames
# trimmed so features align with transcripts.
ENCODER_DELAY_FRAMES = 2


def load_data_config(config_path):
    """Load data source configuration from YAML file."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_librispeech_transcripts(librispeech_dir):
    """Load transcripts from LibriSpeech .trans.txt files."""
    ls_dir = Path(librispeech_dir)
    transcripts = {}
    for trans_file in ls_dir.glob("*/*/*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1].upper()
    return transcripts


def _scan_file(args):
    """Scan a single NPZ file. Returns (npz_path, n_frames, tokens, group_key) or None."""
    npz_path_str, group_key, transcript_source, transcripts = args
    try:
        d = np.load(npz_path_str, allow_pickle=True)
        if "raw_params" not in d:
            return None
        n_frames = d["raw_params"].shape[0]

        utt_id = Path(npz_path_str).stem
        if transcript_source == "librispeech":
            if utt_id not in transcripts:
                return None
            text = transcripts[utt_id]
        elif transcript_source == "embedded":
            if "transcript" not in d:
                return None
            text = str(d["transcript"]).strip().upper()
        else:
            return None
    except Exception:
        return None

    tokens = encode(text)
    if len(tokens) == 0:
        return None

    return (npz_path_str, n_frames, tokens, group_key)


def _get_cache_path(pairs_dir):
    """Cache file sits next to the pairs directory."""
    return Path(str(pairs_dir) + ".scan_cache.pkl")


def _scan_source(source_cfg, min_frames=10, max_frames=2000, scan_workers=12):
    """Scan one data source and return list of (npz_path, tokens, group_key).

    Caches the full scan (with n_frames) to a pickle file next to pairs_dir.
    On subsequent runs, loads from cache and only applies min/max frame filtering.

    Args:
        source_cfg: Dict with pairs_dir, transcript_source, optional librispeech_dir
        min_frames: Skip utterances shorter than this
        max_frames: Skip utterances longer than this
        scan_workers: Number of threads for parallel NPZ loading

    Returns:
        List of (npz_path_str, tokens_list, group_key_str) tuples
    """
    pairs_dir = Path(source_cfg["pairs_dir"])
    transcript_source = source_cfg.get("transcript_source", "embedded")

    if not pairs_dir.exists():
        print("Warning: pairs_dir not found: %s" % pairs_dir)
        return []

    cache_path = _get_cache_path(pairs_dir)
    all_entries = None

    # Try loading from cache
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                all_entries = pickle.load(f)
            print("    (cached: %d entries from %s)" %
                  (len(all_entries), cache_path.name))
        except Exception:
            all_entries = None

    # Full scan if no cache
    if all_entries is None:
        # Load transcripts if LibriSpeech-style
        transcripts = {}
        if transcript_source == "librispeech":
            ls_dir = source_cfg.get("librispeech_dir")
            if ls_dir:
                transcripts = _load_librispeech_transcripts(ls_dir)

        # Collect all file paths first
        file_args = []
        for group_dir in sorted(pairs_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            group_key = group_dir.name
            for npz_path in sorted(group_dir.glob("**/*.npz")):
                file_args.append((
                    str(npz_path), group_key, transcript_source, transcripts,
                ))

        # Parallel scan -- np.load releases the GIL for I/O
        all_entries = []
        with ThreadPoolExecutor(max_workers=scan_workers) as pool:
            for result in pool.map(_scan_file, file_args, chunksize=256):
                if result is not None:
                    all_entries.append(result)

        # Save cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(all_entries, f, protocol=4)
        except Exception as e:
            print("    Warning: failed to write cache: %s" % e)

    # Apply frame filtering (cache stores unfiltered entries)
    samples = [(path, tokens, group_key)
               for path, n_frames, tokens, group_key in all_entries
               if min_frames <= n_frames <= max_frames]

    return samples


class UnifiedIMBEDataset(Dataset):
    """Multi-source IMBE dataset with group-based splitting.

    Args:
        data_config: Dict with 'sources' list and optional settings,
                     or path to YAML config file.
        split: 'train' or 'val' (None for all data)
        normalize: If True, standardize features per-dimension
        stats: Tuple of (mean, std). If None and normalize=True,
               computed from this dataset.
    """

    def __init__(self, data_config, split=None, normalize=True, stats=None,
                 scan_workers=12):
        if isinstance(data_config, (str, Path)):
            data_config = load_data_config(data_config)

        val_fraction = data_config.get("val_fraction", 0.05)
        min_frames = data_config.get("min_frames", 10)
        max_frames = data_config.get("max_frames", 2000)
        seed = data_config.get("seed", 42)

        self.normalize = normalize

        # Scan all sources
        all_samples = []
        for src in data_config["sources"]:
            src_samples = _scan_source(src, min_frames, max_frames,
                                       scan_workers=scan_workers)
            print("  %s: %d samples" % (src["pairs_dir"], len(src_samples)))
            all_samples.extend(src_samples)

        # Group-based split
        if split is not None:
            groups = sorted(set(s[2] for s in all_samples))
            rng = np.random.RandomState(seed)
            rng.shuffle(groups)
            n_val = max(1, int(len(groups) * val_fraction))
            val_groups = set(groups[:n_val])
            train_groups = set(groups[n_val:])

            if split == "val":
                keep_groups = val_groups
            else:
                keep_groups = train_groups

            all_samples = [s for s in all_samples if s[2] in keep_groups]

        # Store as (npz_path, tokens) -- drop group_key
        self.samples = [(s[0], s[1]) for s in all_samples]

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


class MmapIMBEDataset(Dataset):
    """Memory-mapped IMBE dataset for fast loading.

    Reads pre-packed binary files produced by scripts/pack_dataset.py:
      - {name}.features.bin  -- flat float32 array of all raw_params
      - {name}.meta.pkl      -- per-utterance metadata (offset, n_frames, tokens, group_key)

    No per-sample file I/O: __getitem__ is a simple memmap slice.

    Args:
        bin_path: Path to the .features.bin file
        meta_path: Path to the .meta.pkl file (default: inferred from bin_path)
        split: 'train' or 'val' (None for all data)
        val_fraction: Fraction of groups to hold out for validation
        seed: Random seed for group split
        normalize: If True, standardize features per-dimension
        stats: Tuple of (mean, std). If None and normalize=True,
               computed from this dataset.
    """

    def __init__(self, bin_path, meta_path=None, split=None,
                 val_fraction=0.05, seed=42,
                 normalize=True, stats=None,
                 data_fraction=1.0):
        bin_path = Path(bin_path)
        if meta_path is None:
            # all.features.bin -> all.meta.pkl
            meta_path = bin_path.parent / (
                bin_path.name.replace(".features.bin", ".meta.pkl"))
        else:
            meta_path = Path(meta_path)

        # Load metadata
        with open(meta_path, "rb") as f:
            all_meta = pickle.load(f)

        if len(all_meta) == 0:
            raise ValueError("Empty metadata file: %s" % meta_path)

        # Compute total frames from last entry
        last = all_meta[-1]
        total_frames = last["offset"] + last["n_frames"]

        # Memory-map the binary feature file (read-only, float16 on disk)
        self.mmap = np.memmap(
            str(bin_path), dtype=np.float16, mode="r",
            shape=(total_frames, RAW_PARAM_DIM),
        )

        # Group-based split
        if split is not None:
            groups = sorted(set(m["group_key"] for m in all_meta))
            rng = np.random.RandomState(seed)
            rng.shuffle(groups)
            n_val = max(1, int(len(groups) * val_fraction))
            val_groups = set(groups[:n_val])
            train_groups = set(groups[n_val:])

            if split == "val":
                keep_groups = val_groups
            else:
                keep_groups = train_groups

            all_meta = [m for m in all_meta if m["group_key"] in keep_groups]

        # Subsample training data for faster sweep trials
        if data_fraction < 1.0 and split == "train" and len(all_meta) > 0:
            rng = np.random.RandomState(seed)
            n_keep = max(1, int(len(all_meta) * data_fraction))
            indices = rng.choice(len(all_meta), n_keep, replace=False)
            all_meta = [all_meta[i] for i in sorted(indices)]

        self.meta = all_meta
        self.normalize = normalize

        if normalize:
            if stats is not None:
                self.mean, self.std = stats
            else:
                self.mean, self.std = self._compute_stats()

    def _compute_stats(self, max_samples=2000):
        """Compute per-dimension mean/std from a subset of utterances."""
        rng = np.random.RandomState(42)
        indices = rng.choice(len(self.meta),
                             min(max_samples, len(self.meta)), replace=False)
        all_feats = []
        for i in indices:
            m = self.meta[i]
            feats = self.mmap[m["offset"]:m["offset"] + m["n_frames"]]
            all_feats.append(np.array(feats, dtype=np.float32))

        all_feats = np.concatenate(all_feats, axis=0)
        mean = all_feats.mean(axis=0)
        std = all_feats.std(axis=0)
        std = np.maximum(std, 1e-6)
        return mean, std

    def get_stats(self):
        return (self.mean, self.std)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        m = self.meta[idx]
        offset = m["offset"]
        n_frames = m["n_frames"]
        tokens = m["tokens"]

        # Skip first ENCODER_DELAY_FRAMES to compensate for IMBE encoder delay
        skip = min(ENCODER_DELAY_FRAMES, n_frames - 1)
        feats = np.array(
            self.mmap[offset + skip:offset + n_frames], dtype=np.float32)

        if self.normalize:
            feats = (feats - self.mean) / self.std

        return {
            "features": torch.from_numpy(feats),
            "tokens": torch.tensor(tokens, dtype=torch.long),
        }
