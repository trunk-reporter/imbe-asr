#!/usr/bin/env python3
"""Transcribe IMBE vocoder data using imbe-asr-large-1024d (290M).

Downloads model + LM from Hugging Face on first run (~1.6GB).
Supports .dvcf files (SymbolStream v2), .npz files (raw_params), and
watch mode for a directory.

Usage:
    # Single file
    python3 scripts/transcribe_large.py path/to/call.dvcf

    # Multiple files
    python3 scripts/transcribe_large.py *.dvcf

    # Watch directory
    python3 scripts/transcribe_large.py --watch /path/to/dvcf/dir

    # Greedy decode (faster, no LM download)
    python3 scripts/transcribe_large.py --greedy call.dvcf

    # Use local model (skip HF download)
    python3 scripts/transcribe_large.py --model-dir ./my_model call.dvcf
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HF_REPO = "trunk-reporter/imbe-asr-large-1024d"
MODEL_FILE = "model_int8.onnx"
LM_FILE = "lm/5gram.bin"
UNIGRAMS_FILE = "lm/unigrams.txt"
STATS_FILE = "stats.npz"
LM_ALPHA = 0.7
LM_BETA = 2.0
BEAM_WIDTH = 100


def download_model(model_dir, greedy=False):
    """Download model files from HuggingFace if not present."""
    from huggingface_hub import hf_hub_download

    os.makedirs(model_dir, exist_ok=True)
    files = [MODEL_FILE, STATS_FILE]
    if not greedy:
        files += [LM_FILE, UNIGRAMS_FILE]

    for f in files:
        dest = os.path.join(model_dir, f)
        if not os.path.exists(dest):
            print(f"Downloading {f} from {HF_REPO}...")
            os.makedirs(os.path.dirname(dest) or model_dir, exist_ok=True)
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f,
                local_dir=model_dir,
            )
            print(f"  -> {path}")


def load_model(model_dir, greedy=False):
    import onnxruntime as ort
    from src.inference import OnnxModel
    from src.decode import BeamDecoder

    sess = ort.InferenceSession(os.path.join(model_dir, MODEL_FILE))
    model = OnnxModel(sess)
    stats = np.load(os.path.join(model_dir, STATS_FILE))
    mean, std = stats["mean"], stats["std"]

    decoder = None
    if not greedy:
        decoder = BeamDecoder(
            lm_path=os.path.join(model_dir, LM_FILE),
            unigrams_path=os.path.join(model_dir, UNIGRAMS_FILE),
            alpha=LM_ALPHA,
            beta=LM_BETA,
            beam_width=BEAM_WIDTH,
        )

    return model, mean, std, decoder


def transcribe_file(path, model, mean, std, decoder):
    from src.tokenizer import decode_greedy

    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        if "raw_params" not in d:
            print(f"WARNING: {path} has no raw_params, skipping")
            return None
        raw = d["raw_params"].astype(np.float32)
    elif path.endswith(".dvcf"):
        from src.inference import _read_dvcf_file
        frames = _read_dvcf_file(path)
        if not frames:
            print(f"WARNING: {path} yielded no frames, skipping")
            return None
        raw = np.array(frames, dtype=np.float32)
    else:
        print(f"WARNING: unsupported format {path}, skipping")
        return None

    feats = ((raw - mean) / std).astype(np.float32)
    log_probs, lengths = model(feats[None], np.array([len(feats)]))
    lp = log_probs[0, :lengths[0]]

    if decoder:
        return decoder.decode(lp)
    else:
        return decode_greedy(lp)


def watch_directory(watch_dir, model, mean, std, decoder, interval=1.0):
    print(f"Watching {watch_dir} for new .dvcf files... (Ctrl+C to stop)")
    seen = set(os.listdir(watch_dir))
    while True:
        try:
            time.sleep(interval)
            current = set(os.listdir(watch_dir))
            new_files = [f for f in (current - seen) if f.endswith(".dvcf")]
            for fname in sorted(new_files):
                path = os.path.join(watch_dir, fname)
                text = transcribe_file(path, model, mean, std, decoder)
                if text:
                    print(f"{fname}: {text}")
            seen = current
        except KeyboardInterrupt:
            print("\nStopped.")
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help=".dvcf or .npz files to transcribe")
    parser.add_argument("--watch", metavar="DIR", help="Watch directory for new .dvcf files")
    parser.add_argument("--greedy", action="store_true", help="Greedy decode (no LM, faster)")
    parser.add_argument("--model-dir", default=os.path.expanduser("~/.cache/imbe-asr/large-1024d"),
                        help="Local model directory (default: ~/.cache/imbe-asr/large-1024d)")
    args = parser.parse_args()

    if not args.files and not args.watch:
        parser.print_help()
        sys.exit(1)

    download_model(args.model_dir, greedy=args.greedy)
    model, mean, std, decoder = load_model(args.model_dir, greedy=args.greedy)

    if args.watch:
        watch_directory(args.watch, model, mean, std, decoder)
    else:
        for path in args.files:
            text = transcribe_file(path, model, mean, std, decoder)
            if text:
                print(f"{os.path.basename(path)}: {text}")


if __name__ == "__main__":
    main()
