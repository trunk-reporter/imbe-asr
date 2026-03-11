"""WER/CER evaluation utilities.

Provides edit distance, WER/CER computation, and greedy CTC batch decoding.
Can be run standalone to evaluate a checkpoint on the LibriSpeech validation set.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .tokenizer import decode_greedy, ID_TO_CHAR, VOCAB_SIZE


def edit_distance(ref, hyp):
    """Levenshtein edit distance between two sequences."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def compute_wer_cer(refs, hyps):
    """Compute WER and CER over a list of (reference, hypothesis) pairs.

    Returns:
        wer: Word Error Rate (%)
        cer: Character Error Rate (%)
    """
    total_words = 0
    total_word_errors = 0
    total_chars = 0
    total_char_errors = 0

    for ref, hyp in zip(refs, hyps):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_words += len(ref_words)
        total_word_errors += edit_distance(ref_words, hyp_words)

        total_chars += len(ref)
        total_char_errors += edit_distance(list(ref), list(hyp))

    wer = 100.0 * total_word_errors / max(total_words, 1)
    cer = 100.0 * total_char_errors / max(total_chars, 1)
    return wer, cer


def decode_batch(log_probs, output_lengths, targets, target_lengths):
    """Decode a batch and return (references, hypotheses) lists.

    Args:
        log_probs: (B, T, V) log probabilities
        output_lengths: (B,) actual output frame counts
        targets: (sum(L_i),) concatenated target tokens
        target_lengths: (B,) target token counts

    Returns:
        refs: list of reference strings
        hyps: list of hypothesis strings
    """
    refs = []
    hyps = []
    offset = 0

    for i in range(log_probs.size(0)):
        T = output_lengths[i].item()
        hyp = decode_greedy(log_probs[i, :T])
        hyps.append(hyp)

        L = target_lengths[i].item()
        ref_tokens = targets[offset:offset + L].tolist()
        ref = "".join(ID_TO_CHAR.get(t, "") for t in ref_tokens)
        refs.append(ref)
        offset += L

    return refs, hyps


def main():
    """Evaluate a checkpoint on the validation set."""
    parser = argparse.ArgumentParser(description="Evaluate IMBE-to-text model")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--pairs-dir", default="data/pairs")
    parser.add_argument("--librispeech-dir",
                        default="data/LibriSpeech/train-clean-100")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show-examples", type=int, default=20)
    # Beam search options
    parser.add_argument("--beam", action="store_true")
    parser.add_argument("--lm-path", default=None)
    parser.add_argument("--unigrams", default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--hotwords", default=None)
    parser.add_argument("--hotword-weight", type=float, default=10.0)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    from .model import ConformerCTC
    model = ConformerCTC(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        n_layers=config["n_layers"],
        conv_kernel=config["conv_kernel"],
        vocab_size=config["vocab_size"],
        dropout=0.0,
        subsample=config.get("subsample", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Model: %s parameters" % f"{model.count_parameters():,}")
    print("Checkpoint: epoch %d" % (ckpt["epoch"] + 1))

    # Load stats and data
    import numpy as np
    stats_path = os.path.join(os.path.dirname(args.checkpoint), "stats.npz")
    stats_data = np.load(stats_path)
    stats = (stats_data["mean"], stats_data["std"])

    from .dataset import IMBEDataset, collate_fn, get_speaker_split
    _, val_speakers = get_speaker_split(args.pairs_dir)

    val_ds = IMBEDataset(
        args.pairs_dir, args.librispeech_dir,
        speaker_ids=val_speakers, normalize=True, stats=stats,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.workers,
    )
    print("Validation: %d utterances" % len(val_ds))

    # Set up decoder
    beam_decoder = None
    if args.beam:
        from .decode import BeamDecoder, load_hotwords
        hotwords = None
        if args.hotwords:
            hotwords = load_hotwords(args.hotwords)
        beam_decoder = BeamDecoder(
            lm_path=args.lm_path,
            unigrams_path=args.unigrams,
            alpha=args.alpha,
            beta=args.beta,
            beam_width=args.beam_width,
            hotwords=hotwords,
            hotword_weight=args.hotword_weight,
        )

    # Evaluate
    all_refs = []
    all_hyps = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            targets = batch["targets"]
            target_lengths = batch["target_lengths"]

            log_probs, output_lengths = model(features, input_lengths)

            if beam_decoder is not None:
                hyps = beam_decoder.decode_batch(log_probs, output_lengths)
                offset = 0
                refs = []
                for i in range(log_probs.size(0)):
                    L = target_lengths[i].item()
                    ref_tokens = targets[offset:offset + L].tolist()
                    ref = "".join(ID_TO_CHAR.get(t, "") for t in ref_tokens)
                    refs.append(ref)
                    offset += L
            else:
                refs, hyps = decode_batch(log_probs, output_lengths,
                                          targets, target_lengths)

            all_refs.extend(refs)
            all_hyps.extend(hyps)

            if (batch_idx + 1) % 20 == 0:
                print("  [%d/%d]" % (batch_idx + 1, len(val_loader)),
                      flush=True)

    wer, cer = compute_wer_cer(all_refs, all_hyps)
    decode_mode = "beam" if args.beam else "greedy"
    print("\nResults (%s): WER=%.1f%%  CER=%.1f%%" % (decode_mode, wer, cer))

    if args.show_examples > 0:
        print("\nExamples (first %d):" % args.show_examples)
        for i in range(min(args.show_examples, len(all_refs))):
            print("\n  REF: %s" % all_refs[i])
            print("  HYP: %s" % all_hyps[i])


if __name__ == "__main__":
    main()
