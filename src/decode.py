"""CTC decoding: greedy and beam search with KenLM language model.

Greedy decode is in tokenizer.py (decode_greedy). This module adds beam
search via pyctcdecode with optional KenLM n-gram language model.

Usage:
    from src.decode import BeamDecoder

    decoder = BeamDecoder(
        lm_path="data/lm/5gram.bin",
        unigrams_path="data/lm/unigrams.txt",
    )
    text = decoder.decode(log_probs)  # log_probs: (T, V) numpy array
"""

from typing import List, Optional

import numpy as np

from .tokenizer import VOCAB, BLANK, VOCAB_SIZE


def _build_labels():
    """Build the labels list for pyctcdecode.

    pyctcdecode expects a list of strings where index i corresponds to
    the character for logit index i. The blank token is "".
    """
    labels = [""] * VOCAB_SIZE
    for i, char in enumerate(VOCAB):
        labels[i + 1] = char
    return labels


class BeamDecoder:
    """CTC beam search decoder with optional KenLM language model.

    Args:
        lm_path: Path to KenLM model (.arpa or .bin). None for LM-free.
        unigrams_path: Path to unigrams file (one word per line).
        alpha: LM weight (higher = more LM influence). Default 0.5.
        beta: Word insertion bonus (higher = longer outputs). Default 1.5.
        beam_width: Max beams to keep. Default 100.
        beam_prune_logp: Prune beams worse than best by this much.
        token_min_logp: Skip tokens below this log-prob.
        hotwords: Optional list of hotwords to boost.
        hotword_weight: Weight for hotword boosting.
    """

    def __init__(
        self,
        lm_path: Optional[str] = None,
        unigrams_path: Optional[str] = None,
        alpha: float = 0.5,
        beta: float = 1.5,
        beam_width: int = 100,
        beam_prune_logp: float = -10.0,
        token_min_logp: float = -5.0,
        hotwords: Optional[List[str]] = None,
        hotword_weight: float = 10.0,
    ):
        from pyctcdecode import build_ctcdecoder

        self.beam_width = beam_width
        self.beam_prune_logp = beam_prune_logp
        self.token_min_logp = token_min_logp
        self.hotwords = hotwords
        self.hotword_weight = hotword_weight

        labels = _build_labels()

        unigrams = None
        if unigrams_path:
            with open(unigrams_path) as f:
                unigrams = [line.strip() for line in f if line.strip()]
            print("BeamDecoder: loaded %d unigrams" % len(unigrams))

        self.decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=lm_path if lm_path else None,
            unigrams=unigrams,
            alpha=alpha,
            beta=beta,
        )

        lm_info = "LM=%s" % lm_path if lm_path else "no LM"
        print("BeamDecoder: %s, alpha=%s, beta=%s, beam_width=%d" %
              (lm_info, alpha, beta, beam_width))

    def decode(self, log_probs) -> str:
        """Decode a single utterance.

        Args:
            log_probs: (T, V) log probabilities (torch.Tensor or numpy).

        Returns:
            Decoded text string (uppercase).
        """
        if hasattr(log_probs, 'numpy'):
            logits = log_probs.cpu().numpy()
        elif hasattr(log_probs, 'detach'):
            logits = log_probs.detach().cpu().numpy()
        else:
            logits = np.asarray(log_probs)

        logits = logits.astype(np.float32)

        text = self.decoder.decode(
            logits,
            beam_width=self.beam_width,
            beam_prune_logp=self.beam_prune_logp,
            token_min_logp=self.token_min_logp,
            hotwords=self.hotwords,
            hotword_weight=self.hotword_weight,
        )

        return text.upper()

    def decode_batch(self, log_probs_batch, output_lengths) -> List[str]:
        """Decode a batch of utterances.

        Args:
            log_probs_batch: (B, T, V) log probabilities
            output_lengths: (B,) actual output frame counts

        Returns:
            List of decoded text strings.
        """
        if hasattr(log_probs_batch, 'numpy'):
            batch_np = log_probs_batch.cpu().numpy()
        elif hasattr(log_probs_batch, 'detach'):
            batch_np = log_probs_batch.detach().cpu().numpy()
        else:
            batch_np = np.asarray(log_probs_batch)

        if hasattr(output_lengths, 'tolist'):
            lengths = output_lengths.tolist()
        else:
            lengths = list(output_lengths)

        logit_list = []
        for i in range(batch_np.shape[0]):
            T = lengths[i]
            logit_list.append(batch_np[i, :T].astype(np.float32))

        try:
            from pyctcdecode import BeamSearchDecoderCTC
            pool = BeamSearchDecoderCTC.get_pool()
            texts = self.decoder.decode_batch(
                pool,
                logit_list,
                beam_width=self.beam_width,
                beam_prune_logp=self.beam_prune_logp,
                token_min_logp=self.token_min_logp,
            )
        except Exception:
            texts = [self.decode(logits) for logits in logit_list]

        return [t.upper() for t in texts]


def load_hotwords(hotwords_path: str) -> List[str]:
    """Load hotwords from a text file (one per line).

    Useful for P25 radio domain words like unit IDs, 10-codes, etc.
    """
    with open(hotwords_path) as f:
        return [line.strip().upper() for line in f if line.strip()]
