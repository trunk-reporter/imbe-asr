"""Character-level CTC tokenizer for IMBE-to-text.

Vocabulary: blank (0), A-Z (1-26), 0-9 (27-36), space (37), apostrophe (38).
Total: 40 classes.
"""


BLANK = 0
VOCAB = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '")
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i + 1: c for i, c in enumerate(VOCAB)}
ID_TO_CHAR[BLANK] = ""
VOCAB_SIZE = len(VOCAB) + 1  # 39 chars + blank = 40


def encode(text: str) -> list[int]:
    """Encode text to token IDs. Unknown characters are skipped."""
    return [CHAR_TO_ID[c] for c in text.upper() if c in CHAR_TO_ID]


def decode_greedy(log_probs) -> str:
    """Greedy CTC decode: argmax per frame, collapse repeats, remove blanks."""
    prev = -1
    chars = []
    for t in range(log_probs.shape[0]):
        idx = log_probs[t].argmax().item()
        if idx != prev:
            if idx != BLANK:
                chars.append(ID_TO_CHAR[idx])
            prev = idx
    return "".join(chars)
