"""Character-level n-gram language model with Kneser-Ney-ish smoothing.

Built from English wordlist + FSWild train labels (NEVER touches dev or test).
Used by the CTC beam-search decoder to bias toward English-like spellings.

We keep the implementation simple — a stupid-backoff n-gram with add-k
smoothing. Good enough for ~1-3 LER point gains, and easy to tune.
"""

import math
import os
import string
from collections import Counter, defaultdict
from typing import Dict, Iterable, Tuple

import numpy as np


_LETTERS = set(string.ascii_uppercase)
START = "<s>"  # padding token
END = "</s>"


def _iter_chars(text: str) -> Iterable[str]:
    return (c for c in text.upper() if c in _LETTERS)


def _word_to_chars(word: str):
    word = word.upper()
    return [c for c in word if c in _LETTERS]


class CharNgramLM:
    def __init__(self, n: int = 6, smoothing: float = 0.5, backoff_alpha: float = 0.4):
        self.n = n
        self.smoothing = smoothing
        self.backoff_alpha = backoff_alpha
        self.counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.context_totals: Dict[Tuple[str, ...], float] = defaultdict(float)
        self.vocab = sorted(_LETTERS) + [END]

    def _train_on_word(self, chars):
        seq = [START] * (self.n - 1) + chars + [END]
        for i in range(len(seq) - 1):
            for order in range(1, self.n + 1):
                if i + 1 - order < 0:
                    continue
                ctx = tuple(seq[i + 1 - order:i + 1])
                nxt = seq[i + 1]
                self.counts[ctx][nxt] += 1
                self.context_totals[ctx] += 1

    def fit(self, words: Iterable[str]):
        for w in words:
            chars = _word_to_chars(w)
            if not chars:
                continue
            self._train_on_word(chars)
        return self

    def log_prob_next(self, context: Tuple[str, ...], nxt: str) -> float:
        """Log prob of the next character given the (variable-length) context.

        Uses stupid-backoff: shorten the context until we find a non-zero count.
        """
        ctx = tuple(context[-(self.n - 1):])
        order_factor = 1.0
        while True:
            if ctx in self.counts and self.counts[ctx][nxt] > 0:
                p = (self.counts[ctx][nxt] + self.smoothing) / (
                    self.context_totals[ctx] + self.smoothing * len(self.vocab)
                )
                return math.log(p) + math.log(order_factor)
            if not ctx:
                # Uniform fallback at unigram level
                return math.log(1.0 / len(self.vocab)) + math.log(order_factor)
            ctx = ctx[1:]
            order_factor *= self.backoff_alpha

    def log_prob_string(self, s: str) -> float:
        """Log prob of a complete string (with implicit end-of-word)."""
        chars = _word_to_chars(s)
        seq = [START] * (self.n - 1) + chars + [END]
        total = 0.0
        for i in range(self.n - 1, len(seq)):
            ctx = tuple(seq[i - (self.n - 1):i])
            total += self.log_prob_next(ctx, seq[i])
        return total


def build_default_lm(
    wordlist_path: str = "/usr/share/dict/words",
    fswild_train_npz: str = "data/chicagofswild/landmarks/train.npz",
    n: int = 6,
) -> CharNgramLM:
    """Train an LM on English dictionary + FSWild train labels (never dev/test)."""
    words = []
    if os.path.isfile(wordlist_path):
        with open(wordlist_path) as f:
            for line in f:
                w = line.strip()
                if all(c.upper() in _LETTERS for c in w) and w:
                    words.append(w)
    if os.path.isfile(fswild_train_npz):
        data = np.load(fswild_train_npz, allow_pickle=True)
        for label in data["labels"]:
            label = str(label)
            if any(c in _LETTERS for c in label.upper()):
                words.append(label)
    print(f"[CharNgramLM] training on {len(words)} words (n={n})")
    return CharNgramLM(n=n).fit(words)
