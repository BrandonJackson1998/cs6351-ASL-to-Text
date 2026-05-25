"""CTC forced alignment.

Given per-frame log-probs from a trained CTC model and the ground-truth label
sequence, find the most likely state path that produces exactly that label.

Each frame gets a state index ∈ {blank, label[0], label[1], ...}. We use the
standard "extended label" representation: blanks interleaved between every
real letter, plus blanks at the start and end. So label "ASL" becomes:

    [B, A, B, S, B, L, B]   (length 2K+1 = 7)

Viterbi runs over the (T, 2K+1) lattice with the standard CTC transition
constraints:
    - same state continues
    - move to next state
    - skip a blank to non-blank if next non-blank differs from current

We return the frame-by-frame letter assignment ('A'..'Z' or BLANK_CHAR).
"""

from typing import List, Optional, Tuple

import numpy as np

from src.models.ctc_lstm import BLANK_IDX, LETTERS


BLANK_CHAR = "_"  # marker we use externally for blank frames


def _label_to_class_indices(label: str) -> List[int]:
    """Convert a label string to CTC class indices (1..26)."""
    out = []
    for c in label.upper():
        if c in LETTERS:
            out.append(LETTERS.index(c) + 1)  # +1 because BLANK_IDX=0
    return out


def forced_align(log_probs: np.ndarray, label: str) -> Optional[List[str]]:
    """Forced-align CTC log-probs against a label string.

    Args:
        log_probs: (T, V) log-probs from the CTC model. V should be 27.
        label:     ground-truth letter string.

    Returns:
        A list of length T where each entry is either BLANK_CHAR or one of
        'A'..'Z' indicating which letter (or blank) was assigned to that
        frame in the most likely alignment. Returns None if T < 2*K+1 (the
        sequence is too short to spell the label even with no blanks).
    """
    T, V = log_probs.shape
    cls_idx = _label_to_class_indices(label)
    K = len(cls_idx)
    if K == 0:
        return None
    # Extended label: B Y_1 B Y_2 B ... Y_K B  (length 2K+1)
    L = 2 * K + 1
    if T < K:
        return None  # impossible to emit each label letter at least once
    ext = [BLANK_IDX] * L
    for i, c in enumerate(cls_idx):
        ext[2 * i + 1] = c

    NEG_INF = -1e18
    delta = np.full((T, L), NEG_INF, dtype=np.float64)
    bp = np.zeros((T, L), dtype=np.int64)

    # Initialization: only states 0 (blank) and 1 (first label) are reachable
    delta[0, 0] = log_probs[0, ext[0]]
    if L > 1:
        delta[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        for s in range(L):
            # Best previous: s, s-1, or s-2 (s-2 only if current and s-2 are
            # different non-blank labels, with a blank in between).
            best_prev = s
            best_score = delta[t - 1, s]
            if s - 1 >= 0 and delta[t - 1, s - 1] > best_score:
                best_score = delta[t - 1, s - 1]
                best_prev = s - 1
            if s - 2 >= 0:
                # Allow skip if current is non-blank and different from ext[s-2]
                cur_cls = ext[s]
                prev_prev_cls = ext[s - 2]
                if cur_cls != BLANK_IDX and cur_cls != prev_prev_cls:
                    if delta[t - 1, s - 2] > best_score:
                        best_score = delta[t - 1, s - 2]
                        best_prev = s - 2
            delta[t, s] = best_score + log_probs[t, ext[s]]
            bp[t, s] = best_prev

    # Best final state: either L-1 (last blank) or L-2 (last label letter)
    final_options = [L - 1]
    if L - 2 >= 0:
        final_options.append(L - 2)
    final_state = max(final_options, key=lambda s: delta[T - 1, s])
    if delta[T - 1, final_state] <= NEG_INF / 2:
        return None

    # Backtrack
    states = [0] * T
    states[T - 1] = final_state
    for t in range(T - 1, 0, -1):
        states[t - 1] = int(bp[t, states[t]])

    # Convert state indices to letters / BLANK_CHAR
    out: List[str] = []
    for s in states:
        cls = ext[s]
        if cls == BLANK_IDX:
            out.append(BLANK_CHAR)
        else:
            out.append(LETTERS[cls - 1])
    return out
