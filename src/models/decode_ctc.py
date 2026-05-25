"""CTC decoders.

Greedy: argmax per frame, then collapse repeats and remove blanks.
Beam search: optional, returns top-k hypotheses by log-probability.
"""

from typing import List, Optional

import numpy as np

from src.models.ctc_lstm import BLANK_IDX, LETTERS


def fuse_with_ppca(
    ctc_log_probs: np.ndarray,
    ppca_log_likelihoods: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Shallow fusion of CTC log-probs with PPCA per-frame log-likelihoods.

    ctc_log_probs:        (T, 27) — softmax log P(letter | frame, CTC)
                                    column 0 = blank, columns 1..26 = A..Z
    ppca_log_likelihoods: (T, 26) — log p(frame | letter, PPCA), one per letter

    Returns a (T, 27) fused log-prob array. Blank column is untouched.
    Letter columns are: ctc + lam * standardized(ppca).

    PPCA log-likelihoods are zeroed against their per-frame max so they don't
    drown the CTC scores. Effectively we add PPCA's *relative ranking* of
    letters at each frame, weighted by lam.
    """
    if lam <= 0.0:
        return ctc_log_probs

    fused = ctc_log_probs.copy()
    # Standardize PPCA: subtract per-frame max so each frame's best letter is 0.
    p = ppca_log_likelihoods - ppca_log_likelihoods.max(axis=1, keepdims=True)
    fused[:, 1:27] = fused[:, 1:27] + lam * p
    return fused


def greedy_decode(log_probs: np.ndarray) -> str:
    """log_probs: (T, V). Returns decoded letter string."""
    path = log_probs.argmax(axis=-1)
    out = []
    prev = -1
    for idx in path:
        if idx != prev and idx != BLANK_IDX:
            out.append(int(idx))
        prev = int(idx)
    return "".join(LETTERS[i - 1] for i in out if 1 <= i <= len(LETTERS))


def beam_search_decode(log_probs: np.ndarray, beam_size: int = 8) -> List[str]:
    """Vanilla CTC beam search (no LM). Returns top hypotheses, best first."""
    T, V = log_probs.shape
    # Each entry: prefix tuple -> (log_prob_blank_end, log_prob_non_blank_end)
    NEG_INF = float("-inf")
    beam = {(): (0.0, NEG_INF)}

    def logsumexp(a, b):
        if a == NEG_INF:
            return b
        if b == NEG_INF:
            return a
        m = max(a, b)
        return m + np.log(np.exp(a - m) + np.exp(b - m))

    for t in range(T):
        new_beam = {}
        for prefix, (pb, pnb) in beam.items():
            for s in range(V):
                p = log_probs[t, s]
                if s == BLANK_IDX:
                    cur_pb, cur_pnb = new_beam.get(prefix, (NEG_INF, NEG_INF))
                    new_beam[prefix] = (logsumexp(cur_pb, logsumexp(pb + p, pnb + p)), cur_pnb)
                else:
                    if prefix and prefix[-1] == s:
                        # Repeated symbol: extend non-blank only via blank-ended path
                        new_prefix = prefix + (s,)
                        cur_pb_ext, cur_pnb_ext = new_beam.get(new_prefix, (NEG_INF, NEG_INF))
                        new_beam[new_prefix] = (cur_pb_ext, logsumexp(cur_pnb_ext, pb + p))
                        # Same prefix continues (collapsed repeat)
                        cur_pb_same, cur_pnb_same = new_beam.get(prefix, (NEG_INF, NEG_INF))
                        new_beam[prefix] = (cur_pb_same, logsumexp(cur_pnb_same, pnb + p))
                    else:
                        new_prefix = prefix + (s,)
                        cur_pb_ext, cur_pnb_ext = new_beam.get(new_prefix, (NEG_INF, NEG_INF))
                        new_beam[new_prefix] = (
                            cur_pb_ext,
                            logsumexp(cur_pnb_ext, logsumexp(pb + p, pnb + p)),
                        )
        # Keep top beam_size by total log-prob
        scored = sorted(new_beam.items(), key=lambda kv: -logsumexp(*kv[1]))[:beam_size]
        beam = dict(scored)

    final = sorted(beam.items(), key=lambda kv: -logsumexp(*kv[1]))
    return ["".join(LETTERS[i - 1] for i in p if 1 <= i <= len(LETTERS)) for p, _ in final]


def beam_search_decode_lm(
    log_probs: np.ndarray,
    char_lm,
    alpha: float = 0.5,
    beta: float = 1.0,
    beam_size: int = 16,
) -> list:
    """CTC beam search with a character-level LM prior.

    Score: log P_ctc(prefix) + alpha * log P_LM(prefix) + beta * len(prefix).

    The LM is queried only when a *new letter* is added to a prefix (so we
    don't double-count repeated emissions of the same letter). Returns top
    hypotheses sorted best-first.
    """
    T, V = log_probs.shape
    NEG_INF = float("-inf")

    def logsumexp(a, b):
        if a == NEG_INF:
            return b
        if b == NEG_INF:
            return a
        m = max(a, b)
        return m + np.log(np.exp(a - m) + np.exp(b - m))

    # entry: prefix -> (pb, pnb, lm_log_prob_so_far)
    beam = {(): (0.0, NEG_INF, 0.0)}

    def lm_step(prefix_tuple, new_letter_idx):
        # Only compute LM for prefix transitions that add a new letter
        ctx = tuple(LETTERS[i - 1] for i in prefix_tuple if 1 <= i <= len(LETTERS))
        nxt = LETTERS[new_letter_idx - 1]
        return char_lm.log_prob_next(ctx, nxt)

    for t in range(T):
        new_beam = {}
        for prefix, (pb, pnb, lm_lp) in beam.items():
            for s in range(V):
                p = log_probs[t, s]
                if s == BLANK_IDX:
                    cur = new_beam.get(prefix, (NEG_INF, NEG_INF, lm_lp))
                    pb_new = logsumexp(cur[0], logsumexp(pb + p, pnb + p))
                    new_beam[prefix] = (pb_new, cur[1], cur[2])
                else:
                    if prefix and prefix[-1] == s:
                        new_prefix = prefix + (s,)
                        added_lm = lm_lp + lm_step(prefix, s)
                        cur_ext = new_beam.get(new_prefix, (NEG_INF, NEG_INF, added_lm))
                        new_beam[new_prefix] = (
                            cur_ext[0],
                            logsumexp(cur_ext[1], pb + p),
                            added_lm,
                        )
                        cur_same = new_beam.get(prefix, (NEG_INF, NEG_INF, lm_lp))
                        new_beam[prefix] = (
                            cur_same[0],
                            logsumexp(cur_same[1], pnb + p),
                            cur_same[2],
                        )
                    else:
                        new_prefix = prefix + (s,)
                        added_lm = lm_lp + lm_step(prefix, s)
                        cur_ext = new_beam.get(new_prefix, (NEG_INF, NEG_INF, added_lm))
                        new_beam[new_prefix] = (
                            cur_ext[0],
                            logsumexp(cur_ext[1], logsumexp(pb + p, pnb + p)),
                            added_lm,
                        )

        def total_score(item):
            prefix, (pb_, pnb_, lmp) = item
            ctc_lp = logsumexp(pb_, pnb_)
            length_bonus = beta * sum(1 for i in prefix if i != BLANK_IDX)
            return ctc_lp + alpha * lmp + length_bonus

        scored = sorted(new_beam.items(), key=lambda kv: -total_score(kv))[:beam_size]
        beam = dict(scored)

    def final_score(item):
        prefix, (pb_, pnb_, lmp) = item
        ctc_lp = logsumexp(pb_, pnb_)
        length_bonus = beta * sum(1 for i in prefix if i != BLANK_IDX)
        return ctc_lp + alpha * lmp + length_bonus

    final = sorted(beam.items(), key=lambda kv: -final_score(kv))
    return [
        "".join(LETTERS[i - 1] for i in p if 1 <= i <= len(LETTERS))
        for p, _ in final
    ]
