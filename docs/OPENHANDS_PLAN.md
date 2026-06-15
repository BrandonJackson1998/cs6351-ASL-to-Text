# Word-Level Sign Recognition via OpenHands — Plan

**Status:** Draft (2026-05-21)
**Replaces:** building Phase 2 LSTM/Transformer from scratch on WLASL
**Phase:** 2 (word recognition)

---

## Why OpenHands

[OpenHands](https://github.com/AI4Bharat/OpenHands) is a PyTorch toolkit for
sign language recognition with **pretrained checkpoints already trained on
WLASL** — the exact dataset our proposal targets. Models available:

- **I3D** (RGB video) — strong baseline, 3D CNN
- **ST-GCN** (skeleton) — graph CNN on hand+pose keypoints
- **MS-G3D** (skeleton) — current state-of-the-art on skeleton input

Skeleton models match our existing MediaPipe pipeline — same input shape,
no additional video model needed.

**The proposal's Phase 2 target:** "Top-1 and Top-5 word accuracy on the
WLASL test split." Published WLASL benchmarks: I3D ~32% top-1 on WLASL2000,
ST-GCN ~37% top-1, MS-G3D ~46% top-1.

Building this from scratch would take 4-6 weeks (per the proposal timeline).
OpenHands gives us a working WLASL2000 demo in **~1 day**.

---

## What This Replaces / Does Not Replace

**Replaces:**
- The unimplemented Phase 2 LSTM training pipeline
- WLASL preprocessing from scratch (OpenHands has it built in)
- Months of model architecture iteration

**Does NOT replace:**
- The Phase 1 PPCA fingerspelling model (different task — letters)
- The Phase 3 LLM gloss-to-sentence layer (still needed)
- The video segmenter (still needed for chaining word predictions in a
  sentence-length video — but only for "word-level chunks", not letters)

---

## Architecture

```
Video clip (one word)
  │
  ▼
MediaPipe Holistic landmarks (hands + pose + face)  ──►  T × 75 keypoints
  │
  ▼
OpenHands ST-GCN or MS-G3D model (pretrained WLASL2000)
  │
  ▼
Top-1 word + top-5 alternatives  ──►  e.g., ["store", "shop", "market", "buy", "place"]
  │
  ▼ (Phase 3)
Claude API: gloss sequence → English sentence
```

---

## Tasks

### Step 1: Install OpenHands

```bash
pip install openhands  # or git clone + pip install -e .
```

Pin to a specific commit. The package needs PyTorch ≥ 1.10 (we have 2.x).

### Step 2: Download a pretrained checkpoint

OpenHands hosts checkpoints on Hugging Face. Pick one:
- `decoupled_stgcn_wlasl2000` (~50 MB, fastest)
- `msg3d_wlasl2000` (~120 MB, highest accuracy)

Cache to `data/openhands_checkpoints/<name>.pt`.

### Step 3: Adapter from our MediaPipe output to OpenHands' input

OpenHands expects a specific keypoint layout (often MMPose body+hand format).
Our MediaPipe Holistic produces a different layout. Write a small adapter:

```python
# src/models/openhands_adapter.py
def mediapipe_to_openhands_keypoints(holistic_result):
    """Convert MediaPipe Holistic output to OpenHands keypoint tensor.

    Shape: (T, num_keypoints, 3)
    Order: [pose_landmarks, left_hand, right_hand, face_landmarks (subset)]
    """
    ...
```

### Step 4: Inference wrapper

```python
# src/models/wlasl_recognizer.py
class WLASLRecognizer:
    def __init__(self, checkpoint_path): ...
    def predict(self, video_path) -> list[(word, prob)]:
        # extract landmarks → adapter → model → top-k
```

### Step 5: End-to-end script + Makefile target

```bash
make recognize-word VIDEO=path/to/clip.mp4
# Output: store (0.84), shop (0.07), market (0.04), ...
```

For a **multi-word video**, chain it with the existing video segmenter (which
works fine when the signer drops their hands between signs — exactly the
sample_asl.mp4 case where we already get 6 segments cleanly):

```bash
make recognize-sentence VIDEO=path/to/sentence.mp4
# Output: glosses=[STORE, I, GO, YESTERDAY]
```

---

## File-by-File Changes

### New files

| File | Purpose |
|------|---------|
| `src/models/openhands_adapter.py` | MediaPipe → OpenHands keypoint format |
| `src/models/wlasl_recognizer.py` | Inference wrapper for OpenHands models |
| `scripts/recognize_word.py` | Single-clip inference CLI |
| `scripts/recognize_sentence.py` | Multi-segment recognition (chains segmenter + recognizer) |
| `tests/test_openhands_adapter.py` | Validate keypoint adapter on a known frame |

### Modified files

| File | Change |
|------|--------|
| `requirements.txt` | Add `openhands-cslr` (or git+ install) |
| `Makefile` | New targets: `download-openhands-ckpt`, `recognize-word`, `recognize-sentence` |
| `src/preprocessing/landmark_extractor.py` | Implement `extract_holistic_landmarks` (currently NotImplementedError) |
| `README.md` | New "Phase 2: Word Recognition" section |

---

## Evaluation

Per the proposal:
- **Top-1 word accuracy** on WLASL test split
- **Top-5 word accuracy**
- **Per-class F1** for low-frequency signs
- Comparison vs. published WLASL benchmarks (we should land at the published
  numbers since we're using their checkpoints)

**Targets** (from published OpenHands results):
- WLASL2000 top-1: ≥30% (ST-GCN) or ≥45% (MS-G3D)
- WLASL2000 top-5: ≥60% (ST-GCN) or ≥75% (MS-G3D)

For the **subset** of 100-300 words mentioned in the proposal, accuracy
will be considerably higher (smaller class space).

**Stretch (Phase 2.5):** fine-tune OpenHands ST-GCN on a custom subset to
improve accuracy on the specific words in our sentence demos.

---

## Risks

| Risk | Mitigation |
|------|-----------|
| OpenHands keypoint layout doesn't match MediaPipe cleanly | Use OpenHands' built-in MediaPipe extractor instead of writing an adapter (slower but correct) |
| Pretrained checkpoint doesn't load (PyTorch version skew) | Pin OpenHands version; use Python 3.10+ |
| WLASL2000 vocabulary doesn't include words we want to demo | Use the WLASL100/300 checkpoint (smaller vocab, higher accuracy on common words) |
| Inference slow on CPU | OpenHands ST-GCN is small enough to run real-time on CPU; MS-G3D needs GPU for fast inference |

---

## Implementation Order

1. **Install OpenHands** + verify a checkpoint loads (<1 hr)
2. **Single-clip inference** on a known WLASL test video, sanity-check
   against published top-1 (~2 hr)
3. **MediaPipe adapter** + inference on `sample_asl.mp4` segment
   (~half day)
4. **Sentence pipeline:** segmenter → per-clip recognizer → gloss list
   (~half day)
5. **Hook to Phase 3 LLM agent** (already on the roadmap — pass gloss list)
6. **Evaluation script** vs. WLASL test split (~half day)

Estimated effort: **~1 day for working demo, ~3 days for a polished Phase 2 deliverable.**

---

## Combining with the [[CTC_PLAN]] (Fingerspelling)

The two pipelines are complementary:

```
Sentence video
  │
  ▼
VideoSegmenter (existing — works when signer drops hands between signs)
  │
  ├─► Segment with continuous hand pose changes  ──►  CTC fingerspelling model  ──►  letter sequence
  │
  └─► Segment with discrete word-shape         ──►  OpenHands WLASL model     ──►  word
  │
  ▼
[STORE, I, GO, "BRANDON"-fingerspelled, YESTERDAY]
  │
  ▼
Phase 3 LLM:  "I went to the store yesterday with Brandon."
```

A simple router: if a segment looks like fingerspelling (rapid hand-shape
oscillation, fixed wrist position), send to CTC. Otherwise send to OpenHands.
Heuristic: average pose-change rate within the segment.

---

## Out of Scope

- Training new WLASL models from scratch (using OpenHands checkpoints)
- Phase 3 LLM (separate plan)
- Replacing PPCA fingerspelling baseline — kept as a Phase 1 deliverable

---

## References

1. AI4Bharat. *OpenHands: Making Sign Language Recognition Accessible.* ACL 2022. https://github.com/AI4Bharat/OpenHands
2. Li, D. et al. *Word-Level Deep Sign Language Recognition.* WACV 2020. (WLASL)
3. Yan, S. et al. *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.* AAAI 2018. (ST-GCN)
4. Liu, Z. et al. *Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition.* CVPR 2020. (MS-G3D)
