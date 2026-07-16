# Word Segmentation

**Status:** ✅ Implemented  
**Date:** 2026-06-29

Adds word boundary detection to the hold-based letter decoder. Detects word boundaries through:
1. **Hand-drop gaps**: When MediaPipe loses hand detection between holds
2. **Long pauses**: Gaps between letter holds exceeding a time threshold

---

## Usage

### Basic Usage
```bash
make transcribe-words VIDEO=path/to/video.mp4
```

### With Parameters
```bash
# Adjust pause threshold for word boundaries (default: 1.0 second)
make transcribe-words VIDEO=video.mp4 PAUSE=1.5

# Adjust minimum letter hold duration (default: 0.5 seconds)
make transcribe-words VIDEO=video.mp4 MIN_HOLD=0.3 PAUSE=1.0

# Use MPPCA instead of Random Forest
make transcribe-words VIDEO=video.mp4 EMISSION=experiments/mppca_balanced MPPCA=1

# Save JSON output with detailed segment data
make transcribe-words VIDEO=video.mp4 OUT_JSON=output.json
```

---

## Output Format

### Console Output
```
Segmented output:
  FIG FIG | DATE DATE | LIME LIME | GUAVA
```

Where `|` marks detected word boundaries.

### JSON Output (with `OUT_JSON=`)
```json
{
  "video": "video.mp4",
  "segmented": "FIG FIG | DATE DATE | LIME",
  "n_frames": 1250,
  "fps": 30.0,
  "n_segments": 3,
  "segments": [
    {
      "letters": "FIGFIG",
      "holds": [
        {"letter": "F", "start": 0, "end": 25, "duration": 25},
        {"letter": "I", "start": 25, "end": 50, "duration": 25},
        ...
      ]
    },
    ...
  ]
}
```

---

## How It Works

### 1. Landmark Extraction with Hand Tracking
Extracts MediaPipe landmarks and tracks hand presence per frame:
- `hand_present[t] = True` when hand detected
- `hand_present[t] = False` when hand absent (dropped, off-screen, etc.)

### 2. Per-Frame Classification
Same as `transcribe_holds.py`:
- Random Forest (default) or MPPCA classifier
- Predicts letter for each frame based on 63D normalized landmarks

### 3. Hold Detection
Groups consecutive frames with same prediction into "holds":
- Applies sliding-window majority smoothing (default: 11 frames)
- Filters out holds shorter than `min_hold_frames` (noise)

### 4. **NEW: Segmentation by Gaps**
Detects word boundaries between holds:

**Hand-drop boundary:**
```
Hold: F (frames 0-29)
Gap: frames 30-44 → hand_present[30:45] < 50% present
Hold: D (frames 45-54)
→ Word boundary detected
```

**Long-pause boundary:**
```
Hold: G (frames 20-30)
Gap: frames 30-70 (40 frames = 1.33s at 30fps)
Hold: D (frames 70-80)
→ Word boundary if gap ≥ pause_threshold
```

### 5. Output Formatting
- Collapses consecutive duplicate letters within each segment
- Joins segments with ` | ` separator

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VIDEO` | (required) | Input video path |
| `EMISSION` | `experiments/rf_balanced` | Classifier model directory |
| `MPPCA` | off | Set to `1` to use MPPCA classifier |
| `MIN_HOLD` | 0.5 | Minimum hold duration (seconds) |
| `PAUSE` | 1.0 | Pause threshold for word boundaries (seconds) |
| `SMOOTH` | 11 | Sliding window size for smoothing |
| `OUT_JSON` | none | Path to save JSON output |

---

## Tuning Guidelines

### For Fast Signing (ChicagoFSWild style)
```bash
make transcribe-words VIDEO=fast.mp4 MIN_HOLD=0.2 PAUSE=0.5
```
- Shorter letter holds → lower `MIN_HOLD`
- Shorter pauses between words → lower `PAUSE`

### For Slow Alphabet Recital
```bash
make transcribe-words VIDEO=alphabet.mp4 MIN_HOLD=0.5 PAUSE=1.5
```
- Longer letter holds → keep default `MIN_HOLD`
- Longer pauses between words → higher `PAUSE`

### For Dense Continuous Signing
```bash
make transcribe-words VIDEO=dense.mp4 PAUSE=2.0
```
- Only segment on long breaks or hand drops
- Increase `PAUSE` to avoid over-segmentation

---

## Algorithm Details

### Hand-Drop Detection
- **Trigger:** Gap where `mean(hand_present[gap_start:gap_end]) < 0.5`
- **Rationale:** Signer drops hand between words
- **Edge case:** If hand goes off-screen mid-word, may create false boundary

### Long-Pause Detection
- **Trigger:** Gap duration ≥ `pause_threshold_frames`
- **Rationale:** Natural pauses between words longer than within-word transitions
- **Edge case:** Very slow signing may have long gaps within words

### Duplicate Collapsing
Within each segment, consecutive duplicate letters are collapsed:
- Input segment holds: `F F I G`
- Output: `FIG` (not `FFIG`)
- **Rationale:** Repeated letter holds (e.g., "BOOK" spelled as B-O-O-K might have two O holds)

---

## Comparison to `transcribe-holds`

| Feature | `transcribe-holds` | `transcribe-words` |
|---------|--------------------|--------------------|
| Letter detection | ✓ | ✓ |
| Hold collapsing | ✓ | ✓ |
| Word boundaries | ✗ | ✓ (hand-drop + pause) |
| Output format | `FIGFIGDATEDATE` | `FIG FIG \| DATE DATE` |
| Performance | Same (identical classifier) | Same |

---

## Testing

### Unit Tests
```bash
python scripts/test_segmentation_logic.py
```

Tests:
1. Hand-drop segmentation
2. Long-pause segmentation
3. No segmentation (continuous signing)
4. Duplicate letter collapsing

All tests passing ✓

### Manual Testing
Record a test video:
1. Sign "FIG"
2. Drop hand
3. Sign "DATE"
4. Drop hand
5. Sign "LIME"

Expected output: `FIG | DATE | LIME`

---

## Next Steps

### 1. Vocabulary Constraint (Optional)
Add optional vocabulary filter:
```bash
make transcribe-words VIDEO=video.mp4 VOCAB=fruits.txt
```
Only accept segments matching known words (exact match, not fuzzy).

### 2. LLM Post-Processing (Phase 3)
Pass segmented output to Claude API for cleanup:
```
Input: "FIG FIG | DATEY | LIMNE | GUAUA"
LLM: "fig, date, lime, guava"
```

### 3. Adaptive Thresholds
Auto-calibrate `pause_threshold` per video based on:
- Median gap duration (within-word vs between-word)
- Hand-drop frequency
- Signing speed detection

### 4. Evaluation Metrics
- **Word Error Rate (WER)**: Compare segmented output to ground truth
- **Boundary F1**: Precision/recall of detected word boundaries
- **Word Recognition Rate**: % of target words recoverable

---

## Known Limitations

### 1. False Boundaries
- **Cause:** Hand briefly off-screen mid-word
- **Mitigation:** Increase hand-drop threshold (currently 50% of gap)

### 2. Missed Boundaries
- **Cause:** No hand-drop and pause < threshold
- **Mitigation:** Lower `PAUSE` threshold or add motion-based detection

### 3. J and Z Letters
- **Cause:** Motion-based letters not well-represented in static holds
- **Status:** Existing limitation from hold decoder (not new)
- **Solution:** Requires temporal model (future work)

### 4. No Motion Detection
- **Current:** Only uses hand presence/absence
- **Future:** Could add velocity-based detection (hand moving vs still)

---

## Files

### Scripts
- `scripts/transcribe_with_segmentation.py` — Main segmentation script
- `scripts/test_segmentation_logic.py` — Unit tests

### Makefile
- `make transcribe-words` — New target for word segmentation
- `make transcribe-holds` — Original (no segmentation)

---

## References

- Original hold decoder: `scripts/transcribe_holds.py`
- MediaPipe hand tracking: `src/preprocessing/landmark_extractor.py`
- Status report: `docs/STATUS_REPORT.md`
