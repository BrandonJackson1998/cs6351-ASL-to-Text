"""
String together frame delta predictions from a video.

Segments the video, extracts keyframes, runs the MLP on each keyframe,
applies temporal deduplication, and outputs an ordered letter sequence.

Usage:
    make string-deltas VIDEO=path/to/alphabet_video.mp4
"""

import argparse
from collections import Counter
import json
import os
import sys
import shutil
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mlp import FingerspellingMLP
from src.preprocessing.keyframe_extractor import KeyframeExtractor
from src.preprocessing.landmark_extractor import extract_hand_landmarks


def _load_model(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: checkpoint not found: '{checkpoint_path}'", file=sys.stderr)
        sys.exit(1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get("num_classes", 29)
    class_map = ckpt.get("class_map", {})
    idx_to_class = {v: k for k, v in class_map.items()}
    model = FingerspellingMLP(input_dim=63, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, idx_to_class


def _predict(model, idx_to_class, landmarks, device):
    x = torch.from_numpy(landmarks).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).squeeze(0)
    idx = probs.argmax().item()
    return idx_to_class.get(idx, str(idx)), probs[idx].item()


def _is_url(path):
    return path.startswith("http://") or path.startswith("https://")


def _download_video(url):
    """Download a YouTube (or other yt-dlp-supported) URL to a temp dir. Returns the file path."""
    try:
        import yt_dlp
    except ImportError:
        print("ERROR: yt-dlp is required for URL input. Run: pip install yt-dlp", file=sys.stderr)
        sys.exit(1)

    tmp_dir = tempfile.mkdtemp()
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(tmp_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
    if not files:
        print("ERROR: yt-dlp did not produce any output file.", file=sys.stderr)
        sys.exit(1)
    return files[0]


def _video_id_from_url(url):
    """Extract a filesystem-safe video ID from a URL (uses the last path component or query param)."""
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    # YouTube short-form: /shorts/<id>  or  ?v=<id>
    qs = urllib.parse.parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    path_parts = [p for p in parsed.path.split("/") if p]
    if path_parts:
        return path_parts[-1]
    return "video"


def _load_cached_keyframes(keyframes_dir):
    """Return Keyframe objects for all .jpg files already in keyframes_dir, sorted by frame index."""
    from src.preprocessing.keyframe_extractor import Keyframe
    images = sorted(
        f for f in os.listdir(keyframes_dir) if f.endswith(".jpg")
    )
    keyframes = []
    for fname in images:
        # filename format: seg<NNNN>_frame<NNNNNN>.jpg
        base = os.path.splitext(fname)[0]
        parts = base.split("_")
        try:
            seg_idx = int(parts[0].lstrip("seg"))
            frame_idx = int(parts[1].lstrip("frame"))
        except (IndexError, ValueError):
            continue
        keyframes.append(Keyframe(
            segment_index=seg_idx,
            frame_index=frame_idx,
            timestamp=0.0,
            image_path=os.path.join(keyframes_dir, fname),
        ))
    return keyframes


def string_deltas(video_path, checkpoint_path, output_dir, confidence_threshold, dedup, verbose):
    tmp_file = None

    # Determine video ID for cache directory before downloading
    if _is_url(video_path):
        video_id = _video_id_from_url(video_path)
    else:
        video_id = os.path.splitext(os.path.basename(video_path))[0]

    keyframes_dir = os.path.join(output_dir, video_id, "keyframes")
    cached_images = (
        [f for f in os.listdir(keyframes_dir) if f.endswith(".jpg")]
        if os.path.isdir(keyframes_dir)
        else []
    )

    if cached_images:
        print(f"Using cached keyframes ({len(cached_images)} frames) from {keyframes_dir}")
        keyframes = _load_cached_keyframes(keyframes_dir)
    else:
        if _is_url(video_path):
            print(f"Downloading video from URL...")
            tmp_file = _download_video(video_path)
            video_path = tmp_file

        if not os.path.isfile(video_path):
            print(f"ERROR: video not found: '{video_path}'", file=sys.stderr)
            sys.exit(1)

        print("=== Segmenting + extracting keyframes ===")
        extractor = KeyframeExtractor()
        keyframes = extractor.extract(video_path, output_dir=output_dir)
        print(f"Extracted {len(keyframes)} keyframes across {len(set(kf.segment_index for kf in keyframes))} segments")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_class = _load_model(checkpoint_path, device)

    print("\n=== Running predictions ===")
    raw_predictions = []
    for kf in keyframes:
        image_path = kf.image_path
        seg_idx = kf.segment_index
        frame_num = kf.frame_index

        landmarks = extract_hand_landmarks(image_path)
        if landmarks is None:
            if verbose:
                print(f"  [seg {seg_idx}] frame {frame_num} → no hand detected, skipping")
            continue

        letter, conf = _predict(model, idx_to_class, landmarks, device)

        if conf < confidence_threshold:
            if verbose:
                print(f"  [seg {seg_idx}] frame {frame_num} → {letter} ({conf:.4f}) below threshold, skipping")
            continue

        if verbose:
            print(f"  [seg {seg_idx}] frame {frame_num} → {letter} ({conf:.4f})")

        raw_predictions.append({
            "segment": seg_idx,
            "frame": frame_num,
            "predicted": letter,
            "confidence": round(conf, 4),
        })

    if dedup:
        # Group consecutive identical predictions into runs.
        # Accept a run if its peak confidence clears the threshold (confidence-based,
        # not frame-count-based). Keep the highest-confidence frame from each run.
        runs = []
        for pred in raw_predictions:
            if runs and pred["predicted"] == runs[-1][-1]["predicted"]:
                runs[-1].append(pred)
            else:
                runs.append([pred])

        deduped = []
        for run in runs:
            best = max(run, key=lambda p: p["confidence"])
            if best["confidence"] >= confidence_threshold:
                deduped.append(best)
    else:
        deduped = raw_predictions

    sequence_str = " ".join(p["predicted"] for p in deduped)

    print(f"\nSequence: {sequence_str}")
    print(f"({len(deduped)} unique predictions from {len(raw_predictions)} total)")

    video_out_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_out_dir, exist_ok=True)
    predictions_path = os.path.join(video_out_dir, "predictions.json")
    with open(predictions_path, "w") as f:
        json.dump(
            {
                "video": video_id,
                "sequence": deduped,
                "deduped_string": sequence_str.replace(" ", ""),
            },
            f,
            indent=2,
        )
    print(f"Results saved to {predictions_path}")

    if tmp_file:
        shutil.rmtree(os.path.dirname(tmp_file), ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="String together frame delta predictions from a video")
    parser.add_argument("--input", type=str, required=True, help="Input video file")
    parser.add_argument(
        "--checkpoint", type=str,
        default="experiments/latest/best_model.pt",
        help="Path to trained .pt checkpoint",
    )
    parser.add_argument("--output-dir", type=str, default="segments/", help="Where to write keyframes and predictions")
    parser.add_argument("--confidence-threshold", type=float, default=0.6,
                        help="Minimum confidence to include a prediction (default: 0.6)")
    parser.add_argument("--no-dedup", action="store_true", help="Disable temporal deduplication")
    parser.add_argument("--verbose", action="store_true", help="Print per-keyframe details")
    args = parser.parse_args()
    string_deltas(
        video_path=args.input,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        dedup=not args.no_dedup,
        verbose=args.verbose,
    )
