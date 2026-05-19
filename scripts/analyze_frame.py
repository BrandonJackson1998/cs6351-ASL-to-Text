"""
Analyze a single frame using the trained MLP fingerspelling model.

Usage:
    make analyze-frame FRAME=path/to/frame.jpg
    make analyze-frame FRAME=path/to/frame.jpg CHECKPOINT=experiments/my_run/best_model.pt
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mlp import FingerspellingMLP
from src.preprocessing.landmark_extractor import extract_hand_landmarks


def analyze_frame(frame_path, checkpoint_path, verbose=False, output_json=None):
    if not os.path.isfile(frame_path):
        print(f"ERROR: frame not found: '{frame_path}'", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: checkpoint not found: '{checkpoint_path}'", file=sys.stderr)
        sys.exit(1)

    landmarks = extract_hand_landmarks(frame_path)
    if landmarks is None:
        print("No hand detected in frame.")
        if output_json:
            with open(output_json, "w") as f:
                json.dump({"letter": None, "confidence": None, "error": "no_hand_detected"}, f)
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    num_classes = ckpt.get("num_classes", 29)
    class_map = ckpt.get("class_map", {})
    idx_to_class = {v: k for k, v in class_map.items()}

    model = FingerspellingMLP(input_dim=63, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = torch.from_numpy(landmarks).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_k = min(3, num_classes)
    top_probs, top_indices = torch.topk(probs, top_k)

    top1_idx = top_indices[0].item()
    top1_letter = idx_to_class.get(top1_idx, str(top1_idx))
    top1_conf = top_probs[0].item()

    if verbose:
        print("Top predictions:")
        for i in range(top_k):
            idx = top_indices[i].item()
            letter = idx_to_class.get(idx, str(idx))
            conf = top_probs[i].item()
            print(f"  {i+1}. {letter:12s}  {conf:.4f}")
    else:
        print(f"Predicted: {top1_letter}  (confidence: {top1_conf:.4f})")

    if output_json:
        result = {"letter": top1_letter, "confidence": round(top1_conf, 4)}
        with open(output_json, "w") as f:
            json.dump(result, f)

    return top1_letter, top1_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a single frame with the trained MLP")
    parser.add_argument("--frame", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint", type=str,
        default="experiments/latest/best_model.pt",
        help="Path to trained .pt checkpoint",
    )
    parser.add_argument("--verbose", action="store_true", help="Print top-3 predictions")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Write {letter, confidence} JSON to this path")
    args = parser.parse_args()
    analyze_frame(args.frame, args.checkpoint, verbose=args.verbose, output_json=args.output_json)
