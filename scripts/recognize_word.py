"""Recognize word-level sign from video clip.

Usage:
    python scripts/recognize_word.py \
        --input video.mp4 \
        --model experiments/synthetic_wlasl20/best_model.pth \
        --vocabulary data/synthetic_wordsigns/vocabulary.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.word_sign_recognizer import create_model
from scripts.extract_wlasl_landmarks import extract_landmarks_from_video


def load_model(checkpoint_path, vocabulary_path, model_type='cnn', device='cpu'):
    """Load trained model."""
    # Load vocabulary
    with open(vocabulary_path) as f:
        vocabulary = json.load(f)

    num_classes = len(vocabulary)
    idx_to_word = {v: k for k, v in vocabulary.items()}

    # Create model
    model = create_model(model_type, num_classes=num_classes, input_dim=63)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Vocabulary: {num_classes} classes")
    print(f"  Best val acc: {checkpoint.get('val_acc', 'N/A')}")

    return model, idx_to_word


def recognize(video_path, model, idx_to_word, device='cpu', top_k=5):
    """Recognize word sign from video."""
    # Extract landmarks
    print(f"Extracting landmarks from {video_path}...")
    landmarks = extract_landmarks_from_video(video_path)

    if landmarks is None or len(landmarks) == 0:
        print("ERROR: No landmarks extracted")
        return None

    print(f"  Extracted {len(landmarks)} frames")

    # Prepare input
    landmarks_tensor = torch.from_numpy(landmarks).unsqueeze(0).to(device)  # (1, T, 63)

    # Inference
    with torch.no_grad():
        logits = model(landmarks_tensor)
        probs = torch.softmax(logits, dim=1)

        top_k_probs, top_k_idx = torch.topk(probs, k=min(top_k, len(idx_to_word)), dim=1)

    # Format results
    results = []
    for i in range(top_k_probs.size(1)):
        class_idx = top_k_idx[0, i].item()
        prob = top_k_probs[0, i].item()
        word = idx_to_word.get(class_idx, f"class{class_idx}")
        results.append({
            'rank': i + 1,
            'word': word,
            'probability': prob,
            'class_idx': class_idx
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Recognize word-sign from video")
    parser.add_argument('--input', required=True, help="Input video path")
    parser.add_argument('--model', required=True, help="Model checkpoint path")
    parser.add_argument('--vocabulary', required=True, help="Vocabulary JSON path")
    parser.add_argument('--model-type', choices=['cnn', 'cnn_lstm'], default='cnn')
    parser.add_argument('--top-k', type=int, default=5, help="Number of top predictions")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--out-json', default=None, help="Save results to JSON")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Video not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    # Load model
    device = torch.device(args.device)
    model, idx_to_word = load_model(args.model, args.vocabulary, args.model_type, device)

    # Recognize
    print(f"\nRecognizing sign from {args.input}...")
    results = recognize(args.input, model, idx_to_word, device, args.top_k)

    if results is None:
        sys.exit(1)

    # Display results
    print(f"\nTop-{args.top_k} Predictions:")
    for r in results:
        print(f"  {r['rank']}. {r['word']}: {r['probability']:.4f}")

    # Save to JSON if requested
    if args.out_json:
        output = {
            'video': args.input,
            'model': args.model,
            'predictions': results
        }
        with open(args.out_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Results saved to {args.out_json}")


if __name__ == "__main__":
    main()
