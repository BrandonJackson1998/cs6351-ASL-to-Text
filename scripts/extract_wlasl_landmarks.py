"""Extract MediaPipe hand landmarks from WLASL videos.

Processes downloaded WLASL videos and extracts hand landmarks for training.

Output format: {word}/{video_id}.npy with shape (T, 63)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.landmark_extractor import _make_landmarker, normalize_landmarks


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def extract_landmarks_from_video(video_path):
    """Extract landmark sequence from video.

    Returns:
        np.ndarray: (T, 63) normalized landmarks, or None if failed
    """
    landmarker = _make_landmarker()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    landmarks_list = []
    last_valid = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            raw = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
            normalized = normalize_landmarks(raw)
            landmarks_list.append(normalized)
            last_valid = normalized
        elif last_valid is not None:
            # Hand temporarily lost, use last valid frame
            landmarks_list.append(last_valid.copy())

    cap.release()
    landmarker.close()

    if not landmarks_list:
        return None

    return np.stack(landmarks_list)


def process_sample(sample, videos_dir, output_dir):
    """Process a single video sample.

    Returns:
        (word, video_id, success, seq_len)
    """
    word = sample['word']
    video_id = sample['video_id']

    video_path = os.path.join(videos_dir, word, f'{video_id}.mp4')
    output_path = os.path.join(output_dir, word, f'{video_id}.npy')

    # Skip if already processed
    if os.path.exists(output_path):
        landmarks = np.load(output_path)
        return (word, video_id, True, len(landmarks))

    # Check video exists
    if not os.path.exists(video_path):
        logging.warning(f'Video not found: {video_path}')
        return (word, video_id, False, 0)

    # Extract landmarks
    try:
        landmarks = extract_landmarks_from_video(video_path)

        if landmarks is None or len(landmarks) == 0:
            logging.error(f'No landmarks extracted from {video_path}')
            return (word, video_id, False, 0)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, landmarks)

        logging.info(f'✓ Extracted {len(landmarks)} frames from {word}/{video_id}')
        return (word, video_id, True, len(landmarks))

    except Exception as e:
        logging.error(f'✗ Failed to process {video_path}: {e}')
        return (word, video_id, False, 0)


def load_splits(splits_dir, split_name):
    """Load samples from split file."""
    split_path = os.path.join(splits_dir, f'{split_name}.json')
    with open(split_path) as f:
        samples = json.load(f)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Extract WLASL landmarks")
    parser.add_argument('--subset', choices=['100', '300', '2000'], default='100')
    parser.add_argument('--videos-dir', default='data/wlasl/videos')
    parser.add_argument('--output-dir', default='data/wlasl/landmarks')
    parser.add_argument('--splits-dir', default='data/wlasl/splits')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='all')
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_args()

    logging.info("="*60)
    logging.info(f"Extracting WLASL{args.subset} landmarks")
    logging.info("="*60)

    # Load samples
    if args.split == 'all':
        splits_to_process = ['train', 'val', 'test']
    else:
        splits_to_process = [args.split]

    all_samples = []
    for split in splits_to_process:
        samples = load_splits(args.splits_dir, split)
        all_samples.extend(samples)
        logging.info(f"Loaded {len(samples)} samples from {split} split")

    logging.info(f"Total samples to process: {len(all_samples)}")

    # Process in parallel
    success_count = 0
    fail_count = 0
    total_frames = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_sample, sample, args.videos_dir, args.output_dir): sample
            for sample in all_samples
        }

        for future in as_completed(futures):
            word, video_id, success, seq_len = future.result()

            if success:
                success_count += 1
                total_frames += seq_len
            else:
                fail_count += 1

            # Progress
            total = success_count + fail_count
            if total % 10 == 0:
                logging.info(f"Progress: {total}/{len(all_samples)} ({success_count} success, {fail_count} failed)")

    # Summary
    avg_frames = total_frames / success_count if success_count > 0 else 0

    logging.info("\n" + "="*60)
    logging.info("Landmark Extraction Complete")
    logging.info("="*60)
    logging.info(f"Total samples: {len(all_samples)}")
    logging.info(f"Successfully processed: {success_count}")
    logging.info(f"Failed: {fail_count}")
    logging.info(f"Success rate: {100*success_count/len(all_samples):.1f}%")
    logging.info(f"Total frames: {total_frames}")
    logging.info(f"Average sequence length: {avg_frames:.1f} frames")
    logging.info("="*60)


if __name__ == "__main__":
    main()
