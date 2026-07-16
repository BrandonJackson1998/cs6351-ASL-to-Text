"""Download WLASL video clips from various sources.

Downloads videos for WLASL dataset from:
- ASL-LEX (primary source)
- ASLPro (flash videos, skip)
- YouTube (via yt-dlp if available)

Usage:
    python scripts/download_wlasl_videos.py --subset 100 --max-workers 4
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'download_wlasl_{int(time.time())}.log')
    ]
)


def request_video(url, referer=''):
    """Download video from URL with proper headers."""
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    headers = {'User-Agent': user_agent}

    if referer:
        headers['Referer'] = referer

    request = urllib.request.Request(url, None, headers)
    response = urllib.request.urlopen(request, timeout=30)
    data = response.read()
    return data


def download_video(url, output_path):
    """Download video and save to output_path."""
    if os.path.exists(output_path):
        logging.info(f'✓ Already exists: {output_path}')
        return True

    try:
        # Skip YouTube and ASLPro videos
        if 'youtube' in url or 'youtu.be' in url:
            logging.warning(f'⊗ Skipping YouTube video: {url}')
            return False

        if 'aslpro' in url:
            logging.warning(f'⊗ Skipping ASLPro (flash) video: {url}')
            return False

        # Download
        logging.info(f'↓ Downloading: {url}')
        data = request_video(url)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(data)

        # Verify it's a valid video
        cap = cv2.VideoCapture(output_path)
        if not cap.isOpened():
            logging.error(f'✗ Invalid video: {output_path}')
            os.remove(output_path)
            return False
        cap.release()

        logging.info(f'✓ Downloaded: {output_path}')
        time.sleep(0.5)  # Be nice to the server
        return True

    except Exception as e:
        logging.error(f'✗ Failed to download {url}: {e}')
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def trim_video(input_path, output_path, frame_start, frame_end):
    """Trim video to specified frame range."""
    if os.path.exists(output_path):
        return True

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set up writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Read and write frames in range
        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        while frame_idx < (frame_end - frame_start):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        return True

    except Exception as e:
        logging.error(f'✗ Failed to trim {input_path}: {e}')
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def download_sample(sample, output_dir, trim=True):
    """Download and process a single sample.

    Returns:
        (word, video_id, success)
    """
    word = sample['word']
    video_id = sample['video_id']
    url = sample['url']
    frame_start = sample.get('frame_start', -1)
    frame_end = sample.get('frame_end', -1)

    # Output paths
    word_dir = os.path.join(output_dir, 'videos', word)
    raw_path = os.path.join(word_dir, f'{video_id}_raw.mp4')
    final_path = os.path.join(word_dir, f'{video_id}.mp4')

    # Download full video
    success = download_video(url, raw_path)
    if not success:
        return (word, video_id, False)

    # Trim to annotated segment if available
    if trim and frame_start >= 0 and frame_end > frame_start:
        success = trim_video(raw_path, final_path, frame_start, frame_end)
        if success:
            os.remove(raw_path)  # Remove raw, keep trimmed
        else:
            # Trimming failed, use full video
            os.rename(raw_path, final_path)
    else:
        # No frame annotations, use full video
        os.rename(raw_path, final_path)

    return (word, video_id, True)


def load_splits(splits_dir, split_name):
    """Load samples from split file."""
    split_path = os.path.join(splits_dir, f'{split_name}.json')
    with open(split_path) as f:
        samples = json.load(f)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Download WLASL videos")
    parser.add_argument('--subset', choices=['100', '300', '2000'], default='100',
                        help="WLASL vocabulary size")
    parser.add_argument('--splits-dir', default='data/wlasl/splits',
                        help="Directory with train/val/test splits")
    parser.add_argument('--output-dir', default='data/wlasl',
                        help="Output directory for videos")
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='all',
                        help="Which split to download")
    parser.add_argument('--max-workers', type=int, default=4,
                        help="Number of parallel downloads")
    parser.add_argument('--max-videos', type=int, default=None,
                        help="Limit number of videos (for testing)")
    parser.add_argument('--no-trim', action='store_true',
                        help="Don't trim videos to annotated segments")
    args = parser.parse_args()

    logging.info("="*60)
    logging.info(f"Downloading WLASL{args.subset} videos")
    logging.info("="*60)

    # Load samples
    if args.split == 'all':
        splits_to_download = ['train', 'val', 'test']
    else:
        splits_to_download = [args.split]

    all_samples = []
    for split in splits_to_download:
        samples = load_splits(args.splits_dir, split)
        all_samples.extend(samples)
        logging.info(f"Loaded {len(samples)} samples from {split} split")

    # Limit for testing
    if args.max_videos:
        all_samples = all_samples[:args.max_videos]
        logging.info(f"Limited to {args.max_videos} videos for testing")

    logging.info(f"Total samples to download: {len(all_samples)}")

    # Download in parallel
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(download_sample, sample, args.output_dir, not args.no_trim): sample
            for sample in all_samples
        }

        for future in as_completed(futures):
            word, video_id, success = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1

            # Progress
            total = success_count + fail_count
            logging.info(f"Progress: {total}/{len(all_samples)} ({success_count} success, {fail_count} failed)")

    # Summary
    logging.info("\n" + "="*60)
    logging.info("Download Complete")
    logging.info("="*60)
    logging.info(f"Total samples: {len(all_samples)}")
    logging.info(f"Successfully downloaded: {success_count}")
    logging.info(f"Failed: {fail_count}")
    logging.info(f"Success rate: {100*success_count/len(all_samples):.1f}%")
    logging.info("="*60)


if __name__ == "__main__":
    main()
