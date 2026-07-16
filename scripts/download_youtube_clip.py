"""Download and trim YouTube video clip with timestamps.

Usage:
    python scripts/download_youtube_clip.py \
        --url "https://youtube.com/watch?v=..." \
        --timestamps "1:23-1:45,2:10-2:30" \
        --output data/youtube_clips
"""

import argparse
import json
import os
import subprocess
import sys

import cv2


def download_youtube_video(url, output_path):
    """Download full YouTube video using yt-dlp."""
    print(f"Downloading YouTube video: {url}")

    # Try to find yt-dlp in virtualenv or system
    yt_dlp_cmd = None
    for path in ['.virtual_environment/bin/yt-dlp', 'yt-dlp']:
        try:
            result = subprocess.run([path, '--version'], capture_output=True)
            if result.returncode == 0:
                yt_dlp_cmd = path
                break
        except FileNotFoundError:
            continue

    if yt_dlp_cmd is None:
        print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        return False

    cmd = [
        yt_dlp_cmd,
        '-f', 'best[ext=mp4]',  # Best quality MP4
        '-o', output_path,
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: yt-dlp failed: {result.stderr}")
        return False

    print(f"✓ Downloaded to {output_path}")
    return True


def parse_timestamp(ts_str):
    """Parse timestamp string (MM:SS or M:SS) to seconds."""
    parts = ts_str.strip().split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:
        raise ValueError(f"Invalid timestamp format: {ts_str}")


def trim_video(input_path, output_path, start_sec, end_sec):
    """Trim video to specified time range."""
    duration = end_sec - start_sec

    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-ss', str(start_sec),
        '-t', str(duration),
        '-c', 'copy',  # Copy codec (fast, no re-encoding)
        '-y',  # Overwrite output
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: ffmpeg failed: {result.stderr}")
        return False

    print(f"✓ Trimmed {start_sec}s-{end_sec}s → {output_path}")
    return True


def get_video_info(video_path):
    """Get video duration, fps, resolution."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        'fps': fps,
        'duration': duration,
        'width': width,
        'height': height,
        'frame_count': frame_count
    }


def main():
    parser = argparse.ArgumentParser(description="Download and trim YouTube clips")
    parser.add_argument('--url', required=True, help="YouTube video URL")
    parser.add_argument('--timestamps', required=True,
                        help="Comma-separated timestamps (e.g., '1:23-1:45,2:10-2:30')")
    parser.add_argument('--output-dir', default='data/youtube_clips',
                        help="Output directory")
    parser.add_argument('--labels', default=None,
                        help="Comma-separated labels for each clip (optional)")
    parser.add_argument('--keep-full', action='store_true',
                        help="Keep full downloaded video")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse timestamps
    timestamp_pairs = []
    for ts_pair in args.timestamps.split(','):
        start_str, end_str = ts_pair.strip().split('-')
        start_sec = parse_timestamp(start_str)
        end_sec = parse_timestamp(end_str)
        timestamp_pairs.append((start_sec, end_sec))

    print(f"Parsed {len(timestamp_pairs)} timestamp pairs")

    # Parse labels if provided
    labels = None
    if args.labels:
        labels = [l.strip() for l in args.labels.split(',')]
        if len(labels) != len(timestamp_pairs):
            print(f"ERROR: Number of labels ({len(labels)}) doesn't match timestamps ({len(timestamp_pairs)})")
            sys.exit(1)

    # Download full video
    video_id = args.url.split('v=')[-1].split('&')[0]
    full_video_path = os.path.join(args.output_dir, f'{video_id}_full.mp4')

    if not os.path.exists(full_video_path):
        success = download_youtube_video(args.url, full_video_path)
        if not success:
            sys.exit(1)
    else:
        print(f"✓ Full video already exists: {full_video_path}")

    # Get video info
    video_info = get_video_info(full_video_path)
    print(f"\nVideo info:")
    print(f"  Duration: {video_info['duration']:.1f}s")
    print(f"  FPS: {video_info['fps']:.1f}")
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")

    # Trim clips
    clips = []
    for i, (start_sec, end_sec) in enumerate(timestamp_pairs):
        label = labels[i] if labels else f"clip{i+1}"
        clip_path = os.path.join(args.output_dir, f'{video_id}_{label}_{start_sec}-{end_sec}.mp4')

        if os.path.exists(clip_path):
            print(f"✓ Clip already exists: {clip_path}")
        else:
            success = trim_video(full_video_path, clip_path, start_sec, end_sec)
            if not success:
                print(f"✗ Failed to trim clip {i+1}")
                continue

        clips.append({
            'label': label,
            'start': start_sec,
            'end': end_sec,
            'duration': end_sec - start_sec,
            'path': clip_path
        })

    # Save metadata
    metadata = {
        'url': args.url,
        'video_id': video_id,
        'full_video': full_video_path if args.keep_full else None,
        'video_info': video_info,
        'clips': clips
    }

    metadata_path = os.path.join(args.output_dir, f'{video_id}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Created {len(clips)} clips")
    print(f"✓ Metadata saved to {metadata_path}")

    # Clean up full video if not keeping
    if not args.keep_full and os.path.exists(full_video_path):
        os.remove(full_video_path)
        print(f"✓ Removed full video (kept clips only)")

    print("\nNext step:")
    print(f"  python scripts/extract_wlasl_landmarks.py --videos-dir {args.output_dir}")


if __name__ == "__main__":
    main()
