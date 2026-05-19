"""
Dataset download helper.
Downloads ASL Alphabet (Kaggle) and/or WLASL datasets.

Requires Kaggle authentication via one of:
  - Access token: ~/.kaggle/access_token (recommended, new method)
  - Legacy API key: ~/.kaggle/kaggle.json
  - Environment variable: KAGGLE_API_TOKEN

Setup: https://www.kaggle.com/settings -> API -> Create New Token
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Kaggle credential helpers
# ---------------------------------------------------------------------------

SETUP_INSTRUCTIONS = """
Kaggle API credentials not found.

To set up (choose one method):

METHOD 1 (Recommended - New Access Token):
  1. Go to https://www.kaggle.com/settings
  2. Scroll to API section, click "Create New Token"
  3. Copy the token command and run it:
     mkdir -p ~/.kaggle && echo KGAT_your_token_here > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

METHOD 2 (Legacy - JSON API Key):
  1. Go to https://www.kaggle.com/settings
  2. Scroll to API section, click "Create New API Token"
  3. Move the downloaded file:
     mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

Then re-run this command.
"""


def _check_kaggle_credentials():
    """Verify Kaggle authentication is available (access_token or kaggle.json)."""
    access_token = os.path.expanduser("~/.kaggle/access_token")
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    env_token = os.environ.get("KAGGLE_API_TOKEN")

    if os.path.isfile(access_token) or os.path.isfile(kaggle_json) or env_token:
        return  # At least one auth method is available

    print(SETUP_INSTRUCTIONS, file=sys.stderr)
    sys.exit(1)


def _kaggle_api():
    """Authenticate and return the Kaggle API client."""
    try:
        from kaggle import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle", file=sys.stderr)
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()
    return api


# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------

def download_alphabet(output_dir="data/asl_alphabet"):
    """Download the ASL Alphabet dataset from Kaggle.

    Dataset: grassknoted/asl-alphabet
    Size:    ~1 GB, 87 000 images across 29 classes (A-Z + SPACE, DELETE, NOTHING)

    Expected output layout after extraction::

        data/asl_alphabet/
        ├── asl_alphabet_train/
        │   ├── A/
        │   ├── B/
        │   └── ... (one subdirectory per class)
        └── asl_alphabet_test/
    """
    _check_kaggle_credentials()
    api = _kaggle_api()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading ASL Alphabet dataset to '{output_dir}' ...")

    api.dataset_download_files(
        "grassknoted/asl-alphabet",
        path=output_dir,
        unzip=True,
        quiet=False,
    )

    print(f"ASL Alphabet dataset ready in '{output_dir}'.")


def download_wlasl(output_dir="data/wlasl", num_words=100):
    """Download the WLASL 2000 (resized) dataset from Kaggle.

    Dataset: sttaseen/wlasl2000-resized
    Size:    ~3 GB; contains video clips for 2 000 ASL words

    After download the ``num_words`` most-common word classes are kept to
    limit disk usage during development. Pass ``num_words=0`` to keep all
    classes.

    Expected output layout after extraction::

        data/wlasl/
        └── WLASL2000/
            ├── bed/
            │   ├── 00001.mp4
            │   └── ...
            └── ... (one subdirectory per word class)
    """
    _check_kaggle_credentials()
    api = _kaggle_api()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading WLASL 2000 dataset to '{output_dir}' ...")

    api.dataset_download_files(
        "sttaseen/wlasl2000-resized",
        path=output_dir,
        unzip=True,
        quiet=False,
    )

    if num_words > 0:
        _prune_wlasl(output_dir, num_words)

    print(f"WLASL dataset ready in '{output_dir}'.")


def _prune_wlasl(output_dir, num_words):
    """Keep only the ``num_words`` largest word-class subdirectories."""
    import shutil

    # Locate the unpacked root (typically output_dir/WLASL2000/)
    candidates = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ]
    if not candidates:
        print("WARNING: could not locate WLASL subdirectory for pruning.", file=sys.stderr)
        return

    wlasl_root = os.path.join(output_dir, candidates[0])
    class_dirs = [
        d for d in os.listdir(wlasl_root)
        if os.path.isdir(os.path.join(wlasl_root, d))
    ]

    # Sort by number of clips (descending) and drop the tail
    class_dirs.sort(
        key=lambda d: len(os.listdir(os.path.join(wlasl_root, d))),
        reverse=True,
    )
    to_remove = class_dirs[num_words:]

    if to_remove:
        print(f"Pruning {len(to_remove)} word classes (keeping top {num_words}) ...")
        for cls in to_remove:
            shutil.rmtree(os.path.join(wlasl_root, cls))

    print(f"Kept {min(num_words, len(class_dirs))} word classes.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ASL datasets from Kaggle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=SETUP_INSTRUCTIONS.strip(),
    )
    parser.add_argument(
        "--dataset",
        choices=["alphabet", "wlasl", "all"],
        required=True,
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--num-words",
        type=int,
        default=100,
        help="Number of most-common WLASL word classes to keep (0 = keep all). Default: 100.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Root directory for downloaded datasets. Default: data/.",
    )
    args = parser.parse_args()

    if args.dataset in ("alphabet", "all"):
        download_alphabet(output_dir=os.path.join(args.output_dir, "asl_alphabet"))
    if args.dataset in ("wlasl", "all"):
        download_wlasl(
            output_dir=os.path.join(args.output_dir, "wlasl"),
            num_words=args.num_words,
        )
