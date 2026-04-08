"""
Dataset download helper.
Downloads ASL Alphabet (Kaggle) and/or WLASL datasets.
"""

import argparse


def download_alphabet(output_dir="data/asl_alphabet"):
    """Download ASL Alphabet dataset from Kaggle."""
    raise NotImplementedError("Implement Kaggle dataset download")


def download_wlasl(output_dir="data/wlasl", num_words=100):
    """Download WLASL dataset subset."""
    raise NotImplementedError("Implement WLASL dataset download")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ASL datasets")
    parser.add_argument("--dataset", choices=["alphabet", "wlasl", "all"], required=True)
    parser.add_argument("--num-words", type=int, default=100, help="WLASL subset size")
    args = parser.parse_args()

    if args.dataset in ("alphabet", "all"):
        download_alphabet()
    if args.dataset in ("wlasl", "all"):
        download_wlasl(num_words=args.num_words)
