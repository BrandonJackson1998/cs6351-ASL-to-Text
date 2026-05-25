"""Download ChicagoFSWild via kagglehub, extract frames into data/chicagofswild/.

Requires Kaggle authentication (~/.kaggle/access_token or ~/.kaggle/kaggle.json).
"""

import os
import shutil
import subprocess
import sys


def main():
    target = os.path.abspath("data/chicagofswild")
    if os.path.isfile(os.path.join(target, "ChicagoFSWild.csv")):
        print(f"FSWild already extracted at {target}")
        return

    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed. Run: pip install kagglehub", file=sys.stderr)
        sys.exit(1)

    print("Downloading ChicagoFSWild via kagglehub (~13 GB) ...")
    src = kagglehub.dataset_download("joebeachcapital/chicagofswild")
    print(f"  Cached to: {src}")

    os.makedirs(target, exist_ok=True)
    for fname in ["ChicagoFSWild.csv", "HandAnnotation.csv"]:
        s = os.path.join(src, fname)
        if os.path.isfile(s):
            shutil.copy2(s, target)

    tarball = os.path.join(src, "ChicagoFSWild-Frames.tgz")
    if not os.path.isfile(tarball):
        print(f"ERROR: expected {tarball}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting frames to {target} ...")
    subprocess.run(["tar", "-xzf", tarball, "-C", target], check=True)
    print(f"Done.")


if __name__ == "__main__":
    main()
