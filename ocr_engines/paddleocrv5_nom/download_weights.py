"""Download MinhDS fine-tuned PaddleOCRv5 Han-Nom weights from Hugging Face.

Usage:
    python -m ocr_engines.paddleocrv5_nom.download_weights

Downloads the inference model package and extracts it to
`ocr_engines/paddleocrv5_nom/weights/`.

Source: https://huggingface.co/spaces/MinhDS/Fine-tuned-PaddleOCRv5
"""

from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

_HF_BASE = (
    "https://huggingface.co/spaces/MinhDS/Fine-tuned-PaddleOCRv5/resolve/main"
)

_FILES = [
    ("PP-OCRv5_server_rec_infer.tar", True),   # tarball — extract
    ("inference.yml", False),
    ("inference.json", False),
]

_OUT_DIR = Path(__file__).parent / "weights"


def _download(url: str, dest: Path) -> None:
    print(f"  [dl] {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _hook(block, size, total):
        if total <= 0:
            return
        pct = min(100, int(block * size * 100 / total))
        sys.stdout.write(f"\r        {pct:3d}%")
        sys.stdout.flush()

    urlretrieve(url, dest, reporthook=_hook)
    sys.stdout.write("\n")


def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[paddleocrv5_nom] target: {_OUT_DIR}")

    for fname, is_tar in _FILES:
        url = f"{_HF_BASE}/{fname}"
        local = _OUT_DIR / fname
        if local.exists():
            print(f"  [skip] {fname} (already exists)")
            continue
        try:
            _download(url, local)
        except Exception as e:
            print(f"  [error] {fname}: {e}", file=sys.stderr)
            return 1

        if is_tar:
            print(f"  [extract] {fname}")
            with tarfile.open(local) as tf:
                tf.extractall(_OUT_DIR)
            # Flatten if extraction created a subdirectory
            extracted = _OUT_DIR / fname.replace(".tar", "")
            if extracted.is_dir():
                for p in extracted.iterdir():
                    shutil.move(str(p), str(_OUT_DIR / p.name))
                extracted.rmdir()

    print(f"\n[paddleocrv5_nom] ready at {_OUT_DIR}")
    print("Contents:")
    for p in sorted(_OUT_DIR.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:40s}  {size_mb:7.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
