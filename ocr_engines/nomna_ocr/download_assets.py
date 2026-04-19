"""Download NomNaOCR assets (vocab.txt + CRNN.h5) from the NomNaSite repo.

Usage:
    python -m ocr_engines.nomna_ocr.download_assets

Both files are served directly from GitHub (raw.githubusercontent.com),
no Google Drive or auth required. Total size ~51 MB.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

_BASE = "https://raw.githubusercontent.com/ds4v/NomNaSite/main/assets"
_OUT_DIR = Path(__file__).parent / "assets"

_FILES = [
    ("vocab.txt", f"{_BASE}/vocab.txt"),
    ("CRNN.h5",   f"{_BASE}/CRNN.h5"),
]


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
    print(f"[nomna_ocr] target: {_OUT_DIR}")

    for fname, url in _FILES:
        local = _OUT_DIR / fname
        if local.exists() and local.stat().st_size > 0:
            print(f"  [skip] {fname} ({local.stat().st_size // 1024} KB)")
            continue
        try:
            _download(url, local)
        except Exception as e:
            print(f"  [error] {fname}: {e}", file=sys.stderr)
            return 1

    print(f"\n[nomna_ocr] ready at {_OUT_DIR}")
    for p in sorted(_OUT_DIR.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:20s}  {size_mb:7.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
