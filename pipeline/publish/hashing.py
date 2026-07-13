"""Content hashing for release integrity + near-duplicate detection.

No external deps (imagehash is absent): sha256 for exact file integrity (fills the
croissant/datapackage sha256 fields that used to be the literal 'n/a'), and a
difference-hash (dHash) for perceptual near-duplicate detection across splits.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

__all__ = ["sha256_file", "sha256_bytes", "dhash", "hamming"]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dhash(path: str | Path, size: int = 8) -> int:
    """64-bit difference hash of an image (grayscale, (size+1)xsize, adjacent-compare).

    Robust to minor scaling/compression; identical or near-identical crops collide or
    sit at small Hamming distance. Returns a non-negative int, or -1 on an unreadable
    image (0 is a valid hash for a monotonic/uniform crop, so it is NOT the error value).
    """
    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    except Exception:
        return -1
    px = list(img.getdata())
    w = size + 1
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * w + col]
            right = px[row * w + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    """Hamming distance between two hash ints."""
    return bin(a ^ b).count("1")
