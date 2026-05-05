#!/usr/bin/env python3
"""Print the local environment status for the DeepSeek-OCR experiment."""

from __future__ import annotations

import importlib
import platform
from pathlib import Path


MODEL_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / (
    "models--deepseek-ai--DeepSeek-OCR"
)


def _version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"missing ({type(exc).__name__}: {exc})"
    return str(getattr(module, "__version__", "installed"))


def main() -> None:
    print("DeepSeek-OCR environment")
    print("=" * 32)
    print(f"Python:       {platform.python_version()}")
    print(f"Platform:     {platform.platform()}")
    print(f"torch:        {_version('torch')}")
    print(f"transformers: {_version('transformers')}")
    print(f"Pillow:       {_version('PIL')}")
    print(f"opencv:       {_version('cv2')}")
    print(f"hf hub:       {_version('huggingface_hub')}")

    try:
        import torch

        print(f"CUDA:         {torch.cuda.is_available()}")
        print(
            "MPS:          "
            f"{hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}"
        )
    except Exception:
        print("CUDA:         unavailable")
        print("MPS:          unavailable")

    print(f"Model cache:  {MODEL_CACHE if MODEL_CACHE.exists() else 'not found'}")
    if not MODEL_CACHE.exists():
        print("Note: first real inference will need to download deepseek-ai/DeepSeek-OCR.")


if __name__ == "__main__":
    main()
