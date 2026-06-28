"""External lexicon loading for Catholic transliteration data.

Saint names, toponyms, loan phrases and OCR-confusion fixes used to be
hard-coded inside core/text/text_utils.py and core/text/loanword.py. Editing
those tables for a newly-scanned book meant touching pipeline source code —
error prone and easy to break.

They now live as plain-data JSON files under ``config/lexicon/`` (override the
directory with the ``GANNHANOCR_LEXICON_DIR`` env var). To add a saint name or
place name for a new book you edit the JSON only — no code change.

Robustness contract (this is on the MAIN pipeline):
  * If the JSON file exists and parses to the expected type, it is the single
    source of truth (you can add AND remove entries there).
  * If it is missing, unreadable, or malformed, we fall back to the built-in
    ``defaults`` and emit one warning to stderr — the pipeline never crashes
    because of a config typo.
  * Keys starting with ``_`` in a mapping file are treated as comments/metadata
    and ignored, so the JSON can document itself.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def lexicon_dir() -> Path:
    """Directory holding the lexicon JSON files (env-overridable)."""
    env = os.environ.get("GANNHANOCR_LEXICON_DIR")
    if env:
        return Path(env)
    # core/text/lexicon.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2] / "config" / "lexicon"


def _read_json(filename: str):
    path = lexicon_dir() / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"[lexicon] WARNING: cannot read {path} ({e}); "
              f"using built-in defaults.", file=sys.stderr)
        return None


def load_mapping(filename: str, defaults: dict[str, str]) -> dict[str, str]:
    """Load a {str: str} mapping from ``filename``; fall back to ``defaults``.

    Keys beginning with ``_`` are ignored (reserved for in-file comments).
    """
    data = _read_json(filename)
    if data is None:
        return dict(defaults)
    if not isinstance(data, dict):
        print(f"[lexicon] WARNING: {filename} is not a JSON object; "
              f"using built-in defaults.", file=sys.stderr)
        return dict(defaults)
    out: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and k.startswith("_"):
            continue
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out or dict(defaults)


def load_list(filename: str, defaults: list[str]) -> list[str]:
    """Load a list[str] from ``filename``; fall back to ``defaults``."""
    data = _read_json(filename)
    if data is None:
        return list(defaults)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        print(f"[lexicon] WARNING: {filename} is not a JSON list of strings; "
              f"using built-in defaults.", file=sys.stderr)
        return list(defaults)
    return list(data) or list(defaults)
