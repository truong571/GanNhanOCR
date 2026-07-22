"""Hugging Face Hub sync — survive Kaggle session resets.

Kaggle GPU sessions die (9h cap, interrupts, "reset"). Without this, a reset loses
all training. With it, train.py pushes a FULL-STATE `last.pt` (model + head +
optimizer + scheduler + epoch + rng) to a private HF repo every epoch, and on
restart pulls it back and CONTINUES from the saved epoch instead of epoch 1.

Everything degrades gracefully: no `huggingface_hub`, no token, or no internet →
functions no-op / return None and training just runs locally (Kaggle "Save
Version" of /kaggle/working is then your only persistence). The repo's existing
weights live at HF `mdnt571/nom-embed`, so the same account/token works here.

Token: pass --hf-token, or set env HF_TOKEN, or on Kaggle add a Secret named
HF_TOKEN (Add-ons → Secrets) and `os.environ` picks it up.
"""
from __future__ import annotations

import os
from pathlib import Path


def _hub():
    try:
        import huggingface_hub  # noqa: F401
        return huggingface_hub
    except Exception:
        return None


def resolve_token(cli_token: str = "") -> str | None:
    if cli_token:
        return cli_token
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    # Kaggle Secrets (if the notebook enabled them)
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def ensure_repo(repo_id: str, token: str) -> bool:
    hub = _hub()
    if hub is None or not token or not repo_id:
        return False
    try:
        hub.create_repo(repo_id, token=token, private=True, exist_ok=True, repo_type="model")
        return True
    except Exception as e:
        print(f"[hf] create_repo skipped: {e}")
        return False


def push(local_path, repo_id: str, path_in_repo: str, token: str) -> bool:
    """Upload one file. True on success, False if HF unavailable/failed (non-fatal)."""
    hub = _hub()
    if hub is None or not token or not repo_id or not Path(local_path).exists():
        return False
    try:
        hub.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
                        repo_id=repo_id, token=token, repo_type="model")
        print(f"[hf] pushed {path_in_repo} -> {repo_id}")
        return True
    except Exception as e:
        print(f"[hf] push failed ({e}) — continuing local-only")
        return False


def pull(repo_id: str, filename: str, token: str, dest_dir) -> str | None:
    """Download a file from the repo → local path, or None if absent/unavailable."""
    hub = _hub()
    if hub is None or not repo_id:
        return None
    try:
        return hub.hf_hub_download(repo_id=repo_id, filename=filename, token=token or None,
                                   repo_type="model", local_dir=str(dest_dir))
    except Exception as e:
        print(f"[hf] no resume checkpoint on hub ({e})")
        return None
