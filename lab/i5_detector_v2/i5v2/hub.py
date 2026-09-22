"""Hugging Face Hub sync (tuỳ chọn) — sống sót khi Kaggle reset; bản rút gọn của ArcFace/hub.py.

Không có huggingface_hub / token / internet -> mọi hàm no-op (trả False/None), huấn luyện vẫn chạy local.
Token: --hf-token > env HF_TOKEN > Kaggle Secret HF_TOKEN.
"""
from __future__ import annotations

import os
from pathlib import Path


def _hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except Exception:
        return None


def resolve_token(cli_token: str = "") -> str | None:
    if cli_token:
        return cli_token
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def ensure_repo(repo_id: str, token: str | None) -> bool:
    hub = _hub()
    if hub is None or not token or not repo_id:
        return False
    try:
        hub.create_repo(repo_id, token=token, private=True, exist_ok=True, repo_type="model")
        return True
    except Exception as e:
        print(f"[hf] create_repo bỏ qua: {e}", flush=True)
        return False


def push(local_path, repo_id: str, token: str | None, path_in_repo: str | None = None) -> bool:
    hub = _hub()
    if hub is None or not token or not repo_id or not Path(local_path).exists():
        return False
    try:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        hub.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo or Path(local_path).name,
                        repo_id=repo_id, token=token, repo_type="model")
        print(f"[hf] đẩy {Path(local_path).name} -> {repo_id}", flush=True)
        return True
    except Exception as e:
        print(f"[hf] đẩy lỗi ({e}) — tiếp tục local", flush=True)
        return False


def pull(repo_id: str, filename: str, token: str | None, dest_dir) -> str | None:
    hub = _hub()
    if hub is None or not repo_id:
        return None
    try:
        return hub.hf_hub_download(repo_id=repo_id, filename=filename, token=token or None,
                                   repo_type="model", local_dir=str(dest_dir))
    except Exception as e:
        print(f"[hf] không có {filename} trên hub ({type(e).__name__})", flush=True)
        return None
