#!/usr/bin/env python3
"""Xóa 2 codepoint Private-Use Area rác trên HF mdnt571/gannhanocr.

Các codepoint này (U+F0009, U+F061A) nằm trong vùng Unicode PUA-A
(U+F0000-U+FFFFD) — không phải chu-Nom Unicode chuẩn, không có trong
char_universe.txt. Có thể là tàn dư từ run cũ với encoding tùy biến.
Vô hại nhưng xóa cho sạch repo.

Yêu cầu: đã `huggingface-cli login` với token Write scope,
hoặc set env HF_TOKEN.
"""
import os
from pathlib import Path
from huggingface_hub import HfApi

REPO = 'mdnt571/gannhanocr'
REPO_TYPE = 'dataset'
TARGETS = ['U+F0009.png', 'U+F061A.png']


def _load_token() -> str:
    """Đọc HF_TOKEN từ env hoặc .env file ở repo root."""
    tok = os.environ.get('HF_TOKEN', '').strip()
    if tok:
        return tok
    env_file = Path(__file__).resolve().parent.parent / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            if k.strip() == 'HF_TOKEN':
                # Strip optional quotes
                v = v.strip().strip('"').strip("'")
                return v
    return ''


def main() -> None:
    token = _load_token()
    if not token:
        raise SystemExit('✗ HF_TOKEN không tìm thấy (env hoặc .env). Cần token Write scope.')
    print(f'Token loaded (length={len(token)}, prefix={token[:4]}...)')
    api = HfApi(token=token)

    # Verify token: gọi whoami trước khi commit
    try:
        who = api.whoami(token=token)
        print(f'Authenticated as: {who.get("name")}  (type={who.get("type")})')
    except Exception as e:
        raise SystemExit(f'✗ whoami failed: {type(e).__name__}: {e}')

    expected_owner = REPO.split('/')[0]
    if who.get('name') != expected_owner:
        print(f'⚠️  Token thuộc "{who.get("name")}" nhưng repo "{REPO}" của "{expected_owner}".')
        print('    Có thể vẫn xóa được nếu token có quyền trên repo này, nếu không sẽ 401.')

    # Xác minh quyền + sự tồn tại
    files = set(api.list_repo_files(repo_id=REPO, repo_type=REPO_TYPE))
    to_delete = [f for f in TARGETS if f in files]
    not_found = [f for f in TARGETS if f not in files]

    print(f'Repo:    {REPO}')
    print(f'Targets: {TARGETS}')
    print(f'Found:   {to_delete}')
    if not_found:
        print(f'Not on HF (đã xóa hoặc chưa tồn tại): {not_found}')
    if not to_delete:
        print('→ Không có gì để xóa.')
        return

    for fname in to_delete:
        try:
            api.delete_file(
                path_in_repo=fname,
                repo_id=REPO,
                repo_type=REPO_TYPE,
                commit_message=f'Cleanup: remove PUA codepoint {fname[:-4]} (not in char_universe)',
            )
            print(f'  ✓ deleted {fname}')
        except Exception as e:
            print(f'  ✗ delete {fname}: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
