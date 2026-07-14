"""Qwen (DashScope) client with multi-key round-robin + failover.

Free-tier keys have small per-minute / per-day quotas, so we spread calls across every
key in .env and fail over when one is throttled.

Keys are read from .env by name: any `Qwen3-VL<N>` (Qwen3-VL1, Qwen3-VL2, ...) in
numeric order, plus the legacy `Qwen3-VL-Flash` / `DASHSCOPE_API_KEY` if present.

Strategy:
  - round-robin: each successful call advances to the next key (even quota depletion);
  - 429 (rate/quota): rotate to the next key and retry; if all keys 429 in one cycle,
    exponential backoff, then keep trying up to the attempt budget;
  - 401/403 (bad/blocked key): disable that key for the run and rotate;
  - 5xx / timeout: exponential backoff, same key;
  - other 4xx: fail loud.

The HTTP opener is injectable so the rotation logic is unit-tested without network.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

__all__ = ["DEFAULT_URL", "read_env_file", "load_keys", "KeyRotator", "post"]

DEFAULT_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"


def read_env_file(env_path) -> dict:
    """Parse a .env file into {name: value} (quotes stripped, comments/blank ignored)."""
    out: dict[str, str] = {}
    p = Path(env_path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip().strip("'").strip('"')
    return out


def load_keys(env_path, names: list[str] | None = None) -> list[str]:
    """Ordered, de-duplicated list of non-empty Qwen keys from .env.

    If `names` is None, auto-detect `Qwen3-VL<N>` (numeric order) + legacy names, so
    adding a `Qwen3-VL3=...` later needs no code change.
    """
    env = read_env_file(env_path)
    if names is None:
        # sort by the number AFTER "VL" (not the "3" in "Qwen3")
        numbered = sorted((k for k in env if re.fullmatch(r"Qwen3-VL\d+", k)),
                          key=lambda k: int(re.search(r"VL(\d+)$", k).group(1)))
        legacy = [k for k in ("Qwen3-VL-Flash", "DASHSCOPE_API_KEY") if k in env]
        names = numbered + legacy
    keys, seen = [], set()
    for n in names:
        v = (env.get(n) or "").strip()
        if v and v not in seen:
            seen.add(v)
            keys.append(v)
    return keys


class KeyRotator:
    """Cycles through keys; tracks keys disabled (401/403) for the run."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("no Qwen keys found in .env (add Qwen3-VL1=..., Qwen3-VL2=...)")
        self.keys = list(keys)
        self.i = 0
        self.dead: set[str] = set()

    def __len__(self) -> int:
        return len(self.keys)

    def current(self) -> str:
        return self.keys[self.i]

    def label(self) -> str:
        return f"key#{self.i + 1}/{len(self.keys)}"

    def advance(self) -> None:
        self.i = (self.i + 1) % len(self.keys)

    def mark_dead(self, key: str) -> None:
        self.dead.add(key)

    def next_alive(self) -> str | None:
        """Advance to the next non-dead key; None if all keys are dead."""
        for _ in range(len(self.keys)):
            self.advance()
            if self.current() not in self.dead:
                return self.current()
        return None


def post(body_dict: dict, rotator: KeyRotator, url: str = DEFAULT_URL, timeout: int = 120,
         max_attempts: int | None = None, sleep=time.sleep, opener=None,
         log=lambda m: None) -> dict:
    """POST a chat-completions body with key rotation + failover. Returns parsed JSON.

    `opener(req, timeout) -> file-like` defaults to urllib; inject for tests.
    Raises RuntimeError when all keys are exhausted or the attempt budget is spent.
    """
    if opener is None:
        def opener(req, timeout):
            return urllib.request.urlopen(req, timeout=timeout)
    body = json.dumps(body_dict).encode()
    budget = max_attempts if max_attempts is not None else 4 * len(rotator)
    attempts = 0
    cycle_429 = 0
    while attempts < budget:
        if rotator.current() in rotator.dead:
            if rotator.next_alive() is None:
                raise RuntimeError("all Qwen keys disabled (401/403)")
        key = rotator.current()
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with opener(req, timeout) as r:
                data = json.loads(r.read())
            rotator.advance()                       # round-robin: spread the next call
            return data
        except urllib.error.HTTPError as e:
            attempts += 1
            code = e.code
            if code in (401, 403):
                log(f"{rotator.label()} auth error {code} -> disabling this key")
                rotator.mark_dead(key)
                if rotator.next_alive() is None:
                    raise RuntimeError("all Qwen keys invalid (401/403)") from e
                continue
            if code == 429:
                cycle_429 += 1
                log(f"{rotator.label()} rate/quota (429) -> rotating")
                if rotator.next_alive() is None:
                    raise RuntimeError("all Qwen keys rate-limited (429)") from e
                if cycle_429 % len(rotator) == 0:   # cycled all keys and all 429
                    sleep(min(60, 2 ** (cycle_429 // len(rotator))))
                continue
            if 500 <= code < 600:
                sleep(min(30, 2 ** attempts))
                continue
            raise                                   # other 4xx: caller's fault
        except (urllib.error.URLError, TimeoutError) as e:
            attempts += 1
            log(f"{rotator.label()} network error ({e}) -> backoff")
            sleep(min(30, 2 ** attempts))
            continue
    raise RuntimeError(f"Qwen POST failed after {attempts} attempts (budget {budget})")


# --------------------------------------------------------------------------- #
# Live key tester:  python evaluation/tri_consensus/qwen_client.py [model]
# --------------------------------------------------------------------------- #
def _ping(key: str, model: str, timeout: int = 30):
    """One minimal live call with a single key. Returns (status, info)."""
    body = json.dumps({"model": model, "max_tokens": 1,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    req = urllib.request.Request(DEFAULT_URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return "ok", data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except urllib.error.HTTPError as e:
        return "http", e.code
    except Exception as e:  # noqa: BLE001 — surface any network/DNS error verbatim
        return type(e).__name__, str(e)


def _main(argv=None):
    import sys
    from pathlib import Path
    model = (argv or sys.argv[1:] or ["qwen3-vl-flash"])[0]
    env = Path(__file__).resolve().parents[2] / ".env"
    keys = load_keys(env)
    print(f"[qwen-test] {len(keys)} key(s) from {env}  (model={model})")
    for i, k in enumerate(keys):
        print(f"  key#{i + 1}: {k[:6]}…{k[-4:]}")

    print("\n-- per-key live ping (1 token each) --")
    alive = 0
    for i, k in enumerate(keys):
        status, info = _ping(k, model)
        if status == "ok":
            alive += 1
            print(f"  key#{i + 1}: ✅ OK   reply={info!r}")
        elif status == "http":
            tag = {401: "auth", 403: "auth", 429: "rate/quota"}.get(info, "")
            print(f"  key#{i + 1}: ❌ HTTP {info} {tag}")
        else:
            print(f"  key#{i + 1}: ⚠️  {status}: {info}")
    print(f"  => {alive}/{len(keys)} key hợp lệ")

    print(f"\n-- round-robin thật qua {len(keys) + 1} call (phải xoay vòng key) --")
    rot = KeyRotator(keys)
    body = {"model": model, "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]}
    order = []
    for i in range(len(keys) + 1):
        used = rot.i
        try:
            post(body, rot, max_attempts=2, sleep=lambda s: None)
            order.append(used + 1)
            print(f"  call {i + 1}: dùng key#{used + 1} -> OK, kế tiếp key#{rot.i + 1}")
        except Exception as e:  # noqa: BLE001
            print(f"  call {i + 1}: dùng key#{used + 1} -> {e}")
    distinct = len(set(order[:len(keys)]))
    print(f"  => {len(keys)} call đầu dùng {distinct} key khác nhau "
          f"({'XOAY TUA OK' if distinct == len(keys) else 'chưa xoay đủ (có key lỗi?)'})")
    return 0 if alive == len(keys) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
