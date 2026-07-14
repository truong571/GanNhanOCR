"""Self-test for the Qwen multi-key rotation client (no network).

    .venv/bin/python evaluation/tri_consensus/qwen_client_selftest.py
Mocks the HTTP opener to drive success / 429 / 401 / all-exhausted scenarios and checks
which key each request used. Exit 0 = all pass.
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_client as qc

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}  {detail}")


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def read(self):
        return json.dumps(self._p).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _key_of(req):
    return req.get_header("Authorization").split("Bearer ", 1)[1]


def make_opener(handler, seen):
    def opener(req, timeout):
        k = _key_of(req)
        seen.append(k)
        return handler(k, len(seen))          # returns _FakeResp or raises HTTPError
    return opener


def http429(*a):
    raise urllib.error.HTTPError("http://x", 429, "rate", {}, None)


def http401(*a):
    raise urllib.error.HTTPError("http://x", 401, "unauth", {}, None)


def test_load_keys():
    print("[load_keys]")
    with tempfile.TemporaryDirectory() as td:
        env = Path(td) / ".env"
        env.write_text(
            "# comment\n"
            "SN_OCR_TOKEN=abc\n"
            "Qwen3-VL2='K2'\n"
            "Qwen3-VL1=\"K1\"\n"
            "Qwen3-VL10=K10\n"
            "EMPTY=\n"
            "Qwen3-VL-Flash=KLEG\n")
        keys = qc.load_keys(env)
        check("numeric order 1,2,10 then legacy", keys == ["K1", "K2", "K10", "KLEG"], str(keys))
        check("explicit names respected",
              qc.load_keys(env, ["Qwen3-VL1", "Qwen3-VL2"]) == ["K1", "K2"])
        env2 = Path(td) / ".env2"
        env2.write_text("Qwen3-VL1=DUP\nQwen3-VL2=DUP\n")
        check("dedup identical keys", qc.load_keys(env2) == ["DUP"], str(qc.load_keys(env2)))


def test_rotator():
    print("[KeyRotator]")
    r = qc.KeyRotator(["a", "b", "c"])
    check("len", len(r) == 3)
    check("current a", r.current() == "a")
    r.advance(); check("advance -> b", r.current() == "b")
    r.advance(); r.advance(); check("wraps to a", r.current() == "a")
    r.mark_dead("b")
    r.i = 0
    check("next_alive skips dead b -> c", r.next_alive() == "c", r.current())
    r2 = qc.KeyRotator(["x"]); r2.mark_dead("x")
    check("all dead -> next_alive None", r2.next_alive() is None)
    try:
        qc.KeyRotator([])
        check("empty raises", False)
    except ValueError:
        check("empty raises", True)


def test_post():
    print("[post — rotation + failover]")
    # round-robin: two successful calls use k1 then k2
    r = qc.KeyRotator(["k1", "k2"])
    seen = []
    op = make_opener(lambda k, n: _FakeResp({"ok": k}), seen)
    d1 = qc.post({"m": 1}, r, opener=op, sleep=lambda s: None)
    d2 = qc.post({"m": 1}, r, opener=op, sleep=lambda s: None)
    check("round-robin uses k1 then k2", seen == ["k1", "k2"], str(seen))
    check("returns parsed json", d1 == {"ok": "k1"} and d2 == {"ok": "k2"})

    # 429 on k1 -> failover to k2 succeeds
    r = qc.KeyRotator(["k1", "k2"]); seen = []
    op = make_opener(lambda k, n: http429() if k == "k1" else _FakeResp({"ok": k}), seen)
    d = qc.post({"m": 1}, r, opener=op, sleep=lambda s: None)
    check("429 fails over to k2", d == {"ok": "k2"}, str(d))
    check("k1 tried before k2", seen[:2] == ["k1", "k2"], str(seen))

    # 401 on k1 -> disable k1, use k2
    r = qc.KeyRotator(["k1", "k2"]); seen = []
    op = make_opener(lambda k, n: http401() if k == "k1" else _FakeResp({"ok": k}), seen)
    d = qc.post({"m": 1}, r, opener=op, sleep=lambda s: None)
    check("401 disables k1, uses k2", d == {"ok": "k2"} and "k1" in r.dead)

    # all keys 429 -> RuntimeError after budget
    r = qc.KeyRotator(["k1", "k2"]); seen = []
    op = make_opener(lambda k, n: http429(), seen)
    try:
        qc.post({"m": 1}, r, opener=op, sleep=lambda s: None, max_attempts=6)
        check("all 429 -> raises", False)
    except RuntimeError:
        check("all 429 -> raises", True)

    # both keys dead (401) -> RuntimeError
    r = qc.KeyRotator(["k1", "k2"]); seen = []
    op = make_opener(lambda k, n: http401(), seen)
    try:
        qc.post({"m": 1}, r, opener=op, sleep=lambda s: None)
        check("all 401 -> raises", False)
    except RuntimeError:
        check("all 401 -> raises", True)

    # 5xx then success on same key (backoff, no rotation)
    r = qc.KeyRotator(["k1", "k2"]); seen = []
    state = {"n": 0}

    def h5xx(k, n):
        state["n"] += 1
        if state["n"] == 1:
            raise urllib.error.HTTPError("http://x", 503, "busy", {}, None)
        return _FakeResp({"ok": k})
    op = make_opener(h5xx, seen)
    d = qc.post({"m": 1}, r, opener=op, sleep=lambda s: None)
    check("5xx retried then success", d == {"ok": "k1"}, str(d))
    check("5xx retried same key", seen == ["k1", "k1"], str(seen))


def main() -> int:
    print("=" * 60)
    print("QWEN-CLIENT SELFTEST")
    print("=" * 60)
    test_load_keys()
    test_rotator()
    test_post()
    print("=" * 60)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 60)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
