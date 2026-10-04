"""Rà soát gói Kaggle TN11 (hướng chay_thu_dau_cuoi) — tiện ích dùng chung cho các script r_chay_thu_dau_cuoi_*.py.

Ràng buộc: CHỈ ĐỌC gói/bộ chạy; mọi đầu ra nằm dưới measure_out/_tn11/full/review/chay_thu_dau_cuoi/ (R). Không sinh __pycache__ ở lab/
(sys.dont_write_bytecode + PYTHONDONTWRITEBYTECODE cho tiến trình con); mọi tiến trình con chạy với cwd dưới R để các tệp .log của
font_diffusion (FileHandler 'inference.sample_optimized.log') không rơi vào repo.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import importlib.util
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FULL = REPO / "measure_out" / "_tn11" / "full"
PACK = FULL / "kaggle_pack_20261004"
RUNNER = PACK / "w_runner_khung_chay.py"          # đúng bản sẽ chạy trên Kaggle (cmp == bản ở lab/, kiểm ở script 00)
R = FULL / "review" / "chay_thu_dau_cuoi"
R.mkdir(parents=True, exist_ok=True)


def env(extra: dict | None = None) -> dict:
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HOME=str(R / "hf_home"), HF_HUB_DISABLE_TELEMETRY="1",
             HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME"):
        e.pop(k, None)                               # không bao giờ dùng/đưa thông tin xác thực của người dùng
    if extra:
        e.update(extra)
    return e


def load_runner(name: str = "pack_runner"):
    """Nạp w_runner_khung_chay.py của GÓI làm mô-đun (không ghi .pyc)."""
    spec = importlib.util.spec_from_file_location(name, str(RUNNER))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def sh(args: list, cwd: Path, extra_env: dict | None = None, timeout: float | None = None) -> tuple:
    """Chạy bộ chạy của gói như người dùng gõ lệnh. Trả (mã thoát, stdout+stderr, giây)."""
    cwd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, str(RUNNER), *map(str, args)], cwd=cwd, env=env(extra_env), capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr, round(time.time() - t0, 2)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def shard_hashes(out: Path) -> dict:
    base = out / "shards"
    return {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(base.rglob("*.tar"))} if base.exists() else {}


def read_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def task_rows(out: Path) -> list:
    rows = []
    for f in sorted((out / "status").glob("tasks-*.jsonl")):
        rows += read_jsonl(f)
    return rows


def status_of(out: Path) -> dict:
    return json.loads((out / "status" / "status.json").read_text(encoding="utf-8"))


def leftover(out: Path) -> dict:
    return dict(locks=len(list((out / "locks").glob("*.lock"))) if (out / "locks").exists() else 0,
                stale_locks=len([p for p in (out / "locks").glob("*") if ".stale-" in p.name]) if (out / "locks").exists() else 0,
                parts=len(list((out / "shards").rglob("*.part"))) if (out / "shards").exists() else 0)


class Rep:
    """Bộ ghi bất biến (PASS/FAIL) + dữ liệu thô, ghi ra JSON."""

    def __init__(self, name: str):
        self.name = name
        self.d = dict(tao=time.strftime("%Y-%m-%d %H:%M:%S"), invariants={}, du_lieu={})

    def inv(self, key: str, ok: bool, detail=None):
        self.d["invariants"][key] = dict(ok=bool(ok), detail=detail)
        print(("PASS" if ok else "FAIL"), key, "" if detail is None else json.dumps(detail, ensure_ascii=False, default=str)[:420], flush=True)

    def data(self, key: str, val):
        self.d["du_lieu"][key] = val

    def save(self):
        v = self.d["invariants"]
        self.d["tong"] = dict(pass_=sum(x["ok"] for x in v.values()), fail=sum(not x["ok"] for x in v.values()), n=len(v))
        (R / f"{self.name}.json").write_text(json.dumps(self.d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print("TỔNG", self.name, self.d["tong"], flush=True)


def tar_header_report(p: Path) -> dict:
    """Mọi thành viên: mtime=0, uid=gid=0, uname/gname rỗng, mode 0o644, loại thường; _meta.json là thành viên cuối."""
    bad = []
    with tarfile.open(p) as tf:
        ms = tf.getmembers()
        fmt = tf.format
        for m in ms:
            if not (m.mtime == 0 and m.uid == 0 and m.gid == 0 and m.uname == "" and m.gname == "" and m.mode == 0o644 and m.isreg() and not m.pax_headers):
                bad.append(m.name)
        return dict(n=len(ms), last=ms[-1].name, format=fmt, bad=bad[:3])
