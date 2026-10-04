"""v_pb_nguon_goc_01 (04/10) — PHẢN BIỆN hướng "nguồn gốc/tái lập p03-p05": chạy HÀM GỐC của p05 (import, không sao chép).

Mục đích (độc lập với v_nguon_goc_*):
  (1) --builtin : chạy `evaluate_single_book` GỐC với `hash(book)` nguyên bản — dùng dưới PYTHONHASHSEED=0 để xem bảng gốc
      (measure_out/_tn11/p05_full_corpus/ket_qua_full_10_bo.json) có tái lập được không khi hash chuỗi bị ghim như run_pipeline.sh.
  (2) --seeds   : chạy hàm GỐC với hạt giống KHÁC (vá `hash` ở mức mô-đun thành hằng) để đo nhiễu lấy mẫu; thu cả bản ghi từng ô
      (bắt DataFrame `df_b` mà hàm gốc dựng) để gộp ô duy nhất + McNemar.
Không sửa p05 (OUT_DIR của mô-đun được trỏ sang thư mục tạm của tôi trước khi gọi — tránh ghi đè montage_*.png gốc). CPU, 0 API.

    PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .../v_pb_nguon_goc_01_chay_goc.py --builtin --books stt2 ... --tag hs0
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .../v_pb_nguon_goc_01_chay_goc.py --seeds 10 11 12 --books B34 L16 --tag s10_19
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np  # noqa: E402
import pandas as real_pd  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
OUT.mkdir(parents=True, exist_ok=True)
P05 = HERE / "p05_full_corpus_he_quy_chieu.py"


def load_p05():
    spec = importlib.util.spec_from_file_location("p05_goc", str(P05))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _PDProxy:
    """Bọc pandas để bắt DataFrame cuối cùng (df_b = bản ghi từng ô) mà hàm gốc dựng; mọi thứ khác chuyển thẳng."""

    def __init__(self):
        self.captured = []

    def __getattr__(self, k):
        return getattr(real_pd, k)

    def DataFrame(self, *a, **k):  # noqa: N802
        df = real_pd.DataFrame(*a, **k)
        self.captured.append(df)
        return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="*", default=[])
    ap.add_argument("--pairs", nargs="*", default=[], help="book:seed ...  (chạy đúng các cặp này)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[])
    ap.add_argument("--builtin", action="store_true", help="dùng hash() nguyên bản (đặt PYTHONHASHSEED ở ngoài)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    M = load_p05()
    tmp = Path(tempfile.mkdtemp(prefix="pb_p05_"))
    M.OUT_DIR = tmp  # montage_*.png sẽ ghi vào đây, không vào thư mục p05 gốc
    proxy = _PDProxy()
    M.pd = proxy

    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    scorers = SI.Scorers(Assets(), "cpu", M.EMB_DIR, False, lambda m: None)
    encoder = scorers.enc()

    out_path = OUT / f"chay_goc_{a.tag}.json"
    res = {"tag": a.tag, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "device": "cpu", "runs": []}
    jobs = []
    for pr in a.pairs:
        bk, sd = pr.split(":")
        jobs.append((bk, int(sd)))
    if a.builtin:
        for b in a.books:
            jobs.append((b, None))
    else:
        for s in a.seeds:
            for b in a.books:
                jobs.append((b, s))
    for b, s in jobs:
        t0 = time.time()
        proxy.captured.clear()
        if s is not None:
            M.hash = (lambda _x, _s=s: _s)  # hash(book) % 100000 == s  (s < 100000)
        elif hasattr(M, "hash"):
            del M.hash
        seed_used = (s if s is not None else int(hash(b) % 100000))
        r = M.evaluate_single_book(b, encoder, n_sample=20, k_cand=3)
        df = proxy.captured[-1] if proxy.captured else None
        recs = df.to_dict("records") if df is not None else []
        res["runs"].append({"book": b, "seed": seed_used, "res": r, "cells": recs, "sec": round(time.time() - t0, 1)})
        out_path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {len(jobs)} lượt -> {out_path}")


if __name__ == "__main__":
    main()
