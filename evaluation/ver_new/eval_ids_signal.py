"""TEST tín hiệu IDS (cấu trúc/bộ-thủ) cứu ĐUÔI HIẾM — KHÔNG train.

Ý tưởng: chữ hiếm (vd 明) ít mẫu nên encoder yếu; NHƯNG bộ thủ của nó (日, 月) là chữ
PHỔ BIẾN encoder thuộc rõ. Với chữ phân rã nhị phân (⿰AB trái-phải / ⿱AB trên-dưới),
ta TÁCH crop theo layout, match vùng-A với glyph-A, vùng-B với glyph-B (cosine). Hợp với
điểm encoder toàn-chữ → xem retrieval@1 trên đuôi hiếm cải thiện bao nhiêu.

Run: .venv/bin/python evaluation/ver_new/eval_ids_signal.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from pipeline.step0_setup import load_config                       # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402
from evaluation.ver_new.bbox_fix import tighten_box                # noqa: E402

HERE = Path(__file__).resolve().parent
D = HERE / "dataset_out"
FD = REPO / "gannhanocr-fd"
IDC2 = {"⿰", "⿱"}                                  # nhị phân: trái-phải / trên-dưới


def fd_path(ch):
    hx = f"{ord(ch):X}"
    for q in (FD / f"U+{hx}.png", FD / hx[:2] / f"U+{hx}.png"):
        if q.exists():
            return q
    return None


def load_ids():
    """char -> (IDC, A, B) nếu phân rã nhị phân đơn giản; bỏ tag nguồn [..]."""
    out = {}
    f = HERE / "ids_data" / "ids.txt"
    for ln in open(f, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        ch = p[1]
        ids = p[2].split("[")[0].strip()             # IDS đầu, bỏ [GTKV..]
        if len(ids) == 3 and ids[0] in IDC2 and ids[1] != ch and ids[2] != ch:
            out[ch] = (ids[0], ids[1], ids[2])
    return out


def cos(a, b):
    return float(np.dot(a, b))


def tight(g):
    t = tighten_box(g)
    if t is not None:
        a, c, b, d = t
        if b - a >= 6 and d - c >= 6:
            return g[c:d, a:b]
    return g if g.size and min(g.shape) >= 6 else None


def main():
    cfg = load_config(str(REPO / "config" / "pipeline.yaml"))
    qn = load_qn_to_nom(str(REPO / cfg["paths"]["qn_to_nom_dict"]))
    enc = VisualS3(REPO, fd_dir="").enc
    ids = load_ids()
    gemb = {}
    def glyph_emb(ch):
        if ch not in gemb:
            p = fd_path(ch); gemb[ch] = enc.embed_path(str(p)) if p else None
        return gemb[ch]

    rows = list(csv.DictReader(open(D / "labels.csv", encoding="utf-8")))
    gold = Counter(r["label"] for r in rows if r["tier"] == "GOLD" and r["label"])
    # đuôi hiếm: <5 GOLD crop; test crop của lớp hiếm
    test = [r for r in rows if r["tier"] == "GOLD" and r["split"] == "test"
            and r["image"] and r["label"] and _is_cjk(r["label"]) and gold.get(r["label"], 0) < 5]
    print(f"crop test thuộc lớp hiếm(<5): {len(test)}", flush=True)

    def comp_score(crop_gray, ch):
        """điểm compositional cho ứng viên ch (nếu phân rã nhị phân + có glyph 2 bộ thủ)."""
        if ch not in ids:
            return None
        idc, A, B = ids[ch]
        eA, eB = glyph_emb(A), glyph_emb(B)
        if eA is None or eB is None:
            return None
        h, w = crop_gray.shape
        best = -1.0
        for r in (0.4, 0.5, 0.6):
            if idc == "⿰":
                cut = int(w * r); rA, rB = crop_gray[:, :cut], crop_gray[:, cut:]
            else:
                cut = int(h * r); rA, rB = crop_gray[:cut, :], crop_gray[cut:, :]
            rA, rB = tight(rA), tight(rB)
            if rA is None or rB is None:
                continue
            sA = cos(enc.embed_gray(rA), eA); sB = cos(enc.embed_gray(rB), eB)
            best = max(best, min(sA, sB))            # cả 2 bộ thủ phải khớp
        return best if best > -1 else None

    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    hit = {a: 0 for a in alphas}; n = 0; n_ids_cand = 0; n_decomp = 0
    for r in test:
        true = r["label"]; syl = (r["syllable"] or "").lower()
        R = [c for c in qn.get(syl, []) if _is_cjk(c) and fd_path(c)]
        if true not in R:
            R = R + [true]
        if len(R) < 2:
            continue
        g = cv2.imread(str(D / r["image"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        gt = tight(g)
        if gt is None:
            continue
        emb = enc.embed_gray(gt)
        enc_s = {c: cos(emb, glyph_emb(c)) for c in R}
        ids_s = {c: comp_score(gt, c) for c in R}
        if any(v is not None for v in ids_s.values()):
            n_ids_cand += 1
        if ids_s.get(true) is not None:
            n_decomp += 1
        n += 1
        for a in alphas:
            fused = {c: enc_s[c] + (a * ids_s[c] if ids_s[c] is not None else 0.0) for c in R}
            if max(fused, key=fused.get) == true:
                hit[a] += 1

    print("\n" + "=" * 60)
    print(f" IDS compositional rescue trên ĐUÔI HIẾM (n={n} crop, ứng viên ≥2)")
    print("=" * 60)
    print(f"  crop có ≥1 ứng viên phân-rã-được: {n_ids_cand} | chữ-đúng phân-rã-được: {n_decomp}")
    print(f"  {'α (trọng số IDS)':>18} | retrieval@1")
    for a in alphas:
        tag = " (encoder thuần)" if a == 0 else ""
        print(f"  {a:>18.1f} | {hit[a]/max(n,1):.3f}{tag}")
    base = hit[0.0]/max(n,1); best_a = max(alphas, key=lambda a: hit[a]); best = hit[best_a]/max(n,1)
    print(f"\n  >>> encoder thuần {base:.3f} -> +IDS (α={best_a}) {best:.3f}  (Δ {best-base:+.3f})")
    print(f"      cứu thêm ~{hit[best_a]-hit[0.0]} / {n} crop đuôi hiếm")

    out = {"n": n, "n_decomposable_cand": n_ids_cand, "n_true_decomposable": n_decomp,
           "retrieval_by_alpha": {str(a): hit[a]/max(n,1) for a in alphas},
           "best_alpha": best_a, "gain": best - base}
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results" / "eval_ids_signal.json", "w"), indent=2)
    print(f"  -> results/eval_ids_signal.json")


if __name__ == "__main__":
    main()
