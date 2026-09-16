"""BENCHMARK GÁC B-2 — rụng-1-chữ trên cột m = n với mô hình OOF THẬT (KhoiB/v3, Khối B, K4).

Chạy lại benchmark của lab/tham_dinh_2026-09-16/dp_vis5.py nhưng bằng ĐÚNG đường mã sản xuất:
  pipeline.align_engine.visual_emission.VisualEmitter (fold theo trang, hộp OCR thô, cut() == lab)
  + build_dataset.anchored_cost_fn (neo ngữ liệu LOO, ANCHOR_CAP của engine = 2,0 — dp_vis5 dùng cap 0)
  + anchor_align.realign_column(cost_ij=…, cost_del=…, cost_ins=…) (cùng DP như PASS 1b khi --visual-emission).

Kịch bản (thuc_nghiem.perturb, seed 2026 như dp_vis5): drop_char (OCR rụng 1 chữ), drop_syl (QN rụng 1 âm),
drop2_char (OCR rụng 2 chữ liền; khe đúng = ins đúng [k, k+1]; K5), none.
Số đo: ghép sai = cặp (i, j) khác đáp án / tổng cặp; khe đúng = cột mà khe ins/del nằm đúng chỗ (none: không khe).
Cấu hình: văn bản thuần (CALIB), +corpus (neo LOO cap 2,0), +corpus+ảnh λ ∈ {0,25 (mặc định), 0,5, 1,0}, +ảnh không corpus.

ĐIỀU KIỆN CHẤP NHẬN (K4): ở 'none' +ảnh λ=0,25 KHÔNG tệ hơn +corpus (ghép sai ≤, khe đúng ≥) VÀ ở drop_char
TỐT HƠN (ghép sai <, khe đúng >). In PASS/FAIL, ghi KhoiB/v3/bench_b2_results.json.

Tập cột: (1) `--set valtest` (mặc định) = trang val+test theo hash split cũ (như dp_vis5: 523 cột, so được với số cũ);
(2) `--set all` = mọi cột m = n ≥ 10 (mọi trang đều out-of-fold với mô hình 5 fold).

    .venv/bin/python KhoiB/v3/bench_b2.py                     # ≈ 2–4 phút (MPS)
    .venv/bin/python KhoiB/v3/bench_b2.py --set all --lams 0.25
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "lab/gan_nhan_2026-09-13"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(LAB))

from pipeline.align_engine import anchor_align as aa                      # noqa: E402
from pipeline.align_engine.build_dataset import anchored_cost_fn           # noqa: E402
from pipeline.align_engine.visual_emission import VisualEmitter, load_page_gray, page_fold   # noqa: E402
from core.text.dictionary import load_qn_to_nom, load_similarity_dict     # noqa: E402

BOOKDIR = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "stt11": "SachThanhTruyen11"}
KINDS = ("drop_char", "drop_syl", "drop2_char", "none")   # K5: thêm drop2_char (OCR rụng 2 chữ liền)


def split_of(book, page):
    hv = int(hashlib.md5(f"{book}|{page}".encode()).hexdigest(), 16) % 100
    return "train" if hv < 80 else ("val" if hv < 90 else "test")


def ocr_raw_boxes(need_pages):
    """{(book, page, column): [bbox theo thứ tự chữ]} từ *_ocr_cache.json (nom_cols_hybrid như cols.pkl/dp_vis5)."""
    from core.align.run_full import nom_cols_hybrid
    from pipeline.step2_align import _get_qn_lines
    import thuc_nghiem as TN
    out = {}
    for code, book in BOOKDIR.items():
        dd = REPO / "prepared" / book
        for (b, page) in sorted(need_pages):
            if b != code:
                continue
            od = json.load(open(dd / "detected" / f"{page}_ocr_cache.json", encoding="utf-8"))
            lines, _ = _get_qn_lines(dd, page, TN.qs); keys = sorted(lines)
            cs = nom_cols_hybrid(od["columns"], min_len=4)
            if len(cs) != 9:
                cs = nom_cols_hybrid(od["columns"], min_len=1)
            for i in range(min(len(cs), len(keys))):
                out[(code, page, keys[i])] = [ch["bbox"] for ch in cs[i]["chars"]]
    return out


def dp(chars, syls, text_cost, em, logP):
    """Một lần DP như PASS 1b (+ảnh nếu logP != None). Trả (pairs, dels, inss)."""
    kw = {}
    if logP is not None:
        cd, ci = em.gap_costs()
        kw = {"cost_ij": em.make_cost_ij(text_cost, logP), "cost_del": cd, "cost_ins": ci}
    ops = aa.realign_column(chars, syls, {}, None, cost_fn=text_cost, **kw)
    pairs = [(o["nom_idx"], o["syl_idx"]) for o in ops if o["op"] == "match"]
    dels = [o["nom_idx"] for o in ops if o["op"] == "del"]
    inss = [o["syl_idx"] for o in ops if o["op"] == "ins"]
    return pairs, dels, inss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="valtest", choices=["valtest", "all"])
    ap.add_argument("--lams", default="0.25,0.5,1.0", help="λ cho +corpus+ảnh (λ đầu = cấu hình gác)")
    ap.add_argument("--models", default=None, help="thư mục fold0..4.pt (mặc định tìm KhoiB/v3/…)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(REPO / "KhoiB/v3/bench_b2_results.json"))
    ap.add_argument("--min-len", type=int, default=10)
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]

    em = VisualEmitter(models_dir=args.models)
    if not em.available:
        print(f"[bench_b2] KHÔNG chạy được: {em.reason}")
        json.dump({"available": False, "reason": em.reason}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return 0
    print(f"[bench_b2] {em.describe()} | device {em.dev}", flush=True)
    import thuc_nghiem as TN
    qn = load_qn_to_nom(str(REPO / "dict/QuocNgu_SinoNom.csv")); sim = load_similarity_dict(str(REPO / "dict/SinoNom_Similar.csv"))
    TN.set_matrix(TN.CALIB)
    cols = TN.load_cols(); C = TN.Corpus(cols)
    eq = [c for c in cols if len(c["chars"]) == len(c["syl"]) >= args.min_len and all(o["op"] == "match" for o in c["ops"])]
    if args.set == "valtest":
        eq = [c for c in eq if split_of(c["book"], c["page"]) in ("val", "test")]
    print(f"  cột m=n≥{args.min_len} toàn match: {len(eq)} ({args.set})", flush=True)

    # hộp OCR thô + LP một lần / cột (mô hình fold của trang: out-of-fold thật)
    t0 = time.time()
    need = {(c["book"], c["page"]) for c in eq}
    boxes = ocr_raw_boxes(need)
    LPc, miss, gray_cache = {}, 0, {}
    n_fold = [0] * 5
    for c in eq:
        key = (c["book"], c["page"], c["column"])
        bbs = boxes.get(key)
        if bbs is None or len(bbs) != len(c["chars"]):
            miss += 1; continue
        pk = (c["book"], c["page"])
        if pk not in gray_cache:
            gray_cache.clear()
            gray_cache[pk] = load_page_gray(REPO / "prepared" / BOOKDIR[c["book"]] / "pages" / f"{c['page']}.png")
        LP, classes, cid, fold = em.logprobs(gray_cache[pk], bbs, c["book"], c["page"])
        LPc[key] = (LP, cid); n_fold[fold] += 1
    eq = [c for c in eq if (c["book"], c["page"], c["column"]) in LPc]
    print(f"  có hộp OCR khớp số chữ: {len(eq)} · lệch {miss} · hộp {em.n_boxes:,} (không cắt được {em.n_cut_fail}) "
          f"· cột theo fold {n_fold} · {time.time()-t0:.0f}s", flush=True)
    # top-1 âm trên hộp OCR thô (đối chứng với dp_vis5: crop hộp OCR thô 0,7xx)
    t1 = n1 = 0
    for c in eq:
        LP, cid = LPc[(c["book"], c["page"], c["column"])]
        for i, s in enumerate(c["syl"]):
            if s.lower() in cid and not np.isnan(LP[i, 0]):
                n1 += 1; t1 += int(LP[i].argmax()) == cid[s.lower()]
    print(f"  top-1 âm trên hộp OCR thô (OOF): {t1}/{n1} = {t1/max(1,n1):.3f}", flush=True)

    all_chars = [ch for col in cols for ch in col["chars"] if ch]
    configs = [("văn bản thuần (CALIB)", 0.0, False), ("+corpus (neo LOO cap 2,0)", 0.0, True)]
    configs += [(f"+corpus+ảnh λ={lam}", lam, True) for lam in lams]
    configs += [(f"+ảnh λ={lams[0]} không corpus", lams[0], False)]

    deviations = {}   # (kind, cấu hình) -> [cột lệch ở 'none'] để người xem (đáp án đường chéo có thể sai: QN dính token)

    def run(kind, lam, corpus, name=""):
        rng = random.Random(args.seed)
        n = wrong = gap_ok = ncol = 0
        emx = VisualEmitter(models_dir=em.models_dir, device=em.dev, lam=lam) if lam > 0 else None
        for col in eq:
            chars, syls = col["chars"], col["syl"]; m = len(chars)
            key = (col["book"], col["page"], col["column"]); LP, cid = LPc[key]
            st = rng.getstate(); k = rng.randrange(2, m - 2); rng.setstate(st)
            c2, s2, truth = TN.perturb(chars, syls, kind, rng, all_chars)
            # hàng LP của từng chữ còn lại (m=n nên truth[i] == chỉ số chữ gốc khi rụng chữ)
            orig = [truth[i] for i in range(len(c2))] if kind in ("drop_char", "drop2_char") else list(range(len(c2)))
            here = (col["book"], col["page"])
            if corpus:
                text_cost = anchored_cost_fn(C.pair_pages, here, qn, sim)     # ANCHOR_CAP engine (2,0)
            else:
                text_cost = lambda c, s: aa.substitution_cost(c, s, qn, sim)  # noqa: E731
            logP = emx.emission_from_LP(LP, cid, s2, rows=orig) if emx is not None else None
            pairs, dels, inss = dp(c2, s2, text_cost, emx, logP)
            ncol += 1
            nw = sum(truth.get(i) != j for (i, j) in pairs)
            n += len(pairs); wrong += nw
            if kind == "none" and (nw or dels or inss):
                LPf = LP
                deviations.setdefault(name, []).append({
                    "col": list(key), "m": m, "dels": dels, "inss": inss, "pairs_off": [(i, j) for i, j in pairs if i != j],
                    "chars": chars, "syl": syls,
                    "argmax": [(list(cid)[int(LPf[i].argmax())] if not np.isnan(LPf[i, 0]) else "") for i in range(m)],
                    "p_argmax": [round(float(np.exp(LPf[i].max())), 2) if not np.isnan(LPf[i, 0]) else None for i in range(m)]})
            if kind == "drop_char":
                gap_ok += (inss == [k]) and not dels
            elif kind == "drop_syl":
                gap_ok += (dels == [k]) and not inss
            elif kind == "drop2_char":
                gap_ok += (sorted(inss) == [k, k + 1]) and not dels
            else:
                gap_ok += (not dels and not inss)
        return {"n_pairs": n, "wrong": wrong, "wrong_rate": wrong / max(1, n), "gap_ok": gap_ok, "n_cols": ncol,
                "gap_ok_rate": gap_ok / max(1, ncol)}

    res = {"set": args.set, "n_cols": len(eq), "seed": args.seed, "anchor_cap": aa.ANCHOR_CAP, "emission_sec": round(time.time() - t0, 1),
           "models_dir": str(em.models_dir), "lambda_gate": lams[0], "cap": em.cap, "gap_vis": em.gap_vis,
           "top1_ocr_raw_box": {"hit": t1, "n": n1}, "fold_cols": n_fold, "by_kind": {}}
    t0 = time.time()
    for kind in KINDS:
        res["by_kind"][kind] = {}
        parts = []
        for name, lam, corpus in configs:
            tc = time.time()
            r = run(kind, lam, corpus, name)
            r["sec"] = round(time.time() - tc, 2)
            res["by_kind"][kind][name] = r
            parts.append(f"{name}: sai {r['wrong']}/{r['n_pairs']} ({r['wrong_rate']:.2%}) khe {r['gap_ok_rate']:.1%} {r['sec']:.1f}s")
        print(f"  {kind:10s} | " + " | ".join(parts), flush=True)
    print(f"  DP {len(KINDS) * len(configs)} lượt × {len(eq)} cột: {time.time()-t0:.0f}s")

    # điều kiện chấp nhận
    base = "+corpus (neo LOO cap 2,0)"; gate = f"+corpus+ảnh λ={lams[0]}"
    nn, dc = res["by_kind"]["none"], res["by_kind"]["drop_char"]
    ok_none = nn[gate]["wrong"] <= nn[base]["wrong"] and nn[gate]["gap_ok"] >= nn[base]["gap_ok"]
    ok_drop = dc[gate]["wrong"] < dc[base]["wrong"] and dc[gate]["gap_ok"] > dc[base]["gap_ok"]
    res["acceptance"] = {"none_not_worse": ok_none, "drop_char_better": ok_drop, "PASS": bool(ok_none and ok_drop),
                         "gate_config": gate, "baseline": base,
                         "none": {"base": [nn[base]["wrong"], nn[base]["gap_ok"]], "gate": [nn[gate]["wrong"], nn[gate]["gap_ok"]]},
                         "drop_char": {"base": [dc[base]["wrong"], dc[base]["gap_ok"]], "gate": [dc[gate]["wrong"], dc[gate]["gap_ok"]]}}
    print(f"  CHẤP NHẬN B-2 ({gate} so {base}): none không tệ hơn = {ok_none} (sai {nn[base]['wrong']}→{nn[gate]['wrong']}, "
          f"khe {nn[base]['gap_ok']}→{nn[gate]['gap_ok']}/{len(eq)}) · drop_char tốt hơn = {ok_drop} (sai {dc[base]['wrong']}→"
          f"{dc[gate]['wrong']}, khe {dc[base]['gap_ok']}→{dc[gate]['gap_ok']}/{len(eq)}) -> {'PASS' if ok_none and ok_drop else 'FAIL'}")
    d2 = res["by_kind"]["drop2_char"]
    res["acceptance"]["drop2_char"] = {"base": [d2[base]["wrong"], d2[base]["gap_ok"]], "gate": [d2[gate]["wrong"], d2[gate]["gap_ok"]],
                                       "better": bool(d2[gate]["wrong"] < d2[base]["wrong"] and d2[gate]["gap_ok"] > d2[base]["gap_ok"])}
    res["none_deviations"] = deviations
    dv = deviations.get(gate, [])
    if dv:
        print(f"  CỘT LỆCH Ở 'none' với {gate} ({len(dv)}): đáp án = đường chéo m=n, có thể SAI nếu QN dính/tách token —"
              " xem argmax ảnh từng hộp:")
        for d in dv[:10]:
            print(f"    {d['col']} dels {d['dels']} inss {d['inss']} off {d['pairs_off']}")
            for i in range(d["m"]):
                mark = "*" if (i in d["dels"] or any(i == a for a, _ in d["pairs_off"])) else " "
                print(f"     {mark}{i:2d} {d['chars'][i]} {d['syl'][i]:10s} ảnh→ {d['argmax'][i]} {d['p_argmax'][i]}")
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
