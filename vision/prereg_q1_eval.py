#!/usr/bin/env python3
"""Đánh giá Q1 của prereg_vision.json (chọn cấu hình ghép từ pilot) — đúng theo tiêu chí đã đăng ký, OFFLINE (0 yêu cầu).

  .venv/bin/python vision/prereg_q1_eval.py stage1     # n=1 · n=4 không trần · n=4 trần 5 MP · n=3 trần 5 MP (thư mục _pilot/* có manifest n,layout,cap_mp)
  .venv/bin/python vision/prereg_q1_eval.py stage2 --base <cid>   # so không gợi ý với zh-Hant tại cấu hình chọn

Điểm cuối chính: tỉ lệ ô GOLD mà Vision ghép 1–1 và khớp nhãn (agree) trên mọi ô GOLD của trang; so cặp theo trang với n=1; bootstrap theo trang (10 trang).
Chọn n lớn nhất sao cho (i) cận dưới KTC95 % của hiệu ≥ −3 điểm %; (ii) số ký hiệu/trang ≥ 0,90 × n=1 trên ≥ 9/10 trang; (iii) mọi nhóm frame == 'canvas', không ký hiệu thiếu đỉnh.
Cấu hình không trần MP vẫn đủ điều kiện nếu đạt (i)–(iii) (đúng chữ prereg); nếu thắng thì cấu hình CHẠY dùng trần = MP lớn nhất đã kiểm chứng trong pilot (không ngoại suy).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vc_q1", HERE / "vision_crosscheck.py")
vc = importlib.util.module_from_spec(spec)
sys.modules["vc_q1"] = vc
spec.loader.exec_module(vc)
ROOT = vc.ROOT


def per_page_stats(cid: str, R, sim):
    out = {}
    for pf in sorted((ROOT / cid / "pages").glob("*/*.json")):
        pj = json.loads(pf.read_text(encoding="utf-8"))
        bk, pg = pf.parent.name, pf.stem
        d = vc.load_cells(bk)
        d = d[d["page"] == pg]
        recs = pd.DataFrame(vc.analyse_page(d, pj["sym"], R, sim=sim))
        G = recs[recs["tier"] == "GOLD"]
        out[(bk, pg)] = dict(agree=float((G["status"] == "agree").mean()), read=float(G["status"].isin(vc.READ).mean()), n_gold=int(len(G)), symbols=len(pj["sym"]),
                             frame=pj.get("frame", "canvas"), no_vertices=pj.get("no_vertices", 0), scale_g=pj.get("scale_g"), canvas=(pj.get("canvas_w"), pj.get("canvas_h")),
                             vision=(pj.get("vision_w"), pj.get("vision_h")), hi=float(((G["status"] == "agree") & (G["vconf"].astype(float) >= 0.8)).mean()))
    return out


def boot_ci(diff, B=20000, seed=11):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    m = rng.choice(d, size=(B, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cfg_dirs():
    res = []
    for mf in sorted((ROOT / "_pilot").glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        res.append(("_pilot/" + mf.parent.name, m["cfg"]))
    return res


def stage1(log=print):
    R, sim = vc.load_dict(), vc.load_similar()
    cfgs = cfg_dirs()
    st = {cid: per_page_stats(cid, R, sim) for cid, _ in cfgs}
    ref = next((cid for cid, c in cfgs if c["n"] == 1), None)
    if ref is None:
        sys.exit("chưa có cấu hình n=1 trong _pilot — chạy pilot --stage 1 trước")
    pages = sorted(st[ref].keys())
    rows = []
    for cid, c in cfgs:
        if len(st[cid]) != len(pages) or set(st[cid]) != set(pages):
            log(f"⚠ {cid}: thiếu trang ({len(st[cid])}/{len(pages)}) — bỏ qua")
            continue
        d = [100 * (st[cid][p]["agree"] - st[ref][p]["agree"]) for p in pages]
        mean, lo, hi = boot_ci(d)
        ratio = [st[cid][p]["symbols"] / max(1, st[ref][p]["symbols"]) for p in pages]
        ok_ii = sum(r >= 0.90 for r in ratio) >= 9
        frames = {st[cid][p]["frame"] for p in pages}
        nov = sum(st[cid][p]["no_vertices"] for p in pages)
        ok_iii = frames == {"canvas"} and nov == 0
        diag = c["cap_mp"] > 1e6
        rows.append(dict(cid=cid, n=c["n"], layout=c["layout"], cap_mp=c["cap_mp"], hints=c["hints"], agree_mean=100 * float(np.mean([st[cid][p]["agree"] for p in pages])),
                         read_mean=100 * float(np.mean([st[cid][p]["read"] for p in pages])), hi_mean=100 * float(np.mean([st[cid][p]["hi"] for p in pages])),
                         symbols_mean=float(np.mean([st[cid][p]["symbols"] for p in pages])), d_mean=mean, d_lo=lo, d_hi=hi, ok_i=lo >= -3.0, sym_ratio_min=min(ratio), ok_ii=ok_ii,
                         frames=sorted(frames), no_vertices=nov, ok_iii=ok_iii, diagnostic=diag,
                         canvas_max_mp=max((st[cid][p]["canvas"][0] or 0) * (st[cid][p]["canvas"][1] or 0) / 1e6 for p in pages),
                         vision_dims_equal=all(tuple(st[cid][p]["vision"]) == tuple(st[cid][p]["canvas"]) for p in pages)))
    log(f"{'cấu hình':46s} n  MP≤  agree/GOLD  đọc  conf≥0,8  ký hiệu/trang | Δ vs n=1 [KTC95 theo trang]   (i)  (ii)  (iii) frame  Vision=ảnh gửi")
    for r in rows:
        log(f"{r['cid']:46s} {r['n']}  {r['canvas_max_mp']:4.1f}  {r['agree_mean']:6.1f} %   {r['read_mean']:4.1f}  {r['hi_mean']:5.1f} %   {r['symbols_mean']:6.1f}     | "
            f"{r['d_mean']:+5.1f} [{r['d_lo']:+5.1f}, {r['d_hi']:+5.1f}] pp   {'✓' if r['ok_i'] else '✗'}    {'✓' if r['ok_ii'] else '✗'}    {'✓' if r['ok_iii'] else '✗'}   {','.join(r['frames'])}  {r['vision_dims_equal']}"
            f"{'  (không trần MP)' if r['diagnostic'] else ''}{'  (đối chiếu n=1)' if r['cid'] == ref else ''}")
    elig = [r for r in rows if r["n"] > 1 and r["ok_i"] and r["ok_ii"] and r["ok_iii"]]
    if elig:
        best = max(elig, key=lambda r: (r["n"], r["agree_mean"]))
        run_cap = best["cap_mp"] if best["cap_mp"] < 1e6 else round(best["canvas_max_mp"] + 0.05, 1)
        verdict = dict(chon=best["cid"], n=best["n"], layout=best["layout"], cap_mp_chay=run_cap, ly_do="đạt (i),(ii),(iii); n lớn nhất, rồi agree/GOLD cao nhất"
                       + ("; cấu hình thắng không trần → chạy với trần = MP lớn nhất đã kiểm chứng" if best["cap_mp"] >= 1e6 else ""))
    else:
        verdict = dict(chon="n=2 trần 5 MP (dự phòng của prereg)", n=2, ly_do="không cấu hình n≥3 nào đạt cả (i),(ii),(iii)")
    log("QUYẾT ĐỊNH Q1:", json.dumps(verdict, ensure_ascii=False))
    out = vc.OUT / f"q1_stage1_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(rows=rows, verdict=verdict, per_page={cid: {f"{k[0]}/{k[1]}": v for k, v in s.items()} for cid, s in st.items()}), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log("->", out)
    return rows, verdict


def stage2(base_n, log=print):
    R, sim = vc.load_dict(), vc.load_similar()
    cfgs = [(cid, c) for cid, c in cfg_dirs() if c["n"] == base_n and c["cap_mp"] < 1e6]
    by_hint = {tuple(c["hints"]): cid for cid, c in cfgs}
    a, b = by_hint.get(("zh-Hant",)), by_hint.get(())
    if not (a and b):
        sys.exit("cần cả cấu hình gợi ý zh-Hant và không gợi ý ở cùng n — chạy pilot --stage 2 --yes")
    sa, sb = per_page_stats(a, R, sim), per_page_stats(b, R, sim)
    pages = sorted(set(sa) & set(sb))
    d = [100 * (sb[p]["agree"] - sa[p]["agree"]) for p in pages]
    mean, lo, hi = boot_ci(d)
    n_up = sum(x >= 1.0 for x in d)
    log(f"không gợi ý − zh-Hant: Δ agree/GOLD {mean:+.1f} [{lo:+.1f}, {hi:+.1f}] pp; số trang tăng ≥ 1 điểm: {n_up}/{len(pages)}; "
        f"phủ trung bình zh-Hant {100 * np.mean([sa[p]['read'] for p in pages]):.1f} % · không gợi ý {100 * np.mean([sb[p]['read'] for p in pages]):.1f} %")
    verdict = "ĐỔI sang không gợi ý" if n_up >= 7 else "GIỮ zh-Hant"
    log("QUYẾT ĐỊNH gợi ý (prereg: đổi chỉ nếu ≥ 7/10 trang tăng ≥ 1 điểm):", verdict)
    return dict(delta=mean, lo=lo, hi=hi, n_up=n_up, verdict=verdict)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1", "stage2"])
    ap.add_argument("--base-n", type=int, default=4)
    a = ap.parse_args(argv)
    stage1() if a.cmd == "stage1" else stage2(a.base_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
