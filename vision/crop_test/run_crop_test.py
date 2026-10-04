#!/usr/bin/env python3
"""Chạy thí nghiệm crop theo prereg_crop_test.json (cục bộ, 0 yêu cầu Vision).

  .venv/bin/python vision/crop_test/run_crop_test.py --exp E1            # nới cửa sổ L16
  .venv/bin/python vision/crop_test/run_crop_test.py --exp E2            # dời tâm ô lệch (Chr, stt2, stt4, stt11)
  .venv/bin/python vision/crop_test/run_crop_test.py --exp ctrl          # kiểm soát: adapt trên TK, KVK, L83
  .venv/bin/python vision/crop_test/run_crop_test.py --exp all
Đầu ra: vision/ket_qua/crop_test/<exp>_<sách>.csv.gz (điểm từng ô × biến thể) + log. Phân tích: analyze_crop_test.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crop_lib as cl                                        # noqa: E402

spec = importlib.util.spec_from_file_location("vc_ct", cl.VISION / "vision_crosscheck.py")
vc = importlib.util.module_from_spec(spec)
sys.modules["vc_ct"] = vc
spec.loader.exec_module(vc)

CFG = "n2-grid-zhHant-h80-d9e1aa"
OUT = cl.VISION / "ket_qua" / "crop_test"


def vision_attrs(key: str, cells: pd.DataFrame) -> pd.DataFrame:
    """Gắn status/harm/dy_pitch/vconf/hộp Vision cho từng ô (theo trang đã có trong cache Vision)."""
    R, sim = vc.load_dict(), vc.load_similar()
    keep = ["status", "harm", "dy_pitch", "dx", "dy", "vconf", "pitch", "vx0", "vy0", "vx1", "vy1"]
    out = pd.DataFrame(index=cells.index, columns=keep, dtype=object)
    base = cl.VISION / "cache" / CFG / "pages" / key
    for pg, g in cells.groupby("page"):
        f = base / f"{pg}.json"
        if not f.exists():
            continue
        pj = json.loads(f.read_text(encoding="utf-8"))
        recs = vc.analyse_page(g[["page", "column", "x0", "y0", "x1", "y1", "tier", "label", "ocr_char", "syllable", "book_key"]].copy(), pj["sym"], R, sim=sim)
        for idx, r in zip(g.index, recs):
            for k in keep:
                out.at[idx, k] = r.get(k)
    for k in keep:
        out[k] = pd.to_numeric(out[k], errors="coerce") if k != "status" else out[k]
    return pd.concat([cells, out], axis=1)


def gen_crop(variant: str, r, pg, rng_seed: int):
    bbox = (int(r.x0), int(r.y0), int(r.x1), int(r.y1))
    pv, nx = r.prev_bbox, r.next_bbox
    if variant == "base":
        return cl.make_crop(pg, bbox, pv, nx)
    if variant.startswith("py"):
        return cl.make_crop(pg, bbox, pv, nx, pad_y=int(variant[2:]) / 100.0)
    if variant == "adapt":
        return cl.make_crop_adaptive(pg, bbox, pv, nx)[0]
    cx, cy = (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0
    vcx, vcy = (r.vx0 + r.vx1) / 2.0, (r.vy0 + r.vy1) / 2.0
    if variant in ("rc_full", "rc_half"):
        f = 1.0 if variant == "rc_full" else 0.5
        return cl.make_crop(pg, cl.shifted_bbox(bbox, f * (vcx - cx), f * (vcy - cy)), pv, nx)
    if variant == "rc_rand":
        mag = float(np.hypot(vcx - cx, vcy - cy))
        ang = np.random.default_rng(rng_seed).uniform(0, 2 * np.pi)
        return cl.make_crop(pg, cl.shifted_bbox(bbox, mag * np.cos(ang), mag * np.sin(ang)), pv, nx)
    if variant == "vis_union":
        w = cl.vision_union_window(pg, bbox, (r.vx0, r.vy0, r.vx1, r.vy1))
        return cl.make_crop(pg, bbox, pv, nx, window=w)
    raise ValueError(variant)


def score_variants(key: str, cells: pd.DataFrame, variants, S: cl.Scorer, log):
    """cells đã có cột 'Leval'. → DataFrame (một dòng/ô) với s_<v>, m_<v>, t_<v>, chg_<v>."""
    pc = cl.PageCache(key)
    labels, syls = cells["Leval"].tolist(), cells["syllable"].tolist()
    S.prepare(labels, syls)
    res = cells[["page", "column", "syl_idx", "tier", "label", "Leval", "syllable", "status", "harm", "dy_pitch", "vconf"]].copy().reset_index(drop=True)
    order = cells.sort_values(["page"]).index
    base_md5 = {}
    for v in variants:
        t0 = time.time()
        grays, pos = [], []
        md5s = {}
        for i in order:
            r = cells.loc[i]
            pg = pc.get(r["page"])
            g = gen_crop(v, r, pg, int(i) + 1000003)
            if g is not None:
                grays.append(g)
                pos.append(i)
                md5s[i] = hashlib.md5(g.tobytes() + str(g.shape).encode()).hexdigest()
        E = S.embed(grays)
        full = np.zeros((len(cells), E.shape[1]), np.float32)
        have = np.zeros(len(cells), bool)
        ci = {i: k for k, i in enumerate(cells.index)}
        for k, i in enumerate(pos):
            full[ci[i]] = E[k]
            have[ci[i]] = True
        s, m, t = S.metrics(full, labels, syls)
        s[~have], m[~have], t[~have] = np.nan, np.nan, np.nan
        res[f"s_{v}"], res[f"m_{v}"], res[f"t_{v}"] = s, m, t
        if v == "base":
            base_md5 = md5s
        res[f"chg_{v}"] = [float(md5s.get(i) != base_md5.get(i)) if (i in md5s and i in base_md5) else np.nan for i in cells.index]
        log(f"  [{key}] {v:9s} {len(pos)}/{len(cells)} crop, {time.time() - t0:.0f}s, top1={np.nanmean(t) * 100:.2f} %")
    return res


def load_eval_cells(key: str):
    d = cl.load_cells(key)
    d = d[(d["tier"] == "GOLD") & (d["label"] != "")].copy()
    d["Leval"] = d["gt"].where(d["gt"].notna() & (d["gt"].astype(str).str.len() == 1), d["label"]) if cl.BOOKS[key]["ihr"] else d["label"]
    if cl.BOOKS[key]["ihr"]:
        d = d[d["gt"].notna() & (d["gt"].astype(str).str.len() == 1)]          # E1: chỉ ô có nhãn người
    return vision_attrs(key, d)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["E1", "E2", "ctrl", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="thử khói: giới hạn số ô mỗi sách (KHÔNG diễn giải số)")
    ap.add_argument("--sample", type=int, default=3000, help="ctrl: số ô GOLD ngẫu nhiên mỗi sách")
    ap.add_argument("--only", default="", help="chỉ các sách này (vd stt2,stt4,stt11,TK,KVK,L83); mặc định tất cả; kết quả tất định nên chạy lại phần dở an toàn")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"nhat_ky_{time.strftime('%Y%m%d_%H%M%S')}.log", "a", encoding="utf-8")

    def log(*m):
        s = " ".join(str(x) for x in m)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    S = cl.Scorer()
    log(f"prereg sha256: {open(cl.VISION / 'prereg' / 'prereg_crop_test.sha256').read().split()[0][:16]}…  | cfg Vision {CFG}")
    only = {x for x in a.only.split(",") if x}
    want = lambda k: (not only) or k in only
    if a.exp in ("E1", "all") and want("L16"):
        cells = load_eval_cells("L16")
        if a.limit:
            cells = cells.sample(min(a.limit, len(cells)), random_state=1)
        log(f"E1 L16: {len(cells)} ô GOLD có nhãn người; Vision khớp {int((cells['status'] == 'agree').sum())}, bị cắt (agree & hại=1) {int((cells['harm'] == 1).sum())}")
        res = score_variants("L16", cells, ["base", "py20", "py30", "py45", "adapt"], S, log)
        res.to_csv(OUT / f"E1_L16{'_khoi' if a.limit else ''}.csv.gz", index=False)
    if a.exp in ("E2", "all"):
        for key in [k for k in ("Chr", "stt2", "stt4", "stt11") if want(k)]:
            cells = load_eval_cells(key)
            cells = cells[(cells["status"] == "agree") & (cells["dy_pitch"].abs() > 0.10) & cells["vx0"].notna()]
            if a.limit:
                cells = cells.sample(min(a.limit, len(cells)), random_state=1)
            log(f"E2 {key}: {len(cells)} ô lệch tâm (|dy|>0,10) Vision khớp; trong đó bị cắt (hại=1) {int((cells['harm'] == 1).sum())}")
            res = score_variants(key, cells, ["base", "rc_full", "rc_half", "rc_rand", "vis_union", "adapt"], S, log)
            res.to_csv(OUT / f"E2_{key}{'_khoi' if a.limit else ''}.csv.gz", index=False)
    if a.exp in ("ctrl", "all"):
        for key in [k for k in ("TK", "KVK", "L83") if want(k)]:
            cells = load_eval_cells(key)
            n = a.limit or a.sample
            cells = cells.sample(min(n, len(cells)), random_state=3)
            log(f"ctrl {key}: {len(cells)} ô GOLD (mẫu ngẫu nhiên)")
            res = score_variants(key, cells, ["base", "adapt"], S, log)
            res.to_csv(OUT / f"ctrl_{key}{'_khoi' if a.limit else ''}.csv.gz", index=False)
    log("xong")
    return 0


if __name__ == "__main__":
    sys.exit(main())
