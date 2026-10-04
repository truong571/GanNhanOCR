#!/usr/bin/env python3
"""Vision so với NHÃN NGƯỜI IHR (L16, TK) theo từng ô — điểm cuối Q2/phần 'cứu' của prereg_vision.json. OFFLINE (0 yêu cầu).

  .venv/bin/python vision/vision_vs_human.py --cfg <cid> [--books L16,TK]

Ô = dòng của prepared/<Bộ>/dataset_out/labels_gated.csv (có syl_idx, bbox) ghép với measure_out/<Bộ>/ihr_endtoend/cells.csv (gt = chữ người, theo (trang, cột, syl_idx)).
Báo: độ đúng của nhãn pipeline và của chữ kim thô; Vision đọc được bao nhiêu ô; độ chính xác của ô 'agree & conf ≥ 0,8' (xác nhận) và phần ô GOLD được xác nhận;
bất đồng URO không-dị-thể: ai đúng; ô kim thô sai mà Vision đọc: Vision đúng bao nhiêu (tiềm năng 'cứu')."""
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
spec = importlib.util.spec_from_file_location("vc_h", HERE / "vision_crosscheck.py")
vc = importlib.util.module_from_spec(spec)
sys.modules["vc_h"] = vc
spec.loader.exec_module(vc)
REPO, ROOT = vc.REPO, vc.ROOT
IHR = {"L16": "LucVanTien1916", "TK": "TruyenKieu1872"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def book_frame(bk: str, cid: str, R, sim):
    bdir = IHR[bk]
    g = pd.read_csv(REPO / "prepared" / bdir / "dataset_out" / "labels_gated.csv", usecols=["page", "column", "syl_idx", "ocr_char", "label", "syllable", "tier", "bbox"])
    ih = pd.read_csv(REPO / "measure_out" / bdir / "ihr_endtoend" / "cells.csv", usecols=["page", "column", "syl_idx", "gt"])
    pages = sorted(p.stem for p in (ROOT / cid / "pages" / bk).glob("*.json"))
    rows = []
    for pg in pages:
        gp = g[(g.page == pg) & g.bbox.notna()].merge(ih[ih.page == pg], on=["page", "column", "syl_idx"], how="left")
        if gp.empty:
            continue
        bb = gp["bbox"].map(json.loads)
        for i, nm in enumerate(["x0", "y0", "x1", "y1"]):
            gp[nm] = bb.map(lambda b, i=i: b[i])
        gp["book_key"] = bk
        pj = json.loads((ROOT / cid / "pages" / bk / f"{pg}.json").read_text(encoding="utf-8"))
        recs = pd.DataFrame(vc.analyse_page(gp.reset_index(drop=True), pj["sym"], R, sim=sim))
        recs["gt"] = gp["gt"].values
        rows.append(recs)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate(cid: str, books, log=print):
    R, sim = vc.load_dict(), vc.load_similar()
    res = {}
    for bk in books:
        D = book_frame(bk, cid, R, sim)
        if D.empty:
            log(f"[{bk}] chưa có trang nào trong {cid}")
            continue
        T = D[D["gt"].notna()]
        G = T[T["tier"] == "GOLD"]
        lab_ok = (G["label"] == G["gt"])
        kim_ok = (T["kim"] == T["gt"])
        read = G[G["status"].isin(vc.READ)]
        conf = G[(G["status"] == "agree") & (G["vconf"].astype(float) >= 0.8)]
        conf_ok = int((conf["label"] == conf["gt"]).sum())
        dis = G[(G["status"] == "disagree") & (G["label_class"] == "URO") & (G["v_class"] == "URO")]
        dis_v, dis_l = int((dis["v"] == dis["gt"]).sum()), int((dis["label"] == dis["gt"]).sum())
        dis_hi = dis[dis["vconf"].astype(float) >= 0.8]
        kw = T[(T["kim"] != T["gt"]) & T["status"].isin(vc.READ)]
        kw_all = T[T["kim"] != T["gt"]]
        r = dict(book=bk, pages=int(D["page"].nunique()), o_co_gt=int(len(T)), gold_co_gt=int(len(G)),
                 nhan_dung_gold=float(lab_ok.mean()), kim_tho_dung=float(kim_ok.mean()), vision_doc_tren_gold=float(len(read) / max(1, len(G))),
                 xac_nhan_n=int(len(conf)), xac_nhan_chinh_xac=(conf_ok / len(conf) if len(conf) else None), xac_nhan_ci=wilson(conf_ok, len(conf)),
                 xac_nhan_phan_gold=float(len(conf) / max(1, len(G))),
                 bat_dong_URO_n=int(len(dis)), bat_dong_vision_dung=dis_v, bat_dong_nhan_dung=dis_l, bat_dong_hi_n=int(len(dis_hi)),
                 bat_dong_hi_vision_dung=int((dis_hi["v"] == dis_hi["gt"]).sum()),
                 kim_sai_n=int(len(kw_all)), kim_sai_ma_vision_doc=int(len(kw)), kim_sai_ma_vision_dung=int((kw["v"] == kw["gt"]).sum()))
        res[bk] = r
        ci = r["xac_nhan_ci"]
        log(f"[{bk}] {r['pages']} trang · {r['o_co_gt']} ô có nhãn người ({r['gold_co_gt']} GOLD): nhãn GOLD đúng {100 * r['nhan_dung_gold']:.1f} % · kim thô đúng {100 * r['kim_tho_dung']:.1f} % · "
            f"Vision đọc {100 * r['vision_doc_tren_gold']:.1f} % ô GOLD")
        log(f"      xác nhận (agree & conf≥0,8): {r['xac_nhan_n']} ô = {100 * r['xac_nhan_phan_gold']:.1f} % GOLD; chính xác {('–' if r['xac_nhan_chinh_xac'] is None else f'{100 * r['xac_nhan_chinh_xac']:.1f} % [Wilson {100 * ci[0]:.1f}; {100 * ci[1]:.1f}]')}"
            f" | bất đồng URO không-dị-thể {r['bat_dong_URO_n']}: Vision đúng {dis_v}, nhãn đúng {dis_l} (conf≥0,8: {r['bat_dong_hi_n']}, Vision đúng {r['bat_dong_hi_vision_dung']})"
            f" | kim thô sai {r['kim_sai_n']} ô, Vision đọc {r['kim_sai_ma_vision_doc']}, đúng {r['kim_sai_ma_vision_dung']}")
    return res


def borg_eval(cid: str, bk: str, log=print):
    """Borg (B18/B34): nhãn NGƯỜI theo ô ở dataset/_BORG_NHAN_NGUOI/labels.csv (page, bbox, char). Ghép 1–1 ký hiệu Vision với hộp người; báo phủ/độ đúng."""
    bdir = {"B18": "SachKinhThayCaBinh", "B34": "SachDungLyHoThan"}[bk]
    hb = pd.read_csv(REPO / "dataset" / "_BORG_NHAN_NGUOI" / "labels.csv", usecols=["book", "page", "char", "bbox"])
    hb = hb[(hb["book"].str.lower() == bdir.lower()) & hb["char"].notna() & hb["bbox"].notna()]
    sim = vc.load_similar()
    pages = sorted(p.stem for p in (ROOT / cid / "pages" / bk).glob("*.json"))
    n = rd = ok = okv = 0
    hi_n = hi_ok = 0
    for pg in pages:
        s_ = hb[hb["page"] == pg]
        if s_.empty:
            continue
        pj = json.loads((ROOT / cid / "pages" / bk / f"{pg}.json").read_text(encoding="utf-8"))
        syms = [x for x in pj["sym"] if vc.is_cjk(x[0])]
        cells = np.array([json.loads(b) for b in s_["bbox"]], float)
        assign, _ = vc.match_page(cells, syms)
        for i, (ch, j) in enumerate(zip(s_["char"].tolist(), assign)):
            n += 1
            if j >= 0:
                rd += 1
                v, conf = syms[j][0], syms[j][1]
                good = vc.same_char(v, ch)
                ok += int(good)
                okv += int(good or vc.near_variant(ch, v, sim))
                if conf is not None and conf >= 0.8:
                    hi_n += 1
                    hi_ok += int(good)
    r = dict(book=bk, pages=len(pages), o_nguoi=n, vision_doc=rd, dung=ok, dung_ke_ca_di_the=okv, hi_n=hi_n, hi_dung=hi_ok)
    log(f"[{bk}] nhãn NGƯỜI {n} ô trên {len(pages)} trang: Vision đọc {100 * rd / max(1, n):.1f} %; đúng {100 * ok / max(1, n):.1f} % mọi ô ({100 * ok / max(1, rd):.1f} % trên ô có đọc; +dị thể {100 * okv / max(1, rd):.1f} %); "
        f"conf≥0,8: {hi_n} ô ({100 * hi_n / max(1, n):.1f} %), đúng {100 * hi_ok / max(1, hi_n):.1f} %")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--books", default="L16,TK")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    bl = [b for b in a.books.split(",") if b]
    res = evaluate(a.cfg, [b for b in bl if b in IHR])
    for b in bl:
        if b in ("B18", "B34"):
            res[b] = borg_eval(a.cfg, b)
    out = Path(a.out) if a.out else vc.OUT / f"vs_human_{a.cfg.replace('/', '__')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
