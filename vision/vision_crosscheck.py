#!/usr/bin/env python3
"""Đối chứng OFFLINE kết quả Google Vision đã thu hoạch (vision_harvest.py) với nhãn/hộp của pipeline (kim).
Ghép ký hiệu Vision với TỪNG Ô (1–1, Hungarian trên khoảng cách tâm chuẩn hoá) rồi đo:
  • độ phủ, khớp nhãn (chuẩn hoá NFKC; dị thể giản/phồn/hình gần ghi riêng là 'variant', KHÔNG tính bất đồng), theo sách × tầng × lớp mã nhãn (URO/ExtA/ExtB+/PUA);
  • độ tin cậy Vision → tỉ lệ khớp; xác nhận ở conf cao;
  • LỆCH HỘP đo theo BƯỚC CỘT (không theo cao hộp) trên ô Vision khớp chữ + chỉ số HẠI hình học: phần hộp glyph Vision nằm ngoài cửa sổ crop (bbox ± pad 0,12) > 10 %
    (bbox là hộp detector/cửa sổ theo bước, KHÔNG phải hộp chữ chặt — độ lệch tâm một mình chưa phải 'crop hỏng');
  • mô hình hiệu chỉnh theo cột (không dịch / hằng / tuyến tính co rút) ĐÁNH GIÁ bằng bỏ-một-ô (column_model.csv, KPI trong summary) — chỉ là ứng viên, cần kiểm bằng so ảnh;
  • lệch ảnh–chữ (chữ nhãn nằm ở ký hiệu lân cận duy nhất ngoài hộp) — độ nhạy thấp: 'không thấy' KHÔNG chứng minh 'không lệch';
  • ứng viên bất đồng (disagree_candidates.csv): chỉ nhãn URO, đã loại dị thể — KHÔNG phải 'kim sai' (trên nhãn người IHR Vision đúng 1/83 ô bất đồng).
Không gọi mạng, không sửa pipeline/dataset. Ghi vào vision/ket_qua/<cfg>/.

  .venv/bin/python vision/vision_crosscheck.py cells   --cfg <cid> [--books KVK,L83]     (cid có thể là '_pilot/<cid>')
  .venv/bin/python vision/vision_crosscheck.py compare --cfgs <cid1>,<cid2>,…           # so cấu hình trên các trang chung
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent                                           # .../GanNhanOCR/vision
REPO = Path(os.environ.get("VISION_REPO_ROOT", HERE.parent))
ROOT = Path(os.environ.get("VISION_CACHE_ROOT", HERE / "cache"))
OUT = Path(os.environ.get("VISION_AUDIT_OUT", HERE / "ket_qua"))
BOOKS = {"KVK": "KimVanKieu1884", "L83": "LucVanTien1883", "TK": "TruyenKieu1872", "L16": "LucVanTien1916", "Chr": "Chrestomathie1872",
         "stt11": "SachThanhTruyen11", "stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "B34": "SachDungLyHoThan", "B18": "SachKinhThayCaBinh"}
BIG = 1e6
PAD = 0.12          # pad của cửa sổ crop quanh bbox (build_dataset.save_crop)
READ = ("agree", "variant", "disagree")
_SIM: dict = {}


def is_cjk(ch: str) -> bool:
    if not ch:
        return False
    c = ch[0]
    return "CJK" in unicodedata.name(c, "") or unicodedata.category(c) == "Co"


def char_class(ch: str) -> str:
    if not ch:
        return ""
    o = ord(ch[0])
    if 0x4E00 <= o <= 0x9FFF:
        return "URO"
    if 0x3400 <= o <= 0x4DBF:
        return "ExtA"
    if 0xF900 <= o <= 0xFAFF:
        return "Compat"
    if 0xE000 <= o <= 0xF8FF or o >= 0xF0000:
        return "PUA"
    if 0x20000 <= o <= 0x3FFFF:
        return "ExtB+"
    return "other"


def _dict_dir() -> Path:
    for n in ("dict", "Dict"):
        if (REPO / n).is_dir():
            return REPO / n
    return REPO / "dict"


def load_dict():
    R = defaultdict(set)
    f = _dict_dir() / "QuocNgu_SinoNom.csv"
    if f.exists():
        with open(f, encoding="utf-8-sig") as fh:
            rd = csv.reader(fh)
            next(rd)
            for row in rd:
                if len(row) >= 2:
                    R[row[0].strip().lower()].add(row[1].strip())
    else:
        print(f"⚠ không thấy từ điển {f} — v_in_R sẽ luôn False", flush=True)
    return R


def load_similar():
    """dict/SinoNom_Similar.csv: {chữ: {20 chữ giống hình nhất}} — dùng để coi dị thể giản/phồn/hình gần là 'variant' thay vì bất đồng."""
    if _SIM:
        return _SIM
    f = _dict_dir() / "SinoNom_Similar.csv"
    if f.exists():
        with open(f, encoding="utf-8-sig") as fh:
            rd = csv.reader(fh)
            next(rd)
            for row in rd:
                if len(row) >= 2:
                    try:
                        _SIM[row[0]] = set(ast.literal_eval(row[1]))
                    except Exception:  # noqa: BLE001
                        pass
    return _SIM


def same_char(a: str, b: str) -> bool:
    return a == b or unicodedata.normalize("NFKC", a) == unicodedata.normalize("NFKC", b)


def near_variant(label: str, v: str, sim: dict) -> bool:
    return v in sim.get(label, ()) or label in sim.get(v, ())


def match_page(cells: np.ndarray, syms: list, near: float = 0.15):
    """cells: n×4 (x0,y0,x1,y1). syms: [(ch, conf, x0,y0,x1,y1,…)] (chỉ CJK/PUA). → (assign[n]=chỉ số ký hiệu|-1, cost[n])."""
    n, m = len(cells), len(syms)
    if n == 0 or m == 0:
        return np.full(n, -1), np.full(n, np.nan)
    S = np.array([[s[2], s[3], s[4], s[5]] for s in syms], float)
    cw, ch_ = cells[:, 2] - cells[:, 0], cells[:, 3] - cells[:, 1]
    cc = np.stack([(cells[:, 0] + cells[:, 2]) / 2, (cells[:, 1] + cells[:, 3]) / 2], 1)
    sc = np.stack([(S[:, 0] + S[:, 2]) / 2, (S[:, 1] + S[:, 3]) / 2], 1)
    ix = np.clip(np.minimum(cells[:, None, 2], S[None, :, 2]) - np.maximum(cells[:, None, 0], S[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(cells[:, None, 3], S[None, :, 3]) - np.maximum(cells[:, None, 1], S[None, :, 1]), 0, None)
    inter = ix * iy
    s_area = np.maximum((S[:, 2] - S[:, 0]) * (S[:, 3] - S[:, 1]), 1e-6)
    cont = inter / s_area[None, :]
    dx = (sc[None, :, 0] - cc[:, None, 0]) / np.maximum(cw, 1)[:, None]
    dy = (sc[None, :, 1] - cc[:, None, 1]) / np.maximum(ch_, 1)[:, None]
    dist = np.hypot(dx, dy)
    inside = (np.abs(dx) <= 0.5 + near) & (np.abs(dy) <= 0.5 + near)
    cand = (cont >= 0.5) | inside
    cost = np.where(cand, dist, BIG)
    r, c = linear_sum_assignment(cost)
    assign = np.full(n, -1)
    out_cost = np.full(n, np.nan)
    for i, j in zip(r, c):
        if cost[i, j] < BIG:
            assign[i] = j
            out_cost[i] = dist[i, j]
    return assign, out_cost


def column_pitch(page_df: pd.DataFrame) -> dict:
    """Bước dọc trung vị giữa tâm các ô liền kề của mỗi cột (px); cột <3 ô → None."""
    out = {}
    for c, g in page_df.groupby("column"):
        cy = np.sort(((g["y0"] + g["y1"]) / 2).to_numpy(float))
        d = np.diff(cy)
        d = d[d > 1]
        out[c] = float(np.median(d)) if len(d) >= 2 else None
    return out


def analyse_page(page_df: pd.DataFrame, sym_rows: list, R: dict, conf_hi=0.8, shift_min=0.5, shift_max=2.5, sim: dict | None = None):
    """→ list[dict] mỗi ô. Hai lượt: (1) ghép 1–1, xếp agree/variant/disagree; (2) với ô disagree/no_vision tìm chữ nhãn ở ký hiệu LÂN CẬN DUY NHẤT
    (loại ký hiệu đã được ô 'agree' giữ; bỏ chữ xuất hiện ≥ 4 lần trên trang) → 'shifted' kèm độ lệch theo đơn vị ô."""
    sim = load_similar() if sim is None else sim
    syms = [s for s in sym_rows if is_cjk(s[0])]
    freq = defaultdict(int)
    for s in syms:
        freq[s[0]] += 1
    cells = page_df[["x0", "y0", "x1", "y1"]].to_numpy(float)
    assign, cost = match_page(cells, syms)
    pitch = column_pitch(page_df)
    recs = []
    agree_syms = set()
    for k, (idx, r) in enumerate(page_df.iterrows()):
        x0, y0, x1, y1 = cells[k]
        cw, chh = max(x1 - x0, 1), max(y1 - y0, 1)
        label = r["label"] if isinstance(r["label"], str) else ""
        kim = r["ocr_char"] if isinstance(r["ocr_char"], str) else ""
        syl = str(r["syllable"]).lower()
        pit = pitch.get(r["column"]) or chh
        rec = dict(idx=idx, book=r["book_key"], page=r["page"], column=r["column"], tier=r["tier"], syl=syl, label=label, kim=kim, label_class=char_class(label),
                   x0=x0, y0=y0, x1=x1, y1=y1, pitch=pit, v="", v_class="", vconf=np.nan, ang=np.nan, dx=np.nan, dy=np.nan, dy_pitch=np.nan, vw=np.nan, vh=np.nan,
                   harm=np.nan, status="no_vision", shift_dx=np.nan, shift_dy=np.nan, shift_conf=np.nan, v_in_R=False,
                   vx0=np.nan, vy0=np.nan, vx1=np.nan, vy1=np.nan, _k=k)
        j = int(assign[k])
        if j >= 0:
            s = syms[j]
            ch, conf, sx0, sy0, sx1, sy1 = s[:6]
            ang = s[6] if len(s) > 6 else np.nan
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            scx, scy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
            rec.update(v=ch, v_class=char_class(ch), vconf=conf if conf is not None else np.nan, ang=ang, dx=(scx - cx) / cw, dy=(scy - cy) / chh, dy_pitch=(scy - cy) / pit,
                       vw=(sx1 - sx0) / cw, vh=(sy1 - sy0) / chh, v_in_R=ch in R.get(syl, set()), vx0=sx0, vy0=sy0, vx1=sx1, vy1=sy1)
            if not label:
                rec["status"] = "syllable_seen"
            elif same_char(ch, label):
                rec["status"] = "agree"
                agree_syms.add(j)
                if not (isinstance(ang, float) and abs(ang) > 3.0):      # hộp AABB của ký hiệu nghiêng bị phình → không đo hại
                    wx0, wx1, wy0, wy1 = x0 - PAD * cw, x1 + PAD * cw, y0 - PAD * chh, y1 + PAD * chh
                    ix = max(0.0, min(sx1, wx1) - max(sx0, wx0))
                    iy = max(0.0, min(sy1, wy1) - max(sy0, wy0))
                    area = max((sx1 - sx0) * (sy1 - sy0), 1e-6)
                    rec["harm"] = float((area - ix * iy) / area > 0.10)
            elif near_variant(label, ch, sim):
                rec["status"] = "variant"
            else:
                rec["status"] = "disagree"
        recs.append(rec)
    for rec in recs:
        if rec["status"] in ("no_vision", "disagree") and rec["label"] and freq.get(rec["label"], 0) < 4:
            x0, y0, x1, y1 = cells[rec["_k"]]
            cw, chh = max(x1 - x0, 1), max(y1 - y0, 1)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            cands = []
            for jj, s in enumerate(syms):
                ch, conf, sx0, sy0, sx1, sy1 = s[:6]
                if not same_char(ch, rec["label"]) or jj in agree_syms:
                    continue
                sdx, sdy = ((sx0 + sx1) / 2 - cx) / cw, ((sy0 + sy1) / 2 - cy) / chh
                if abs(sdx) <= 0.9 and abs(sdy) <= 3.0:
                    cands.append((float(np.hypot(sdx, sdy)), sdx, sdy, conf))
            if len(cands) == 1 and cands[0][0] >= shift_min and abs(cands[0][2]) <= shift_max:
                d, sdx, sdy, conf = cands[0]
                rec.update(status="shifted", shift_dx=sdx, shift_dy=sdy, shift_conf=conf if conf is not None else np.nan)
    for rec in recs:
        rec.pop("_k", None)
    return recs


def load_cells(book_key: str):
    bdir = BOOKS[book_key]
    d = pd.read_csv(REPO / "dataset" / bdir / "labels.csv")
    d = d[d["bbox"].notna()].copy()
    bb = d["bbox"].map(json.loads)
    for i, nm in enumerate(["x0", "y0", "x1", "y1"]):
        d[nm] = bb.map(lambda b, i=i: b[i])
    d["book_key"] = book_key
    return d


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    sl = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x)) if abs(x[j] - x[i]) > 1e-9]
    return float(np.median(sl)) if sl else 0.0


def column_models(A: pd.DataFrame, min_n: int = 4, k0: float = 3.0):
    """Mỗi cột (sách, trang, cột) có ≥ min_n ô agree: bỏ-một-ô cho 3 mô hình (không dịch · hằng · tuyến tính co rút theo vị trí) → sai số |dy_pitch − dự đoán|.
    → (bảng cột, {mô hình: [sai số mọi ô]})."""
    rows, errs = [], {"zero": [], "const": [], "lin": []}
    if A.empty:
        return pd.DataFrame(rows), errs
    for (bk, pg, col), g in A.groupby(["book", "page", "column"]):
        if len(g) < min_n:
            continue
        cy = ((g["y0"] + g["y1"]) / 2).to_numpy(float)
        pos = (cy - cy.mean()) / np.maximum(g["pitch"].to_numpy(float), 1)
        dy = g["dy_pitch"].to_numpy(float)
        for i in range(len(g)):
            m = np.ones(len(g), bool)
            m[i] = False
            a = float(np.median(dy[m]))
            lam = m.sum() / (m.sum() + k0)
            b = _theil_sen(pos[m], dy[m] - a)
            errs["zero"].append(abs(dy[i]))
            errs["const"].append(abs(dy[i] - a))
            errs["lin"].append(abs(dy[i] - lam * (a + b * (pos[i] - pos[m].mean()))))
        a_all = float(np.median(dy))
        rows.append(dict(book=bk, page=pg, column=col, n_agree=len(g), a=a_all, b=_theil_sen(pos, dy - a_all), shrink=len(g) / (len(g) + k0), pos_mean=float(pos.mean()),
                         span_pitch=float(cy.max() - cy.min()) / max(float(np.median(g["pitch"])), 1)))
    return pd.DataFrame(rows), errs


def _frac_le(e, thr=0.1):
    e = np.asarray(e, float)
    return float((e <= thr).mean()) if len(e) else None


def run_cells(cid: str, books: list[str] | None, conf_hi=0.8, shift_min=0.5, out_dir: Path | None = None, log=print):
    R = load_dict()
    sim = load_similar()
    base = ROOT / cid / "pages"
    rows = []
    for bk in (books or [b for b in BOOKS if (base / b).exists()]):
        if not (base / bk).exists():
            continue
        d = load_cells(bk)
        for pf in sorted((base / bk).glob("*.json")):
            pj = json.loads(pf.read_text(encoding="utf-8"))
            pd_ = d[d["page"] == pj["page"]]
            if pd_.empty:
                continue
            rows += analyse_page(pd_, pj["sym"], R, conf_hi, shift_min, sim=sim)
    D = pd.DataFrame(rows)
    if D.empty:
        log("không có dữ liệu")
        return None
    D["hi"] = D["vconf"].astype(float) >= conf_hi
    out = (out_dir or (OUT / cid.replace("/", "__")))
    out.mkdir(parents=True, exist_ok=True)
    G = D[D["tier"] == "GOLD"]
    read = G[G["status"].isin(READ)]
    summ = dict(cfg=cid, o=len(D), gold=len(G), phu_gold=float(len(read) / max(1, len(G))), khop_tren_o_co=float((read["status"] == "agree").mean()) if len(read) else None,
                variant=int((G["status"] == "variant").sum()))
    summ["theo_lop_nhan"] = {}
    for c, s in G.groupby("label_class"):
        rd = s[s["status"].isin(READ)]
        summ["theo_lop_nhan"][c] = dict(n=int(len(s)), phu=float(len(rd) / len(s)), khop=float((rd["status"] == "agree").mean()) if len(rd) else None)
    A = D[(D["status"] == "agree") & (D["vconf"].astype(float) >= 0.5) & D["dy_pitch"].notna()]
    cm, errs = column_models(A)
    cm.to_csv(out / "column_model.csv", index=False)
    per_book = {}
    for bk, s in G.groupby("book"):
        rd = s[s["status"].isin(READ)]
        ag = s[s["status"] == "agree"]
        sub = A[A["book"] == bk]
        _, eb = column_models(sub)
        per_book[bk] = dict(
            n=len(s), phu=float(len(rd) / len(s)), khop_tren_o_co=float((rd["status"] == "agree").mean()) if len(rd) else None,
            xac_nhan_conf_cao=float(len(ag[ag["hi"]]) / len(s)), shifted=float((s["status"] == "shifted").mean()),
            dy_pitch_trung_vi=float(np.nanmedian(sub["dy_pitch"])) if len(sub) else None, dx_trung_vi=float(np.nanmedian(sub["dx"])) if len(sub) else None,
            ty_le_hai_hinh_hoc=float(np.nanmean(ag["harm"])) if ag["harm"].notna().any() else None, n_do_hai=int(ag["harm"].notna().sum()),
            cot_du_mau=int((cm["book"] == bk).sum()) if len(cm) else 0, loo_n=len(eb["zero"]),
            loo_zero=_frac_le(eb["zero"]), loo_const=_frac_le(eb["const"]), loo_lin=_frac_le(eb["lin"]))
    summ["theo_cuon_GOLD"] = per_book
    summ["loo_tong"] = dict(n=len(errs["zero"]), zero=_frac_le(errs["zero"]), const=_frac_le(errs["const"]), lin=_frac_le(errs["lin"]))
    cols = ["book", "page", "column", "tier", "syl", "label", "label_class", "kim", "x0", "y0", "x1", "y1", "pitch", "v", "v_class", "vconf", "ang", "dx", "dy", "dy_pitch", "harm",
            "status", "v_in_R", "vx0", "vy0", "vx1", "vy1"]
    D[cols + ["shift_dx", "shift_dy"]].to_csv(out / "cells.csv.gz", index=False)
    dis = G[(G["status"] == "disagree") & G["hi"] & (G["label_class"] == "URO") & (G["v_class"] == "URO")]
    dis[cols].to_csv(out / "disagree_candidates.csv", index=False)
    G[G["status"] == "shifted"][cols + ["shift_dx", "shift_dy"]].to_csv(out / "drift.csv", index=False)
    if len(A):
        off = (A.groupby(["book", "page", "column"]).agg(n=("dx", "size"), dx_med=("dx", "median"), dy_pitch_med=("dy_pitch", "median"),
                                                          dy_pitch_mad=("dy_pitch", lambda x: float(np.median(np.abs(x - np.median(x)))))).reset_index())
        off[off["n"] >= 3].to_csv(out / "offsets.csv", index=False)
    summ["ung_vien_bat_dong"] = int(len(dis))
    summ["lech_anh_chu"] = int((G["status"] == "shifted").sum())
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pct = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f} %"
    log(f"[{cid}] {len(D)} ô ({len(G)} GOLD): Vision phủ {100 * summ['phu_gold']:.1f} %, khớp {100 * (summ['khop_tren_o_co'] or 0):.1f} % trên ô có đọc (+{summ['variant']} dị thể) | "
        f"ứng viên bất đồng URO {summ['ung_vien_bat_dong']} · lệch ảnh–chữ {summ['lech_anh_chu']} → {out}")
    lo = summ["loo_tong"]
    log(f"   bỏ-một-ô (|sai số| ≤ 0,1 bước, n={lo['n']}): không dịch {pct(lo['zero'])} · hằng {pct(lo['const'])} · tuyến tính co rút {pct(lo['lin'])}  (tiêu chí đăng ký trước: ≥ 80 %)")
    for bk, v in per_book.items():
        log(f"   {bk:6s} n={v['n']:5d} phủ {100 * v['phu']:5.1f} % khớp {100 * (v['khop_tren_o_co'] or 0):5.1f} % xác nhận≥conf {100 * v['xac_nhan_conf_cao']:5.1f} % "
            f"| dy trung vị {'–' if v['dy_pitch_trung_vi'] is None else round(v['dy_pitch_trung_vi'], 3)} bước · hại hình học {pct(v['ty_le_hai_hinh_hoc'])} (n={v['n_do_hai']}) "
            f"| cột đủ mẫu {v['cot_du_mau']} · bỏ-một-ô ≤0,1 (không dịch/hằng/tuyến tính): {pct(v['loo_zero'])}/{pct(v['loo_const'])}/{pct(v['loo_lin'])} (n={v['loo_n']})")
    return D, summ


def run_compare(cids: list[str], log=print):
    """So cấu hình trên CÁC TRANG CHUNG: độ phủ GOLD, khớp, xác nhận conf cao, ký hiệu/trang."""
    R = load_dict()
    sim = load_similar()
    sets = {}
    for cid in cids:
        sets[cid] = {(b.parent.name, b.stem) for b in (ROOT / cid / "pages").glob("*/*.json")}
    common = set.intersection(*sets.values()) if sets else set()
    log(f"trang chung: {len(common)}")
    res = {}
    for cid in cids:
        rows = []
        nsym = 0
        for bk, pg in sorted(common):
            pj = json.loads((ROOT / cid / "pages" / bk / f"{pg}.json").read_text(encoding="utf-8"))
            nsym += len(pj["sym"])
            d = load_cells(bk)
            d = d[d["page"] == pg]
            rows += analyse_page(d, pj["sym"], R, sim=sim)
        D = pd.DataFrame(rows)
        G = D[D["tier"] == "GOLD"]
        m = G[G["status"].isin(READ)]
        ag = G[G["status"] == "agree"]
        hi = ag[ag["vconf"].astype(float) >= 0.8]
        res[cid] = dict(phu=len(m) / max(1, len(G)), khop=(m["status"] == "agree").mean() if len(m) else float("nan"), xac_nhan_cao=len(hi) / max(1, len(G)),
                        agree_all=len(ag) / max(1, len(G)), ky_hieu_moi_trang=nsym / max(1, len(common)))
        log(f"  {cid:46s} phủ {100 * res[cid]['phu']:5.1f} % | khớp/ô có đọc {100 * res[cid]['khop']:5.1f} % | khớp trên mọi ô GOLD {100 * res[cid]['agree_all']:5.1f} % | "
            f"xác nhận conf≥0,8 {100 * res[cid]['xac_nhan_cao']:5.1f} % | {res[cid]['ky_hieu_moi_trang']:.0f} ký hiệu/trang")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cells", "compare"])
    ap.add_argument("--cfg")
    ap.add_argument("--cfgs")
    ap.add_argument("--books", default="")
    ap.add_argument("--conf-hi", type=float, default=0.8)
    ap.add_argument("--shift-min", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.cmd == "cells":
        run_cells(a.cfg, [b for b in a.books.split(",") if b] or None, a.conf_hi, a.shift_min)
    else:
        run_compare([c for c in a.cfgs.split(",") if c])
    return 0


if __name__ == "__main__":
    sys.exit(main())
