"""TN6 x06 — TÍN HIỆU GIÁN TIẾP (không cần sự thật) cho vị trí hộp của một bản dựng: bộ kiểm thị giác m_win (chép ý
pipeline.gold_exact.signals_img.score_vis, WIN=2): m = s(nhãn) − max s(chữ kim/nhãn của ô lân cận n±1..2 cùng cột), s = cos(nhúng
crop [v1,v2], glyph font/FD); hộp trượt ±1 ô thì chữ láng giềng thắng (m < 0). Crop cắt lại từ ảnh trang theo bbox (pad 0,12 +
tighten, như borg_human.geom.crop_gray) cho MỌI dòng có bbox — không phụ thuộc tầng/ảnh đã lưu.
CẢNH BÁO THIÊN LỆCH: visual_dp chọn hộp bằng CHÍNH họ encoder + glyph R(âm) ⇒ m_win THIÊN VỀ visual_dp (SUY ĐOÁN, không phải sự
thật); kiểm độ tin của tín hiệu trên Borg (có hộp người) bằng --truth.
Thêm: tỉ lệ bbox trùng (cùng trang) và cờ crop_quality_flag theo tầng (độc lập encoder).

  .venv/bin/python lab/thu_nghiem_anh_chu/TN6_hop_anh/x06_indirect.py --book Chrestomathie1872 --variant current --data prepared/Chrestomathie1872
"""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).parent))


def labels_path(book, variant):
    if variant == "current":
        for p in (REPO / f"prepared/_auto/{book}/dataset_out/labels_gated.csv", REPO / f"prepared/{book}/dataset_out/labels_gated.csv",
                  REPO / "dataset_out/labels_gated.csv"):
            if p.exists():
                return p
    return REPO / f"measure_out/_tn6/{book}/{variant}/dataset_out/labels_gated.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--data", required=True, help="thư mục prepared của sách (pages/*.png)")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--book-code", default="", help="lọc cột book (STT: stt2|stt4|stt11)")
    ap.add_argument("--out-tag", default="", help="hậu tố tên tệp kết quả (STT: theo sách)")
    ap.add_argument("--truth", action="store_true", help="Borg: ghép hộp người để đo độ tin của m_win")
    a = ap.parse_args()
    from pipeline.borg_human.encoders import Glyphs, MultiEnc, device
    from pipeline.borg_human.geom import crop_gray
    lab = (REPO / a.labels) if a.labels else labels_path(a.book, a.variant)
    L = pd.read_csv(lab, dtype=str, keep_default_na=False)
    if a.book_code:            # bộ nhiều sách (STT): chỉ các dòng của một sách (cột book = stt2/stt4/stt11)
        L = L[L.book == a.book_code].copy()
    L = L[L.bbox.str.startswith("[")].reset_index(drop=True)
    enc = MultiEnc(["v1", "v2"], device("auto"), True)
    E = np.zeros((len(L), 512), np.float32)
    data = REPO / a.data
    for pg, idx in L.groupby("page").groups.items():
        g = cv2.imread(str(data / "pages" / f"{pg}.png"), cv2.IMREAD_GRAYSCALE)
        crops = []
        for i in idx:
            b = json.loads(L.bbox[i])
            crops.append(crop_gray(g, dict(x1=b[0], y1=b[1], x2=b[2], y2=b[3])))
        E[list(idx)] = enc.embed(crops)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    # ngữ cảnh: (page, column, nom_idx) -> (ocr_char, label)
    L["ni"] = pd.to_numeric(L.nom_idx, errors="coerce").fillna(-99).astype(int)
    ctx = {(p, c, n): (o, l) for p, c, n, o, l in zip(L.page, L.column, L.ni, L.ocr_char, L.label)}
    need = set()
    wins = []
    for p, c, n, lb in zip(L.page, L.column, L.ni, L.label):
        w = set()
        for k in range(n - 2, n + 3):
            if k == n:
                continue
            for ch in ctx.get((p, c, k), ("", "")):
                if len(ch) == 1 and ch != lb:
                    w.add(ch)
        wins.append(w); need |= w
        if len(lb) == 1:
            need.add(lb)
    G = Glyphs()
    chars = sorted(need)
    imgs, keys = [], []
    for ch in chars:
        for im in (G.font(ch), G.fdimg(ch)):
            if im is not None:
                imgs.append(im); keys.append(ch)
    GE = enc.embed(imgs).astype(np.float32); GE /= np.linalg.norm(GE, axis=1, keepdims=True) + 1e-8
    ref = {}
    for ch, v in zip(keys, GE):
        ref.setdefault(ch, []).append(v)
    m = np.full(len(L), np.nan)
    for i, (lb, W) in enumerate(zip(L.label, wins)):
        if lb not in ref or not W:
            continue
        s = lambda ch: float(np.mean([E[i] @ v for v in ref[ch]]))
        others = [s(c) for c in W if c in ref]
        if others:
            m[i] = s(lb) - max(others)
    L["m_win"] = m
    L["dup_bbox"] = L.groupby(["page", "bbox"]).bbox.transform("count") > 1
    out = dict(book=a.book, variant=a.variant, labels=str(lab.relative_to(REPO)), n=int(len(L)))
    for t in ("ALL", "GOLD", "SYLLABLE", "REVIEW"):
        s = L if t == "ALL" else L[L.tier == t]
        mm = s.m_win.dropna()
        out[t] = dict(n=int(len(s)), n_m=int(len(mm)), win=round(float((mm > 0).mean()), 4) if len(mm) else None,
                      m_med=round(float(mm.median()), 4) if len(mm) else None, dup_bbox=round(float(s.dup_bbox.mean()), 4) if len(s) else None,
                      cq=s.crop_quality_flag.value_counts(normalize=True).round(4).to_dict() if "crop_quality_flag" in s else None)
    if a.truth:
        import tn6lib as T
        J = T.join(a.book, L)
        k = J[J.has_h & J.keep_level.isin(T.KEEP_OK) & J.m_win.notna()]
        from scipy.stats import rankdata

        def roc_auc_score(y, x):
            y = np.asarray(y, bool); r = rankdata(np.asarray(x, float)); n1 = y.sum(); n0 = len(y) - n1
            return (r[y].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)
        out["truth"] = dict(n=int(len(k)), slot=round(float(k.slot_ok.mean()), 4), win=round(float((k.m_win > 0).mean()), 4),
                            auc_mwin_slot=round(float(roc_auc_score(k.slot_ok, k.m_win)), 4) if k.slot_ok.nunique() > 1 else None,
                            slot_given_win=round(float(k[k.m_win > 0].slot_ok.mean()), 4),
                            slot_given_lose=round(float(k[k.m_win <= 0].slot_ok.mean()), 4) if (k.m_win <= 0).any() else None)
    od = REPO / f"measure_out/_tn6/{a.book}/{a.variant}/eval"; od.mkdir(parents=True, exist_ok=True)
    L[["page", "column", "nom_idx", "syl_idx", "tier", "label", "bbox", "m_win", "dup_bbox"]].to_pickle(od / f"indirect{a.out_tag}.pkl")
    json.dump(out, open(od / f"tn6_indirect{a.out_tag}.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(out, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
