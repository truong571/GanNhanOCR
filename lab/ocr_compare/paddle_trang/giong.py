"""GIÓNG đầu ra chế độ TRANG về ô kinhhannom + chấm M1/M2/M3 (không dùng hộp kinhhannom khi đọc, chỉ khi gióng).
  .venv/bin/python giong.py --raw raw_a --tag a
Đầu ra: ket_qua/<tag>_<set>.csv (từng ô), ket_qua/<tag>_trang.csv (từng trang), tong_hop_<tag>.json.
Định dạng raw JSON/trang: rec_texts, rec_scores, rec_polys, text_word (list[list[str]]), text_word_boxes (list[list[[x0,y0,x1,y1]]]).
Nếu text_word_boxes = None cho hộp nào → chia đều hộp theo chiều dọc cho các ký tự (như kinhhannom làm).
Gióng:  pred_iou  = ký tự engine có tâm nằm trong bbox ô, gần tâm ô nhất (mỗi ký tự chỉ được gán 1 ô: ô có IoU lớn nhất)
        pred_mono = DP đơn điệu theo y giữa dãy ký tự của hộp-cột engine và dãy ô kinhhannom của cột giao x nhiều nhất
Lọc rác: hộp không có ký tự CJK (số cột/số trang in máy) bị bỏ, đếm riêng; ký tự phi-CJK trong hộp CJK bị bỏ.
"""
import os, sys, json, argparse, collections, random
import numpy as np, pandas as pd
REPO = "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"
HERE = f"{REPO}/lab/ocr_compare/paddle_trang"
MAU = f"{REPO}/lab/ocr_compare/mau"
BOOK_DIR = {"stt11": "SachThanhTruyen11", "stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4"}
sys.path.insert(0, f"{REPO}/lab/ocr_compare/paddle")
from score import is_cjk, cjk_chars


def load_cache(b, p):
    return json.load(open(f"{REPO}/prepared/{BOOK_DIR[b]}/detected/{p}_ocr_cache.json"))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0: return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def mono_align(ys, centers, cell_h, gap=0.75):
    n, m = len(ys), len(centers)
    S = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): S[i][0] = i * gap
    for j in range(1, m + 1): S[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            S[i][j] = min(S[i - 1][j - 1] + abs(ys[i - 1] - centers[j - 1]) / cell_h, S[i - 1][j] + gap, S[i][j - 1] + gap)
    i, j, mp = n, m, {}
    while i > 0 and j > 0:
        if abs(S[i][j] - (S[i - 1][j - 1] + abs(ys[i - 1] - centers[j - 1]) / cell_h)) < 1e-9:
            mp[j - 1] = i - 1; i -= 1; j -= 1
        elif abs(S[i][j] - (S[i - 1][j] + gap)) < 1e-9: i -= 1
        else: j -= 1
    return mp


def engine_chars(d):
    """Trả về: chars = [(char, box[x0,y0,x1,y1], score, box_idx)], n_junk_boxes, junk_texts, col_boxes"""
    chars, junk, col_boxes = [], [], []
    tw = d.get("text_word"); twb = d.get("text_word_boxes")
    for bi, (t, s, poly) in enumerate(zip(d["rec_texts"], d["rec_scores"], d["rec_polys"])):
        P = np.array(poly, dtype=float); x0, y0, x1, y1 = P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()
        if not cjk_chars(t):
            junk.append(t); continue
        words = tw[bi] if tw and tw[bi] is not None else list(t)
        boxes = twb[bi] if twb and twb[bi] is not None else None
        if boxes is None or len(boxes) != len(words):
            # chia đều theo chiều dọc (cột) hoặc ngang
            n = len(words); vert = (y1 - y0) >= (x1 - x0)
            boxes = [[x0, y0 + k * (y1 - y0) / n, x1, y0 + (k + 1) * (y1 - y0) / n] if vert else
                     [x0 + k * (x1 - x0) / n, y0, x0 + (k + 1) * (x1 - x0) / n, y1] for k in range(n)]
        cb = []
        for w, bx in zip(words, boxes):
            w = cjk_chars(w)
            if not w: continue
            for c in w:  # nếu một "word" gộp nhiều CJK thì cùng hộp
                cb.append((c, [float(v) for v in bx], float(s), bi))
        chars += cb
        col_boxes.append(dict(idx=bi, box=[x0, y0, x1, y1], n=len(cb), chars=cb, text=t))
    return chars, junk, col_boxes


def align_page(d, cache):
    chars, junk, col_boxes = engine_chars(d)
    cells = []  # (col_idx, cell_idx, bbox, char)
    for ci, col in enumerate(cache["columns"]):
        for k, c in enumerate(col):
            cells.append((ci, k, c["bbox"], c["char"]))
    # ---- gióng IoU/tâm: mỗi ký tự engine → ô có IoU lớn nhất (>0) ; ô nhận nhiều ký tự → lấy ký tự gần tâm nhất
    cell_hits = collections.defaultdict(list)
    for ch, bx, sc, bi in chars:
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        best, bv = None, 0.0
        for j, (ci, k, bb, _) in enumerate(cells):
            v = iou(bx, bb)
            if v > bv: bv, best = v, j
        if best is None:  # thử tâm trong hộp
            for j, (ci, k, bb, _) in enumerate(cells):
                if bb[0] <= cx <= bb[2] and bb[1] <= cy <= bb[3]: best = j; break
        if best is not None:
            bb = cells[best][2]
            dist = abs(cy - (bb[1] + bb[3]) / 2) + abs(cx - (bb[0] + bb[2]) / 2)
            cell_hits[best].append((dist, ch, sc))
    pred_iou = {}
    for j, hs in cell_hits.items():
        hs.sort(); pred_iou[j] = dict(pred=hs[0][1], score=hs[0][2], all="".join(h[1] for h in hs))
    # ---- gióng đơn điệu: hộp-cột engine (n>=3 ký tự, dọc) ↔ cột kinhhannom giao x nhiều nhất
    pred_mono = {}
    kcols = []
    for ci, col in enumerate(cache["columns"]):
        if not col: kcols.append(None); continue
        xs0 = min(c["bbox"][0] for c in col); xs1 = max(c["bbox"][2] for c in col)
        kcols.append((xs0, xs1))
    n_col_detected = 0
    col_map = {}
    for cb in col_boxes:
        x0, y0, x1, y1 = cb["box"]
        if cb["n"] < 3 or (y1 - y0) < 2 * (x1 - x0): continue
        n_col_detected += 1
        best, bv = None, 0
        for ci, kc in enumerate(kcols):
            if kc is None: continue
            ov = max(0, min(x1, kc[1]) - max(x0, kc[0]))
            if ov > bv: bv, best = ov, ci
        if best is None: continue
        col_map.setdefault(best, []).append(cb)
    for ci, cbs in col_map.items():
        col = cache["columns"][ci]
        cents = [(c["bbox"][1] + c["bbox"][3]) / 2 for c in col]
        ch_h = max(1.0, col[0]["bbox"][3] - col[0]["bbox"][1])
        allc = sorted([c for cb in cbs for c in cb["chars"]], key=lambda c: (c[1][1] + c[1][3]) / 2)
        ys = [(c[1][1] + c[1][3]) / 2 for c in allc]
        mp = mono_align(ys, cents, ch_h)
        for k, i in mp.items():
            j = cells.index((ci, k, col[k]["bbox"], col[k]["char"]))
            pred_mono[j] = dict(pred=allc[i][0], score=allc[i][2])
    n_kinh = len(cells)
    stats = dict(n_boxes=len(d["rec_texts"]), n_junk_boxes=len(junk), junk="|".join(junk), n_col_boxes=n_col_detected,
                 n_kinh_cols=len(cache["columns"]), n_kcols_hit=len(col_map), n_chars=len(chars), n_kinh=n_kinh,
                 n_cells_hit_iou=len(pred_iou), n_cells_hit_mono=len(pred_mono), sec=d.get("sec", None))
    return cells, pred_iou, pred_mono, stats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", default="raw_a"); ap.add_argument("--tag", default="a")
    a = ap.parse_args()
    os.makedirs(f"{HERE}/ket_qua", exist_ok=True)
    pages = {}
    page_rows = []
    sets = {s: pd.read_csv(f"{MAU}/{s}.csv", dtype=str, keep_default_na=False) for s in ["M1", "M2", "M3"]}
    need = sorted(set().union(*[set(zip(m.book, m.page)) for m in sets.values()]))
    for b, p in need:
        f = f"{HERE}/{a.raw}/{b}_{p}.json"
        if not os.path.exists(f): continue
        d = json.load(open(f)); cache = load_cache(b, p)
        cells, pi, pm, st = align_page(d, cache)
        pages[(b, p)] = (cells, pi, pm)
        page_rows.append(dict(book=b, page=p, **st))
    pdf = pd.DataFrame(page_rows); pdf.to_csv(f"{HERE}/ket_qua/{a.tag}_trang.csv", index=False)
    print("pages aligned", len(pdf))
    freq = collections.Counter(pd.read_csv(f"{REPO}/dataset_out/labels.csv", dtype=str, keep_default_na=False).ocr_char)
    tot = sum(freq.values())
    summary = dict(tag=a.tag, n_pages=len(pdf))
    if len(pdf):
        summary["trang"] = dict(
            sec_per_page=float(pd.to_numeric(pdf.sec, errors="coerce").mean()),
            n_boxes_mean=float(pdf.n_boxes.mean()), n_junk_boxes_mean=float(pdf.n_junk_boxes.mean()),
            n_col_boxes_mean=float(pdf.n_col_boxes.mean()), pct_pages_9_col_boxes=float((pdf.n_col_boxes == 9).mean()),
            n_kcols_hit_mean=float(pdf.n_kcols_hit.mean()), pct_pages_all9_kcols_hit=float((pdf.n_kcols_hit == pdf.n_kinh_cols).mean()),
            n_chars_mean=float(pdf.n_chars.mean()), n_kinh_mean=float(pdf.n_kinh.mean()),
            chars_ratio=float(pdf.n_chars.sum() / pdf.n_kinh.sum()),
            cells_hit_iou_ratio=float(pdf.n_cells_hit_iou.sum() / pdf.n_kinh.sum()),
            cells_hit_mono_ratio=float(pdf.n_cells_hit_mono.sum() / pdf.n_kinh.sum()),
            col_box_hist={int(k): int(v) for k, v in pdf.n_col_boxes.value_counts().sort_index().items()},
            junk_examples=list(pdf.junk.head(5)),
        )
    rng = random.Random(20260911)
    for s, m in sets.items():
        rows = []
        for _, r in m.iterrows():
            key = (r.book, r.page)
            if key not in pages:
                rows.append(dict(sample_id=r.sample_id, pred_iou="", pred_mono="", pred_all="", note="no_page")); continue
            cells, pi, pm = pages[key]
            bb = json.loads(r.bbox); ci = int(r.column) - 1
            cand = [j for j, c in enumerate(cells) if c[0] == ci]
            if not cand:
                rows.append(dict(sample_id=r.sample_id, pred_iou="", pred_mono="", pred_all="", note="no_col")); continue
            j = min(cand, key=lambda j: abs((cells[j][2][1] + cells[j][2][3]) / 2 - (bb[1] + bb[3]) / 2))
            rows.append(dict(sample_id=r.sample_id, pred_iou=pi.get(j, {}).get("pred", ""), score_iou=pi.get(j, {}).get("score", ""),
                             pred_mono=pm.get(j, {}).get("pred", ""), pred_all=pi.get(j, {}).get("all", ""),
                             cell_char=cells[j][3], note="ok" if cells[j][3] == r.ocr_char else "cell_char_mismatch"))
        res = pd.DataFrame(rows); res.to_csv(f"{HERE}/ket_qua/{a.tag}_{s}.csv", index=False)
        d = m.merge(res, on="sample_id")
        R = d.R_candidates.map(lambda x: set(v for v in x.split("|") if v))
        p_freq = float(np.mean([sum(freq[c] for c in r) / tot for r in R]))
        out = dict(n=len(d), n_page_missing=int((d.note == "no_page").sum()), cell_mismatch=int((d.note == "cell_char_mismatch").sum()),
                   p_random_freq=p_freq)
        for col in ["pred_iou", "pred_mono"]:
            pr = d[col].fillna("")
            hit = pr != ""
            inR = np.array([p != "" and p in r for p, r in zip(pr, R)])
            eqk = (pr == d.ocr_char).values
            o = dict(coverage=float(hit.mean()), in_R=float(inR.mean()), in_R_given_hit=float(inR[hit].mean()) if hit.any() else None,
                     eq_kinh=float(eqk.mean()), n_in_R=int(inR.sum()))
            # hoán vị 500 lần
            perms = []
            vals = list(pr)
            for _ in range(500):
                rng.shuffle(vals); perms.append(np.mean([p != "" and p in r for p, r in zip(vals, R)]))
            o["perm_in_R"] = float(np.mean(perms)); o["lift_perm"] = o["in_R"] / o["perm_in_R"] if o["perm_in_R"] else None
            o["lift_freq"] = o["in_R"] / p_freq
            if s == "M2":
                o["rescue"] = f"{int(inR.sum())}/{len(d)}"
                o["by_tier"] = {t: dict(n=int(len(g)), in_R=int(inR[g.index].sum()), rate=float(inR[g.index].mean())) for t, g in d.groupby("tier")}
                o["rescue_eq_pipeline_label_SILVER"] = int(((pr == d.label) & inR & (d.tier == "SILVER")).sum())
                o["eq_kinh_both_notR"] = int(eqk.sum())
            if s == "M3":
                cnt = collections.Counter(pr)
                o["top"] = cnt.most_common(10); o["n_𠊚"] = int((pr == "𠊚").sum()); o["n_㝵"] = int((pr == "㝵").sum())
            out[col] = o
        summary[s] = out
    json.dump(summary, open(f"{HERE}/tong_hop_{a.tag}.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
