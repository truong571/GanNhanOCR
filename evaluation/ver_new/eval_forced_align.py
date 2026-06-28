"""TEST ý tưởng FORCED-ALIGNMENT: thay vì cắt crop rồi so, ta TRƯỢT glyph của chữ
KỲ VỌNG (từ QN) dọc cột để tìm vị trí khớp nhất → lấy cửa sổ đó làm crop. Né lỗi
'dính chữ hàng xóm' của cắt-mù.

Hiệu quả: KHÔNG duyệt cả folder 89k glyph. Mỗi vị trí chỉ có vài chữ ứng viên (QN
gợi ý) → chỉ so ~N glyph/cột. Thuật toán:
  - cắt vùng cột; chia lưới vị trí top (bước nhỏ); EMBED mỗi cửa sổ 1 LẦN (cache).
  - score[i][y] = max cosine(window_y, glyph ứng_viên của vị trí i).
  - DP đơn điệu (chữ dưới phải nằm dưới chữ trên) tối đa tổng score → N hộp.
So với midpoint (A) trên CÙNG cột diverged: two_blob / MLS / match-cosine.

Run:
  .venv/bin/python evaluation/ver_new/eval_forced_align.py --limit 40
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from evaluation.ver_new.align_production import _detect, _reseg_column  # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402
from evaluation.ver_new.seg_valley_n_ab import two_blob, metrics   # noqa: E402
from evaluation.ver_new.bbox_fix import tighten_box                # noqa: E402

HERE = Path(__file__).resolve().parent
FD = REPO / "gannhanocr-fd"


def fd_path(ch):
    hx = f"{ord(ch):X}"
    for q in (FD / f"U+{hx}.png", FD / hx[:2] / f"U+{hx}.png"):
        if q.exists():
            return q
    return None


def forced_align(col_gray, N, cand_embs, enc):
    """col_gray: HxW vùng cột. cand_embs: list theo vị trí, mỗi phần tử = mảng (k,256)
    embedding các glyph ứng viên (có thể rỗng). Trả N (y_top, h) trong toạ độ cột."""
    Hc, Wc = col_gray.shape
    ch = max(12, Hc // N)
    if Hc - ch < 1:
        return [(int(i * Hc / N), ch) for i in range(N)]
    stride = max(2, ch // 5)
    grid = list(range(0, Hc - ch + 1, stride))
    # embed mỗi cửa sổ lưới 1 lần
    g_emb = {}
    for y in grid:
        w = col_gray[y:y + ch]
        t = tighten_box(w)
        if t is not None:
            a, c, b, d = t
            if b - a >= 8 and d - c >= 8:
                w = w[c:d, a:b]
        if w.size and (w < 128).mean() > 0.02:
            g_emb[y] = enc.embed_gray(w)
    # score[i][gy]
    G = len(grid)
    score = np.full((N, G), -1.0)
    for i in range(N):
        E = cand_embs[i]
        for j, y in enumerate(grid):
            if y not in g_emb:
                continue
            if E is not None and len(E):
                score[i][j] = float(np.max(E @ g_emb[y]))
            else:
                score[i][j] = 0.0
    # DP đơn điệu: chữ i+1 top >= chữ i top + 0.6*ch
    min_step = int(0.6 * ch)
    NEG = -1e9
    dp = np.full((N, G), NEG); back = np.full((N, G), -1, int)
    for j in range(G):
        dp[0][j] = score[0][j]
    for i in range(1, N):
        for j in range(G):
            if score[i][j] <= -1.0:
                continue
            # vị trí trước phải <= grid[j]-min_step
            best, bk = NEG, -1
            for jp in range(j):
                if grid[j] - grid[jp] >= min_step and dp[i - 1][jp] > best:
                    best, bk = dp[i - 1][jp], jp
            if bk >= 0:
                dp[i][j] = best + score[i][j]; back[i][j] = jp if False else bk
    # backtrack
    jend = int(np.argmax(dp[N - 1]))
    if dp[N - 1][jend] <= NEG / 2:
        return [(int(i * Hc / N), ch) for i in range(N)], 0.0
    path = [jend]
    for i in range(N - 1, 0, -1):
        path.append(back[i][path[-1]])
    path = path[::-1]
    boxes = [(grid[j], ch) for j in path]
    avg = float(np.mean([score[i][path[i]] for i in range(N) if score[i][path[i]] > -1.0]))
    return boxes, avg


def valley_glyph_dp(col_gray, N, cand_embs, enc, cand_asp=None, lam=0.0):
    """#3: điểm cắt ứng viên = KHE MỰC thật (valley); DP chọn N đoạn sao cho mỗi đoạn
    khớp glyph kỳ vọng nhất. Kết hợp ưu điểm valley (cắt sạch, không vỡ) + danh-tính
    glyph (chọn đúng khe). Trả [(top,h)], avg-cosine.
    lam>0 + cand_asp: HEIGHT-PRIOR — phạt đoạn lệch chiều cao kỳ vọng (tỉ-lệ glyph × Wc)."""
    Hc, Wc = col_gray.shape
    ch = max(12, Hc // N)
    ink = (col_gray < 128).sum(axis=1).astype(float)
    k = max(3, Hc // 40) | 1
    ink_s = np.convolve(ink, np.ones(k) / k, mode="same")
    # valley = cực tiểu địa phương của mực
    val = [y for y in range(1, Hc - 1) if ink_s[y] <= ink_s[y - 1] and ink_s[y] < ink_s[y + 1]]
    if len(val) > 3 * N:                                   # giữ 3N khe sâu nhất (thấp mực nhất)
        val = sorted(sorted(val, key=lambda y: ink_s[y])[: 3 * N])
    B = sorted(set([0] + val + [Hc]))
    m = len(B)
    if m < N + 1:                                          # không đủ khe -> fallback cửa sổ đều
        return [(int(i * Hc / N), ch) for i in range(N)], 0.0
    seg_emb = {}
    def seg_score(p, q, i):
        h = B[q] - B[p]
        if h < 0.45 * ch or h > 1.8 * ch:                  # chiều cao 1-chữ hợp lý
            return -1.0
        if (p, q) not in seg_emb:
            w = col_gray[B[p]:B[q]]
            t = tighten_box(w)
            if t is not None:
                a, c, b, d = t
                if b - a >= 8 and d - c >= 8:
                    w = w[c:d, a:b]
            seg_emb[(p, q)] = enc.embed_gray(w) if (w.size and (w < 128).mean() > 0.02) else None
        e = seg_emb[(p, q)]
        if e is None:
            return -1.0
        E = cand_embs[i]
        if E is None or not len(E):
            return 0.0
        cosines = E @ e
        if lam > 0 and cand_asp is not None and cand_asp[i] is not None:
            h_exp = np.maximum(cand_asp[i] * Wc, 1.0)        # cao kỳ vọng = tỉ-lệ glyph × bề-rộng-cột
            cosines = cosines - lam * np.abs(h - h_exp) / h_exp
        return float(np.max(cosines))

    NEG = -1e9
    dp = np.full((N, m), NEG); back = np.full((N, m), -1, int)
    for q in range(1, m):
        s = seg_score(0, q, 0)
        if s > -1.0:
            dp[0][q] = s
    for i in range(1, N):
        for q in range(i + 1, m):
            best, bk = NEG, -1
            for p in range(i, q):
                if dp[i - 1][p] <= NEG / 2:
                    continue
                s = seg_score(p, q, i)
                if s <= -1.0:
                    continue
                if dp[i - 1][p] + s > best:
                    best, bk = dp[i - 1][p] + s, p
            dp[i][q] = best; back[i][q] = bk
    if dp[N - 1][m - 1] <= NEG / 2:
        return [(int(i * Hc / N), ch) for i in range(N)], 0.0
    ends = [0] * N; ends[N - 1] = m - 1
    for i in range(N - 1, 0, -1):
        ends[i - 1] = back[i][ends[i]]
    boxes, sb, scores = [], 0, []
    for i in range(N):
        eb = ends[i]
        boxes.append((B[sb], B[eb] - B[sb]))
        scores.append(seg_score(sb, eb, i)); sb = eb
    return boxes, float(np.mean([s for s in scores if s > -1.0]) if scores else 0.0)


def hill_climb(col_gray, boxes, cand_embs, enc, max_px=6):
    """Box Adjuster không-train (ECCV'22 ý tưởng): trượt biên trên/dưới ±max_px px để
    TỐI ĐA cosine với glyph kỳ vọng → tự gạt mẩu chữ hàng xóm (glyph sạch cosine cao hơn)."""
    Hc, Wc = col_gray.shape
    nb = max(len(boxes), 1)
    minh = max(8, int(0.4 * (Hc / nb)))
    out = []
    for i, (top, h) in enumerate(boxes):
        E = cand_embs[i] if i < len(cand_embs) else None
        t0, b0 = int(top), int(top + h)
        if E is None or not len(E) or b0 - t0 < 8:
            out.append((top, h)); continue
        def sc(t, b):
            t, b = max(0, t), min(Hc, b)
            if b - t < minh:
                return -1e9
            w = col_gray[t:b]
            tb = tighten_box(w)
            if tb is not None:
                a, c, bb, d = tb
                if bb - a >= 8 and d - c >= 8:
                    w = w[c:d, a:bb]
            if not w.size or (w < 128).mean() <= 0.02:
                return -1e9
            return float(np.max(E @ enc.embed_gray(w)))
        ct, cb = t0, b0; cur = sc(ct, cb)
        for _ in range(max_px):
            bm, best = None, cur
            for nt, nb_ in [(ct - 1, cb), (ct + 1, cb), (ct, cb - 1), (ct, cb + 1)]:
                if abs(nt - t0) <= max_px and abs(nb_ - b0) <= max_px:
                    s = sc(nt, nb_)
                    if s > best:
                        best, bm = s, (nt, nb_)
            if bm is None:
                break
            ct, cb = bm; cur = best
        out.append((ct, cb - ct))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40, help="pages per book")
    args = ap.parse_args()
    cfg = load_config(str(REPO / "config" / "pipeline.yaml")); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"])); qn_set = set(qn.keys())
    data_root = REPO / paths["data_dir"]
    enc = VisualS3(REPO, fd_dir="").enc

    LAM = 0.3                                     # trọng số height-prior
    glyph_cache, asp_cache = {}, {}
    def glyph_emb(ch):
        if ch not in glyph_cache:
            p = fd_path(ch)
            glyph_cache[ch] = enc.embed_path(str(p)) if p else None
        return glyph_cache[ch]
    def glyph_asp(ch):                            # tỉ-lệ cao/rộng glyph (đã siết ink)
        if ch not in asp_cache:
            p = fd_path(ch); a = None
            if p:
                g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    tb = tighten_box(g)
                    if tb is not None:
                        x0, y0, x1, y1 = tb
                        if x1 - x0 >= 4 and y1 - y0 >= 4:
                            a = (y1 - y0) / (x1 - x0)
            asp_cache[ch] = a
        return asp_cache[ch]

    TAGS = ("A", "F", "V", "VP", "H")
    agg = {t: defaultdict(float) for t in TAGS}
    nb = {t: 0 for t in TAGS}; n_div = 0; cos_sum = 0.0; cos_n = 0
    for b in cfg["books"]:
        data_dir = data_root / b["name"]
        trans = [t for t in sorted(glob.glob(str(data_dir / "transcriptions" / "page_*.json")))
                 if not t.endswith("_qn_ocr_cache.json")]
        if args.limit:
            trans = trans[: args.limit]
        for tf in trans:
            page = Path(tf).stem
            try:
                det = _detect(page, data_dir, qn_set)
            except Exception:
                det = None
            if not det:
                continue
            cols, qn_lines, iter_pairs, binary, _ = det
            page_bgr = cv2.imread(str(data_dir / "pages" / f"{page}.png"), cv2.IMREAD_COLOR)
            if page_bgr is None:
                continue
            page_gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
            for nom_idx, line_id in iter_pairs:
                cluster = cols[nom_idx]; syl = qn_lines[line_id]
                if not syl or not cluster.get("chars"):
                    continue
                N = len(syl)
                if N < 2 or len(cluster["chars"]) == N:
                    continue                       # bỏ cột không diverged
                chars = cluster["chars"]
                if cluster.get("x_range"):
                    cx1, cx2 = int(cluster["x_range"][0]), int(cluster["x_range"][1])
                else:
                    cx1 = min(int(c["bbox"][0]) for c in chars); cx2 = max(int(c["bbox"][2]) for c in chars)
                cy1 = min(int(c["bbox"][1]) for c in chars); cy2 = max(int(c["bbox"][3]) for c in chars)
                if cx2 - cx1 < 10 or cy2 - cy1 < 20:
                    continue
                # ứng viên glyph mỗi vị trí (từ QN). Bỏ cột thiếu glyph quá nửa.
                cand, cand_asp = [], []
                have = 0
                for i in range(N):
                    s = str(syl[i]).strip().lower()
                    embs, asps = [], []
                    for c in qn.get(s, []):
                        if _is_cjk(c):
                            e = glyph_emb(c)
                            if e is not None:
                                embs.append(e); asps.append(glyph_asp(c) or 1.1)
                    cand.append(np.stack(embs) if embs else None)
                    cand_asp.append(np.array(asps) if asps else None)
                    have += bool(embs)
                if have < max(2, N // 2):
                    continue
                n_div += 1
                col_gray = page_gray[cy1:cy2, cx1:cx2]
                fboxes, avgcos = forced_align(col_gray, N, cand, enc)
                cos_sum += avgcos; cos_n += 1
                # F boxes -> page coords
                F = [(cx1, cy1 + y, cx2, cy1 + y + h) for (y, h) in fboxes]
                vboxes, _ = valley_glyph_dp(col_gray, N, cand, enc)                       # V: base
                V = [(cx1, cy1 + y, cx2, cy1 + y + h) for (y, h) in vboxes]
                vpboxes, _ = valley_glyph_dp(col_gray, N, cand, enc, cand_asp, LAM)        # VP: + height-prior
                VP = [(cx1, cy1 + y, cx2, cy1 + y + h) for (y, h) in vpboxes]
                hboxes = hill_climb(col_gray, vpboxes, cand, enc)                          # H: VP + edge hill-climb
                Hb = [(cx1, cy1 + y, cx2, cy1 + y + h) for (y, h) in hboxes]
                A = _reseg_column(cluster) or []
                for tag, boxes in (("A", A), ("F", F), ("V", V), ("VP", VP), ("H", Hb)):
                    for bx in boxes:
                        m = metrics(page_bgr, bx, enc)
                        if m is None:
                            continue
                        nb[tag] += 1
                        for k in ("tall", "two_blob", "mls"):
                            agg[tag][k] += m[k]
        print(f"  [{b['name']}] diverged cols dùng được {n_div}", flush=True)

    def av(t, k): return agg[t][k] / max(nb[t], 1)
    print("\n" + "=" * 78)
    print(f" 5 lớp trên {n_div} cột diverged  (A=midpoint  F=forced-align  V=valley+glyph-DP")
    print("                                  VP=V+height-prior  H=VP+edge-hill-climb)")
    print("=" * 78)
    print(f"  {'':9s}   A       F       V      VP       H")
    for k in ("two_blob", "tall", "mls"):
        print(f"  {k:9s} " + "   ".join(f"{av(t,k):.3f}" for t in TAGS))
    print(f"  boxes:    " + "   ".join(f"{nb[t]:>5d}" for t in TAGS))
    tb = [(av(t, "two_blob"), t) for t in TAGS]; ml = [(av(t, "mls"), t) for t in TAGS]
    print(f"\n  >>> two_blob THẤP nhất: {min(tb)[1]} ({min(tb)[0]:.3f}) | MLS CAO nhất: {max(ml)[1]} ({max(ml)[0]:.3f})")
    print(f"  >>> tiến triển two_blob: A {av('A','two_blob'):.3f} → V {av('V','two_blob'):.3f} "
          f"→ VP {av('VP','two_blob'):.3f} → H {av('H','two_blob'):.3f}")
    import json
    (HERE / "results").mkdir(exist_ok=True)
    json.dump({"diverged_cols": n_div,
               **{t: {k: av(t, k) for k in ("two_blob", "tall", "mls")} for t in TAGS},
               "F_match_cosine": cos_sum/max(cos_n, 1)},
              open(HERE / "results" / "eval_forced_align.json", "w"), indent=2)
    print("  -> results/eval_forced_align.json")


if __name__ == "__main__":
    main()
