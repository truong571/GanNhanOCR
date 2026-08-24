"""T4-A — quét cấu hình CẮT CROP trên bộ hộp CỐ ĐỊNH, đo bằng họ D (không cần nhãn).

VÌ SAO TÁCH RA KHỎI "144 CẤU HÌNH" CỦA ĐẶC TẢ
---------------------------------------------
Đặc tả T4.2 nhân 5 trục thành 144 cấu hình. Nhưng 4 trục đầu (pad · ngưỡng siết ·
carve · resolve) là HẬU-PHÁT-HIỆN: chúng nhận cùng một bộ hộp rồi cắt khác nhau, nên
so sánh được HÀNG-ĐỐI-HÀNG. Trục thứ 5 (bộ tách) đổi CHÍNH BỘ HỘP → đổi số ký tự →
đổi căn chỉnh → đổi bộ nhãn; không có phép so hàng-đối-hàng nào hợp lệ. Gộp chung 144
là trộn hai đại lượng khác bản chất.

Nên T4 chia hai:
  * T4-A (tệp này)      4 × 3 × 2 × 2 = **48** cấu hình, bộ hộp cố định, đo họ D.
  * T4-B (crop_iou.py)  3 bộ tách, đo IoU THẬT trên ngữ liệu tổng hợp có đáp án hộp.

MỐC THẬT (đo 2026-08-24, KHÁC đặc tả)
-------------------------------------
`config/pipeline.yaml:70` khai `crop_pad_frac: 0.18`, nhưng `build_dataset` KHÔNG đọc
dòng đó: `--pad` default **0.12** và `run_pipeline.sh` không truyền. Toàn bộ crop đã
giao cắt ở **pad = 0,12**. Đặc tả T4.2 in đậm 0,18 là "hiện tại" — sai mốc.
`resolve_overlap` cũng là MÃ CHẾT: đường build chưa từng gọi.

CẠM BẪY SUY BIẾN (bài học T3)
-----------------------------
`flag_ok` = (stray <= 0,08) & (border <= 0,20) & (ink >= 0,05). Nếu tăng pad làm
border giảm mà không làm gì tăng, thì trục pad SUY BIẾN: tối ưu flag_ok chỉ là chọn
pad lớn nhất, đúng lớp lỗi đã bác bỏ bảng T3 lần đầu. Nên `--diagnose` PHẢI chạy
trước và phải chứng minh có LỰC KÉO NGƯỢC (ink_pct giảm khi pad tăng), nếu không thì
kết quả không được dùng để chốt cấu hình.

    python -m pipeline.lab.crop_grid --diagnose            # kiểm trục trước
    python -m pipeline.lab.crop_grid --pages 60            # quét 48 cấu hình
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LABELS = REPO / "dataset_out" / "labels_final.csv"
PREPARED = REPO / "prepared"
OUT = REPO / "lab" / "t4a_grid.csv"
BOOKS = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4",
         "stt11": "SachThanhTruyen11"}

# ---- lưới 48 cấu hình -------------------------------------------------------
PADS = (0.12, 0.15, 0.18, 0.22)            # 0,12 = MỐC THẬT (không phải 0,18)
THRS = ("fixed128", "otsu", "sauvola")     # ngưỡng siết hộp
CARVE = (True, False)
RESOLVE = (False, True)                    # False = MỐC (mã chết trong production)
BASELINE = {"pad": 0.12, "thr": "fixed128", "carve": True, "resolve": False}

_CHAR_IDX = re.compile(r"_c(\d+)_(\d+)\.png$")


def _img_key(image: str) -> tuple[int, int]:
    """(chỉ số cột, chỉ số ký tự trong cột) từ tên tệp crop."""
    m = _CHAR_IDX.search(image or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def load_boxes(labels: Path = LABELS, n_pages: int = 60, seed: int = 20260824):
    """Bộ hộp CỐ ĐỊNH, giữ nguyên NGUYÊN CỘT (carve cần hộp trước/sau cùng cột).

    Lấy mẫu theo TRANG chứ không theo hàng: cắt một crop cần hộp láng giềng, nên
    lấy mẫu hàng lẻ sẽ phá ngữ cảnh cột và làm trục `carve` mất tác dụng giả tạo.
    """
    import pandas as pd
    df = pd.read_csv(labels, usecols=["image", "book", "page", "column", "bbox", "tier"])
    df = df[df["bbox"].notna()]
    pages = sorted({(b, p) for b, p in zip(df["book"], df["page"])})
    if n_pages and n_pages < len(pages):
        # tất định, phủ đều 3 sách: bước nhảy đều trên danh sách đã sắp
        step = len(pages) / n_pages
        pages = [pages[int(i * step)] for i in range(n_pages)]
    keep = set(pages)
    df = df[[(b, p) in keep for b, p in zip(df["book"], df["page"])]]

    cols: dict[tuple, list] = {}
    for r in df.itertuples(index=False):
        ci, xi = _img_key(r.image)
        cols.setdefault((r.book, r.page, ci), []).append(
            (xi, json.loads(r.bbox), r.tier))
    out = []
    for (book, page, ci), items in sorted(cols.items()):
        items.sort(key=lambda t: t[0])
        out.append({"book": book, "page": page, "col": ci,
                    "bboxes": [b for _, b, _ in items],
                    "tiers": [t for _, _, t in items]})
    return out, len(pages)


def _tighten(gray, mode: str):
    """Siết hộp về mực, với 3 chế độ ngưỡng."""
    from pipeline.align_engine.bbox_fix import tighten_box
    import numpy as np
    if mode == "fixed128":
        return tighten_box(gray)
    if mode == "otsu":
        import cv2
        if gray.size == 0 or gray.std() < 1e-6:
            return tighten_box(gray)
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return tighten_box(gray, thr=int(max(1, min(254, t))))
    if mode == "sauvola":
        from skimage.filters import threshold_sauvola
        h, w = gray.shape[:2]
        win = max(3, (min(h, w) // 2) * 2 + 1)      # lẻ, <= cạnh ngắn
        try:
            loc = threshold_sauvola(gray, window_size=win, k=0.2)
        except Exception:
            return tighten_box(gray)
        # Sauvola cho ngưỡng CỤC BỘ; tighten_box nhận ngưỡng vô hướng, nên nhị
        # phân hoá trước rồi đưa ảnh 0/255 vào với thr=128 (tương đương).
        bw = (gray < loc).astype(np.uint8)
        return tighten_box(np.where(bw, 0, 255).astype(np.uint8))
    raise ValueError(mode)


def cut(img, gray_full, bbox, pad, thr, carve, prev_bbox, next_bbox):
    """Bản song song của `build_dataset.save_crop` nhưng THAM SỐ HOÁ và trong bộ nhớ.

    Giữ ĐÚNG thứ tự phép toán của production (pad -> carve -> tighten); đổi thứ tự
    sẽ làm kết quả không nói được gì về pipeline thật.
    """
    import cv2
    from pipeline.align_engine.bbox_fix import carve_neighbor_ink
    if img is None or not bbox:
        return None
    H, W = img.shape[:2]
    ox1, oy1, ox2, oy2 = (int(v) for v in bbox)
    pw, ph = int((ox2 - ox1) * pad), int((oy2 - oy1) * pad)
    x1, y1 = max(0, ox1 - pw), max(0, oy1 - ph)
    x2, y2 = min(W, ox2 + pw), min(H, oy2 + ph)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = crop.copy()
    if carve and gray_full is not None and (prev_bbox is not None or next_bbox is not None):
        crop = carve_neighbor_ink(crop, gray_full, x1, y1, x2, y2, (oy1, oy2),
                                  prev_bbox, next_bbox)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    tb = _tighten(gray, thr)
    if tb is not None:
        a, c, b, d = tb
        gray = gray[c:d, a:b]
    return gray if gray.size else None


def run_config(columns, cfg, cache) -> dict:
    """Cắt lại mọi crop của `columns` theo `cfg`, gộp thống kê họ D."""
    import numpy as np
    from pipeline.align_engine import crop_quality
    n = ok = bleed = trunc = blank = 0
    s_sum = b_sum = i_sum = 0.0
    ars = []
    for col in columns:
        img, gray_full = cache(col["book"], col["page"])
        if img is None:
            continue
        boxes = col["bboxes"]
        if cfg["resolve"]:
            boxes, _ = crop_quality.resolve_column(gray_full, boxes)
        for k, bb in enumerate(boxes):
            g = cut(img, gray_full, bb, cfg["pad"], cfg["thr"], cfg["carve"],
                    boxes[k - 1] if k else None,
                    boxes[k + 1] if k + 1 < len(boxes) else None)
            if g is None:
                blank += 1
                n += 1
                continue
            m = crop_quality.measure(g)
            n += 1
            s_sum += m["stray_ink"]; b_sum += m["border_ink"]; i_sum += m["ink_pct"]
            f = m["crop_quality_flag"]
            if f == "ok":
                ok += 1
            elif f == "bleed":
                bleed += 1
            elif f == "truncated":
                trunc += 1
            else:
                blank += 1
            h, w = g.shape[:2]
            ars.append(h / max(w, 1))
    d = max(n, 1)
    ar = np.asarray(ars) if ars else np.zeros(1)
    return {"n": n, "flag_ok": ok / d, "flag_bleed": bleed / d,
            "flag_truncated": trunc / d, "flag_blank": blank / d,
            "stray_ink": s_sum / d, "border_ink": b_sum / d, "ink_pct": i_sum / d,
            "aspect_outlier": float(((ar < 0.5) | (ar > 2.0)).mean())}


def page_cache():
    """Đọc mỗi trang ĐÚNG MỘT LẦN cho cả 48 cấu hình (445 trang × 3 MB)."""
    import cv2
    store: dict = {}

    def get(book, page):
        key = (book, page)
        if key not in store:
            p = PREPARED / BOOKS.get(book, book) / "pages" / f"{page}.png"
            img = cv2.imread(str(p)) if p.exists() else None
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
            store[key] = (img, g)
        return store[key]
    return get


def diagnose(columns, cache) -> dict:
    """Kiểm SUY BIẾN trước khi tin bất kỳ xếp hạng nào.

    Điều kiện phải thoả: khi pad tăng, `border_ink` giảm (lực thuận) NHƯNG một
    thước đo khác phải TĂNG (lực kéo ngược). Không có lực kéo ngược thì `flag_ok`
    đơn điệu theo pad và trục pad vô nghĩa — đúng lớp lỗi của trục `yield` ở T3.
    """
    rows = []
    for pad in PADS:
        cfg = dict(BASELINE, pad=pad)
        r = run_config(columns, cfg, cache)
        r["pad"] = pad
        rows.append(r)
        print(f"    pad={pad:.2f}  flag_ok={r['flag_ok']:.4f}  border={r['border_ink']:.4f}"
              f"  stray={r['stray_ink']:.4f}  ink={r['ink_pct']:.4f}", flush=True)
    b = [r["border_ink"] for r in rows]
    s = [r["stray_ink"] for r in rows]
    i = [r["ink_pct"] for r in rows]
    o = [r["flag_ok"] for r in rows]
    mono_ok = all(o[k] <= o[k + 1] for k in range(len(o) - 1))
    counter = (s[-1] > s[0] + 1e-4) or (i[-1] < i[0] - 1e-4)
    verdict = ("SUY BIẾN — flag_ok đơn điệu tăng theo pad và không có lực kéo ngược"
               if mono_ok and not counter else
               "DÙNG ĐƯỢC — có lực kéo ngược, flag_ok không chỉ là hàm của pad")
    print(f"\n  border_ink {b[0]:.4f} -> {b[-1]:.4f} | stray {s[0]:.4f} -> {s[-1]:.4f}"
          f" | ink_pct {i[0]:.4f} -> {i[-1]:.4f}")
    print(f"  flag_ok đơn điệu theo pad: {mono_ok} | có lực kéo ngược: {counter}")
    print(f"  => {verdict}")
    return {"rows": rows, "monotone_ok": mono_ok, "counterforce": counter,
            "verdict": verdict}


# =============================================================================
# LUẬT QUYẾT ĐỊNH — TIỀN ĐĂNG KÝ, commit TRƯỚC khi chạy lưới (bài học T3)
# =============================================================================
# T3 sập vì luật quyết định chỉ so TỔNG mức lợi với nhiễu, nên một trục suy biến
# (swap_syl) tự nó kéo cả kết luận. Nên T4 khoá 4 điều kiện, phải thoả HẾT:
#
#   (1) HỢP LỆ    `--diagnose` phải trả "DÙNG ĐƯỢC" (có lực kéo ngược). Nếu
#                 "SUY BIẾN" thì DỪNG, không xếp hạng gì.
#   (2) HƠN THẬT  flag_ok của cấu hình thắng phải hơn MỐC ít nhất DELTA_MIN =
#                 0,005 (0,5 điểm phần trăm). Dưới mức đó là nhiễu lấy mẫu:
#                 n ~ 9.000, sai số chuẩn của một tỷ lệ ~ 0,93% -> 0,5pp không
#                 phân biệt được nếu chỉ có một phép đo, nên thêm (3).
#   (3) KHÔNG ĐÁNH ĐỔI NGẦM  flag_truncated (chữ bị cắt mất nét — hỏng KHÔNG
#                 cứu được) không được tăng quá TRUNC_MAX_RISE = 0,002 so với
#                 MỐC. `bleed` còn cứu được bằng carve; `truncated` thì không.
#   (4) BỀN       lợi thế không được đến từ MỘT trục duy nhất: nếu cố định trục
#                 mạnh nhất về mức MỐC mà lợi thế biến mất, coi là KHÔNG BỀN.
#
# Không thoả đủ 4 -> GIỮ MỐC (pad 0,12 · fixed128 · carve bật · resolve tắt).
# Đây là kết luận ÂM hoàn toàn chấp nhận được, và T3 đã cho một tiền lệ.
DELTA_MIN = 0.005
TRUNC_MAX_RISE = 0.002


def decide(rows: list[dict], diag: dict | None = None) -> dict:
    """Áp 4 điều kiện tiền đăng ký. Trả kết luận + lý do từng điều kiện."""
    base = next((r for r in rows if r.get("is_baseline")), None)
    if base is None:
        return {"verdict": "KHÔNG KẾT LUẬN", "why": "thiếu hàng MỐC"}
    if diag is not None and not (diag.get("counterforce") or not diag.get("monotone_ok")):
        return {"verdict": "DỪNG — TRỤC SUY BIẾN", "why": diag.get("verdict", "")}

    cand = sorted(rows, key=lambda r: -r["flag_ok"])[0]
    gain = cand["flag_ok"] - base["flag_ok"]
    trunc_rise = cand["flag_truncated"] - base["flag_truncated"]
    checks = {
        "2_hon_that": gain >= DELTA_MIN,
        "3_khong_danh_doi": trunc_rise <= TRUNC_MAX_RISE,
    }
    # (4) bền: với mỗi trục, ép cấu hình thắng về mức MỐC ở TRỤC ĐÓ và xem lợi
    #     thế còn lại bao nhiêu. Trục nào một mình chiếm > 70% lợi thế thì cấu
    #     hình đó dựa vào một trục -> không bền.
    contrib = {}
    for ax in ("pad", "thr", "carve", "resolve"):
        forced = dict(cand); forced[ax] = BASELINE[ax]
        m = next((r for r in rows if all(r[k] == forced[k]
                                        for k in ("pad", "thr", "carve", "resolve"))), None)
        if m is not None and gain > 1e-12:
            contrib[ax] = (gain - (m["flag_ok"] - base["flag_ok"])) / gain
    top_ax = max(contrib, key=contrib.get) if contrib else None
    checks["4_ben"] = bool(top_ax) and contrib[top_ax] <= 0.70
    verdict = ("ĐỔI CẤU HÌNH" if all(checks.values()) else "GIỮ MỐC")
    return {"verdict": verdict, "gain": gain, "trunc_rise": trunc_rise,
            "candidate": {k: cand[k] for k in ("pad", "thr", "carve", "resolve")},
            "baseline_flag_ok": base["flag_ok"], "cand_flag_ok": cand["flag_ok"],
            "checks": checks, "axis_contribution": contrib, "dominant_axis": top_ax}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.lab.crop_grid")
    ap.add_argument("--pages", type=int, default=60)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--diagnose", action="store_true")
    args = ap.parse_args(argv)

    columns, n_pages = load_boxes(n_pages=args.pages)
    n_box = sum(len(c["bboxes"]) for c in columns)
    print(f"[T4-A] {n_pages} trang · {len(columns):,} cột · {n_box:,} hộp")
    cache = page_cache()

    if args.diagnose:
        print("\n[T4-A] kiểm suy biến trục pad (4 mức, cấu hình còn lại = MỐC):")
        d = diagnose(columns, cache)
        Path(REPO / "lab").mkdir(exist_ok=True)
        (REPO / "lab" / "t4a_diagnose.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if d["counterforce"] or not d["monotone_ok"] else 1

    cfgs = [{"pad": p, "thr": t, "carve": c, "resolve": r}
            for p in PADS for t in THRS for c in CARVE for r in RESOLVE]
    print(f"[T4-A] {len(cfgs)} cấu hình × {n_box:,} hộp = {len(cfgs)*n_box:,} lần cắt\n")
    rows = []
    for k, cfg in enumerate(cfgs, 1):
        r = run_config(columns, cfg, cache)
        r.update(cfg)
        r["is_baseline"] = cfg == BASELINE
        r["cfg_id"] = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
        rows.append(r)
        print(f"  [{k:2d}/{len(cfgs)}] pad={cfg['pad']:.2f} {cfg['thr']:<8} "
              f"carve={int(cfg['carve'])} res={int(cfg['resolve'])} "
              f"-> ok={r['flag_ok']:.4f} bleed={r['flag_bleed']:.4f} "
              f"trunc={r['flag_truncated']:.4f}{'  <-- MỐC' if r['is_baseline'] else ''}",
              flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    diag_p = REPO / "lab" / "t4a_diagnose.json"
    diag = json.loads(diag_p.read_text(encoding="utf-8")) if diag_p.exists() else None
    d = decide(rows, diag)
    print(f"\n[T4-A] -> {out}")
    print("\n=== LUẬT TIỀN ĐĂNG KÝ ===")
    for k, v in (d.get("checks") or {}).items():
        print(f"  {k:22} {'THOẢ' if v else 'KHÔNG THOẢ'}")
    if "gain" in d:
        print(f"  lợi thế flag_ok: {d['baseline_flag_ok']:.4f} -> {d['cand_flag_ok']:.4f} "
              f"({d['gain']:+.4f}) | truncated {d['trunc_rise']:+.4f}")
        print(f"  ứng viên: {d['candidate']} | trục trội: {d['dominant_axis']} "
              f"({100*(d['axis_contribution'] or {}).get(d['dominant_axis'] or '', 0):.0f}% lợi thế)")
    print(f"  => {d['verdict']}")
    (REPO / "lab" / "t4a_decision.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
