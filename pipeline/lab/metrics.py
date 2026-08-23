"""Thư viện THƯỚC ĐO KHÔNG CẦN NHÃN NGƯỜI — mỗi cấu hình -> một dòng số.

VÌ SAO TỒN TẠI
--------------
Đề tài hiện có 0 phán quyết người dùng được (docs/KE_HOACH_TONG_THE_2026-08-22.md §0),
nên precision không đo được. Nhưng RẤT NHIỀU thứ vẫn đo được mà không cần một nhãn nào:
hình học crop, độ sạch âm Quốc ngữ, cấu trúc cột, độ neo của cột. Module này gom chúng
lại thành một hàng số duy nhất để so cấu hình với cấu hình.

BỐN HỌ TIÊU CHÍ (xem docs/CHUONG_TRINH_THI_NGHIEM_2026-08-22.md §1) — ở đây là họ D
(hình học & thống kê thuần). Họ A (ngữ liệu tổng hợp) và B (nhiễu loạn có đáp án) nằm ở
pipeline/lab/perturb.py.

ĐỪNG TỐI ƯU THEO `n_gold`. Đó chính là luật gán nhãn; cấu hình nới lỏng nhất sẽ luôn
"thắng" trong khi sinh ra nhiều nhãn sai nhất. `n_gold` có mặt ở đây để BÁO CÁO kèm, và
để dựng đường cong sản lượng–độ bền, không phải để cực đại hoá.
"""
from __future__ import annotations

import collections
import csv
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Ngưỡng "gần như trắng" dùng khi đếm nhanh, khớp crop_quality.INK_PCT_MIN.
_BLANK_INK = 0.05


# --------------------------------------------------------------------------- #
# họ D-1: hình học crop  (lần đầu tiên crop_quality.py được dùng thật)
# --------------------------------------------------------------------------- #
def _measure_one(args):
    """Đo 1 crop. Tách ra ngoài để multiprocessing pickle được."""
    import numpy as np
    from PIL import Image
    from pipeline.align_engine import crop_quality
    path, = args
    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L"))
    except Exception:
        return None
    if g.size == 0:
        return None
    m = crop_quality.measure(g)
    h, w = g.shape[:2]
    m["h"], m["w"] = int(h), int(w)
    m["aspect"] = round(w / h, 4) if h else 0.0
    return m


def crop_geometry(images: list[str], root: Path | str = REPO / "dataset_out",
                  workers: int = 0) -> dict:
    """Đo hình học TOÀN BỘ danh sách crop. Trả một dict phẳng.

    `workers=0` -> tự chọn (cpu-2). Kết quả KHÔNG phụ thuộc thứ tự nên tất định.
    """
    root = Path(root)
    paths = [str(root / im) for im in images if im]
    if not paths:
        return {"n_crops": 0}
    if workers == 0:
        import os
        workers = max(1, (os.cpu_count() or 2) - 2)

    rows = []
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_measure_one, ((p,) for p in paths), chunksize=256):
                if r:
                    rows.append(r)
    else:
        for p in paths:
            r = _measure_one((p,))
            if r:
                rows.append(r)

    n = len(rows)
    out = {"n_crops": n, "n_crops_unreadable": len(paths) - n}
    if not n:
        return out
    flags = collections.Counter(r["crop_quality_flag"] for r in rows)
    for k in ("ok", "bleed", "truncated", "blank"):
        out[f"flag_{k}"] = flags.get(k, 0)
        out[f"flag_{k}_pct"] = round(100.0 * flags.get(k, 0) / n, 3)
    for col in ("stray_ink", "border_ink", "ink_pct", "aspect"):
        vals = sorted(r[col] for r in rows)
        out[f"{col}_p50"] = round(_pct(vals, 50), 4)
        out[f"{col}_p95"] = round(_pct(vals, 95), 4)
        out[f"{col}_p99"] = round(_pct(vals, 99), 4)
    # ngoại lai tỉ lệ khung: chữ Nôm gần vuông, lệch mạnh = hộp hỏng
    asp = sorted(r["aspect"] for r in rows)
    lo, hi = _pct(asp, 1), _pct(asp, 99)
    out["aspect_outlier_pct"] = round(
        100.0 * sum(1 for a in asp if a < lo or a > hi) / n, 3)
    return out


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q / 100.0
    f = math.floor(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


# --------------------------------------------------------------------------- #
# họ D-2: chất lượng âm Quốc ngữ  (nhãn của cả đề tài neo vào đây)
# --------------------------------------------------------------------------- #
_GARBAGE = re.compile(r"^[\W\d_]+$")


def syllable_quality(rows: list[dict], qn_keys: set[str]) -> dict:
    """Âm QN có nằm trong từ điển không — thước đo chính của chương trình T1."""
    import sys
    sys.path.insert(0, str(REPO))
    from core.text.text_utils import is_plausible_qn_syllable

    syl = [str(r.get("syllable", "")).lower() for r in rows]
    syl = [s for s in syl if s]
    n = len(syl)
    if not n:
        return {"n_syllable": 0}
    in_dict = sum(1 for s in syl if s in qn_keys)
    plausible = sum(1 for s in syl if is_plausible_qn_syllable(s))
    garbage = sum(1 for s in syl if _GARBAGE.match(s))
    return {
        "n_syllable": n,
        "syl_in_dict_pct": round(100.0 * in_dict / n, 4),
        "syl_out_of_dict": n - in_dict,
        "syl_plausible_pct": round(100.0 * plausible / n, 4),
        "syl_garbage": garbage,
    }


# --------------------------------------------------------------------------- #
# họ D-3: cấu trúc cột + độ neo  (thước đo cho T2 và T3)
# --------------------------------------------------------------------------- #
ANCHOR_RULE = "s1_inter_s2_direct"


def structure(rows: list[dict]) -> dict:
    """Cột/trang, ô/cột, và TỈ LỆ Ô NEO — nền của chuẩn nhiễu loạn ở perturb.py."""
    per_page = collections.defaultdict(set)
    per_col = collections.Counter()
    anchors = collections.Counter()
    total = collections.Counter()
    for r in rows:
        col = r.get("column")
        if not col:
            continue
        key = (r.get("book"), r.get("page"))
        per_page[key].add(col)
        ck = (r.get("book"), r.get("page"), col)
        per_col[ck] += 1
        total[ck] += 1
        if str(r.get("rule", "")).split("|")[0] == ANCHOR_RULE:
            anchors[ck] += 1

    ncols = collections.Counter(len(v) for v in per_page.values())
    sizes = sorted(per_col.values())
    frac = sorted(anchors[k] / total[k] for k in total if total[k])
    out = {
        "n_pages": len(per_page),
        "n_columns": len(per_col),
        "pages_with_9_cols": ncols.get(9, 0),
        "pages_with_9_cols_pct": round(100.0 * ncols.get(9, 0) / max(len(per_page), 1), 3),
        "cells_per_col_p50": _pct(sizes, 50),
        "cells_per_col_min": sizes[0] if sizes else 0,
        "cells_per_col_max": sizes[-1] if sizes else 0,
    }
    if frac:
        out["anchor_frac_p10"] = round(_pct(frac, 10), 4)
        out["anchor_frac_p50"] = round(_pct(frac, 50), 4)
        out["anchor_frac_p90"] = round(_pct(frac, 90), 4)
    return out


# --------------------------------------------------------------------------- #
# họ D-4: tier / rule / lớp  (BÁO CÁO kèm, KHÔNG phải mục tiêu tối ưu)
# --------------------------------------------------------------------------- #
USABLE = ("GOLD", "SYLLABLE")


def composition(rows: list[dict]) -> dict:
    tiers = collections.Counter(r.get("tier", "") for r in rows)
    out = {"n_rows": len(rows)}
    for t, n in tiers.items():
        out[f"tier_{t}"] = n
    out["n_usable"] = sum(tiers.get(t, 0) for t in USABLE)
    out["n_classes_usable"] = len(
        {r.get("label") for r in rows if r.get("tier") in USABLE and r.get("label")})
    # mâu thuẫn tự thân: cùng ảnh (md5) mang >1 nhãn -> LỖI CHẮC CHẮN, không cần nhãn người
    by_md5 = collections.defaultdict(set)
    for r in rows:
        if r.get("image_md5") and r.get("label"):
            by_md5[r["image_md5"]].add(r["label"])
    out["md5_label_conflicts"] = sum(1 for v in by_md5.values() if len(v) > 1)
    return out


# --------------------------------------------------------------------------- #
def collect(labels_csv: Path | str, src_root: Path | str = REPO / "dataset_out",
            with_crops: bool = True, crop_sample: int = 0, workers: int = 0) -> dict:
    """Gom mọi thước đo về MỘT dòng. `crop_sample>0` -> chỉ đo mẫu (để chạy nhanh)."""
    import sys
    sys.path.insert(0, str(REPO))
    from core.text.dictionary import dict_dir, load_qn_to_nom

    with open(labels_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    qn = load_qn_to_nom(str(dict_dir() / "QuocNgu_SinoNom.csv"))

    out: dict = {"labels_csv": str(labels_csv)}
    out.update(composition(rows))
    out.update(syllable_quality(rows, set(qn)))
    out.update(structure(rows))
    if with_crops:
        imgs = [r["image"] for r in rows if r.get("image") and r.get("tier") in USABLE]
        if crop_sample and crop_sample < len(imgs):
            step = len(imgs) / crop_sample          # lấy mẫu ĐỀU -> tất định
            imgs = [imgs[int(i * step)] for i in range(crop_sample)]
            out["crop_sampled"] = len(imgs)
        out.update(crop_geometry(imgs, src_root, workers=workers))
    return out
