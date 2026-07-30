"""Đo và sửa 2 khuyết tật hình học của crop ký tự — ĐO BẰNG PIXEL, không bằng bbox.

BỐI CẢNH (đo 2026-07-29 trên 69.344 crop thật):

  · DÍNH CHỮ HÀNG XÓM   1,27% (chặt) … 11,06% (lỏng)
  · CẮT THIẾU NỬA CHỮ    0,71% (chặt) …  5,99% (lỏng)

VÌ SAO KHÔNG DÙNG `bbox`: cột bbox KHÔNG phải hộp detector — `_reseg_column`
(align_production.py:157) tổng hợp hộp midpoint với phần chồng lấn CỐ Ý
`m = pitch*0.10` để giữ đuôi nét chữ cao. Đo được: trung vị chồng lấn 0,208,
p99 0,32 → **99,5% dòng "chồng lấn" là do thiết kế**. Mọi chỉ số bleed suy ra
từ bbox (IoU, overlap-fraction) vì thế vô nghĩa. Phải đọc pixel.

BẪY ĐÃ ĐO — chữ Nôm ghép ⿱ vốn CÓ SẴN 2 dải mực (trên/dưới) là một ký tự duy
nhất. Ngưỡng tách dải lỏng cho p50 = 2 dải trên toàn corpus → báo động giả hàng
loạt (chính là cảnh báo giả "76% cắt dính" mà VLM từng đưa ra và đã bị bác).
Nên khe hở tách dải bắt buộc >= max(6px, 0.15*h).

QUAN HỆ VỚI `carve_neighbor_ink` (bbox_fix.py): hàm đó CHỈ dọn được mực hàng xóm
nằm trong vùng PAD, vì nó tự chặn cứng không đụng vào [oy1,oy2] — hộp gốc của
chính ký tự (bbox_fix.py:175-183). Chặn đó là đúng và phải giữ: bỏ nó ra thì
seam "rẻ nhất" cắt xuyên giữa ký tự, đo được ink_pct=0.0 hàng loạt.
`resolve_overlap` dưới đây giải quyết phần carve KHÔNG với tới: khi 2 hộp gốc
chồng nhau thật, nó dời RANH GIỚI HỘP trước khi cắt, nên carve/tighten sau đó
làm việc trên 2 hộp đã rời nhau — không cần nới lỏng chặn nào cả.
"""
from __future__ import annotations

import numpy as np

from pipeline.align_engine.bbox_fix import _seam_boundary

# ---------------------------------------------------------------------------
# Ngưỡng — đặt ở ĐUÔI phân phối GOLD đo được, không phải số tự nghĩ ra.
#   stray_ink : GOLD p90=0, p95=0.101, p99=0.286  -> 0.08 nằm giữa p90 và p99
#   border_ink: GOLD p50=0.012, p95=0.100, p99=0.171 -> 0.20 nằm TRÊN p99
#   ink_pct   : GOLD p1=0.101 -> 0.05 là crop gần như trắng
STRAY_INK_MAX = 0.08
BORDER_INK_MAX = 0.20
INK_PCT_MIN = 0.05

# resolve_overlap
OVERLAP_DESIGN_MAX = 0.25   # <= mức này là biên chủ ý (pitch*0.10), không đụng
MIN_HEIGHT_RATIO = 0.45     # 2 hộp lệch cao hơn mức này => nghi hộp hỏng
MIN_KEEP_RATIO = 0.50       # seam không được ăn quá nửa chiều cao gốc hộp nào

_INK_THR = 128


def _bw(gray: "np.ndarray") -> "np.ndarray":
    return gray < _INK_THR


def border_ink(gray: "np.ndarray") -> float:
    """Tỷ lệ mực chạm mép TRÊN/DƯỚI của crop -> dấu hiệu bị CẮT THIẾU.

    save_crop() luôn kết thúc bằng tighten_box(), hàm này co về hộp bao mực rồi
    cộng margin=4px. Nên một crop lành lặn có ĐÚNG 4px trắng trước khi tới mực ở
    mọi phía (đã kiểm chứng: mode của khe hở = 4 ở cả 4 phía). Mực chạm sát mép
    nghĩa là hộp đã cắt vào giữa nét — tighten_box không thể nới ra được nữa.

    Chỉ xét mép trên/dưới: ký tự xếp DỌC trong cột nên đây là trục bị cắt.
    Lấy 2 hàng biên (không phải 1) để 1 pixel nhiễu lẻ không thổi phồng chỉ số.
    """
    bw = _bw(gray)
    if bw.size == 0 or bw.shape[0] < 4:
        return 0.0
    return float(max(bw[:2, :].mean(), bw[-2:, :].mean()))


def stray_ink(gray: "np.ndarray") -> float:
    """Tỷ lệ mực thuộc về ký tự HÀNG XÓM còn sót trong crop -> dấu hiệu DÍNH CHỮ.

    Chiếu mực theo hàng, tách thành các dải cách nhau bởi khe trắng
    >= max(6px, 0.15*h). Dải "lạ" = dải (a) giữ < 35% tổng mực VÀ (b) nằm hẳn ở
    30% trên cùng hoặc 30% dưới cùng. Trả về phần mực của các dải lạ đó.

    Điều kiện (b) là thứ phân biệt được mực hàng xóm với ký tự ghép ⿱ hợp lệ:
    hai thành phần của một chữ ⿱ chia nhau phần mực khá đều và nằm ở GIỮA khung,
    còn mực hàng xóm sót lại luôn bị đẩy ra sát rìa.
    """
    bw = _bw(gray)
    h = bw.shape[0]
    total = int(bw.sum())
    if total == 0 or h < 12:
        return 0.0

    rows = bw.sum(axis=1)
    gap_min = max(6, int(0.15 * h))

    # gom các hàng có mực thành dải, cắt dải khi gặp khe trắng đủ rộng
    bands: list[tuple[int, int]] = []
    start = None
    blank = 0
    for y in range(h):
        if rows[y] > 0:
            if start is None:
                start = y
            blank = 0
        else:
            if start is not None:
                blank += 1
                if blank >= gap_min:
                    bands.append((start, y - blank + 1))
                    start = None
    if start is not None:
        bands.append((start, h))
    if len(bands) < 2:
        return 0.0

    top_zone, bot_zone = 0.30 * h, 0.70 * h
    stray = 0
    for y0, y1 in bands:
        mass = int(rows[y0:y1].sum())
        if mass / total >= 0.35:
            continue                       # dải chính -> là ký tự của chính nó
        if y1 <= top_zone or y0 >= bot_zone:
            stray += mass                  # nhỏ VÀ nằm sát rìa -> mực hàng xóm
    return float(stray / total)


def crop_quality_flag(stray: float, border: float, ink_pct: float) -> str:
    """'ok' | 'bleed' | 'truncated' | 'blank' — ưu tiên khuyết tật nặng nhất.

    Tên cột phải là `crop_quality_flag`, KHÔNG phải `quality_flag`:
    suspicion.py:129 đã dùng `quality_flag` làm tên TẦNG lấy mẫu
    (sampling.py:25) và add_suspicion() sẽ ghi đè âm thầm nếu trùng tên.
    """
    if ink_pct < INK_PCT_MIN:
        return "blank"
    if border > BORDER_INK_MAX:
        return "truncated"
    if stray > STRAY_INK_MAX:
        return "bleed"
    return "ok"


def measure(gray: "np.ndarray") -> dict:
    """3 chỉ số + cờ cho một crop đã cắt xong (ảnh xám)."""
    bw = _bw(gray)
    ink = float(bw.mean()) if bw.size else 0.0
    s, b = stray_ink(gray), border_ink(gray)
    return {"stray_ink": round(s, 4), "border_ink": round(b, 4),
            "ink_pct": round(ink, 4), "crop_quality_flag": crop_quality_flag(s, b, ink)}


def resolve_overlap(gray_full: "np.ndarray", bbox_a, bbox_b):
    """Tách ranh giới 2 hộp LIỀN KỀ cùng cột đang chồng lấn THẬT (a ở trên b).

    Trả (bbox_a_mới, bbox_b_mới, verdict) với verdict:
      'clean'      chồng lấn <= biên chủ ý -> KHÔNG đụng gì
      'fixed'      chồng lấn thật -> đã dời ranh giới về đường seam ít mực nhất
      'degenerate' hộp lồng nhau / lệch kích thước / seam ăn quá sâu -> TỪ CHỐI
                   tự sửa, để người soát (thường là detection hỏng, không phải
                   2 chữ chạm nhau)

    Đo trên 200 cặp xấu nhất: 147 'fixed' (chồng lấn về đúng 0), 53 'degenerate'.

    CHẶN 50%: nếu seam làm một trong hai hộp mất quá nửa chiều cao gốc thì từ
    chối. Không có chặn này, cặp stt2/page_0050 c7 (tỷ lệ cao 0,459 lọt qua chặn
    lệch-kích-thước) bị seam cắt sát đỉnh dải tranh chấp, ảnh "sau" gần như trắng.
    """
    ax1, ay1, ax2, ay2 = (int(v) for v in bbox_a)
    bx1, by1, bx2, by2 = (int(v) for v in bbox_b)
    ha, hb = ay2 - ay1, by2 - by1
    if ha <= 0 or hb <= 0:
        return bbox_a, bbox_b, "degenerate"

    lo, hi = max(ay1, by1), min(ay2, by2)
    overlap = hi - lo
    if overlap <= 0 or overlap / min(ha, hb) <= OVERLAP_DESIGN_MAX:
        return bbox_a, bbox_b, "clean"

    nested = (ay1 <= by1 and ay2 >= by2) or (by1 <= ay1 and by2 >= ay2)
    if nested or min(ha, hb) / max(ha, hb) < MIN_HEIGHT_RATIO:
        return bbox_a, bbox_b, "degenerate"

    x1, x2 = max(ax1, bx1), min(ax2, bx2)
    if x2 - x1 < 4:                       # 2 cột lệch nhau bất thường
        x1, x2 = min(ax1, bx1), max(ax2, bx2)

    seam = _seam_boundary(gray_full, x1, x2, lo, hi)
    if seam is None:
        return bbox_a, bbox_b, "degenerate"
    cut = max(lo + 1, min(hi - 1, int(np.median(seam))))

    if (cut - ay1) < MIN_KEEP_RATIO * ha or (by2 - cut) < MIN_KEEP_RATIO * hb:
        return bbox_a, bbox_b, "degenerate"
    return [ax1, ay1, ax2, cut], [bx1, cut, bx2, by2], "fixed"


def resolve_column(gray_full: "np.ndarray", bboxes: list) -> tuple[list, list]:
    """Áp resolve_overlap dọc một cột đã sắp theo y. Trả (bboxes_mới, verdicts).

    Duyệt tuần tự và mang kết quả sang cặp sau (hộp i đã bị cặp i-1 cắt thì cặp
    i,i+1 phải nhìn hộp ĐÃ cắt) — nếu không, 3 hộp chồng nhau liên tiếp sẽ nhận
    2 quyết định mâu thuẫn về cùng một ranh giới.
    """
    out = [list(b) if b else b for b in bboxes]
    verdicts = []
    for i in range(len(out) - 1):
        if not out[i] or not out[i + 1]:
            verdicts.append("clean")
            continue
        na, nb, v = resolve_overlap(gray_full, out[i], out[i + 1])
        out[i], out[i + 1] = na, nb
        verdicts.append(v)
    return out, verdicts
