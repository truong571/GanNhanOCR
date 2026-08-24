"""Production-faithful page alignment, with a swappable pairing block.

This mirrors pipeline/step2_align.py::process_page_structural EXACTLY for the
column-detection + QN-parsing front end (so the comparison is apples-to-apples
with what the real labeler sees), and swaps ONLY the inner Nôm↔QN pairing:

    mode="old" : the current logic — force-equalize counts (truncate leading
                 extras / re-segment when too few) then emit 1-1 by INDEX.
                 Faithful copy of step2_align.py lines ~164-214 + emit.
    mode="new" : banded dictionary-anchored DP (anchor_align.realign_column).
                 Genuine gaps are NOT emitted as labels (they go to REVIEW);
                 the column is never tail-shifted.

No crops / no JSON are written here — this is the evaluation harness. The
production wiring (where this replaces the pairing in step2 and feeds step3/4)
is documented in FLOW.md §8.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# Production front-end pieces (identical to step2_align.py) ------------------
from pipeline.step2_align import _get_qn_lines           # parse_v5 QN parsing
from core.align.nom_detect_v3 import detect_nom_columns_v3
from core.align.export_dataset_v4 import resegment_col
from core.image.char_segmenter import segment_characters_in_column
from core.image.image_processing import load_and_binarize

from pipeline.align_engine.anchor_align import realign_column, matched_pairs
from pipeline.align_engine.syllable_normalize import build_readings, normalize_column
from pipeline.align_engine.consensus import decide_label
from pipeline.align_engine.bbox_fix import frame_offset, correct_columns


def _detect(page_name: str, data_dir: Path, qn_dict_set: set):
    """Replicate the detection + QN parse + iter-plan of process_page_structural.

    Returns (cols, qn_lines, iter_pairs, binary, page_ok) or None if the page
    cannot be processed.
    """
    pages_dir = data_dir / "pages"
    denoised_dir = data_dir / "pages_denoised"
    img_path = pages_dir / f"{page_name}.png"
    if not img_path.exists():
        return None
    color_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if color_img is None:
        return None

    ocr_path = data_dir / "detected" / f"{page_name}_ocr_cache.json"
    if not ocr_path.exists():
        return None
    ocr_data = json.load(open(ocr_path, encoding="utf-8"))
    ocr_columns = ocr_data.get("columns", [])
    # FIX bbox offset: OCR ran on a frame-cropped image, so bboxes are in
    # cropped coords. Map them back to full-page coords before any cropping.
    # Skip if the cache is already full-page (new ocr_api format) -> no double-shift.
    if ocr_data.get("coords_space") != "fullpage":
        ox, oy = frame_offset(str(img_path), ocr_data.get("framed"),
                              ocr_data.get("frame_pad", 12))
        correct_columns(ocr_columns, ox, oy)

    qn_lines, _ = _get_qn_lines(data_dir, page_name, qn_dict_set)
    qn_keys = sorted(qn_lines.keys())
    if not qn_keys:
        return None

    bin_src = denoised_dir / f"{page_name}.png"
    if not bin_src.exists():
        bin_src = img_path
    try:
        _, binary = load_and_binarize(str(bin_src))
    except Exception:
        binary = None

    if binary is not None:
        cols, col_method = detect_nom_columns_v3(binary, ocr_columns, 9)
    else:
        from core.align.run_full import nom_cols_hybrid
        cols, col_method = nom_cols_hybrid(ocr_columns, min_len=4), "hybrid_no_image"

    page_col_match = (len(cols) == len(qn_keys))
    qn_parse_ok = (len(qn_lines) == 9)
    nom_suspect = (col_method == "suspect")
    max_qn = max(qn_keys) if qn_keys else 0
    partial_recovery = (not qn_parse_ok and not nom_suspect
                        and len(cols) >= max_qn and max_qn > 0)
    page_ok = ((page_col_match and qn_parse_ok and not nom_suspect)
               or partial_recovery)

    if partial_recovery:
        iter_pairs = [(lid - 1, lid) for lid in qn_keys if (lid - 1) < len(cols)]
    else:
        n_align = min(len(cols), len(qn_keys))
        iter_pairs = [(i, qn_keys[i]) for i in range(n_align)]
    return cols, qn_lines, iter_pairs, binary, page_ok


def _pair_old(cluster: dict, syllables: list[str], binary) -> list[dict]:
    """Faithful copy of step2's count force-equalize + index emit."""
    actual, expected = len(cluster["chars"]), len(syllables)
    if actual > expected:                    # too many -> drop LEADING extras
        chars_used = [{"bbox": c["bbox"], "ocr_char": c.get("char")}
                      for c in cluster["chars"][actual - expected:]]
    elif actual < expected:                  # too few -> re-segment the image
        chars_used = None
        if binary is not None and cluster["chars"]:
            res = resegment_col(binary, cluster, expected)
            if res:
                chars_used = [{"bbox": r["bbox"], "ocr_char": r.get("char")} for r in res]
        if chars_used is None and binary is not None and cluster.get("bbox"):
            try:
                bb = segment_characters_in_column(binary, cluster["bbox"],
                                                  expected_count=expected)
                if len(bb) == expected:
                    chars_used = [{"bbox": list(b), "ocr_char": None} for b in bb]
            except Exception:
                pass
        if chars_used is None:
            chars_used = [{"bbox": c["bbox"], "ocr_char": c.get("char")}
                          for c in cluster["chars"]]
    else:
        chars_used = [{"bbox": c["bbox"], "ocr_char": c.get("char")}
                      for c in cluster["chars"]]
    out = []
    for k in range(min(len(chars_used), len(syllables))):
        out.append({"ocr_char": chars_used[k]["ocr_char"],
                    "bbox": chars_used[k]["bbox"], "syllable": syllables[k]})
    return out


# =============================================================================
# BOX_OVERLAP_FRAC — biên nới dọc của hộp ký tự, tính theo BƯỚC LẶP của cột.
# =============================================================================
# Hộp = [trung điểm với chữ trên − m, trung điểm với chữ dưới + m] với m = pitch*F,
# nên CHIỀU CAO HỘP = pitch × (1 + 2F).
#
# F = 0,10 (từ đầu đến 2026-08-24) làm hộp cao 1,20 × bước lặp. Đo T4 trên 537 cột /
# 3 sách: 99,44% cột có hộp cao hơn bước lặp (trung vị 1,2069), và **84,46% cặp chữ
# liền kề trong cùng cột CHỒNG nhau** (khe hở trung vị −0,171 chiều cao hộp). Cộng
# đệm 0,12 lúc cắt thì CỬA SỔ CROP = 1,49 × bước lặp, tức luôn trùm sang ~24,5% chữ
# trên và chữ dưới; chỉ `carve_neighbor_ink` giữ cho crop còn dùng được.
#
# Chú thích cũ nói mức nới ấy để "chữ cao giữ được đuôi nét". ĐO T4.e BÁC ĐIỀU ĐÓ:
# quét F ∈ {−0,10 … 0,30} trên 9.404 ô / 60 trang, tỷ lệ BỊ CẮT gần như KHÔNG đổi
# theo F (0,0165 ở F=0,10 vs 0,0179 ở F=0 — chênh 0,0014, KHÔNG vượt khoảng tin cậy
# 95%). Mức nới không cứu được đuôi nét nào — `--pad` lúc cắt đã lo việc đó — nó chỉ
# đổi lấy −3,81 điểm mực CỦA CHÍNH CHỮ (tinh khiết 0,9453 → 0,9072).
#
# 🔴 ĐÃ THỬ F = 0 VÀ HOÀN NGUYÊN VỀ 0,10 (2026-08-25). Ghi lại để không ai thử lại.
#
# Lập luận "hộp = đúng một ô bước lặp" nghe hợp lý và ĐÃ CHẠY THẬT, nhưng sai vì BỎ SÓT
# MỘT RÀNG BUỘC. Đo sau khi chạy, trên 15.457 ô mà F=0 thật sự chạm tới:
#
#   1. XÉN VÀO THÂN CHỮ. Mực CỦA CHÍNH CHỮ (tách bằng liên thông, không phải tổng mực)
#      mất trung vị 11,5%; 17,3% số ô mất HƠN 20% mực thân chữ (≈2.675 ô bộ giao nộp).
#      Đối chứng trên hộp không đổi: 0,00% — nên hoàn toàn quy được cho F.
#
#   2. KHUNG HÌNH VỠ LÀM HAI. Production chạy `--reseg detector`, nên F chỉ quyết ~26%
#      số hộp; 74% còn lại là hộp CenterNet, KHÔNG chịu chi phối của hằng số này và cao
#      1,2308 × pitch. Ở F=0,10 hộp midpoint cao 1,2000 — hai nguồn khớp nhau trong
#      2,5%. Ở F=0 chúng lệch 18,8%, và phần bị thu nhỏ chính là các ô KHÓ (bộ dò bất
#      đồng với tâm OCR), nên khung hình trở thành biến gây nhiễu TƯƠNG QUAN VỚI ĐỘ KHÓ.
#      Giá trị làm midpoint khớp CenterNet là (1,2308−1)/2 = 0,1154 — tức 0,10 gốc gần
#      như CHÍNH LÀ giá trị đúng, và nó đúng vì ràng buộc liên-backend này.
#
# VÌ SAO PHÉP ĐO T4.e KHÔNG BẮT ĐƯỢC: (a) chỉ số "bị cắt" đo SAU `carve_neighbor_ink`,
# mà carve xoá trắng 98,6% tín hiệu mực-chạm-mép, nên nó ĐÃ BÃO HOÀ và không phân biệt
# được F — kết luận "chênh 0,0014, không vượt KTC 95%" là chỉ số chết chứ không phải
# bằng chứng; (b) phép quét dựng lại TOÀN BỘ hộp từ bước lặp, ngầm giả định 100% hộp là
# midpoint, nên vừa thổi phồng lợi ích (+3,81 điểm, đo lại sau khi chạy chỉ +1,23) vừa
# giấu mất ràng buộc ở mục 2.
#
# BÀI HỌC: mực láng giềng thừa thì mắt người bỏ qua được; nét đã mất thì không lấy lại
# được. Với mục tiêu chấm tay, hai loại hỏng KHÔNG cân xứng — đừng tối ưu một chỉ số
# gộp chúng làm một.
#
# Đọc từ `config/pipeline.yaml: step2.box_overlap_frac`; build_dataset gán vào đây lúc
# khởi động.
BOX_OVERLAP_FRAC = 0.10


def _reseg_column(cluster) -> list | None:
    """Rebuild per-char boxes from the OCR y-CENTERS (which are reliable) with
    MIDPOINT boundaries between consecutive chars, so no crop can span into a
    neighbouring character (the cause of 'merged 2-char' crops). x-range = the
    robust column width (median of char x). Returns one box per OCR char (same
    order), or None. Combined with bbox_fix.tighten_box (x/y ink trim) this gives
    a clean single-glyph crop. Valley re-segmentation was tried but mis-packs
    when the column x-window catches neighbour ink — midpoints are far more
    robust because they trust only the detected centres.
    """
    chars = cluster.get("chars") or []
    if not chars:
        return None
    cys = [(c["bbox"][1] + c["bbox"][3]) / 2.0 for c in chars]
    x1 = int(np.median([c["bbox"][0] for c in chars]))
    x2 = int(np.median([c["bbox"][2] for c in chars]))
    if x2 <= x1:
        return None
    n = len(cys)
    pitch = float(np.median(np.diff(cys))) if n >= 2 else 80.0
    if not (pitch > 0):
        pitch = 80.0
    # BIÊN NỚI DỌC — xem BOX_OVERLAP_FRAC ở đầu tệp. Trước 2026-08-25 ghim cứng 0,10.
    m = pitch * BOX_OVERLAP_FRAC
    boxes = []
    for i, cy in enumerate(cys):
        top = (cys[i - 1] + cy) / 2.0 - m if i > 0 else cy - pitch / 2.0
        bot = (cys[i + 1] + cy) / 2.0 + m if i < n - 1 else cy + pitch / 2.0
        boxes.append([x1, int(round(top)), x2, int(round(bot))])
    return boxes


_DETECTOR = None
_DETECTOR_TRIED = False


class DetectorUnavailableError(FileNotFoundError):
    """Yêu cầu reseg=detector nhưng không dựng được detector — KHÔNG rơi ngầm."""


_HINT = ("tải: huggingface-cli download mdnt571/nom-char-det detector_r34.best.pt "
         "--local-dir train_crop/  — hoặc chạy lại với --reseg midpoint nếu CỐ Ý "
         "muốn dùng trung điểm (bộ crop sẽ KHÁC HẲN, phải đo lại).")


def _get_detector(strict: bool = True):
    """Lazy, cached CenterNet detector (train_crop/detector_r34.best.pt).

    strict=True (mặc định): thiếu checkpoint -> NÉM DetectorUnavailableError.
    Trước 2026-08-23 hàm này chỉ IN một dòng log rồi lặng lẽ rơi về midpoint: toàn bộ
    hộp ký tự bị tách bằng trung điểm thay vì CenterNet, cho ra bộ crop khác hẳn, mà
    labels.csv KHÔNG ghi lại backend nào đã dùng -> không ai truy được về sau.
    Chỉ dùng bởi reseg_mode='detector'.
    """
    global _DETECTOR, _DETECTOR_TRIED
    if _DETECTOR_TRIED:
        if strict and (_DETECTOR is None or not getattr(_DETECTOR, "trained", False)):
            raise DetectorUnavailableError(f"reseg=detector nhưng detector không dùng được. {_HINT}")
        return _DETECTOR
    _DETECTOR_TRIED = True
    err = None
    try:
        from pipeline.align_engine.char_detector.detector_infer import DetectorInfer
        # thr 0.2->0.3: bớt detection tin cậy thấp (dễ lệch vị trí) lọt vào
        # enforce_count/column_boxes -> ít box sai hơn feed cho _monotone_assign.
        _DETECTOR = DetectorInfer(thr=0.3)  # tự tìm ckpt v1 ở train_crop/detector_r34.best.pt
        if not _DETECTOR.trained:
            err = "không thấy train_crop/detector_r34.best.pt"
        else:
            print(f"  [reseg detector] CenterNet v1 (img {_DETECTOR.img}, seam) — N = #âm tiết.", flush=True)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        _DETECTOR = None
    if err:
        if strict:
            raise DetectorUnavailableError(f"reseg=detector nhưng {err}. {_HINT}")
        print(f"  [reseg detector] {err} -> midpoint fallback.", flush=True)
    return _DETECTOR


def preflight_detector(reseg_mode: str) -> str:
    """Kiểm detector MỘT LẦN trước khi duyệt trang. Trả tên backend sẽ dùng.

    Fail fast: thiếu checkpoint thì dừng ngay ở trang đầu tiên chứ không âm thầm
    tách bằng trung điểm cho cả 445 trang.
    """
    if reseg_mode != "detector":
        return reseg_mode
    _get_detector(strict=True)
    return "detector_centernet_v1"


def _valley_boxes(cluster, binary, n):
    """N valley boxes for the column (char_segmenter force-N), or None.
    Column box = x_range × full y-extent of the detected chars."""
    if binary is None or n < 1:
        return None
    chars = cluster.get("chars") or []
    if not chars:
        return None
    if cluster.get("x_range"):
        cx1, cx2 = int(cluster["x_range"][0]), int(cluster["x_range"][1])
    else:
        cx1 = min(int(c["bbox"][0]) for c in chars); cx2 = max(int(c["bbox"][2]) for c in chars)
    cy1 = min(int(c["bbox"][1]) for c in chars); cy2 = max(int(c["bbox"][3]) for c in chars)
    try:
        bb = segment_characters_in_column(binary, (cx1, cy1, cx2, cy2), expected_count=n)
    except Exception:
        return None
    return bb if len(bb) == n else None


def _mean_mls(boxes, page_bgr, encoder):
    if page_bgr is None or encoder is None or not boxes:
        return None
    from pipeline.align_engine.bbox_fix import tighten_box
    H, W = page_bgr.shape[:2]
    vals = []
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 6 or y2 - y1 < 6:
            continue
        crop = page_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        tb = tighten_box(g)
        if tb is not None:
            a, c, bb2, d = tb
            if bb2 - a >= 8 and d - c >= 8:
                g = g[c:d, a:bb2]
        m = encoder.mls(encoder.embed_gray(g))
        if m is not None:
            vals.append(m)
    return float(np.mean(vals)) if vals else None


def _monotone_assign(cys, boxes, mid, guard=0.35, pitch=None):
    """Assign each char y-center in `cys` to a DISTINCT box, monotone (non-crossing),
    minimising total |cy - box_center_y| via DP. Returns boxes in the ORIGINAL char
    order, or None if there are fewer boxes than chars (caller falls back to midpoint).

    A char whose assigned box sits farther than guard*pitch from it is replaced by its
    midpoint box mid[i]. This is the fix for the independent-argmin defect (AE-1): two
    chars can no longer grab the same box, and a badly-placed box degrades to the robust
    midpoint instead of a neighbour glyph.
    """
    n = len(cys)
    m = len(boxes)
    if n == 0:
        return []
    if m < n:
        return None
    bcy = [(b[1] + b[3]) / 2.0 for b in boxes]
    border = sorted(range(m), key=lambda j: bcy[j])
    sb = [boxes[j] for j in border]
    sbcy = [bcy[j] for j in border]
    corder = sorted(range(n), key=lambda i: cys[i])
    scy = [cys[i] for i in corder]

    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[-1] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = 0.0
    for i in range(1, n + 1):
        for j in range(i, m + 1):
            best, choice = dp[i][j - 1], -2                 # -2 = skip box j-1
            cost = dp[i - 1][j - 1] + abs(scy[i - 1] - sbcy[j - 1])
            if cost < best:
                best, choice = cost, j - 1                  # assign char i-1 -> box j-1
            dp[i][j], back[i][j] = best, choice

    assign_sorted = [None] * n
    i, j = n, m
    while i > 0:
        bj = back[i][j]
        if bj == -2:
            j -= 1
        else:
            assign_sorted[i - 1] = sb[bj]
            i -= 1
            j = bj

    if pitch is None:
        gaps = [scy[k + 1] - scy[k] for k in range(n - 1)]
        pitch = float(np.median(gaps)) if gaps else None
    out = [None] * n
    for k, i_orig in enumerate(corder):
        box = assign_sorted[k]
        if box is None:
            out[i_orig] = mid[i_orig] if mid else None
        elif pitch and abs(cys[i_orig] - (box[1] + box[3]) / 2.0) > guard * pitch:
            out[i_orig] = mid[i_orig] if mid else box
        else:
            out[i_orig] = box
    return out


def _pick_reseg(cluster, syllables, binary, reseg_mode, encoder=None, page_bgr=None,
                det=None, page_boxes=None):
    """Per-OCR-char boxes for the emitted pairs, per reseg_mode:
      midpoint        — OCR y-center midpoints (default, robust; _reseg_column)
      valley_n        — force-N valley boxes, each OCR char -> nearest valley box
      valley_guarded  — valley only on UNDER-counted cols AND only if it does NOT
                        lower mean MLS (needs encoder+page_bgr); else midpoint.
      detector        — count-constrained CenterNet boxes (needs a trained
                        detector.pt; det+page_boxes from align_page). The real fix.
    Returns a list indexed like cluster['chars'] (or None)."""
    mid = _reseg_column(cluster)
    chars = cluster["chars"]; n = len(syllables)
    if reseg_mode == "detector" and det is not None and page_boxes is not None and cluster.get("x_range"):
        cb = det.column_boxes(page_boxes, cluster["x_range"], n)
        if len(cb) == n:
            cys = [(c["bbox"][1] + c["bbox"][3]) / 2.0 for c in chars]
            assigned = _monotone_assign(cys, cb, mid)        # monotone 1-1 (fixes AE-1)
            if assigned is None:
                return mid
            # x-range guard (fixes F1): reject a box whose center-x falls outside this
            # column, replacing it with the midpoint box rather than a neighbour glyph.
            # Tightened 0.15->0.10 (+ guard 0.5->0.35 above): fewer marginal detector
            # boxes accepted, more (safer) midpoint fallback — trims residual bleed.
            x1, x2 = cluster["x_range"]
            xtol = 0.10 * (x2 - x1)
            out = []
            for i, box in enumerate(assigned):
                if box is None:
                    out.append(mid[i] if mid else None)
                    continue
                bcx = (box[0] + box[2]) / 2.0
                out.append((mid[i] if mid else box)
                           if (bcx < x1 - xtol or bcx > x2 + xtol) else box)
            return out
        return mid
    if reseg_mode == "midpoint" or mid is None or reseg_mode not in ("valley_n", "valley_guarded"):
        return mid
    if reseg_mode == "valley_guarded" and len(chars) >= n:
        return mid                                   # only act on OCR under-count
    vb = _valley_boxes(cluster, binary, n)
    if not vb:
        return mid
    cys = [(c["bbox"][1] + c["bbox"][3]) / 2.0 for c in chars]
    mapped = _monotone_assign(cys, vb, mid)                  # monotone 1-1 (fixes AE-1)
    if mapped is None:
        return mid
    if reseg_mode == "valley_n":
        return mapped
    # valley_guarded: accept valley for the whole column only if MLS not worse
    mv, mm = _mean_mls(vb, page_bgr, encoder), _mean_mls(mid, page_bgr, encoder)
    if mv is None or mm is None:
        return mid
    return mapped if mv >= mm else mid


def _pair_new(cluster: dict, syllables: list[str], qn_to_nom, similar,
              binary=None, reseg: bool = True, reseg_mode: str = "midpoint",
              encoder=None, page_bgr=None, det=None, page_boxes=None) -> tuple[list[dict], int]:
    """Banded anchored DP — emit only matches, gaps go unlabelled (REVIEW).

    With reseg=True the emitted bbox comes from a fresh column re-segmentation
    instead of the loose OCR per-char box. reseg_mode selects the method (see
    _pick_reseg); default 'midpoint' is the measured-best general choice (valley
    modes traded merging for fragments on diverged cols — seg_valley_n_ab.py).
    Falls back to the OCR box if re-segmentation is unavailable.
    """
    ops = realign_column(cluster["chars"], syllables, qn_to_nom, similar)
    mp = matched_pairs(ops)
    nom_chars = cluster["chars"]
    reseg_boxes = _pick_reseg(cluster, syllables, binary, reseg_mode, encoder, page_bgr,
                              det, page_boxes) if reseg else None
    out = []
    for p in mp:
        i = p["nom_idx"]
        bbox = reseg_boxes[i] if reseg_boxes else nom_chars[i].get("bbox")
        out.append({"ocr_char": p["ocr_char"], "bbox": bbox,
                    "syllable": p["syllable"], "confirmed": p["confirmed"]})
    # number of syllables left without a Nôm box (Nôm OCR misses) -> REVIEW
    n_gap = sum(1 for o in ops if o["op"] == "ins")
    return out, n_gap


def align_page(page_name: str, data_dir: Path, qn_dict_set: set,
               qn_to_nom: dict, similar: dict, mode: str,
               reseg_mode: str = "midpoint", encoder=None) -> dict | None:
    """Align one page in the given mode. Returns per-page record with pairs.

    reseg_mode (only used when mode != 'old'): 'midpoint' (default) | 'valley_n' |
    'valley_guarded'. valley_guarded needs `encoder` (NomEncoder) + loads the page
    image to apply the MLS guard. See _pick_reseg.
    """
    det = _detect(page_name, data_dir, qn_dict_set)
    if det is None:
        return None
    cols, qn_lines, iter_pairs, binary, page_ok = det
    page_bgr = None
    detector = None
    page_boxes = None
    if reseg_mode in ("valley_guarded", "detector"):
        import cv2 as _cv2
        page_bgr = _cv2.imread(str(data_dir / "pages" / f"{page_name}.png"), _cv2.IMREAD_COLOR)
    seg_backend = reseg_mode
    if reseg_mode == "detector":
        if page_bgr is None:
            raise DetectorUnavailableError(
                f"reseg=detector nhưng không đọc được ảnh trang {page_name}.png. {_HINT}")
        detector = _get_detector(strict=True)     # thiếu ckpt -> ném lỗi, KHÔNG rơi ngầm
        page_boxes = detector.boxes_for_page(page_bgr)   # all char boxes, once per page
        seg_backend = "detector_centernet_v1"

    pairs: list[dict] = []
    n_gap_total = 0
    n_norm_total = 0
    _readings = build_readings(qn_to_nom)
    for nom_idx, line_id in iter_pairs:
        cluster = cols[nom_idx]
        syllables = qn_lines[line_id]
        if not syllables:
            continue
        # CHUẨN HOÁ TRƯỚC KHI CĂN CHỈNH — vá dấu phụ VietOCR rụng, dùng đọc âm của
        # chính các chữ trong cột này. Đặt SAU build thì +0 ô vào bộ giao nộp; đặt ở
        # đây thì các ô được vá đủ điều kiện s1_inter_s2_direct = GOLD.
        syllables, _norm_log = normalize_column(cluster.get("chars"), syllables,
                                                qn_dict_set, _readings)
        n_norm_total += sum(1 for e in _norm_log if e.get("action") == "fixed")
        matched = (len(cluster["chars"]) == len(syllables))
        if mode == "old":
            col_pairs = _pair_old(cluster, syllables, binary)
            for p in col_pairs:
                p.update(column=line_id, matched=matched)
            pairs.extend(col_pairs)
        else:
            col_pairs, n_gap = _pair_new(cluster, syllables, qn_to_nom, similar,
                                         binary=binary, reseg_mode=reseg_mode,
                                         encoder=encoder, page_bgr=page_bgr,
                                         det=detector, page_boxes=page_boxes)
            n_gap_total += n_gap
            # anchored flag: a pair flanked by a confirmed neighbour. Its LOCAL
            # register is certain even if the whole column's counts diverged, so
            # it is GOLD/SILVER-eligible (gold_ok in consensus). NOTE: dropped the
            # old `p["confirmed"] and` conjunct — that made `anchored` imply
            # `confirmed`, but a confirmed pair already returns at the GOLD-direct
            # branch BEFORE SILVER, so the flag could never unblock anything. Now
            # an UN-confirmed pair next to a confirmed one can finally reach the
            # similar-bridge GOLD / SILVER(S3) paths in a count-diverged column.
            conf = [p["confirmed"] for p in col_pairs]
            for k, p in enumerate(col_pairs):
                nbr = (k > 0 and conf[k - 1]) or (k + 1 < len(col_pairs) and conf[k + 1])
                p.update(column=line_id, matched=matched,
                         anchored=bool(nbr))
            pairs.extend(col_pairs)
    return {"page": page_name, "page_ok": page_ok, "pairs": pairs,
            "n_review_gap": n_gap_total, "seg_backend": seg_backend,
            "n_syllable_normalized": n_norm_total}
