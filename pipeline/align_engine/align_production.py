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
from pipeline.align_engine.book_layout import (BookLayout, DEFAULT_LAYOUT,
                                                lithograph_gate, expected_qn_counts)


def _detect(page_name: str, data_dir: Path, qn_dict_set: set,
            layout: BookLayout | None = None, gate_out: dict | None = None):
    """Replicate the detection + QN parse + iter-plan of process_page_structural.

    Returns (cols, qn_lines, iter_pairs, binary, page_ok) or None if the page
    cannot be processed (5 phần tử như cũ — pipeline/lab/extract_columns.py còn unpack 5).
    `layout` (pipeline.align_engine.book_layout) cho số cột kỳ vọng: None/mặc định
    = 9 (STT, byte-identical); layout=lithograph thêm cổng page_ok (số cột ==
    n_columns, mỗi cột QN 14 âm hoặc như transcriptions ghi). `gate_out`: dict do
    bên gọi đưa vào để nhận chi tiết cổng (chỉ được ghi khi layout=lithograph).
    """
    lay = layout or DEFAULT_LAYOUT
    n_exp = lay.n_columns
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

    qn_lines, _ = _get_qn_lines(data_dir, page_name, qn_dict_set, n_columns=n_exp)
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
        cols, col_method = detect_nom_columns_v3(binary, ocr_columns, n_exp)
    else:
        from core.align.run_full import nom_cols_hybrid
        cols, col_method = nom_cols_hybrid(ocr_columns, min_len=4), "hybrid_no_image"

    page_col_match = (len(cols) == len(qn_keys))
    qn_parse_ok = (len(qn_lines) == n_exp)
    nom_suspect = (col_method == "suspect")
    max_qn = max(qn_keys) if qn_keys else 0
    partial_recovery = (not qn_parse_ok and not nom_suspect
                        and len(cols) >= max_qn and max_qn > 0)
    page_ok = ((page_col_match and qn_parse_ok and not nom_suspect)
               or partial_recovery)
    if lay.is_lithograph:
        # Cổng thạch bản: đủ n_columns cột Nôm VÀ QN, mỗi cột QN đủ 14 âm (6⧺8) hoặc
        # đúng num_syllables ghi trong transcriptions/<page>.json. Không partial_recovery.
        # projection_fallback LUÔN trả đúng n_expected cột (ép chiếu ảnh) nên đếm cột là
        # tautology ở nhánh đó -> thạch bản đòi phương pháp hybrid* (SPEC §7: >= 95 % trang).
        g_ok, gate = lithograph_gate(cols, qn_lines, lay,
                                     expected_qn_counts(data_dir, page_name))
        gate["col_method"] = col_method
        # "hybrid_no_image" = không nhị phân hoá được ảnh (cột chỉ từ cache OCR): không
        # phải kết quả dò trên ảnh -> không qua cổng (crop cũng không dựng được).
        page_ok = bool(g_ok and str(col_method).startswith("hybrid")
                       and col_method != "hybrid_no_image")
        if gate_out is not None:
            gate_out.update(gate)

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

# --- A-6 (flow N3f/N4d): HỘP THÔ DETECTOR + GÁN HỘP 3 NHÁNH ---------------------
# Trước 2026-09-16: ngưỡng tin cậy 0,3 ghim trong _get_detector, biên x ±0,5w ghim
# trong column_boxes, và hộp LUÔN bị ép về N = #âm tiết (enforce_count) rồi gán bằng
# _monotone_assign — kể cả 70% cột mà detector đã đếm đúng. Đo (DANH_GIA_KE_HOACH_
# NANG_CAP_2026-09-15, cột OCR=QN): thr 0,2 ±0,25w cho M==N 69,9→93,2%, M<N 27,1→1,9%,
# M>N 3,0→4,8% — nên phần lớn cột gán hộp THẲNG theo chỉ số (assign_boxes), chỉ cột
# lệch đếm mới đi đường ép đếm cũ.
# Đọc từ config/pipeline.yaml step2.det_thr / step2.det_xmargin (build_dataset gán vào
# đây lúc khởi động, như BOX_OVERLAP_FRAC). `--box-rule legacy` KHÔNG đọc hai hằng này:
# nó dùng trọn bộ LEGACY_* để tái lập bộ cũ (selftest: bbox khớp 100% trên 5 trang).
DETECTOR_THR = 0.2
DETECTOR_XMARGIN = 0.25
MONOTONE_GUARD = 0.35       # _monotone_assign: hộp xa tâm OCR quá guard×pitch -> midpoint
# Bộ hằng luật CŨ, dùng TRỌN GÓI (thr + biên x + đường ép đếm _pick_reseg) cho
# --box-rule legacy và cho cột có ô khoá QĐ-01 trong luật mới (không trộn hộp cũ của ô
# khoá với hộp mới của hàng xóm -> trùng bbox -> census AE-1 cách ly cả hai).
LEGACY_DETECTOR_THR = 0.3
LEGACY_DETECTOR_XMARGIN = 0.5
BOX_RULES = ("syl_index", "legacy")


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


_DETECTORS: dict = {}          # thr -> DetectorInfer | None — cache THEO thr (A-6)
_DETECTOR_TRIED: set = set()


class DetectorUnavailableError(FileNotFoundError):
    """Yêu cầu reseg=detector nhưng không dựng được detector — KHÔNG rơi ngầm."""


_HINT = ("tải: huggingface-cli download mdnt571/nom-char-det detector_r34.best.pt "
         "--local-dir train_crop/  — hoặc chạy lại với --reseg midpoint nếu CỐ Ý "
         "muốn dùng trung điểm (bộ crop sẽ KHÁC HẲN, phải đo lại).")


def _get_detector(strict: bool = True, thr: float | None = None):
    """Lazy, cached CenterNet detector (train_crop/detector_r34.best.pt), cache theo thr.

    thr=None -> DETECTOR_THR (đọc lúc gọi, nên config ghi đè có hiệu lực); --box-rule
    legacy truyền LEGACY_DETECTOR_THR để tái lập đúng bộ cũ (0,3) — hai đối tượng
    detector sống song song, mỗi ngưỡng một cache.
    strict=True (mặc định): thiếu checkpoint -> NÉM DetectorUnavailableError.
    Trước 2026-08-23 hàm này chỉ IN một dòng log rồi lặng lẽ rơi về midpoint: toàn bộ
    hộp ký tự bị tách bằng trung điểm thay vì CenterNet, cho ra bộ crop khác hẳn, mà
    labels.csv KHÔNG ghi lại backend nào đã dùng -> không ai truy được về sau.
    Chỉ dùng bởi reseg_mode='detector'.
    """
    thr = float(DETECTOR_THR if thr is None else thr)
    if thr in _DETECTOR_TRIED:
        d = _DETECTORS.get(thr)
        if strict and (d is None or not getattr(d, "trained", False)):
            raise DetectorUnavailableError(f"reseg=detector nhưng detector không dùng được. {_HINT}")
        return d
    _DETECTOR_TRIED.add(thr)
    err = None
    d = None
    try:
        from pipeline.align_engine.char_detector.detector_infer import DetectorInfer
        # thr: trước A-6 ghim 0,3 (bớt detection tin cậy thấp lọt vào enforce_count);
        # nay DETECTOR_THR 0,2 vì hộp thô không còn bị ép đếm ở cột đếm đúng.
        d = DetectorInfer(thr=thr)  # tự tìm ckpt v1 ở train_crop/detector_r34.best.pt
        if not d.trained:
            err = "không thấy train_crop/detector_r34.best.pt"
        else:
            print(f"  [reseg detector] CenterNet v1 (img {d.img}, seam, thr {thr}).", flush=True)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        d = None
    _DETECTORS[thr] = d
    if err:
        if strict:
            raise DetectorUnavailableError(f"reseg=detector nhưng {err}. {_HINT}")
        print(f"  [reseg detector] {err} -> midpoint fallback.", flush=True)
    return d


def _legacy_page_boxes(page_boxes, thr_run: float, page_bgr=None):
    """Hộp trang theo NGƯỠNG CŨ (LEGACY_DETECTOR_THR) từ lần chạy ở thr_run.

    decode() của CenterNet = top-k(1024) đỉnh heatmap rồi lọc score >= thr, nên lọc lại
    theo score từ lần chạy ở ngưỡng thấp hơn CHO ĐÚNG tập hộp của lần chạy ở 0,3
    (selftest kiểm bằng detector thứ hai trên 5 trang). Chỉ khi thr_run > 0,3 mới
    phải chạy detector thứ hai.
    """
    if thr_run <= LEGACY_DETECTOR_THR:
        return [b for b in page_boxes if b[4] >= LEGACY_DETECTOR_THR]
    return _get_detector(strict=True, thr=LEGACY_DETECTOR_THR).boxes_for_page(page_bgr)


def preflight_detector(reseg_mode: str, box_rule: str = "syl_index") -> str:
    """Kiểm detector MỘT LẦN trước khi duyệt trang. Trả tên backend sẽ dùng.

    Fail fast: thiếu checkpoint thì dừng ngay ở trang đầu tiên chứ không âm thầm
    tách bằng trung điểm cho cả 445 trang. box_rule quyết ngưỡng (legacy = 0,3).
    """
    if reseg_mode != "detector":
        return reseg_mode
    _get_detector(strict=True, thr=LEGACY_DETECTOR_THR if box_rule == "legacy" else None)
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
                        A-6: đây là ĐƯỜNG CŨ trọn gói (ép đếm về N, ±0,5w) — chỉ còn
                        dùng cho --box-rule legacy và cột có ô khoá QĐ-01; luật mới
                        đi qua assign_boxes.
    Returns a list indexed like cluster['chars'] (or None)."""
    mid = _reseg_column(cluster)
    chars = cluster["chars"]; n = len(syllables)
    if reseg_mode == "detector" and det is not None and page_boxes is not None and cluster.get("x_range"):
        cb = det.column_boxes(page_boxes, cluster["x_range"], n, x_margin=LEGACY_DETECTOR_XMARGIN)
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


def assign_boxes(G, ops, n_ocr, n_qn, cluster=None, cb=None, det=None, guard=None):
    """Gán hộp ảnh cho từng chữ OCR theo LUẬT 3 NHÁNH (A-6, flow N4d). Hàm thuần —
    engine (PASS 1 với ops lượt 1, PASS 1b với ops lượt 2) và lab `thuc_nghiem.py geo`
    cùng gọi, để số đo của lab là số đo của chính đường sản xuất.

    G     : hộp thô detector của cột (DetectorInfer.raw_column_boxes: đã NMS, sắp theo y,
            [x1,y1,x2,y2,score]).
    ops   : ops DP (lượt cuối) của cột — nhánh (a) cần syl_idx của từng match.
    cb    : hộp đã ép đếm G -> n_qn (DetectorInfer.enforce_count) cho nhánh (c); PASS 1
            tính sẵn theo ảnh xám của ĐÚNG trang (det chỉ giữ ảnh xám của trang cuối),
            PASS 1b truyền lại. cb=None và det có -> tự tính.
    Trả (boxes, box_source, count_source):
      boxes[i]      = hộp cho nom_idx i (None nếu không có / chữ không được ghép ở nhánh a)
      box_source[i] ∈ {'detector', 'split', 'midpoint', ''}
      count_source  ∈ {'equal_qn', 'equal_ocr', 'conflict'} (hai giá trị còn lại của enum,
                      'legacy' / 'legacy_locked_col', do _pair_new_state ghi khi cột đi
                      đường trọn gói luật cũ — không qua hàm này)
      (a) |G| == n_qn          -> boxes[i] = G[j] với (i, j) match — theo THỨ TỰ ÂM (khe
                                  của DP văn bản đặt đúng chỗ chữ rụng; lab: 96–97% hộp đúng)
      (b) |G| == n_ocr != n_qn -> boxes[i] = G[i] — theo THỨ TỰ CHỮ (âm rụng, chữ đủ)
      (c) còn lại              -> enforce_count(G -> n_qn) + _monotone_assign + x-guard
                                  y hệt luật cũ (_pick_reseg); hộp do enforce_count bổ đôi
                                  = 'split', hộp rơi về trung điểm = 'midpoint'.
    """
    G = G or []
    boxes: list = [None] * n_ocr
    src: list = [""] * n_ocr
    if n_qn > 0 and len(G) == n_qn:
        for o in ops:
            if o.get("op") == "match":
                i = o["nom_idx"]
                if 0 <= i < n_ocr:
                    boxes[i] = list(G[o["syl_idx"]][:4])
                    src[i] = "detector"
        return boxes, src, "equal_qn"
    if n_ocr > 0 and len(G) == n_ocr:
        return [list(g[:4]) for g in G], ["detector"] * n_ocr, "equal_ocr"
    # (c) — đường ép đếm cũ; cần cluster (tâm OCR, x_range) và cb
    mid = _reseg_column(cluster) if cluster else None
    if cb is None and det is not None and n_qn > 0:
        cb = det.enforce_count(G, n_qn)
    gset = {tuple(int(v) for v in g[:4]) for g in G}
    if cluster and cb and len(cb) == n_qn and cluster.get("x_range"):
        chars = cluster.get("chars") or []
        cys = [(c["bbox"][1] + c["bbox"][3]) / 2.0 for c in chars]
        assigned = _monotone_assign(cys, cb, mid, guard=MONOTONE_GUARD if guard is None else guard)
        if assigned is not None:
            x1, x2 = cluster["x_range"]
            xtol = 0.10 * (x2 - x1)
            for i, box in enumerate(assigned):
                if box is None:
                    boxes[i], src[i] = (mid[i], "midpoint") if mid else (None, "")
                    continue
                bcx = (box[0] + box[2]) / 2.0
                if bcx < x1 - xtol or bcx > x2 + xtol:
                    box = mid[i] if mid else box
                boxes[i] = list(box)
                if mid and list(box) == list(mid[i]):
                    src[i] = "midpoint"
                else:
                    src[i] = "detector" if tuple(int(v) for v in box[:4]) in gset else "split"
            return boxes, src, "conflict"
    if mid:
        return [list(b) for b in mid], ["midpoint"] * n_ocr, "conflict"
    return boxes, src, "conflict"


def _pair_new_state(cluster: dict, syllables: list[str], qn_to_nom, similar,
                    binary=None, reseg: bool = True, reseg_mode: str = "midpoint",
                    encoder=None, page_bgr=None, det=None, page_boxes=None,
                    box_rule: str = "syl_index", legacy_page_boxes=None
                    ) -> tuple[list[dict], int, list[dict], list | None, dict]:
    """Như `_pair_new` nhưng trả thêm (ops lượt 1, reseg_boxes, box_info) — trạng thái
    cột cho PASS 1b (flow N3g): build_dataset chạy DP lại với `cost_fn` neo ngữ liệu
    trên đúng `chars`/`syllables`/hộp này, không phải dò lại trang lần hai.

    box_rule (A-6): 'syl_index' = hộp thô G (thr DETECTOR_THR, ±DETECTOR_XMARGIN) + 3
    nhánh assign_boxes; 'legacy' / 'legacy_locked_col' = trọn gói luật cũ (_pick_reseg
    trên legacy_page_boxes = hộp ở ngưỡng 0,3, biên ±0,5w; count_source == box_source ==
    box_rule, tức 'legacy' hoặc 'legacy_locked_col'). box_info = {G, cb, n_ocr, n_qn,
    n_det, count_source, box_source (theo nom_idx), box_rule} để PASS 1b gán lại hộp
    theo ops lượt 2 mà không cần detector.
    """
    ops = realign_column(cluster["chars"], syllables, qn_to_nom, similar)
    mp = matched_pairs(ops)
    nom_chars = cluster["chars"]
    n_ocr, n_qn = len(nom_chars), len(syllables)
    use_det = (reseg_mode == "detector" and det is not None and page_boxes is not None
               and bool(cluster.get("x_range")))
    G = cb = None
    n_det = ""
    count_source = ""
    box_source = None
    if not reseg:
        reseg_boxes = None
    elif use_det and box_rule == "syl_index":
        G = det.raw_column_boxes(page_boxes, cluster["x_range"], DETECTOR_XMARGIN)
        n_det = len(G)
        if len(G) != n_qn and len(G) != n_ocr:
            cb = det.enforce_count(G, n_qn)       # nhánh (c): ép đếm trên ảnh xám ĐÚNG trang
        reseg_boxes, box_source, count_source = assign_boxes(G, ops, n_ocr, n_qn,
                                                             cluster=cluster, cb=cb)
    else:
        pb = legacy_page_boxes if (use_det and legacy_page_boxes is not None) else page_boxes
        reseg_boxes = _pick_reseg(cluster, syllables, binary, reseg_mode, encoder, page_bgr,
                                  det, pb)
        if use_det:
            # n_det ghi theo đúng bộ hằng cũ (thr 0,3 ±0,5w) mà cột này thực dùng
            n_det = len(det.raw_column_boxes(pb, cluster["x_range"], LEGACY_DETECTOR_XMARGIN))
            # count_source đi cùng box_source: 'legacy' (--box-rule legacy) hoặc
            # 'legacy_locked_col' (cột khoá QĐ-01 trong luật syl_index) — enum FLOW §172
            count_source = box_rule if box_rule != "syl_index" else "legacy"
            box_source = [count_source] * n_ocr
        else:
            box_source = [reseg_mode if reseg_mode != "detector" else "midpoint"] * n_ocr
    box_info = {"G": G, "cb": cb, "n_ocr": n_ocr, "n_qn": n_qn, "n_det": n_det,
                "count_source": count_source, "box_source": box_source,
                "box_rule": box_rule if use_det else reseg_mode}
    out = []
    for p in mp:
        i = p["nom_idx"]
        bbox = reseg_boxes[i] if reseg_boxes else nom_chars[i].get("bbox")
        # nom_idx/syl_idx (N0b): vị trí ô trong cột Nôm / âm tiết trong dòng QN —
        # khoá bền để đối soát thế hệ và khoá QĐ-01; chỉ PHÁT THÊM, không đổi hành vi.
        out.append({"ocr_char": p["ocr_char"], "bbox": bbox,
                    "syllable": p["syllable"], "confirmed": p["confirmed"],
                    "nom_idx": i, "syl_idx": p["syl_idx"],
                    # A-6 (N4e): số đếm cột + nguồn hộp — sang columns.csv ở S9
                    "n_ocr": n_ocr, "n_qn": n_qn, "n_det": n_det,
                    "count_source": count_source,
                    "box_source": box_source[i] if box_source else ""})
    # number of syllables left without a Nôm box (Nôm OCR misses) -> REVIEW
    n_gap = sum(1 for o in ops if o["op"] == "ins")
    return out, n_gap, ops, reseg_boxes, box_info


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
    out, n_gap, _ops, _boxes, _info = _pair_new_state(
        cluster, syllables, qn_to_nom, similar, binary=binary, reseg=reseg,
        reseg_mode=reseg_mode, encoder=encoder, page_bgr=page_bgr, det=det,
        page_boxes=page_boxes)
    return out, n_gap


def align_page(page_name: str, data_dir: Path, qn_dict_set: set,
               qn_to_nom: dict, similar: dict, mode: str,
               reseg_mode: str = "midpoint", encoder=None,
               box_rule: str = "syl_index", locked_columns=None,
               legacy_also_columns=None, layout: BookLayout | None = None) -> dict | None:
    """Align one page in the given mode. Returns per-page record with pairs.

    layout (pipeline.align_engine.book_layout.BookLayout, tuỳ chọn): số cột kỳ vọng
    và kiểu trang của sách; None = STT 9 cột (mặc định, không đổi kết quả).

    reseg_mode (only used when mode != 'old'): 'midpoint' (default) | 'valley_n' |
    'valley_guarded'. valley_guarded needs `encoder` (NomEncoder) + loads the page
    image to apply the MLS guard. See _pick_reseg.

    box_rule (A-6, chỉ với reseg_mode='detector'): 'syl_index' (mặc định) = hộp thô ở
    DETECTOR_THR/±DETECTOR_XMARGIN + assign_boxes 3 nhánh; 'legacy' = trọn gói luật cũ
    (thr 0,3, ±0,5w, _pick_reseg) tái lập bộ cũ. locked_columns = tập line_id (cột QN)
    có ô khoá QĐ-01 trên trang này: cột ấy chạy trọn gói luật cũ dù box_rule=syl_index
    (box_source='legacy_locked_col'). legacy_also_columns (B-5, --lock-scope cell*):
    tập line_id chạy luật MỚI như thường nhưng tính THÊM hộp trọn gói luật cũ vào
    col_state['legacy_boxes'] (theo nom_idx) để PASS 1c có thể lùi cả cột về luật cũ
    khi ô khoá QĐ-01 cho thấy hộp 3 nhánh lệch.

    Kết quả (mode='new') còn có `col_states` (flow N3g) — mỗi cột một dict
    {line_id, cluster, syllables (sau normalize_column), syllable_ocr (VietOCR
    nguyên văn, lower), matched, reseg_boxes (theo nom_idx), ops1 (DP lượt 1),
    G, cb, n_ocr, n_qn, n_det, count_source, box_source, box_rule} để build_dataset
    PASS 1b chạy DP lại với neo ngữ liệu và gán lại hộp mà không dò lại trang.
    """
    layout_gate: dict = {}          # chỉ được ghi khi layout=lithograph
    det = _detect(page_name, data_dir, qn_dict_set, layout=layout, gate_out=layout_gate)
    if det is None:
        return None
    cols, qn_lines, iter_pairs, binary, page_ok = det
    page_bgr = None
    detector = None
    page_boxes = None
    legacy_page_boxes = None
    locked_columns = set(locked_columns or ())
    legacy_also_columns = set(legacy_also_columns or ())
    if reseg_mode in ("valley_guarded", "detector"):
        import cv2 as _cv2
        page_bgr = _cv2.imread(str(data_dir / "pages" / f"{page_name}.png"), _cv2.IMREAD_COLOR)
    seg_backend = reseg_mode
    if reseg_mode == "detector":
        if page_bgr is None:
            raise DetectorUnavailableError(
                f"reseg=detector nhưng không đọc được ảnh trang {page_name}.png. {_HINT}")
        thr = LEGACY_DETECTOR_THR if box_rule == "legacy" else DETECTOR_THR
        detector = _get_detector(strict=True, thr=thr)   # thiếu ckpt -> ném lỗi, KHÔNG rơi ngầm
        page_boxes = detector.boxes_for_page(page_bgr)   # all char boxes, once per page
        if box_rule == "legacy":
            legacy_page_boxes = page_boxes
        elif locked_columns or legacy_also_columns:
            # cột có ô khoá QĐ-01: hộp ở ngưỡng cũ 0,3 (lọc lại từ lần chạy 0,2 — cùng tập)
            legacy_page_boxes = _legacy_page_boxes(page_boxes, thr, page_bgr)
        seg_backend = "detector_centernet_v1"

    pairs: list[dict] = []
    # col_states (flow N3g): trạng thái từng cột sau lượt DP 1 — build_dataset PASS 1b
    # chạy DP lại với neo ngữ liệu trên chính trạng thái này (detector 1 lần/trang).
    col_states: list[dict] = []
    n_gap_total = 0
    n_norm_total = 0
    _readings = build_readings(qn_to_nom)
    for nom_idx, line_id in iter_pairs:
        cluster = cols[nom_idx]
        syllables = qn_lines[line_id]
        if not syllables:
            continue
        # âm VietOCR NGUYÊN VĂN (lower) TRƯỚC normalize_column — cột `syllable_ocr`
        # (flow N3c): "ghi đè bản in = 0" phải đo trên chuỗi này, không phải chuỗi đã vá.
        syllable_ocr = [str(x).lower() for x in syllables]
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
            # A-6 (N4d): cột có ô khoá QĐ-01 -> trọn gói luật cũ cho CẢ cột
            col_rule = box_rule
            if box_rule == "syl_index" and line_id in locked_columns:
                col_rule = "legacy_locked_col"
            col_pairs, n_gap, ops1, reseg_boxes, box_info = _pair_new_state(
                cluster, syllables, qn_to_nom, similar,
                binary=binary, reseg_mode=reseg_mode,
                encoder=encoder, page_bgr=page_bgr,
                det=detector, page_boxes=page_boxes,
                box_rule=col_rule, legacy_page_boxes=legacy_page_boxes)
            n_gap_total += n_gap
            # syllable_raw = âm SAU normalize_column (đầu vào DP); syllable_ocr = âm
            # VietOCR nguyên văn. Chỉ PHÁT THÊM vào pair, không đổi hành vi ghép.
            for p in col_pairs:
                j = p["syl_idx"]
                p["syllable_raw"] = syllables[j]
                p["syllable_ocr"] = syllable_ocr[j] if len(syllable_ocr) == len(syllables) else ""
            legacy_boxes = None
            if (line_id in legacy_also_columns and col_rule == "syl_index"
                    and reseg_mode == "detector" and legacy_page_boxes is not None):
                # B-5: hộp trọn gói luật cũ (thr 0,3, ±0,5w, ép đếm + _monotone_assign)
                # của cùng cột — theo nom_idx, không phụ thuộc đường ghép
                legacy_boxes = _pick_reseg(cluster, syllables, binary, reseg_mode, encoder,
                                           page_bgr, detector, legacy_page_boxes)
            col_states.append({
                "line_id": line_id, "cluster": cluster, "syllables": syllables,
                "syllable_ocr": syllable_ocr, "matched": matched,
                "reseg_boxes": reseg_boxes, "ops1": ops1,
                "legacy_boxes": legacy_boxes,     # B-5: chỉ khác None ở cột legacy_also_columns
                **box_info,          # A-6: G, cb, n_ocr, n_qn, n_det, count/box_source, box_rule
            })
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
    rec = {"page": page_name, "page_ok": page_ok, "pairs": pairs,
           "n_review_gap": n_gap_total, "seg_backend": seg_backend,
           "n_syllable_normalized": n_norm_total,
           # N3g: trạng thái cột cho PASS 1b; `pairs` giữ nguyên để tương thích
           "col_states": col_states}
    if layout_gate:
        rec["layout_gate"] = layout_gate      # chỉ có với layout=lithograph
    return rec
