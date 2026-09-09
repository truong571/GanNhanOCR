"""Build the FINAL labeled dataset end-to-end (NEW pipeline, no A/B).

Two passes:
  PASS 1 — align every page (production-faithful: detect_nom_columns_v3 + parse_v5
           + bbox_fix offset + banded anchored DP + re-segment) and tier each pair
           via consensus.decide_label.
  PROMOTE — cross-page-consistent REVIEW "unconfirmed" pairs -> SYLLABLE tier
           (semantic/nghĩa borrowings the phonetic dict can't contain). [#6]
  SPLIT  — leakage-safe train/val/test by group=(book,page,column). [#4]
  PASS 2 — materialize crops (GOLD + SILVER + SYLLABLE; +REVIEW with --crop-review)
           with per-crop quality columns (ink%, size, md5, seg_flag). [#3,#5]

Tiers (label_level separates char- vs syllable-supervision):
  GOLD     label_level=char     dict-confirmed char (direct, or unique similar-bridge)
  SILVER   label_level=char     visual S3 (OFF until a Nôm-trained model replaces DINOv2)
  SYLLABLE label_level=syllable char unconfirmed but syllable reliable & cross-page-consistent
  REVIEW   label_level=''       not usable as a label (kept in manifest with bbox)

Run:
  cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
  .venv/bin/python evaluation/ver_new/build_dataset.py
  # options: --no-tighten --pad 0.12 --limit N --out <dir> --crop-review --use-s3
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import (                                    # noqa: E402
    build_nom_to_qn, load_qn_to_nom, load_similarity_dict)
from core.text.text_utils import is_plausible_qn_syllable, strip_all  # noqa: E402
from pipeline.align_engine import align_production as ap_mod          # noqa: E402
from pipeline.align_engine.align_production import (                    # noqa: E402
    DetectorUnavailableError, align_page, preflight_detector)
from pipeline.align_engine.consensus import (                         # noqa: E402
    AM_DA_QUYET, am_da_quyet, chuan_am, decide_label)
from pipeline.align_engine.bbox_fix import tighten_box, carve_neighbor_ink  # noqa: E402


def _book_code(name: str) -> str:
    """Short, meaningful book code for the labels.csv `book` field.

    'SachThanhTruyen2' -> 'stt2', 'SachThanhTruyen11' -> 'stt11' (STT = Sách
    Thánh Truyện). Replaces the old `book[12:]` substring hack, which sliced
    'SachThanhTruyen2' into the accidental, meaningless 'yen2'. The audit reverse
    map (audit_grid.book_to_scan_dir) keys off the trailing digits, so it keeps
    working for either code. Falls back to the lowercased name for other books.
    """
    m = re.match(r"SachThanhTruyen(\d+)$", name)
    return f"stt{m.group(1)}" if m else name.lower()

# SYLLABLE-tier gate (cross-page consistency of an unconfirmed char's reading).
SYL_MIN_OCC = 5      # the (char,syllable) must occur >= this many times corpus-wide
SYL_MIN_PAGES = 3    # on >= this many distinct pages
SYL_MIN_PURITY = 0.6  # and be the dominant syllable for that char by this share

# --------------------------------------------------------------------------- #
# NEO GOLD-TRỰC-TIẾP — bằng chứng cấp ngữ liệu dùng chung cho L1 và L3.
# --------------------------------------------------------------------------- #
# Chỉ đếm ô `s1_inter_s2_direct` của PASS 1: đó là luật DUY NHẤT không cần cầu tự
# dạng, không cần S3, không cần cột khớp — nó là ocr_char NẰM SẴN trong từ điển
# đọc âm của chính âm ấy. Vì L1/L3 phát ra rule id KHÁC ("..._am_sua_dau",
# "..._cot_lech") nên ô do chúng sinh KHÔNG BAO GIỜ quay lại làm neo cho chính
# chúng: không có vòng tự khẳng định, dù chạy lại pipeline bao nhiêu lần.
GOLD_DIRECT_RULE = "s1_inter_s2_direct"
ANCHOR_MIN_OCC = 5     # cặp (chữ, âm) phải có >= ngần này ô GOLD-trực-tiếp
ANCHOR_MIN_PAGES = 3   # trên >= ngần này TRANG khác nhau (chặn 1 trang hỏng tự neo mình)
L3_MIN_ATTEST = 1      # L3 chỉ đòi cặp (chữ-cầu, âm) ĐÃ TỪNG được chứng thực GOLD-trực-tiếp

# rule id do bộ vá này sinh ra — gom một chỗ để bảng số liệu/báo cáo bắt được hết
RULE_L1 = "s1_inter_s2_direct_am_sua_dau"
RULE_L3 = "s1_inter_s2_similar_cot_lech"


def gold_direct_anchors(records):
    """{(ocr_char, âm-thường): (số ô, số trang)} trên riêng ô GOLD-TRỰC-TIẾP."""
    cnt = Counter()
    pages_of = defaultdict(set)
    for r in records:
        if r["tier"] == "GOLD" and r["rule"] == GOLD_DIRECT_RULE and r["ocr_char"]:
            k = (r["ocr_char"], str(r["syllable"]).lower())
            cnt[k] += 1
            pages_of[k].add((r["book"], r["page"]))
    return {k: (n, len(pages_of[k])) for k, n in cnt.items()}


def _du_neo(anchors, key, min_occ=ANCHOR_MIN_OCC, min_pages=ANCHOR_MIN_PAGES):
    n, npg = anchors.get(key, (0, 0))
    return n >= min_occ and npg >= min_pages


def apply_am_sua_dau(records, qn_to_nom, nom_to_qn, anchors,
                     min_occ=ANCHOR_MIN_OCC, min_pages=ANCHOR_MIN_PAGES):
    """L1 · SỬA DẤU CỦA ÂM khi chính ocr_char đã có neo GOLD ở âm cùng khung xương.

    VietOCR rụng dấu ("den" cho "đến", "ay" cho "ấy") hoặc lệch thanh. Khi âm sai,
    R = qn_to_nom[âm] sai theo, nên `s1_inter_s2_direct` không thể khớp và ô rơi
    xuống REVIEW/SILVER dù chữ Nôm OCR đọc ra ĐÚNG.

    Luật: giữ nguyên ocr_char (S1), SỬA ÂM (S2) — ngược hẳn với các luật cầu tự
    dạng. Chỉ nhận khi:
      · ô chưa phải GOLD, có ocr_char, và ocr_char ∉ R(âm hiện tại);
      · âm ứng viên là một ĐỌC ÂM CÓ THẬT của chính ocr_char (nom_to_qn), cùng
        KHUNG XƯƠNG (strip_all: bỏ mọi dấu phụ + thanh, đ→d) với âm hiện tại;
      · cặp (ocr_char, âm ứng viên) đã có >= min_occ ô GOLD-trực-tiếp trên
        >= min_pages TRANG — tức chính ngữ liệu này đã chứng thực cách đọc ấy;
      · và còn ĐÚNG MỘT ứng viên. Nhiều hơn một -> ĐỂ NGUYÊN, không đoán.

    `strip_all` gộp cả dấu tạo chữ nên "vua"/"vừa" cùng khung xương — chính vì thế
    ràng buộc neo GOLD (>=5 ô/>=3 trang) là bắt buộc, nó mới là phần mang bằng chứng.
    """
    n_moi = n_nang = 0
    for r in records:
        if r["tier"] == "GOLD":
            continue
        ch = r["ocr_char"]
        if not ch:
            continue
        syl = str(r["syllable"]).lower()
        if am_da_quyet(syl):
            continue            # CHỐT CHẶN: nhường quyền quyết cho glyph_fix (QĐ-01)
        if ch in qn_to_nom.get(syl, []):
            continue            # đã là GOLD-trực-tiếp, không việc gì tới đây
        khung = strip_all(syl)
        cands = []
        for alt in nom_to_qn.get(ch, ()):
            alt = str(alt).lower()
            if alt == syl or strip_all(alt) != khung:
                continue
            if am_da_quyet(alt) or not is_plausible_qn_syllable(alt):
                continue
            if not _du_neo(anchors, (ch, alt), min_occ, min_pages):
                continue
            if alt not in cands:
                cands.append(alt)
        if len(cands) != 1:
            continue
        if r["tier"] == "REVIEW":
            n_moi += 1
        else:
            n_nang += 1          # SILVER (nhãn do S3 quyết) -> GOLD (nhãn do từ điển quyết)
        r["syllable"] = cands[0]
        r["label"] = ch
        r["tier"] = "GOLD"
        r["rule"] = RULE_L1
    return n_moi, n_nang


def apply_cot_lech_cau_xuoi(records, qn_to_nom, similar_dict, anchors,
                            min_attest=L3_MIN_ATTEST):
    """L3 · CỘT LỆCH + ĐÚNG 1 CẦU XUÔI + CẶP ĐÃ CHỨNG THỰC -> GOLD.

    `decide_label` chặn cầu tự dạng khi cột lệch (gold_ok=False) vì khi ấy không
    có gì bảo đảm ô này ứng với âm này. Nhưng cầu tự dạng DUY NHẤT cộng với việc
    cặp (chữ-cầu, âm) ĐÃ được chính ngữ liệu chứng thực ở tier GOLD-trực-tiếp là
    hai điều kiện độc lập nhau: một đến từ hình chữ (Dict/SinoNom_Similar), một
    đến từ thống kê ghép đã xác nhận. Hai cái cùng trỏ một chỗ thì đủ.

    Chạy SAU L1 và chỉ trên ô còn `diverged_column` — ô L1 đã cứu thì không đụng lại.
    """
    n = 0
    for r in records:
        if r["tier"] != "REVIEW" or r["rule"] != "diverged_column":
            continue
        ch = r["ocr_char"]
        if not ch:
            continue
        syl = str(r["syllable"]).lower()
        if am_da_quyet(syl) or not is_plausible_qn_syllable(syl):
            continue            # CHỐT CHẶN + không cho âm rác neo nhãn
        R = qn_to_nom.get(syl, [])
        if not R or ch in R:
            continue
        cau = list(dict.fromkeys(s for s in similar_dict.get(ch, []) if s in R and s != ch))
        if len(cau) != 1:
            continue
        if anchors.get((cau[0], syl), (0, 0))[0] < min_attest:
            continue            # cặp chưa từng được chứng thực -> không nhận
        r["label"] = cau[0]
        r["tier"] = "GOLD"
        r["rule"] = RULE_L3
        n += 1
    return n


def be_day_du(r, unconf) -> bool:
    """L5 · Ô có nằm trong BỂ ĐẦY ĐỦ của cổng âm tiết không?

    Bể = MỌI ô mà `ocr_char ∉ R(âm)` — tức mọi ô mà S1∩S2 KHÔNG tự khớp. Tập đó
    chính là REVIEW ∪ SILVER: một ô có ocr_char ∈ R đã về GOLD-trực-tiếp ở nhánh
    đầu `decide_label`, nên không ô GOLD nào thuộc bể.

    Vì sao PHẢI gộp SILVER vào: bể cũ chỉ lấy REVIEW, mà SILVER chính là phần S3
    ĐÃ NHẶT RA KHỎI REVIEW. Nghĩa là mẫu số của phép tính độ thuần bị cắt theo đúng
    tiêu chí của một tín hiệu mà bước `s3_unwind` đã gỡ quyền quyết (CI95 của AUC
    bắt lỗi = [0,459; 0,672], chứa 0,5). Đo được: bể cũ cho chữ 𭔿 độ thuần 0,79,
    bể đầy đủ cho 0,502 — con số thứ hai mới là độ thuần THẬT của chữ ấy.
    """
    if r["tier"] == "SILVER":
        return True
    return r["tier"] == "REVIEW" and r["rule"] in unconf


def syllable_gate(records, unconf, min_occ=SYL_MIN_OCC, min_pages=SYL_MIN_PAGES,
                  min_purity=SYL_MIN_PURITY, bo_qua_am=AM_DA_QUYET, gom_silver=True):
    """(ocr_char, LOWERCASED syllable) pairs passing the cross-page consistency gate.

    Case-insensitive on the syllable so 'Nhị' and 'nhị' merge into one class — the
    cased keying used to split them, diluting the occurrence/purity thresholds and
    dropping ~1,131 labels. Pure + deterministic; unit-tested in phase1_engine_selftest.

    L5 · BỂ ĐẦY ĐỦ: `unconf` do lời gọi truyền vào, và main() nay truyền CẢ
    "diverged_column"; `gom_silver=True` gộp thêm tier SILVER — xem `be_day_du`.
    `gom_silver=False` khôi phục đúng hành vi cũ (dùng cho kiểm thử hồi quy).

    `bo_qua_am` — âm đã có phán quyết người: VẪN ĐẾM vào mẫu số (nếu loại khỏi mẫu số
    thì một chữ như 㝵 mất 1.183 ô "người" trong denominator và âm phổ biến THỨ HAI
    của nó bỗng đủ độ thuần, tức chốt chặn lại đẻ ra đúng cái nó định chặn), nhưng
    KHÔNG BAO GIỜ được trả ra làm cặp hợp lệ.
    """
    cnt = defaultdict(Counter)
    pages_of = defaultdict(lambda: defaultdict(set))
    for r in records:
        if r["ocr_char"] and (be_day_du(r, unconf) if gom_silver else
                              (r["tier"] == "REVIEW" and r["rule"] in unconf)):
            syl = str(r["syllable"]).lower()
            if not is_plausible_qn_syllable(syl):
                continue      # garbage ('0'/'2017') must never become a SYLLABLE target
            cnt[r["ocr_char"]][syl] += 1
            pages_of[r["ocr_char"]][syl].add((r["book"], r["page"]))
    syl_ok = set()
    for ch, c in cnt.items():
        syl, n = c.most_common(1)[0]
        if chuan_am(syl) in (bo_qua_am or ()):
            continue          # CHỐT CHẶN: nhường quyền quyết cho glyph_fix (QĐ-01)
        if (n >= min_occ and len(pages_of[ch][syl]) >= min_pages
                and n / sum(c.values()) >= min_purity):
            syl_ok.add((ch, syl))
    return syl_ok


def maybe_s3(p, page_png, qn_to_nom, vs3):
    if vs3 is None or not p.get("ocr_char"):
        return None
    if not (p.get("matched") or p.get("anchored")):
        return None
    cands = qn_to_nom.get((p["syllable"] or "").lower(), [])
    if p["ocr_char"] in cands:
        return None
    return vs3.compute(page_png, p.get("bbox"), p["ocr_char"], cands)


def _seg_flag(crop_gray) -> str:
    """Cheap advisory flag: 'tall' crops may be a merged 2-glyph or a tall char."""
    h, w = crop_gray.shape[:2]
    return "tall" if h > 1.8 * max(w, 1) else "ok"


def save_crop(img, gray_full, bbox, pad, path: Path, tighten: bool = True,
              prev_bbox=None, next_bbox=None) -> dict | None:
    """Cut + carve-neighbour-ink + (tighten) + save a crop; return per-crop quality
    stats or None. prev_bbox/next_bbox = raw bbox of the char immediately before/after
    THIS one in the SAME column (by y-order) — used to erase neighbour ink that
    bled into the padded window (tighten_box alone can't separate 2 touching glyphs)."""
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
    if gray_full is not None and (prev_bbox is not None or next_bbox is not None):
        crop = carve_neighbor_ink(crop, gray_full, x1, y1, x2, y2, (oy1, oy2),
                                  prev_bbox, next_bbox)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if tighten:
        tb = tighten_box(gray)
        if tb is not None:
            a, c, b, d = tb
            crop = crop[c:d, a:b]
            gray = gray[c:d, a:b]
    if crop.size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)
    ch, cw = crop.shape[:2]
    return {
        "ink": round(float((gray < 128).mean()), 3),
        "w": cw, "h": ch,
        "md5": hashlib.md5(path.read_bytes()).hexdigest()[:12],
        "seg": _seg_flag(gray),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--out", default=str(REPO / "dataset_out"))
    ap.add_argument("--use-s3", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="fail loud if S3 can't load (else SILVER silently -> REVIEW)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--no-tighten", action="store_true")
    ap.add_argument("--no-carve", action="store_true",
                    help="skip seam-carve neighbour-ink erasure (debug/compare only; "
                         "carving is ON by default — it's what fixes crops bleeding "
                         "into the char above/below in the same column)")
    ap.add_argument("--crop-review", action="store_true",
                    help="also materialize REVIEW crops (kept in labels.csv either way)")
    # None = LẤY TỪ CONFIG (step2.crop_pad_frac). Truyền --pad chỉ để ghi đè khi thí
    # nghiệm; đường chạy sản xuất phải để cấu hình quyết, nếu không lại tái diễn lớp
    # lỗi "cấu hình khai một đằng, mã chạy một nẻo" mà T4.a tìm ra.
    ap.add_argument("--pad", type=float, default=None)
    ap.add_argument("--reseg", default="midpoint",
                    choices=["midpoint", "valley_n", "valley_guarded", "detector"],
                    help="column re-segmentation for crop boxes (default midpoint; valley_* are "
                         "opt-in experiments — see seg_valley_n_ab.py / seg_smart_ab.py). "
                         "valley_guarded needs the encoder (auto-loaded). detector uses a trained "
                         "char_detector/detector.pt (Kaggle; falls back to midpoint if absent).")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    paths = config["paths"]

    # --- HÌNH HỌC CẮT ẢNH: cấu hình -> mã (nối 2026-08-25) --------------------
    _s2 = config.get("step2") or {}
    if args.pad is None:
        args.pad = float(_s2.get("crop_pad_frac", 0.12))
    _ov = float(_s2.get("box_overlap_frac", ap_mod.BOX_OVERLAP_FRAC))
    ap_mod.BOX_OVERLAP_FRAC = _ov
    print(f"  [hình học] đệm cắt = {args.pad} | biên nới dọc F = {_ov} "
          f"-> hộp cao {1 + 2 * _ov:.2f} × bước lặp", flush=True)
    qn_to_nom = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    qn_dict_set = set(qn_to_nom.keys())
    # từ điển NGƯỢC {chữ Nôm: [âm QN từ điển công nhận]} — L1 cần nó để biết một
    # ocr_char còn được đọc là những âm nào khác cùng khung xương.
    nom_to_qn = build_nom_to_qn(qn_to_nom)
    similar = load_similarity_dict(str(REPO / paths["similar_dict"]))
    data_root = REPO / paths["data_dir"]

    vs3 = None
    if args.use_s3:
        from pipeline.align_engine.visual_signal import VisualS3
        try:
            print("Loading S3 (trained Nôm embedder + FD) ...", flush=True)
            # fd_cache_similar (optional): a glyph cache in a font SIMILAR to the
            # crops. When present it becomes the "simfont" reference tier (smaller
            # domain gap than FD). Absent -> simfont tier off (current state).
            simfont = str(REPO / paths["fd_cache_similar"]) if paths.get("fd_cache_similar") else ""
            vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]),
                           simfont_dir=simfont)
        except Exception as e:
            if getattr(args, "strict", False):
                raise RuntimeError(
                    f"[S3 STRICT] S3 failed to load ({type(e).__name__}: {e}). "
                    "SILVER would silently collapse to REVIEW. Fix the nom-embed/best.pt "
                    "+ gannhanocr-fd checkpoints, or drop --strict to build GOLD-only."
                ) from e
            print(f"  [S3 OFF] {type(e).__name__}: {e}\n  -> SILVER bỏ qua; GOLD/SYLLABLE "
                  "vẫn chạy. (Cần checkpoint nom-embed/best.pt — train ở nom_classifier/.) "
                  "Dùng --strict để fail loud thay vì degrade âm thầm.",
                  flush=True)
            vs3 = None

    # encoder for reseg_mode=valley_guarded (the MLS guard) — reuse S3's if loaded,
    # else load just the encoder; absent -> _pick_reseg falls back to midpoint.
    reseg_encoder = vs3.enc if vs3 is not None else None
    if args.reseg == "valley_guarded" and reseg_encoder is None:
        try:
            from pipeline.align_engine.nom_classifier.infer import NomEncoder
            from pipeline.align_engine.visual_signal import _find_ckpt
            reseg_encoder = NomEncoder(_find_ckpt(REPO))
            print("  [reseg] valley_guarded: encoder loaded for the MLS guard.", flush=True)
        except Exception as e:
            print(f"  [reseg] valley_guarded needs the encoder ({e}) -> midpoint fallback.", flush=True)
    if args.reseg != "midpoint":
        print(f"  [reseg] mode = {args.reseg}", flush=True)
        # FAIL FAST: dựng detector NGAY, trước khi duyệt trang nào. Thiếu checkpoint mà
        # chạy tiếp = lặng lẽ tách chữ bằng trung điểm cho cả 445 trang.
        _backend = preflight_detector(args.reseg)
        print(f"  [reseg] backend thực dùng = {_backend}", flush=True)

    # ---------- PASS 1: align all pages, collect records (no crop yet) ----------
    records = []
    pages_done = 0
    for b in config["books"]:
        book = b["name"]
        data_dir = data_root / book
        trans = sorted(glob.glob(str(data_dir / "transcriptions" / "page_*.json")))
        trans = [t for t in trans if not t.endswith("_qn_ocr_cache.json")]
        if args.limit:
            trans = trans[:args.limit]
        print(f"[align] {book}: {len(trans)} pages ...", flush=True)
        for pi, tf in enumerate(trans):
            page = Path(tf).stem
            try:
                rec = align_page(page, data_dir, qn_dict_set, qn_to_nom, similar, "new",
                                 reseg_mode=args.reseg, encoder=reseg_encoder)
            except DetectorUnavailableError:
                # KHÔNG nuốt: thiếu detector mà vẫn chạy tiếp = lặng lẽ tách chữ bằng
                # trung điểm cho TOÀN BỘ corpus. Phải dừng hẳn.
                raise
            except Exception as e:
                print(f"   [warn] {book}/{page}: {type(e).__name__}: {e}", flush=True)
                continue
            if rec is None:
                continue
            pages_done += 1
            page_png = str(data_dir / "pages" / f"{page}.png")
            for idx, p in enumerate(rec["pairs"]):
                s3 = maybe_s3(p, page_png, qn_to_nom, vs3) if vs3 else None
                dec = decide_label(p.get("ocr_char"), p["syllable"], p.get("matched", False),
                                   qn_to_nom, similar, s3=s3, anchored=p.get("anchored", False))
                records.append({
                    "book": _book_code(book), "page": page, "column": p["column"], "idx": idx,
                    "page_png": page_png, "ocr_char": p.get("ocr_char") or "",
                    # Canonical lowercase syllable for ALL tiers (not just SYLLABLE
                    # below): the QN→Nôm dict and decide_label are already case-folded,
                    # so keeping raw OCR case here only fragmented the reading vocabulary
                    # (Nhị/nhị/NHỊ → 3 classes) and inflated the distinct-syllable count.
                    # NFC + modern tone-mark placement are already applied upstream (parse_v5).
                    "syllable": str(p["syllable"]).lower(), "bbox": p.get("bbox"),
                    "tier": dec.tier, "rule": dec.rule_id, "label": dec.label or "",
                    "s3_cosine": round(s3.cosine, 3) if s3 else "",
                    "seg_backend": rec.get("seg_backend", ""),
                })

    # ---------- L1 + L3: neo bằng bằng chứng CẤP NGỮ LIỆU (TRƯỚC khối PROMOTE) ----
    # Đặt ở đây, không đặt trong decide_label, vì cả hai luật cần thống kê TOÀN NGỮ
    # LIỆU (cặp nào đã được chứng thực ở GOLD-trực-tiếp) — thứ chỉ có sau PASS 1.
    # Đặt TRƯỚC PROMOTE vì một ô đã được nâng lên nhãn CHỮ thì không được đồng thời
    # đi làm nhãn ÂM TIẾT; nếu chạy sau, cùng một ô sẽ vào hai tier.
    anchors = gold_direct_anchors(records)
    n_l1_moi, n_l1_nang = apply_am_sua_dau(records, qn_to_nom, nom_to_qn, anchors)
    n_l3 = apply_cot_lech_cau_xuoi(records, qn_to_nom, similar, anchors)
    print(f"  [L1 sửa dấu âm] {n_l1_moi + n_l1_nang:,} ô -> GOLD/{RULE_L1} "
          f"({n_l1_moi:,} từ REVIEW, {n_l1_nang:,} từ SILVER)", flush=True)
    print(f"  [L3 cột lệch]   {n_l3:,} ô -> GOLD/{RULE_L3}", flush=True)
    print(f"  [chốt chặn]     âm nhường cho phán quyết người: "
          f"{sorted(AM_DA_QUYET)} (L1/L2/L3/L5 không chạm)", flush=True)

    # ---------- PROMOTE: cross-page-consistent unconfirmed -> SYLLABLE [#6] ----------
    # The unconfirmed pool = REVIEW rows with an ocr_char that S1∩S2 didn't confirm.
    # Without S3 the rule is 'unconfirmed_no_s3'; with S3 ON the S3-failed ones are
    # 'below_visual_threshold'. Both are eligible for the syllable tier (SILVER
    # already took the S3-confirmed ones), so SYLLABLE coexists with SILVER.
    #
    # L5 · BỂ ĐẦY ĐỦ. "diverged_column" nằm trong bể từ nay. Ba rule dưới đây là
    # TOÀN BỘ lý do một ô có thể ở REVIEW sau PASS 1, nên tập này = "mọi ô REVIEW
    # còn ocr_char" = mọi ô có ocr_char ∉ R(âm). Bể cũ thiếu "diverged_column", tức
    # độ thuần của mỗi chữ được tính trên một MẪU BỊ CẮT theo đúng tiêu chí S3 —
    # một tín hiệu mà bước s3_unwind đã gỡ quyền quyết (CI của AUC bắt lỗi chứa 0,5).
    # HỆ QUẢ ĐÚNG, KHÔNG PHẢI HỒI QUY: một số cặp (chữ, âm) đang ở SYLLABLE sẽ RỚT,
    # vì trên bể đầy đủ độ thuần THẬT của chúng dưới ngưỡng 0,6. Chúng chưa bao giờ
    # đạt ngưỡng; chỉ là mẫu số bị giấu mất một phần.
    UNCONF = {"unconfirmed_no_s3", "below_visual_threshold", "diverged_column"}
    # Case-insensitive gate (fixes the case-split; +~1,131 labels). The promoted row
    # also stores the canonical lowercase syllable so its target class is not fragmented
    # into cased variants downstream.
    syl_ok = syllable_gate(records, UNCONF)
    n_promoted = 0        # từ REVIEW (nhãn mới hoàn toàn)
    n_tu_silver = 0       # từ SILVER (đổi nhãn CHỮ do S3 quyết -> nhãn ÂM có bằng chứng)
    for r in records:
        syl_l = str(r["syllable"]).lower()
        if am_da_quyet(syl_l):
            continue      # CHỐT CHẶN: ô âm "người" phải ở lại REVIEW cho glyph_fix (QĐ-01)
        if (r["ocr_char"], syl_l) not in syl_ok:
            continue
        if r["tier"] == "REVIEW" and r["rule"] in UNCONF:
            r["tier"], r["rule"] = "SYLLABLE", "nghia_consensus"
            r["syllable"] = syl_l
            n_promoted += 1
        elif r["tier"] == "SILVER":
            # Ô này đang mang một nhãn CHỮ do S3 quyết. `s3_unwind` sẽ đổi tên tier
            # thành SILVER_uncalibrated và `export_final_dataset` LOẠI hẳn khỏi bộ
            # giao nộp — nên nhãn chữ ấy hiện không đi đâu cả. Cổng âm tiết thì có
            # bằng chứng ĐỘC LẬP với S3 (>=5 ô, >=3 trang, độ thuần >=0,6 trên bể
            # đầy đủ). Đổi một khẳng định KHÔNG dùng được lấy một khẳng định YẾU HƠN
            # NHƯNG DÙNG ĐƯỢC. Nhãn chữ bị xoá (giao ước của tier SYLLABLE), nên
            # xuất xứ nằm ở rule id riêng để đếm lại được, không lẫn vào 'nghia_consensus'.
            r["tier"], r["rule"], r["label"] = "SYLLABLE", "nghia_consensus_tu_silver", ""
            r["syllable"] = syl_l
            n_tu_silver += 1

    # label_level + unicode
    for r in records:
        if r["tier"] in ("GOLD", "SILVER"):
            r["label_level"] = "char"
            lab = r["label"]
            r["unicode"] = f"U+{ord(lab):04X}" if len(lab) == 1 else ""
        elif r["tier"] == "SYLLABLE":
            r["label_level"] = "syllable"   # char unconfirmed; syllable is the target
            r["label"], r["unicode"] = "", ""
        else:
            r["label_level"], r["unicode"] = "", ""

    # ---------- SPLIT: RỜI NHAU THEO TRANG ------------------------------------
    # Trước 2026-08-25 nhóm theo (sách, trang, CỘT). Đo được: 0/3.985 cột nằm ở hai
    # phía — sạch theo đơn vị của chính nó — NHƯNG 360/444 TRANG có cột rơi vào các
    # phía khác nhau. Hai cột cạnh nhau trên cùng một trang dùng chung ván khắc, chung
    # mực, chung lần quét, nên mô hình học DIỆN MẠO TRANG rồi được chấm lại trên chính
    # trang đó -> mọi chỉ số là CẬN TRÊN, không phải hiệu năng trên trang chưa từng thấy.
    # Đây là câu phản biện chắc chắn bị hỏi, nên chia theo TRANG.
    def split_of(group: str) -> str:
        h = int(hashlib.md5(group.encode()).hexdigest(), 16) % 100
        return "train" if h < 80 else ("val" if h < 90 else "test")
    for r in records:
        r["split_group"] = f"{r['book']}|{r['page']}"
        r["split"] = split_of(r["split_group"])

    # VÌ SAO BỎ LUẬT "ép nhóm chứa lớp singleton vào train":
    # ở mức CỘT nó chỉ chạm 375/4.003 cột (9,4%). Ở mức TRANG, 261/445 trang (58,7%)
    # chứa ít nhất một lớp chỉ-xuất-hiện-một-lần -> giữ luật này sẽ ép 59,1% số ô vào
    # train và phá nát chia tách. Bóp méo chia tách để chỉ số đẹp là đánh đổi SAI.
    #
    # Thay vào đó: chia TRUNG THỰC, rồi ghi giới hạn thành DỮ LIỆU. Cột `label_in_train`
    # cho biết lớp chữ của ô này có mặt trong train hay không; ai đánh giá thì lọc theo
    # nó, thay vì để chỉ số im lặng vô định.
    ccnt = Counter(r["label"] for r in records if r["label_level"] == "char" and r["label"])
    train_classes = {r["label"] for r in records
                     if r["split"] == "train" and r["label_level"] == "char" and r["label"]}
    for r in records:
        if r["label_level"] == "char" and r["label"]:
            r["label_in_train"] = "1" if r["label"] in train_classes else "0"
        else:
            r["label_in_train"] = ""
    _n_unseen = sum(1 for r in records if r.get("label_in_train") == "0")
    print(f"  [split] rời nhau theo TRANG | ô có lớp chữ KHÔNG có trong train: "
          f"{_n_unseen:,} (đánh giá phải lọc bằng label_in_train)", flush=True)

    # ---------- PASS 2: materialize crops + quality columns [#3,#5] ----------
    crop_tiers = {"GOLD", "SILVER", "SYLLABLE"} | ({"REVIEW"} if args.crop_review else set())
    by_page = defaultdict(list)
    for r in records:
        by_page[r["page_png"]].append(r)
    labels = []
    for png, recs in by_page.items():
        need = (not args.no_crops) and any(r["tier"] in crop_tiers for r in recs)
        img = cv2.imread(png, cv2.IMREAD_COLOR) if need else None
        gray_full = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if img is not None and not args.no_carve else None)

        # Neighbour bbox (prev/next by y-order) of every record IN THE SAME COLUMN,
        # over ALL records on this page (not just crop_tiers) — a REVIEW-tier char
        # still marks where neighbour ink sits, needed to bound the carve seam even
        # when its own crop isn't materialized. Stashed as temp keys, never emitted.
        by_col = defaultdict(list)
        for r in recs:
            if r.get("bbox"):
                by_col[r["column"]].append(r)
        for col_recs in by_col.values():
            col_recs.sort(key=lambda r: (r["bbox"][1] + r["bbox"][3]) / 2.0)
            for i, r in enumerate(col_recs):
                r["_prev_bbox"] = col_recs[i - 1]["bbox"] if i > 0 else None
                r["_next_bbox"] = col_recs[i + 1]["bbox"] if i < len(col_recs) - 1 else None

        for r in recs:
            img_rel = q = None
            if img is not None and r["tier"] in crop_tiers:
                fn = f"{r['book']}_{r['page']}_c{r['column']:02d}_{r['idx']:03d}.png"
                q = save_crop(img, gray_full, r.get("bbox"), args.pad, out / r["tier"].lower() / fn,
                              tighten=not args.no_tighten,
                              prev_bbox=r.get("_prev_bbox"), next_bbox=r.get("_next_bbox"))
                if q:
                    img_rel = f"{r['tier'].lower()}/{fn}"
            labels.append({
                # khoá sắp xếp, KHÔNG ghi ra CSV (xem `fields`) — chỉ để T6.b
                "_sort": (r["book"], r["page"], int(r["column"]), int(r["idx"])),
                "image": img_rel or "", "book": r["book"], "page": r["page"],
                "column": r["column"], "ocr_char": r["ocr_char"], "syllable": r["syllable"],
                "label": r["label"], "unicode": r["unicode"], "label_level": r["label_level"],
                "tier": r["tier"], "rule": r["rule"],
                # backend tách ký tự THỰC DÙNG — không có cột này thì một lần rơi về
                # midpoint sẽ không để lại dấu vết nào trong bộ nhãn (KHỐI 1.3).
                "seg_backend": r.get("seg_backend", ""),
                "label_in_train": r.get("label_in_train", ""),
                "ink_pct": q["ink"] if q else "", "crop_w": q["w"] if q else "",
                "crop_h": q["h"] if q else "", "image_md5": q["md5"] if q else "",
                "seg_flag": q["seg"] if q else "",
                "s3_cosine": r.get("s3_cosine", ""),
                "split": r["split"], "split_group": r["split_group"],
                "bbox": json.dumps(r.get("bbox")),
            })

    # ---------- write manifest + summary ----------
    fields = ["image", "book", "page", "column", "ocr_char", "syllable", "label",
              "unicode", "label_level", "tier", "rule", "s3_cosine", "ink_pct",
              "crop_w", "crop_h", "image_md5", "seg_flag", "split", "split_group",
              "label_in_train", "bbox",
              "seg_backend"]
    with open(out / "labels.csv", "w", encoding="utf-8", newline="") as f:
        # T6.b — SẮP DÒNG THEO KHOÁ CANON TRƯỚC KHI GHI.
        # build_dataset duyệt `for b in config["books"]` KHÔNG sắp, nên thứ tự sách
        # trong cấu hình quyết định thứ tự dòng: đảo danh sách books cho ra tệp có
        # cùng NỘI DUNG (đo T6: 3.398/3.398 dòng, cùng tập) nhưng KHÁC BYTE. Tiêu chí
        # T6 "đổi thứ tự sách -> byte-identical" vì thế không thể đạt.
        # Sắp theo (sách, trang, cột, chỉ-số-trong-cột) khiến thứ tự dòng KHÔNG còn phụ
        # thuộc cấu hình. Khoá này giữ nguyên thứ tự TRONG cột, nên mọi phân tích dựa
        # vào tính liền kề dọc cột (ví dụ dựng lại `anchored`) vẫn đúng.
        labels.sort(key=lambda r: r["_sort"])
        for r in labels:
            r.pop("_sort", None)
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(labels)

    tiers = Counter(r["tier"] for r in records)
    splits = Counter((r["tier"], r["split"]) for r in records if r["label_level"])
    char_classes = len(set(r["label"] for r in records if r["label_level"] == "char" and r["label"]))
    summary = {
        "pages": pages_done, "total_pairs": len(records),
        "tiers": dict(tiers), "syllable_promoted": n_promoted,
        "syllable_tu_silver": n_tu_silver,
        # rã theo luật để bảng số liệu không phải suy ngược từ tier
        "rules": dict(Counter(r["rule"] for r in records)),
        "luat_moi": {RULE_L1: n_l1_moi + n_l1_nang, "  trong đó từ REVIEW": n_l1_moi,
                     "  trong đó từ SILVER": n_l1_nang, RULE_L3: n_l3,
                     "s1_inter_s2_similar_nguoc":
                         sum(1 for r in records if r["rule"] == "s1_inter_s2_similar_nguoc"),
                     "am_da_quyet": sorted(AM_DA_QUYET)},
        "char_classes": char_classes,
        "usable_char": tiers["GOLD"] + tiers["SILVER"],
        "usable_total": tiers["GOLD"] + tiers["SILVER"] + tiers["SYLLABLE"],
        "split_counts": {f"{t}/{s}": n for (t, s), n in sorted(splits.items())},
    }
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f" DATASET -> {out}")
    print("=" * 64)
    print(f" pages {pages_done} | pairs {len(records)} | char classes {char_classes}")
    for t in ("GOLD", "SILVER", "SYLLABLE", "REVIEW"):
        print(f"   {t:9s}: {tiers.get(t, 0)}")
    print(f" SYLLABLE promoted from REVIEW: {n_promoted}")
    print(f" SYLLABLE chuyển từ SILVER (L5): {n_tu_silver}")
    print(f" USABLE char-level (GOLD+SILVER): {summary['usable_char']}  | "
          f"+syllable: {summary['usable_total']}")
    print(f" manifest: {out}/labels.csv  ({len(fields)} cột)")


if __name__ == "__main__":
    main()
