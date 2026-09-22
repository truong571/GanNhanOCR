"""Build the FINAL labeled dataset end-to-end (NEW pipeline, no A/B).

Two passes:
  PASS 1 — align every page (production-faithful: detect_nom_columns_v3 + parse_v5
           + bbox_fix offset + banded anchored DP + re-segment). Mặc định (--two-pass)
           chỉ GOM `col_states` theo trang; --no-two-pass giữ đường cũ (tier ngay).
  PASS 1b— (flow N4a–N4c) `pair_pages` LOO từ MỌI match lượt 1 → DP lại từng cột với
           cost_fn neo (min(base, ANCHOR_CAP) khi cặp thấy ở ≥2 trang khác) + posterior
           p_register cùng cost_fn → gán hộp 3 nhánh theo ops lượt 2 (A-6, flow N4d,
           align_production.assign_boxes; --box-rule legacy = trọn gói luật cũ; cột có
           ô khoá QĐ-01 luôn chạy luật cũ) → cặp lượt 2 (chưa tier).
  PASS 1c— (A-7, flow N5a–N5h) tier_v3 NGUYÊN VĂN (tier_v3.py: feats LOO + posterior)
           thay decide_label + L2 + L3 + L5; chốt is_plausible trước tier; L1 sửa dấu
           âm CÓ CỔNG 2-gram ÂM–ÂM; KHOÁ QĐ-01 theo (book,page,column,nom_idx)
           (config/qd01_cells.csv + qd01a_decisions.csv); 'người' ngoài khoá mang
           nhãn 㝵 -> REVIEW. A-8 (N5f/N5h/N5i): config/decisions.yaml — mục da_ky
           corpus_readings -> GOLD `corpus_reading:<id>`; lop_nham đọc từ yaml;
           di_the -> cột `label_canonical` (mặc định = label, cả hai đường chạy).
           Chỉ ở --two-pass; --no-two-pass giữ đường cũ
           (decide_label + L1/L3 + PROMOTE) để tái lập bộ 64.525.
  SPLIT  — [BỎ 16/09, A-10] không chia train/val/test; cần thì tự chia theo trang:
           int(md5(f'{book}|{page}').hexdigest(),16)%100 <80 train / <90 val / còn lại test.
  PASS 2 — materialize crops (GOLD + SYLLABLE + ô QĐ-01 pending; +REVIEW with
           --crop-review); ô khoá QĐ-01 cắt bằng bbox_cu/prev/next_cu; dọn thư mục
           đích trước khi cắt; per-crop quality columns (ink%, size, md5, seg_flag).

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
import math
import os
import re
import shutil
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
from pipeline.align_engine import anchor_align as aa                  # noqa: E402
from pipeline.align_engine.align_production import (                    # noqa: E402
    DetectorUnavailableError, align_page, preflight_detector)
from pipeline.align_engine.consensus import (                         # noqa: E402
    AM_DA_QUYET, am_da_quyet, chuan_am, decide_label)
from pipeline.align_engine.bbox_fix import tighten_box, carve_neighbor_ink  # noqa: E402
from pipeline.align_engine.book_layout import book_layout, DEFAULT_LAYOUT, resolve_detector_ckpt  # noqa: E402  (n_columns/layout/ckpt theo sách)
from pipeline.align_engine import recenter_f3g as rf3g                 # noqa: E402  (D-1, chỉ chạy khi --crops-v2)
from pipeline.align_engine import tier_v3 as tv3                      # noqa: E402
from pipeline.align_engine.visual_emission import load_page_gray as vis_load_gray  # noqa: E402  (B-2; torch nạp lười)
from pipeline import decisions as dcs                                 # noqa: E402


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

# --------------------------------------------------------------------------- #
# PASS 1b · ĐỆ QUY HAI LƯỢT (flow N4a–N4c). Neo NGỮ LIỆU, không phải neo từ điển.
# --------------------------------------------------------------------------- #
# Lượt 1 ghép bằng từ điển. Lượt 2 hạ chi phí một cặp (chữ, âm) xuống ANCHOR_CAP
# nếu chính cặp ấy đã được lượt 1 ghép ở >= ANCHOR_MIN_OTHER_PAGES TRANG KHÁC
# (leave-one-out theo (book, page): trang đang xét không tự neo mình). Lấy từ
# MỌI match lượt 1, KHÔNG chỉ confirmed — nếu chỉ lấy confirmed thì cặp ngoài từ
# điển không bao giờ được neo và đệ quy vô tác dụng. Trần 2,0 nat (không phải 0) để
# một cặp neo sai không kéo cả đoạn. Cờ `confirmed` vẫn do từ điển quyết (is_confirmed).
ANCHOR_CAP = aa.ANCHOR_CAP        # đọc từ engine; config step2.anchor_cap ghi đè (main)
ANCHOR_MIN_OTHER_PAGES = 2        # cặp phải thấy ở >= ngần này trang KHÁC trang đang xét


def build_pair_pages(page_states):
    """{(ocr_char, âm-thường): {(book, page)}} từ MỌI op 'match' của ops lượt 1.

    `page_states` = [(book_code, page, rec)] với rec["col_states"] từ align_page.
    """
    pair_pages: dict[tuple[str, str], set] = defaultdict(set)
    for book, page, rec in page_states:
        for cs in rec.get("col_states") or ():
            for o in cs["ops1"]:
                if o["op"] == "match" and o.get("ocr_char"):
                    pair_pages[(o["ocr_char"], str(o["syllable"] or "").lower())].add((book, page))
    return pair_pages


def anchored_cost_fn(pair_pages, here, qn_to_nom, similar, anchor_cap=None,
                     min_other=ANCHOR_MIN_OTHER_PAGES):
    """cost_fn(c, s) cho realign_column/posterior_matches của MỘT cột ở trang `here`:
    min(base, anchor_cap) khi (c, s) thấy ở >= min_other trang khác; else base."""
    cap = ANCHOR_CAP if anchor_cap is None else anchor_cap

    def cost_fn(c, s):
        base = aa.substitution_cost(c, s, qn_to_nom, similar)
        pages = pair_pages.get((c, (s or "").lower()))
        if pages and len(pages) - (1 if here in pages else 0) >= min_other:
            return min(base, cap)
        return base
    return cost_fn


def realign_with_anchors(chars, syllables, qn_to_nom, similar, pair_pages, here,
                         anchor_cap=None, vis=None):
    """DP lượt 2 + posterior cùng cost_fn (N4b, N4c). Trả (ops2, post, band_touched,
    n_anchored) với n_anchored = số cặp match lượt 2 mà neo THẬT SỰ hạ chi phí
    (cost_fn < substitution_cost, tức cặp ngoài từ điển được ngữ liệu neo — "ô được
    neo" N4a); cặp confirmed (0,0) có khoá trong pair_pages KHÔNG tính.

    vis (B-2, --visual-emission): (emitter, logP) với logP (m × n) từ
    visual_emission.VisualEmitter.emission → cost_ij = cost_fn + λ·min(−logP, CAP),
    khe = COST_DEL/INS + λ·GAP_VIS; posterior CÙNG chi phí. None = văn bản thuần."""
    cost_fn = anchored_cost_fn(pair_pages, here, qn_to_nom, similar, anchor_cap)
    kw = {}
    if vis is not None:
        emitter, logP = vis
        c_del, c_ins = emitter.gap_costs()
        kw = {"cost_ij": emitter.make_cost_ij(cost_fn, logP), "cost_del": c_del, "cost_ins": c_ins}
    ops2 = aa.realign_column(chars, syllables, qn_to_nom, similar, cost_fn=cost_fn, **kw)
    post = aa.posterior_matches(chars, syllables, qn_to_nom, similar, T=1.0, cost_fn=cost_fn, **kw)
    touched = aa.band_touched(ops2, len(chars), len(syllables))
    n_anch = 0
    for o in ops2:
        if o["op"] != "match" or not o.get("ocr_char"):
            continue
        if cost_fn(o["ocr_char"], o["syllable"]) < aa.substitution_cost(
                o["ocr_char"], o["syllable"], qn_to_nom, similar):
            n_anch += 1
    return ops2, post, touched, n_anch


def load_locked_columns(path) -> dict[tuple[str, str], set[int]]:
    """{(book, page): {column}} các cột có ô khoá QĐ-01 (config/qd01_cells.csv, schema
    N0d: book,page,column,nom_idx,…). A-6 (N4d): cột ấy chạy trọn gói luật hộp cũ để
    bbox/md5 của ô khoá không đổi VÀ không trộn hộp cũ–mới trong cùng cột (23 cặp trùng
    bbox -> census AE-1 cách ly cả hai). Tệp rỗng/None -> không khoá cột nào."""
    locked: dict[tuple[str, str], set[int]] = defaultdict(set)
    if not path:
        return locked
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            locked[(row["book"], row["page"])].add(int(row["column"]))
    return locked


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
                     min_occ=ANCHOR_MIN_OCC, min_pages=ANCHOR_MIN_PAGES,
                     bigram_syl_pages=None, colmap=None, corpus=None):
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

    CỔNG 2-GRAM ÂM–ÂM (A-7, flow N5e) — khi truyền `bigram_syl_pages` (+ `colmap`,
    `corpus`): âm ứng viên còn phải THẮNG âm gốc về ngữ cảnh dòng QN, đo bằng
    `tier_v3.syl_bigram_count` (2-gram âm kề trái + phải, LOO trang):
      · n(âm mới) > n(âm gốc) -> đổi âm, `l1_support` = hiệu (> 0), tier tính lại
        bằng feats/tier_v3 trên âm mới (direct nên luôn CHAR_A/CHAR_B -> GOLD, rule
        RULE_L1 — KHÔNG phải `s1_inter_s2_direct` nên không bao giờ làm neo);
      · hoà -> GIỮ gốc, `l1_tie` = 1;  · thua -> giữ gốc, `l1_support` < 0.
    `syllable_ocr/syllable_raw` không bao giờ bị đổi (ghi đè bản in đo trên chúng).
    Không truyền -> hành vi cũ (đường --no-two-pass).
    """
    n_moi = n_nang = 0
    n_giu = n_hoa = 0
    v3 = bigram_syl_pages is not None
    for r in records:
        if r["tier"] == "GOLD":
            continue
        ch = r["ocr_char"]
        if not ch:
            continue
        syl = str(r["syllable"]).lower()
        if am_da_quyet(syl):
            # L1 không đổi âm đi/đến âm có phán quyết người: khoá QĐ-01 bám theo âm
            # của DÒNG QN (`syllable_raw`), L1 viết lại `syllable` sẽ làm hai bên lệch.
            continue
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
        alt = cands[0]
        if v3:
            col = colmap.get((r["book"], r["page"], int(r["column"])))
            j = r.get("syl_idx", "")
            if col is None or j == "":
                continue
            pg = (r["book"], r["page"])
            syls = col["syl"]
            n_new = tv3.syl_bigram_count(bigram_syl_pages, syls, int(j), alt, pg)
            n_old = tv3.syl_bigram_count(bigram_syl_pages, syls, int(j), syl, pg)
            r["l1_support"] = n_new - n_old
            if n_new == n_old:
                r["l1_tie"] = 1
                n_hoa += 1
                continue
            if n_new < n_old:
                n_giu += 1
                continue
        if r["tier"] == "REVIEW":
            n_moi += 1
        else:
            n_nang += 1          # SILVER/SYLLABLE -> GOLD (nhãn do từ điển quyết)
        r["syllable"] = alt
        r["label"] = ch
        r["tier"] = "GOLD"
        r["rule"] = RULE_L1
        if v3:
            # tính lại feats/tier_v3 trên âm mới (không chạy lại DP: p_register giữ)
            syls2 = list(col["syl"])
            syls2[int(j)] = alt
            f = corpus.feats(col["chars"], syls2, int(r["nom_idx"]), int(j), pg)
            r["tier_v3"] = tv3.tier_v3(f, float(r["p_register"] or 0.0))
            r["dict_support"] = f["dict_support"]
            r["context_evidence"] = tv3.context_evidence(f)
    if v3:
        return n_moi, n_nang, n_giu, n_hoa
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

    A-7 (N5c): KHÔNG còn gọi trên đường --two-pass — tier_v3 xử lý cầu tự dạng bằng
    `sim_unique ∧ ngữ cảnh LOO ∧ p ≥ 0,8` (CHAR_B, đo được 110 ô L3 -> 83/22/5). Giữ
    hàm cho đường --no-two-pass (tái lập bộ 64.525).
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

    A-7 (N5c): KHÔNG còn gọi trên đường --two-pass — tầng SYLLABLE do tier_v3 quyết
    (bigram | corpus4 | tone, LOO trang, p ≥ 0,5); cổng độ-thuần-theo-chữ này không
    LOO nên tự khẳng định (48 ô hiện hành rớt khi đo lại). Giữ cho --no-two-pass và selftest.
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


# --------------------------------------------------------------------------- #
# PASS 1c (A-7, flow N5a–N5h): tier_v3 + khoá QĐ-01 + 'người' ngoài khoá + flank_gold
# --------------------------------------------------------------------------- #
NGUOI_2029A = "\U0002029A"        # 𠊚 — nhãn QĐ-01 (KHÔNG phải 𠊛 U+2029B)
NGUOI_NHAM = "㝵"                  # nhãn máy hay nhầm cho âm 'người' (lop_nham, N5h)
# A-8: lop_nham đọc từ config/decisions.yaml (mục da_ky); NGUOI_NHAM/RULE_LOP_NHAM chỉ
# còn là mặc định khi gọi apply_nguoi_ngoai_khoa không truyền danh sách (selftest/đường cũ)
RULE_QD01_LOCK = "quyet_dinh_nguoi:qd01_cell_lock"
RULE_QD01_PENDING = "quyet_dinh_nguoi:pending"
RULE_QD01A = "quyet_dinh_nguoi:qd01a"
RULE_LOP_NHAM = "lop_nham:nguoi_2029A_vs_346B"
# cột cờ/chẩn đoán A-7 — mặc định DÀY (0 cho cờ, '' cho chuỗi) ở cả hai đường chạy
# B-2 (--visual-emission): cột sidecar thêm vào CUỐI labels.csv chỉ khi bật cờ
FIELDS_B2 = ["p_visual_syl", "visual_fold", "visual_argmax", "visual_max_p"]
FIELDS_1C = {"tier_v3": "", "dict_support": "", "context_evidence": "", "l1_support": 0,
             "l1_tie": 0, "flank_gold": 0, "qd01_locked": 0, "qd01_excluded": 0,
             "am_da_quyet_ngoai_khoa": 0, "tier_goc": "", "rule_goc": ""}


def _parse_bbox(s):
    """'[x1, y1, x2, y2]' -> [int]*4; ''/None -> None."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return [int(v) for v in json.loads(s)]


def _qd01_key(row) -> tuple:
    return (row["book"], row["page"], int(row["column"]), int(row["nom_idx"]))


def load_qd01_cells_full(path) -> dict[tuple, dict]:
    """{(book,page,column,nom_idx): dòng} của config/qd01_cells.csv (schema N0d)."""
    if not path or str(path).lower() == "none" or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {_qd01_key(row): row for row in csv.DictReader(f)}


def load_qd01a_decisions(path) -> dict[tuple, dict]:
    """{(book,page,column,nom_idx): dòng} của config/qd01a_decisions.csv — phán quyết
    người cho ô trôi (quyet ∈ {giu_2029A, bo, khac:<chữ>, '' = chờ người})."""
    if not path or not Path(path).exists():
        return {}
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            q = (row.get("quyet") or "").strip()
            if q and q not in ("giu_2029A", "bo") and not q.startswith("khac:"):
                raise SystemExit(f"[QĐ-01a] quyet không hợp lệ '{q}' ở {row}")
            row["quyet"] = q
            out[_qd01_key(row)] = row
    return out


def apply_tier_v3(records, corpus, colmap):
    """N5b–N5d: feats LOO + p_register -> tier_v3 -> (tier, rule, label). Chốt trước:
    âm không hợp lý (`is_plausible_qn_syllable`) -> REVIEW `not_plausible`; ô không có
    ocr_char -> REVIEW `no_context`. Ghi tier_v3/dict_support/context_evidence."""
    cross = Counter()
    for r in records:
        col = colmap[(r["book"], r["page"], int(r["column"]))]
        i, j = int(r["nom_idx"]), int(r["syl_idx"])
        pg = (r["book"], r["page"])
        f = corpus.feats(col["chars"], col["syl"], i, j, pg)
        p = float(r["p_register"]) if r.get("p_register", "") != "" else 0.0
        syl = str(r["syllable"]).lower()
        ch = r["ocr_char"]
        if not is_plausible_qn_syllable(syl):
            t, rule, label = "REVIEW", "not_plausible", ""
        elif not ch:
            t, rule, label = "REVIEW", "no_context", ""
        else:
            t = tv3.tier_v3(f, p)
            rule = tv3.rule_of(t, f, p)
            if t == "CHAR_A" or (t == "CHAR_B" and f["direct"]):
                label = ch
            elif t == "CHAR_B":
                label = corpus.bridge_char(ch, syl)      # chữ cầu duy nhất sim(c) ∩ R
            else:
                label = ""
        r["tier"], r["rule"], r["label"] = tv3.TIER_OF[t], rule, label
        r["tier_v3"] = t
        r["dict_support"] = f["dict_support"]
        r["context_evidence"] = tv3.context_evidence(f)
        cross[t] += 1
    return cross


def apply_qd01_lock(records, cells, decisions, colmap, built_pages, seg_backend_of):
    """N5g: khoá QĐ-01 theo (book,page,column,nom_idx) TRONG build.

    Với mỗi ô của qd01_cells.csv thuộc trang đã build:
      (i)   khớp record ∧ (âm dòng QN == 'người' ∨ quyet giu_2029A) -> ghi đè label 𠊚,
            tier GOLD, rule RULE_QD01_LOCK, qd01_locked=1, bbox=bbox_cu, prev/next cho
            PASS 2 = prev/next_bbox_cu, box_source=qd01_locked; giữ tier_goc/rule_goc;
      (ii)  quyet 'bo' -> giữ tier_v3, qd01_excluded=1; quyet 'khac:<chữ>' -> label chữ ấy,
            GOLD, rule RULE_QD01A, khoá hộp như (i);
      (iii) khớp nom_idx nhưng âm khác / thành khe / không khớp -> PENDING: rule
            RULE_QD01_PENDING, qd01_locked=0, tier giữ theo tier_v3 (khe: record tổng
            hợp tier REVIEW), ép cắt crop bằng cả hộp mới lẫn bbox_cu (`_bbox_cu`).
    Dòng của qd01a_decisions NGOÀI cells (A-3: 12 ô GOLD 'người' mã ≠ 𠊚): quyet ''/bo
    -> qd01_excluded=1 (khong_dung_cho); giu_2029A/khac -> áp như trên.
    Trả (thống kê, danh sách pending cho $out/qd01a_pending.csv).
    """
    idx = {(r["book"], r["page"], int(r["column"]), int(r["nom_idx"])): r
           for r in records if r.get("nom_idx", "") != ""}
    max_idx = defaultdict(int)
    for r in records:
        max_idx[(r["book"], r["page"])] = max(max_idx[(r["book"], r["page"])], int(r["idx"]))
    st = Counter()
    pending = []

    def _lock(r, cell, label, rule):
        r["tier_goc"], r["rule_goc"] = r["tier"], r["rule"]
        r["label"], r["tier"], r["rule"] = label, "GOLD", rule
        r["qd01_locked"] = 1
        bcu = _parse_bbox(cell.get("bbox_cu"))
        if bcu:
            # B-5: giữ hộp build gán TRƯỚC khi khoá (đo luật hộp trên ô người đã xem)
            r["_bbox_truoc_lock"] = list(r["bbox"]) if r.get("bbox") else None
            r["_box_source_truoc_lock"] = r.get("box_source", "")
            r["bbox"] = bcu
            r["box_source"] = "qd01_locked"
            r["_prev_bbox_cu"] = _parse_bbox(cell.get("prev_bbox_cu"))
            r["_next_bbox_cu"] = _parse_bbox(cell.get("next_bbox_cu"))
            r["_has_prev_next_cu"] = True

    def _pending(r, cell, ly_do):
        r["tier_goc"], r["rule_goc"] = r["tier"], r["rule"]
        r["rule"] = RULE_QD01_PENDING
        r["qd01_locked"] = 0
        r["_qd01_pending"] = True
        r["_bbox_cu"] = _parse_bbox(cell.get("bbox_cu"))
        pending.append({"book": r["book"], "page": r["page"], "column": r["column"],
                        "nom_idx": r["nom_idx"], "syl_idx": r.get("syl_idx", ""),
                        "bbox_cu": cell.get("bbox_cu", ""), "bbox_moi": json.dumps(r.get("bbox")),
                        "syllable_moi": r.get("syllable", ""), "ocr_char": r.get("ocr_char", ""),
                        "tier_v3": r.get("tier_v3", ""), "tier_goc": r["tier_goc"],
                        "rule_goc": r["rule_goc"], "ly_do": ly_do, "_rec": r})

    for key, cell in cells.items():
        if (key[0], key[1]) not in built_pages:
            continue
        st["n_cells_in_build"] += 1
        dec = decisions.get(key)
        q = dec["quyet"] if dec else ""
        r = idx.get(key)
        if r is None:
            # thành khe (nom_idx không match) hoặc không khớp (cột không có trong build)
            col = colmap.get(key[:3])
            khe = col is not None and key[3] < len(col["chars"])
            st["n_khe" if khe else "n_khong_khop"] += 1
            bbox_moi = None
            if khe and col.get("boxes"):
                bbox_moi = col["boxes"][key[3]]
            max_idx[key[:2]] += 1
            r = {
                "book": key[0], "page": key[1], "column": key[2], "idx": max_idx[key[:2]],
                "page_png": col["page_png"] if col else "", "ocr_char": (col["chars"][key[3]] if khe else cell.get("ocr_char", "")),
                "syllable": "", "bbox": bbox_moi or _parse_bbox(cell.get("bbox_cu")),
                "tier": "REVIEW", "rule": "no_context", "label": "", "s3_cosine": "",
                "seg_backend": seg_backend_of.get(key[:2], ""), "nom_idx": key[3], "syl_idx": "",
                "syllable_ocr": "", "syllable_raw": "", "p_register": "", "band_touched": 0,
                "n_ocr": col["n_ocr"] if col else "", "n_qn": col["n_qn"] if col else "",
                "n_det": col["n_det"] if col else "", "count_source": col["count_source"] if col else "",
                "box_source": (col["box_source"][key[3]] if khe and col.get("box_source") else ""),
                **FIELDS_1C, "tier_v3": "REVIEW",
            }
            records.append(r)
            idx[key] = r
            if q == "giu_2029A":            # người đã quyết ở QĐ-01a -> khoá dù thành khe
                _lock(r, cell, NGUOI_2029A, RULE_QD01_LOCK)
                st["n_locked"] += 1
            elif q.startswith("khac:"):
                _lock(r, cell, q[len("khac:"):].strip(), RULE_QD01A)
                st["n_khac"] += 1
            elif q == "bo":
                r["qd01_excluded"] = 1
                st["n_bo"] += 1
            else:
                _pending(r, cell, "khe" if khe else "khong_khop")
                st["n_pending"] += 1
            continue
        am = chuan_am(r.get("syllable_raw") or r.get("syllable"))
        if q == "giu_2029A" or (not q and am == "người"):
            _lock(r, cell, NGUOI_2029A, RULE_QD01_LOCK)
            st["n_locked"] += 1
        elif q == "bo":
            r["qd01_excluded"] = 1
            st["n_bo"] += 1
        elif q.startswith("khac:"):
            _lock(r, cell, q[len("khac:"):].strip(), RULE_QD01A)
            st["n_khac"] += 1
        else:
            _pending(r, cell, f"am_khac:{am}")
            st["n_pending"] += 1

    # QĐ-01a ngoài cells (A-3)
    for key, dec in decisions.items():
        if key in cells or (key[0], key[1]) not in built_pages:
            continue
        r = idx.get(key)
        if r is None:
            st["n_decisions_ngoai_cells_khong_khop"] += 1
            continue
        q = dec["quyet"]
        if q == "giu_2029A":
            _lock(r, dec, NGUOI_2029A, RULE_QD01_LOCK)
            st["n_decisions_ngoai_cells_lock"] += 1
        elif q.startswith("khac:"):
            _lock(r, dec, q[len("khac:"):].strip(), RULE_QD01A)
            st["n_decisions_ngoai_cells_khac"] += 1
        else:
            r["qd01_excluded"] = 1          # '' (chờ người) hoặc 'bo': khong_dung_cho
            st["n_excluded_ngoai_cells"] += 1
            if not q:
                # chờ người: ép cắt crop (cả bbox_cu nếu khác) để người xem được ở QĐ-01a
                r["_qd01_pending"] = True
                r["_bbox_cu"] = _parse_bbox(dec.get("bbox_cu"))
                pending.append({"book": r["book"], "page": r["page"], "column": r["column"],
                                "nom_idx": r["nom_idx"], "syl_idx": r.get("syl_idx", ""),
                                "bbox_cu": dec.get("bbox_cu", ""), "bbox_moi": json.dumps(r.get("bbox")),
                                "syllable_moi": r.get("syllable", ""), "ocr_char": r.get("ocr_char", ""),
                                "tier_v3": r.get("tier_v3", ""), "tier_goc": r["tier"],
                                "rule_goc": r["rule"], "ly_do": "qd01a_chua_quyet", "_rec": r})
    return st, pending


def _iou(a, b) -> float:
    """IoU hai hộp [x1,y1,x2,y2]; 0 nếu thiếu."""
    if not a or not b:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# col = trọn cột legacy (mặc định); cell = khoá ô + dời hàng xóm về midpoint; cell_fallback =
# như cell nhưng cột mà ô khoá cho thấy hộp 3 nhánh lệch (IoU(hộp 3 nhánh, bbox_cu) < SHIFT_IOU
# hoặc có hàng xóm trùng) thì LÙI cả cột về trọn gói luật cũ (legacy_locked_col)
LOCK_SCOPES = ("col", "cell", "cell_fallback")
SHIFT_IOU = 0.5          # B-5: hộp mới của hàng xóm trùng bbox_cu ô khoá (IoU >= 0,5) -> midpoint


def apply_lock_shift(records, colmap, iou_thr=SHIFT_IOU):
    """B-5 (DANH_MUC 1B, --lock-scope cell): SAU khoá QĐ-01, ô hàng xóm (cùng cột hoặc
    cột kề ±1, cùng trang) KHÔNG khoá mà hộp mới trùng `bbox_cu` của ô khoá (IoU >=
    iou_thr) thì về hộp midpoint cũ của chính nó (`_reseg_column(cluster)[nom_idx]`,
    colmap[...]['mid']) với box_source='shifted_by_lock' — để census AE-1/MD5_DUP không
    cách ly cả hai. Ô pending QĐ-01 (`_qd01_pending`) không dời (người phải nhìn hộp
    mới). Trả Counter: n_locked, n_shifted, n_shifted_same_col, n_shifted_adj_col,
    n_exact_eq (hộp trùng đúng byte), n_locked_locked (hai ô khoá trùng nhau — có sẵn
    trong bộ cũ, không dời), n_no_mid (không có midpoint -> giữ), n_still_overlap
    (sau dời vẫn IoU >= thr — báo cáo)."""
    st = Counter()
    by_page = defaultdict(list)
    for r in records:
        if r.get("bbox"):
            by_page[(r["book"], r["page"])].append(r)
    for (book, page), recs in by_page.items():
        locked = [r for r in recs if r.get("qd01_locked") and r.get("box_source") == "qd01_locked"]
        if not locked:
            continue
        by_col = defaultdict(list)
        for r in recs:
            by_col[int(r["column"])].append(r)
        for L in locked:
            st["n_locked"] += 1
            bcu = L["bbox"]
            cL = int(L["column"])
            for c in (cL - 1, cL, cL + 1):
                for r in by_col.get(c, ()):
                    if r is L or not r.get("bbox"):
                        continue
                    iou = _iou(r["bbox"], bcu)
                    if iou < iou_thr:
                        continue
                    if r.get("qd01_locked"):
                        st["n_locked_locked"] += 1
                        continue
                    if r.get("_qd01_pending"):
                        st["n_pending_overlap"] += 1
                        continue
                    if r.get("box_source") == "shifted_by_lock":
                        st["n_already_shifted"] += 1
                        continue
                    st["n_exact_eq"] += int(list(r["bbox"]) == list(bcu))
                    col = colmap.get((book, page, c))
                    mid = (col or {}).get("mid")
                    ni = r.get("nom_idx", "")
                    if not mid or ni == "" or int(ni) >= len(mid):
                        st["n_no_mid"] += 1
                        continue
                    r["_bbox_truoc_shift"] = list(r["bbox"])
                    r["bbox"] = list(mid[int(ni)])
                    r["box_source"] = "shifted_by_lock"
                    st["n_shifted"] += 1
                    st["n_shifted_same_col" if c == cL else "n_shifted_adj_col"] += 1
                    if _iou(r["bbox"], bcu) >= iou_thr:
                        st["n_still_overlap"] += 1
    return st


def apply_lock_fallback(records, colmap, iou_thr=SHIFT_IOU):
    """B-5 (--lock-scope cell_fallback): cột có ô khoá QĐ-01 mà (i) hộp 3 nhánh của chính
    ô khoá lệch bbox_cu (IoU < iou_thr) hoặc (ii) có hàng xóm cùng cột đã bị dời
    (shifted_by_lock) -> mọi ô KHÔNG khoá trong cột về hộp trọn gói luật cũ
    colmap[...]['legacy_boxes'][nom_idx], box_source = count_source = 'legacy_locked_col'.
    Ô khoá giữ bbox_cu. Trả Counter n_cols_lock, n_cols_fallback (+ lý do), n_cells_fallback."""
    st = Counter()
    by_col = defaultdict(list)
    for r in records:
        if r.get("nom_idx", "") != "":
            by_col[(r["book"], r["page"], int(r["column"]))].append(r)
    for key, recs in by_col.items():
        locks = [r for r in recs if r.get("qd01_locked") and r.get("box_source") == "qd01_locked"]
        if not locks:
            continue
        st["n_cols_lock"] += 1
        lech = any(r.get("_bbox_truoc_lock") is not None
                   and _iou(r["_bbox_truoc_lock"], r["bbox"]) < iou_thr for r in locks)
        doi = any(r.get("box_source") == "shifted_by_lock" for r in recs)
        if not (lech or doi):
            continue
        lb = (colmap.get(key) or {}).get("legacy_boxes")
        if not lb:
            st["n_cols_no_legacy"] += 1
            continue
        st["n_cols_fallback"] += 1
        st["n_cols_fallback_lech"] += int(lech)
        st["n_cols_fallback_doi"] += int(doi)
        for r in recs:
            if r.get("qd01_locked") or r.get("_qd01_pending"):
                if r.get("qd01_locked"):
                    r["count_source"] = "legacy_locked_col"      # như col: cột đi luật cũ
                continue
            ni = int(r["nom_idx"])
            if ni >= len(lb) or lb[ni] is None:
                st["n_cells_no_box"] += 1
                continue
            if r.get("box_source") == "shifted_by_lock":
                st["n_cells_was_shifted"] += 1
            r["_bbox_truoc_fallback"] = list(r["bbox"]) if r.get("bbox") else None
            r["bbox"] = list(lb[ni])
            r["box_source"] = r["count_source"] = "legacy_locked_col"
            st["n_cells_fallback"] += 1
    return st


def apply_nguoi_ngoai_khoa(records, lop_nham=None):
    """N5h: ô âm 'người' KHÔNG khoá — nhãn v3 == 㝵 -> REVIEW `lop_nham:...` (nhãn xoá);
    còn lại giữ tier_v3, cờ am_da_quyet_ngoai_khoa=1 để mẻ chấm ưu tiên. KHÔNG chốt
    AM_DA_QUYET bao trùm (mất 243 ô + hạ 49 ô mốc đã ký).

    A-8: `lop_nham` = danh sách mục da_ky của decisions.yaml [{id, syllable, ocr, den}];
    None -> mặc định một mục ('người', 㝵) với rule RULE_LOP_NHAM (hành vi S7)."""
    if lop_nham is None:
        lop_nham = [{"id": RULE_LOP_NHAM.split(":", 1)[1], "syllable": "người",
                     "ocr": NGUOI_NHAM, "den": "REVIEW"}]
    nham = {(chuan_am(x["syllable"]), x["ocr"]): x for x in lop_nham}
    n_nham = n_ngoai = 0
    for r in records:
        if r.get("qd01_locked") or chuan_am(r.get("syllable")) not in AM_DA_QUYET:
            continue
        if r.get("qd01_excluded"):
            continue
        r["am_da_quyet_ngoai_khoa"] = 1
        n_ngoai += 1
        x = nham.get((chuan_am(r.get("syllable")), r["label"]))
        if x is not None:
            r["tier_goc"], r["rule_goc"] = r["tier"], r["rule"]
            r["tier"], r["rule"], r["label"] = x["den"], f"{dcs.RULE_LOP_NHAM}:{x['id']}", ""
            n_nham += 1
    return n_nham, n_ngoai


def compute_flank_gold(records):
    """N5k: số ô kề (syl_idx ± 1, cùng cột) có tier_v3 == CHAR_A ∈ {0, 1, 2}."""
    t = {(r["book"], r["page"], int(r["column"]), int(r["syl_idx"])): r.get("tier_v3")
         for r in records if r.get("syl_idx", "") != ""}
    for r in records:
        if r.get("syl_idx", "") == "":
            r["flank_gold"] = 0
            continue
        k = (r["book"], r["page"], int(r["column"]), int(r["syl_idx"]))
        r["flank_gold"] = sum(1 for d in (-1, 1)
                              if t.get((k[0], k[1], k[2], k[3] + d)) == "CHAR_A")


def maybe_s3(p, page_png, qn_to_nom, vs3):
    if vs3 is None or not p.get("ocr_char"):
        return None
    if not (p.get("matched") or p.get("anchored")):
        return None
    cands = qn_to_nom.get((p["syllable"] or "").lower(), [])
    if p["ocr_char"] in cands:
        return None
    return vs3.compute(page_png, p.get("bbox"), p["ocr_char"], cands)


def _record(book, page, page_png, idx, p, dec, s3, seg_backend) -> dict:
    """Một bản ghi PASS 1 (một cặp đã tier). Gom một chỗ để đường cũ (--no-two-pass)
    và PASS 1b ghi CÙNG một schema; 4 cột N4e (syllable_ocr, syllable_raw,
    p_register, band_touched) ở đường cũ = ''/''/''/0."""
    return {
        "book": book, "page": page, "column": p["column"], "idx": idx,
        "page_png": page_png, "ocr_char": p.get("ocr_char") or "",
        # Canonical lowercase syllable for ALL tiers (not just SYLLABLE
        # below): the QN→Nôm dict and decide_label are already case-folded,
        # so keeping raw OCR case here only fragmented the reading vocabulary
        # (Nhị/nhị/NHỊ → 3 classes) and inflated the distinct-syllable count.
        # NFC + modern tone-mark placement are already applied upstream (parse_v5).
        "syllable": str(p["syllable"]).lower(), "bbox": p.get("bbox"),
        "tier": dec.tier, "rule": dec.rule_id, "label": dec.label or "",
        "s3_cosine": round(s3.cosine, 3) if s3 else "",
        "seg_backend": seg_backend,
        # N0b: chỉ số ô trong cột Nôm / âm tiết trong dòng QN (từ ops của
        # anchor_align qua _pair_new) — khoá bền cho đối soát thế hệ.
        "nom_idx": p.get("nom_idx", ""), "syl_idx": p.get("syl_idx", ""),
        # N4e: âm VietOCR nguyên văn / âm sau normalize_column / posterior thanh ghi
        # / cột chạm biên băng — cờ ghi DÀY 0/1 (không để trống -> pandas không ép float).
        "syllable_ocr": p.get("syllable_ocr", ""),
        "syllable_raw": p.get("syllable_raw", ""),
        "p_register": (round(float(p["p_register"]), 4)
                       if p.get("p_register", "") != "" else ""),
        "band_touched": int(bool(p.get("band_touched", False))),
        # A-6 (N4d/N4e): số đếm cột (n_* và count_source sang columns.csv ở S9) + nguồn hộp
        "n_ocr": p.get("n_ocr", ""), "n_qn": p.get("n_qn", ""), "n_det": p.get("n_det", ""),
        "count_source": p.get("count_source", ""), "box_source": p.get("box_source", ""),
        # B-2 (--visual-emission): P(âm ghép | crop hộp OCR thô) out-of-fold; '' khi tắt cờ,
        # âm ∉ lớp hoặc hộp không cắt được. CHỈ ĐO — không quyết tier.
        "p_visual_syl": p.get("p_visual_syl", ""), "visual_fold": p.get("visual_fold", ""),
        "visual_argmax": p.get("visual_argmax", ""), "visual_max_p": p.get("visual_max_p", ""),
        # A-7 (PASS 1c): cột chẩn đoán/cờ, mặc định dày; PASS 1c ghi đè trên đường v3
        **FIELDS_1C,
    }


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
    # D-1 (Khối D, mặc định TẮT): crops_v2 F3g tái định tâm theo hình chiếu mực — ghi SONG SONG
    # $out/crops_v2/<tier>/<cùng tên>.png + $out/labels_crops_v2.csv; KHÔNG đổi crop giao nộp/labels.csv.
    ap.add_argument("--crops-v2", action="store_true",
                    help="D-1: thêm crops_v2/ (F3g, pipeline.align_engine.recenter_f3g) + labels_crops_v2.csv "
                         "sidecar; crop giao nộp và labels.csv giữ nguyên byte")
    # A-5 (flow N4a–N4c): mặc định BẬT. --no-two-pass giữ đường cũ (tier ngay trong
    # vòng PASS 1, không neo ngữ liệu, không p_register) để tái lập bộ hiện tại.
    ap.add_argument("--two-pass", action=argparse.BooleanOptionalAction, default=True,
                    help="PASS 1b: pair_pages LOO + DP lại với ANCHOR_CAP + posterior "
                         "(mặc định bật; --no-two-pass = đường cũ)")
    # None = LẤY TỪ CONFIG (step2.crop_pad_frac). Truyền --pad chỉ để ghi đè khi thí
    # nghiệm; đường chạy sản xuất phải để cấu hình quyết, nếu không lại tái diễn lớp
    # lỗi "cấu hình khai một đằng, mã chạy một nẻo" mà T4.a tìm ra.
    ap.add_argument("--pad", type=float, default=None)
    # A-6 (flow N3f/N4d): luật hộp. syl_index = hộp thô thr 0,2 ±0,25w + 3 nhánh theo chỉ
    # số; legacy = TRỌN GÓI luật cũ (thr 0,3 ±0,5w, ép đếm + _monotone_assign) để tái lập
    # bộ 64.525 (selftest: bbox khớp 100%). Chỉ có nghĩa với --reseg detector.
    ap.add_argument("--box-rule", default="syl_index", choices=list(ap_mod.BOX_RULES),
                    help="luật gán hộp ảnh (A-6): syl_index (mặc định) | legacy (trọn gói cũ)")
    ap.add_argument("--qd01-cells", default="none",
                    help="khoá QĐ-01: 'none' (mặc định: tự động 100%%, không áp đặt người) | "
                         "đường dẫn tệp CSV khoá (ví dụ config/qd01_cells.csv để thí nghiệm/tái lập cũ)")
    # B-5 (DANH_MUC 1B): phạm vi khoá QĐ-01. col (mặc định) = cột chứa ô khoá chạy trọn
    # gói luật hộp cũ (legacy_locked_col); cell = cột chạy 3 nhánh bình thường, riêng ô
    # khoá dùng bbox/prev/next_cu, hàng xóm trùng hộp (IoU >= SHIFT_IOU) về midpoint
    # (box_source=shifted_by_lock). THÍ NGHIỆM — mặc định KHÔNG đổi.
    ap.add_argument("--lock-scope", default="col", choices=list(LOCK_SCOPES),
                    help="B-5: col (mặc định, trọn cột legacy) | cell (khoá từng ô + dời hàng xóm) "
                         "| cell_fallback (cell + lùi cả cột về legacy khi ô khoá lộ hộp lệch)")
    ap.add_argument("--pages", default="",
                    help="thí nghiệm: CSV có cột book,page (mã sách stt*) — chỉ build các trang này")
    ap.add_argument("--qd01a-decisions", default=str(REPO / "config" / "qd01a_decisions.csv"),
                    help="phán quyết người cho ô QĐ-01 trôi + 12 ô A-3 (N5g); thiếu tệp = rỗng")
    ap.add_argument("--decisions", default=str(REPO / "config" / "decisions.yaml"),
                    help="A-8: quyết định theo lớp (corpus_readings/di_the/lop_nham); chỉ mục "
                         "da_ky có xuat_xu được áp; 'none' = không áp gì (CHỈ để thí nghiệm)")
    ap.add_argument("--reseg", default="midpoint",
                    choices=["midpoint", "valley_n", "valley_guarded", "detector"],
                    help="column re-segmentation for crop boxes (default midpoint; valley_* are "
                         "opt-in experiments — see seg_valley_n_ab.py / seg_smart_ab.py). "
                         "valley_guarded needs the encoder (auto-loaded). detector uses a trained "
                         "char_detector/detector.pt (Kaggle; falls back to midpoint if absent).")
    # B-2 (DANH_MUC 1B): phát xạ ảnh trong DP PASS 1b. MẶC ĐỊNH TẮT — chỉ đo/sidecar, không đổi
    # tier. Cần models/fold0..4.pt của B-1' (Kaggle); thiếu -> cảnh báo rõ, chạy văn bản thuần
    # (--strict thì dừng). Tham số λ/gap/cap/neutral_p đọc config step2.visual_emission, CLI ghi đè.
    ap.add_argument("--visual-emission", action=argparse.BooleanOptionalAction, default=False,
                    help="B-2: cộng λ·min(−logP, cap) (CNN âm tiết OOF, hộp OCR thô) vào chi phí DP "
                         "PASS 1b + posterior cùng chi phí; ghi p_visual_syl (mặc định TẮT)")
    ap.add_argument("--visual-models", default=None,
                    help="thư mục fold0..4.pt (mặc định: KhoiB/v3/p_visual_oof_v3_results/models rồi KhoiB/v3/models)")
    ap.add_argument("--visual-lambda", type=float, default=None, help="λ (mặc định config, 0,25)")
    ap.add_argument("--visual-gap", type=float, default=None, help="khe ảnh cộng vào COST_DEL/INS ×λ (mặc định 8)")
    ap.add_argument("--visual-cap", type=float, default=None, help="trần −logP (mặc định 12)")
    ap.add_argument("--force", "-f", action="store_true", help="bỏ qua khóa .FROZEN, ép buộc ghi đè")
    args = ap.parse_args()

    out = Path(args.out)
    # HÀNG RÀO: bộ đã đóng băng (dataset_out/.FROZEN) không được build đè — PASS 2 dọn
    # thư mục crop trước khi cắt. Cùng quy ước với run_pipeline.sh (FROZEN_OVERRIDE=GHIDE hoặc cờ --force).
    if (out / ".FROZEN").exists() and os.environ.get("FROZEN_OVERRIDE") != "GHIDE" and not args.force:
        raise SystemExit(f"[build] {out}/.FROZEN tồn tại — không build đè bộ đã đóng băng. "
                         "Dùng thêm cờ --force, hoặc FROZEN_OVERRIDE=GHIDE, hoặc xóa file .FROZEN.")
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    paths = config["paths"]
    # A-8: nạp decisions.yaml NGAY ĐẦU — schema/xuất xứ sai thì dừng trước khi tốn 10 phút align
    # (tên `qdl` — quyết định lớp — vì `dec` đã là LabelDecision trong vòng PASS 1)
    qdl = None
    if str(args.decisions).lower() != "none":
        qdl = dcs.load(args.decisions)
        _rep = qdl.report()
        print(f"  [decisions] {args.decisions}: corpus_readings da_ky {_rep['corpus_readings']['da_ky']}"
              f"/{_rep['corpus_readings']['n']} | di_the {_rep['di_the']['da_ky']}/{_rep['di_the']['n']} | "
              f"lop_nham {_rep['lop_nham']['da_ky']}/{_rep['lop_nham']['n']} | chờ ký {_rep['n_cho_ky']} "
              "(chỉ mục da_ky được áp)", flush=True)
    else:
        print("  [decisions] --decisions none: KHÔNG áp corpus_readings/di_the/lop_nham (thí nghiệm)", flush=True)

    # --- HÌNH HỌC CẮT ẢNH: cấu hình -> mã (nối 2026-08-25) --------------------
    _s2 = config.get("step2") or {}
    if args.pad is None:
        args.pad = float(_s2.get("crop_pad_frac", 0.12))
    _ov = float(_s2.get("box_overlap_frac", ap_mod.BOX_OVERLAP_FRAC))
    ap_mod.BOX_OVERLAP_FRAC = _ov
    print(f"  [hình học] đệm cắt = {args.pad} | biên nới dọc F = {_ov} "
          f"-> hộp cao {1 + 2 * _ov:.2f} × bước lặp", flush=True)
    # A-6 (N3f): ngưỡng detector + biên x đọc từ config (step2.det_thr / det_xmargin),
    # gán ngược vào engine như BOX_OVERLAP_FRAC. legacy KHÔNG đọc: trọn gói hằng cũ.
    locked_cols: dict = {}
    if args.box_rule == "syl_index":
        ap_mod.DETECTOR_THR = float(_s2.get("det_thr", ap_mod.DETECTOR_THR))
        ap_mod.DETECTOR_XMARGIN = float(_s2.get("det_xmargin", ap_mod.DETECTOR_XMARGIN))
        if args.reseg == "detector":
            if args.qd01_cells.lower() == "none":
                print("  [hộp] --qd01-cells none: KHÔNG khoá cột QĐ-01 (thí nghiệm)", flush=True)
            elif not Path(args.qd01_cells).exists():
                raise SystemExit(f"[hộp] --box-rule syl_index cần {args.qd01_cells} (khoá QĐ-01, "
                                 "sinh bằng pipeline/tools/sinh_qd01_cells.py); thiếu tệp thì ô "
                                 "khoá sẽ đổi hộp/md5 âm thầm. Dùng --qd01-cells none nếu CỐ Ý.")
            else:
                locked_cols = load_locked_columns(args.qd01_cells)
        print(f"  [hộp] luật = syl_index | thr = {ap_mod.DETECTOR_THR} | biên x = "
              f"±{ap_mod.DETECTOR_XMARGIN}w | cột có ô khoá QĐ-01 chạy luật cũ: "
              f"{sum(len(v) for v in locked_cols.values()):,} cột / "
              f"{len(locked_cols):,} trang | lock-scope = {args.lock_scope}"
              + (" (cột KHÔNG chạy legacy; khoá từng ô + dời hàng xóm)" if args.lock_scope == "cell" else ""),
              flush=True)
    else:
        print(f"  [hộp] luật = legacy TRỌN GÓI (thr {ap_mod.LEGACY_DETECTOR_THR}, biên x "
              f"±{ap_mod.LEGACY_DETECTOR_XMARGIN}w, ép đếm + _monotone_assign)", flush=True)
    # Giá trị TOÀN CỤC (step2) — sách khai books[].det_xmargin / det_thr (book_layout) ghi
    # đè cho riêng sách đó trong vòng lặp PASS 1 rồi khôi phục; sách STT không khai -> y hệt.
    det_thr_global, det_xmargin_global = ap_mod.DETECTOR_THR, ap_mod.DETECTOR_XMARGIN
    det_params_by_book: dict = {}
    pages_sel: set | None = None
    if args.pages:
        with open(args.pages, encoding="utf-8", newline="") as f:
            pages_sel = {(row["book"], row["page"]) for row in csv.DictReader(f)}
        print(f"  [thí nghiệm] --pages {args.pages}: chỉ build {len(pages_sel)} trang", flush=True)
    # Trần chi phí neo ngữ liệu (PASS 1b): config step2.anchor_cap ghi đè hằng engine;
    # gán ngược vào engine để lab đọc `anchor_align.ANCHOR_CAP` thấy đúng giá trị chạy.
    anchor_cap = float(_s2.get("anchor_cap", aa.ANCHOR_CAP))
    aa.ANCHOR_CAP = anchor_cap
    if args.two_pass:
        print(f"  [đệ quy] PASS 1b bật | ANCHOR_CAP = {anchor_cap} | neo khi cặp thấy ở "
              f">= {ANCHOR_MIN_OTHER_PAGES} trang khác (LOO)", flush=True)
    else:
        print("  [đệ quy] --no-two-pass: đường cũ (tier ngay trong PASS 1, không neo)", flush=True)
    # B-2: emitter ảnh (chỉ khi --visual-emission và --two-pass). Nạp lười theo fold trang.
    vis_em = None
    _vcfg = _s2.get("visual_emission") or {}
    vis_params = {"lam": float(_vcfg.get("lambda", 0.25) if args.visual_lambda is None else args.visual_lambda),
                  "gap_vis": float(_vcfg.get("gap", 8.0) if args.visual_gap is None else args.visual_gap),
                  "cap": float(_vcfg.get("cap", 12.0) if args.visual_cap is None else args.visual_cap),
                  "neutral_p": float(_vcfg.get("neutral_p", 0.5))}
    if args.visual_emission:
        from pipeline.align_engine.visual_emission import VisualEmitter
        if not args.two_pass:
            raise SystemExit("[B-2] --visual-emission cần PASS 1b (--two-pass); bỏ --no-two-pass.")
        try:
            vis_em = VisualEmitter(models_dir=args.visual_models or _vcfg.get("models_dir"),
                                   strict=bool(args.strict), **vis_params)
        except FileNotFoundError as e:
            raise SystemExit(f"[B-2 STRICT] {e}") from e
        if vis_em.available:
            print(f"  [B-2] phát xạ ảnh trong DP BẬT | {vis_em.describe()}", flush=True)
        else:
            print(f"  [B-2] --visual-emission nhưng KHÔNG SẴN SÀNG -> chạy VĂN BẢN THUẦN, "
                  f"p_visual_syl để trống. Lý do: {vis_em.reason}", flush=True)
            vis_em = None
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
        _backend = preflight_detector(args.reseg, box_rule=args.box_rule)
        print(f"  [reseg] backend thực dùng = {_backend}", flush=True)
    # (2026-09-22) ckpt/resize detector THEO SÁCH (books[].detector_ckpt / detector_resize, lab/i5_detector_v2):
    # resolve + kiểm tệp tồn tại + dựng thử detector NGAY (fail fast), trước khi tốn phút align. Sách không
    # khai -> None/linear = ckpt toàn cục v1, STT không đổi byte. Chỉ luật syl_index + reseg detector dùng.
    det_ckpt_by_book: dict = {}
    for _b in config["books"]:
        _lay = book_layout(_b)
        if args.reseg == "detector" and args.box_rule == "syl_index" and (_lay.detector_ckpt or _lay.detector_resize != "linear"):
            _ck = resolve_detector_ckpt(_lay.detector_ckpt, REPO)          # FileNotFoundError nếu thiếu
            det_ckpt_by_book[_b["name"]] = (_ck, _lay.detector_resize)
            _d = ap_mod._get_detector(strict=True, thr=(_lay.det_thr if _lay.det_thr is not None
                                                        else float(_s2.get("det_thr", ap_mod.DETECTOR_THR))),
                                      ckpt=_ck, resize=_lay.detector_resize)
            print(f"  [reseg] {_b['name']}: detector riêng sách ckpt = {_ck or '(toàn cục v1)'} | resize = "
                  f"{_lay.detector_resize} | img {_d.img} | backend = "
                  f"{ap_mod.detector_backend_name(_ck, _lay.detector_resize)}", flush=True)

    # ---------- PASS 1: align all pages, collect records (no crop yet) ----------
    records = []
    pages_done = 0
    page_states = []        # [(book_code, page, page_png, rec)] — chỉ khi --two-pass
    layout_gate_stats: dict = {}   # {book: {...}} chỉ với sách khai layout/n_columns (không STT)
    for b in config["books"]:
        book = b["name"]
        data_dir = data_root / book
        # Bố cục theo sách (khoá tuỳ chọn layout/n_columns; vắng = STT 9 cột). Đọc
        # TRƯỚC khi duyệt trang: giá trị sai -> ValueError ngay, không rơi ngầm về 9.
        lay = book_layout(b)
        if lay is not DEFAULT_LAYOUT:
            print(f"[align] {book}: layout={lay.layout} n_columns={lay.n_columns} "
                  f"qn_syllables_per_column={lay.qn_per_column}", flush=True)
        # Tham số detector THEO SÁCH (2026-09-22): chỉ luật syl_index đọc DETECTOR_*; legacy
        # trọn gói hằng cũ. Vắng khoá -> giá trị step2 toàn cục (STT: 0,2 / ±0,25w, không đổi).
        if args.box_rule == "syl_index":
            ap_mod.DETECTOR_THR = (det_thr_global if lay.det_thr is None else lay.det_thr)
            ap_mod.DETECTOR_XMARGIN = (det_xmargin_global if lay.det_xmargin is None
                                       else lay.det_xmargin)
            if (ap_mod.DETECTOR_THR, ap_mod.DETECTOR_XMARGIN) != (det_thr_global, det_xmargin_global):
                print(f"[align] {book}: detector theo sách thr = {ap_mod.DETECTOR_THR} | "
                      f"biên x = ±{ap_mod.DETECTOR_XMARGIN}w (toàn cục {det_thr_global} / "
                      f"±{det_xmargin_global}w)", flush=True)
            det_params_by_book[book] = {"det_thr": ap_mod.DETECTOR_THR,
                                        "det_xmargin": ap_mod.DETECTOR_XMARGIN,
                                        "box_decoder": lay.box_decoder}
            # ckpt/resize detector theo sách (2026-09-22): gán cho vòng trang của sách này, khôi phục sau PASS 1
            ap_mod.DETECTOR_CKPT, ap_mod.DETECTOR_RESIZE = det_ckpt_by_book.get(book, (None, "linear"))
            if book in det_ckpt_by_book:
                det_params_by_book[book]["detector_ckpt"] = ap_mod.DETECTOR_CKPT
                det_params_by_book[book]["detector_resize"] = ap_mod.DETECTOR_RESIZE
                print(f"[align] {book}: detector ckpt = {ap_mod.DETECTOR_CKPT or '(toàn cục v1)'} | resize = "
                      f"{ap_mod.DETECTOR_RESIZE} -> seg_backend {ap_mod.detector_backend_name()}", flush=True)
            if lay.box_decoder != "legacy":
                print(f"[align] {book}: box_decoder = {lay.box_decoder} (pitch_decode: ứng viên ≥ 0,05 + ô ảo "
                      f"chiếu mực, DP theo bước cột; n_det vẫn = hộp thô ở det_thr)", flush=True)
        trans = sorted(glob.glob(str(data_dir / "transcriptions" / "page_*.json")))
        trans = [t for t in trans if not t.endswith("_qn_ocr_cache.json")]
        if args.limit:
            trans = trans[:args.limit]
        print(f"[align] {book}: {len(trans)} pages ...", flush=True)
        for pi, tf in enumerate(trans):
            page = Path(tf).stem
            if pages_sel is not None and (_book_code(book), page) not in pages_sel:
                continue
            try:
                # B-5 scope=cell: KHÔNG truyền cột khoá -> mọi cột đi 3 nhánh syl_index;
                # ô khoá lấy bbox_cu ở PASS 1c (apply_qd01_lock) rồi apply_lock_shift.
                rec = align_page(page, data_dir, qn_dict_set, qn_to_nom, similar, "new",
                                 reseg_mode=args.reseg, encoder=reseg_encoder,
                                 box_rule=args.box_rule,
                                 locked_columns=(None if args.lock_scope != "col"
                                                 else locked_cols.get((_book_code(book), page))),
                                 legacy_also_columns=(locked_cols.get((_book_code(book), page))
                                                      if args.lock_scope != "col" else None),
                                 layout=lay)
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
            if lay is not DEFAULT_LAYOUT:
                # Cổng bố cục theo sách (chỉ sách khai layout/n_columns): đếm page_ok,
                # phương pháp cột, cột QN lệch âm -> summary.json["layout_gate"][book]
                g = rec.get("layout_gate") or {}
                lg = layout_gate_stats.setdefault(book, {
                    "layout": lay.layout, "n_columns": lay.n_columns,
                    "qn_syllables_per_column": lay.qn_per_column,
                    "pages": 0, "page_ok": 0, "col_method": {}, "pages_not_ok": []})
                lg["pages"] += 1
                lg["page_ok"] += int(bool(rec.get("page_ok")))
                m = g.get("col_method", "?")
                lg["col_method"][m] = lg["col_method"].get(m, 0) + 1
                if not rec.get("page_ok"):
                    lg["pages_not_ok"].append({"page": page, "col_method": m,
                                               "n_nom_cols": g.get("n_nom_cols"),
                                               "n_qn_cols": g.get("n_qn_cols"),
                                               "bad_syl_cols": g.get("bad_syl_cols", [])})
            page_png = str(data_dir / "pages" / f"{page}.png")
            if args.two_pass:
                # N3g: KHÔNG tier trong vòng — chỉ gom trạng thái cột; PASS 1b bên dưới
                # ghép lại rồi mới gọi maybe_s3/decide_label trên cặp lượt 2.
                page_states.append((_book_code(book), page, page_png, rec))
                continue
            for idx, p in enumerate(rec["pairs"]):
                s3 = maybe_s3(p, page_png, qn_to_nom, vs3) if vs3 else None
                dec = decide_label(p.get("ocr_char"), p["syllable"], p.get("matched", False),
                                   qn_to_nom, similar, s3=s3, anchored=p.get("anchored", False))
                records.append(_record(_book_code(book), page, page_png, idx, p, dec, s3,
                                       rec.get("seg_backend", "")))

    if args.box_rule == "syl_index":
        ap_mod.DETECTOR_THR, ap_mod.DETECTOR_XMARGIN = det_thr_global, det_xmargin_global
        ap_mod.DETECTOR_CKPT, ap_mod.DETECTOR_RESIZE = None, "linear"

    # ---------- PASS 1b: đệ quy hai lượt (flow N4a–N4c) ----------
    n_anchor_pairs = 0          # số cặp lượt 2 được neo LOO hạ chi phí thật (N4a "ô được neo")
    n_pairs_ops1 = n_pairs_ops2 = n_pairs_changed = n_cols_1b = n_band_touched = 0
    n_cols_vis = n_cols_vis_changed = n_pairs_vis_changed = 0     # B-2
    pair_pages = {}
    if args.two_pass:
        # N4a: pair_pages từ MỌI match lượt 1 (không chỉ confirmed) — LOO theo (book, page)
        pair_pages = build_pair_pages([(b, pg, r) for b, pg, _png, r in page_states])
        _pp_json = {f"{c}\t{sy}": sorted(f"{b}/{pg}" for b, pg in pages)
                    for (c, sy), pages in sorted(pair_pages.items())}
        json.dump({"anchor_cap": anchor_cap, "min_other_pages": ANCHOR_MIN_OTHER_PAGES,
                   "n_pair_keys": len(pair_pages),
                   # cặp có >= min_other trang: neo được ở MỌI trang chưa chứa nó
                   # (và ở trang chứa nó nếu còn >= min_other trang khác)
                   "n_keys_ge_min_pages": sum(1 for v in pair_pages.values()
                                              if len(v) >= ANCHOR_MIN_OTHER_PAGES),
                   "pair_pages": _pp_json},
                  open(out / "pair_pages.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=0)
        print(f"  [PASS 1b] pair_pages: {len(pair_pages):,} cặp (chữ, âm) từ "
              f"{sum(len(v) for v in pair_pages.values()):,} (cặp, trang) — "
              f"{out / 'pair_pages.json'}", flush=True)
        for bookc, page, page_png, rec in page_states:
            here = (bookc, page)
            page_pairs = []
            # B-2: ảnh xám trang nạp 1 lần; hộp phát xạ = hộp OCR thô của cluster (có sẵn lúc DP)
            vis_gray = None
            if vis_em is not None:
                vis_gray = vis_load_gray(page_png)
                if vis_gray is None:
                    print(f"   [B-2 warn] {bookc}/{page}: không đọc được ảnh -> văn bản thuần", flush=True)
            for cs in rec["col_states"]:
                chars, syllables = cs["cluster"]["chars"], cs["syllables"]
                vis, vinfo = None, None
                if vis_em is not None and vis_gray is not None:
                    _boxes = [c.get("bbox") for c in chars]
                    logP, vinfo = vis_em.emission(vis_gray, _boxes, syllables, bookc, page)
                    vis = (vis_em, logP)
                    n_cols_vis += 1
                # N4b + N4c: DP lại với cost_fn neo + posterior CÙNG cost_fn (+ ảnh nếu B-2)
                ops2, post, touched, n_anch = realign_with_anchors(
                    chars, syllables, qn_to_nom, similar, pair_pages, here, anchor_cap, vis=vis)
                if vis is not None:
                    _ops_txt = realign_with_anchors(chars, syllables, qn_to_nom, similar,
                                                    pair_pages, here, anchor_cap)[0]
                    _mt = {(o["nom_idx"], o["syl_idx"]) for o in _ops_txt if o["op"] == "match"}
                    _mv = {(o["nom_idx"], o["syl_idx"]) for o in ops2 if o["op"] == "match"}
                    n_pairs_vis_changed += len(_mt ^ _mv)
                    n_cols_vis_changed += int(_mt != _mv)
                n_cols_1b += 1
                n_anchor_pairs += n_anch
                n_band_touched += int(touched)
                m1 = {(o["nom_idx"], o["syl_idx"]) for o in cs["ops1"] if o["op"] == "match"}
                m2 = {(o["nom_idx"], o["syl_idx"]) for o in ops2 if o["op"] == "match"}
                n_pairs_ops1 += len(m1); n_pairs_ops2 += len(m2)
                n_pairs_changed += len(m1 ^ m2)
                reseg_boxes, syl_ocr = cs["reseg_boxes"], cs["syllable_ocr"]
                # A-6 (N4d): luật mới gán hộp LẠI theo ops lượt 2 (nhánh a đi theo syl_idx
                # nên phụ thuộc đường ghép); legacy/legacy_locked_col giữ hộp PASS 1 (theo
                # nom_idx, không phụ thuộc đường ghép).
                box_source, count_source = cs.get("box_source"), cs.get("count_source", "")
                if cs.get("box_rule") == "syl_index":
                    reseg_boxes, box_source, count_source = ap_mod.assign_boxes(
                        cs["G"], ops2, cs["n_ocr"], cs["n_qn"], cluster=cs["cluster"],
                        cb=cs.get("cb"))
                elif cs.get("box_rule") == "pitch":
                    # box_decoder=pitch (2026-09-22): hộp đã giải mã theo bước, gán lại theo syl_idx
                    # của ops lượt 2; nguồn hộp từng ô giữ G_src (detector/detector_low/ink_cut)
                    _rb, _bs, _cs = ap_mod.assign_boxes_pitch(cs["G"], cs.get("G_src"), ops2,
                                                              cs["n_ocr"], cs["n_qn"])
                    if _rb is not None:
                        reseg_boxes, box_source, count_source = _rb, _bs, _cs
                # PASS 1c (A-7) cần ops CUỐI + hộp cuối của cột (khe QĐ-01 không có record)
                cs["ops2"], cs["post"], cs["boxes2"] = ops2, post, reseg_boxes
                cs["box_source2"], cs["count_source2"] = box_source, count_source
                col_pairs = []
                for mpair in aa.matched_pairs(ops2):
                    i, j = mpair["nom_idx"], mpair["syl_idx"]
                    bbox = reseg_boxes[i] if reseg_boxes else chars[i].get("bbox")
                    col_pairs.append({
                        "ocr_char": mpair["ocr_char"], "bbox": bbox,
                        "syllable": mpair["syllable"], "confirmed": mpair["confirmed"],
                        "nom_idx": i, "syl_idx": j,
                        "syllable_raw": syllables[j],
                        "syllable_ocr": syl_ocr[j] if len(syl_ocr) == len(syllables) else "",
                        "p_register": post.get((i, j), 0.0), "band_touched": touched,
                        "column": cs["line_id"], "matched": cs["matched"],
                        "n_ocr": cs.get("n_ocr", ""), "n_qn": cs.get("n_qn", ""),
                        "n_det": cs.get("n_det", ""), "count_source": count_source,
                        "box_source": box_source[i] if box_source else "",
                        # B-2: '' khi tắt cờ / âm ∉ lớp / hộp không cắt được
                        **({"p_visual_syl": (round(float(math.exp(logP[i, j])), 4)
                                             if vinfo["syl_in_classes"][j] and vinfo["argmax"][i] != "" else ""),
                            "visual_fold": vinfo["fold"], "visual_argmax": vinfo["argmax"][i],
                            "visual_max_p": (round(float(vinfo["max_p"][i]), 4)
                                             if vinfo["argmax"][i] != "" else "")}
                           if vinfo is not None else {}),
                    })
                # anchored: cặp kề một cặp confirmed (cùng định nghĩa align_page:532-536)
                conf = [q["confirmed"] for q in col_pairs]
                for k, q in enumerate(col_pairs):
                    nbr = (k > 0 and conf[k - 1]) or (k + 1 < len(col_pairs) and conf[k + 1])
                    q["anchored"] = bool(nbr)
                page_pairs.extend(col_pairs)
            # A-7: KHÔNG tier ở đây — PASS 1c (tier_v3) quyết; `decide_label` chỉ còn
            # là chỗ giữ schema (tier/rule bị ghi đè toàn bộ). S3 (nếu nạp) chỉ ghi
            # s3_cosine để tra, không quyết tier. idx theo trang như cũ.
            for idx, p in enumerate(page_pairs):
                s3 = maybe_s3(p, page_png, qn_to_nom, vs3) if vs3 else None
                dec = decide_label(p.get("ocr_char"), p["syllable"], p.get("matched", False),
                                   qn_to_nom, similar, s3=None, anchored=p.get("anchored", False))
                records.append(_record(bookc, page, page_png, idx, p, dec, s3,
                                       rec.get("seg_backend", "")))
        print(f"  [PASS 1b] {n_cols_1b:,} cột | match lượt 1 {n_pairs_ops1:,} → lượt 2 "
              f"{n_pairs_ops2:,} | cặp đổi (hiệu đối xứng) {n_pairs_changed:,} | "
              f"ô được neo LOO {n_anchor_pairs:,} | cột chạm biên băng {n_band_touched:,}",
              flush=True)
        if vis_em is not None:
            print(f"  [B-2] phát xạ ảnh: {n_cols_vis:,} cột | hộp {vis_em.n_boxes:,} (không cắt được "
                  f"{vis_em.n_cut_fail:,}) | cột đổi đường ghép so với văn bản thuần {n_cols_vis_changed:,} "
                  f"| cặp đổi (hiệu đối xứng) {n_pairs_vis_changed:,}", flush=True)

    # ---------- PASS 1c (A-7, flow N5a–N5h): tier_v3 + L1 có cổng + khoá QĐ-01 ----------
    n_l1_moi = n_l1_nang = n_l1_giu = n_l1_hoa = n_l3 = 0
    n_promoted = n_tu_silver = 0
    cross_v3 = Counter()
    qd01_st, qd01_pending = Counter(), []
    lock_shift_st = Counter()
    n_lop_nham = n_nguoi_ngoai = 0
    cells = decisions = {}
    n_cr, n_dt = Counter(), Counter()          # A-8: ô áp corpus_readings / di_the theo id
    if args.two_pass:
        # N5a: thống kê LOO từ ops CUỐI (ops2) của mọi cột; colmap để tra chars/syl/hộp
        cols = tv3.cols_from_page_states([(b, pg, png, r) for b, pg, png, r in page_states])
        colmap = {}
        for (bookc, page, page_png, rec) in page_states:
            for cs in rec["col_states"]:
                col = {"book": bookc, "page": page, "column": cs["line_id"], "page_png": page_png,
                       "chars": [c.get("char") or c.get("ocr_char") or "" for c in cs["cluster"]["chars"]],
                       "syl": list(cs["syllables"]),
                       "boxes": cs.get("boxes2") or cs.get("reseg_boxes"),
                       "box_source": cs.get("box_source2") or cs.get("box_source"),
                       "count_source": cs.get("count_source2") or cs.get("count_source", ""),
                       "n_ocr": cs.get("n_ocr", ""), "n_qn": cs.get("n_qn", ""), "n_det": cs.get("n_det", ""),
                       # B-5: hộp midpoint cũ theo nom_idx (đích dời của apply_lock_shift) +
                       # hộp trọn gói luật cũ (chỉ cột khoá, scope cell*) cho apply_lock_fallback
                       "mid": ap_mod._reseg_column(cs["cluster"]),
                       "legacy_boxes": cs.get("legacy_boxes")}
                colmap[(bookc, page, cs["line_id"])] = col
        corpus = tv3.CorpusStats(cols, qn_to_nom, similar)
        # N5b–N5d: feats + p -> tier_v3 -> tier/rule/label (chốt is_plausible trước)
        cross_v3 = apply_tier_v3(records, corpus, colmap)
        print(f"  [PASS 1c] tier_v3: " + " | ".join(f"{k} {cross_v3[k]:,}" for k in
              ("CHAR_A", "CHAR_B", "SYL", "REVIEW")) + f" | pair_pages {len(corpus.pair_pages):,} "
              f"| bigram_pages {len(corpus.bigram_pages):,}", flush=True)
        # N5f (A-8): corpus_readings da_ky -> GOLD `corpus_reading:<id>` TRƯỚC L1 (L1 bỏ qua
        # GOLD nên không ghi đè âm đã ký); KHÔNG vào qn_to_nom, KHÔNG làm neo (rule khác)
        n_cr = dcs.apply_corpus_readings(records, qdl)
        print(f"  [corpus_readings] {sum(n_cr.values()):,} ô -> GOLD/corpus_reading:* "
              f"({len(n_cr)} mục da_ky" + (": " + ", ".join(f"{k} {v}" for k, v in sorted(n_cr.items()))
                                          if n_cr else "") + ")", flush=True)
        # N5e: L1 có cổng 2-gram ÂM–ÂM (từ col_states, LOO); neo chỉ từ CHAR_A
        # (rule s1_inter_s2_direct — KHÔNG gồm _lowp/corpus_readings)
        anchors = gold_direct_anchors(records)
        bigram_syl_pages = tv3.build_bigram_syl_pages(cols)
        n_l1_moi, n_l1_nang, n_l1_giu, n_l1_hoa = apply_am_sua_dau(
            records, qn_to_nom, nom_to_qn, anchors, bigram_syl_pages=bigram_syl_pages,
            colmap=colmap, corpus=corpus)
        print(f"  [L1 sửa dấu âm] đổi {n_l1_moi + n_l1_nang:,} ô -> GOLD/{RULE_L1} "
              f"({n_l1_moi:,} từ REVIEW, {n_l1_nang:,} từ SYLLABLE) | giữ gốc (thua) "
              f"{n_l1_giu:,} | hoà {n_l1_hoa:,} (l1_tie)", flush=True)
        # L3 (apply_cot_lech_cau_xuoi) và L5 (syllable_gate/PROMOTE) KHÔNG chạy ở đường
        # này — tier_v3 đã thay (N5c); hàm giữ cho --no-two-pass.
        # N5g: KHOÁ QĐ-01 theo (book,page,column,nom_idx) + QĐ-01a (MẶC ĐỊNH TẮT: none)
        cells = {}
        decisions = {}
        qd01_pending = []
        qd01_st = {"n_cells_in_build": 0, "n_locked": 0, "n_pending": 0, "n_khe": 0,
                   "n_khong_khop": 0, "n_bo": 0, "n_khac": 0, "n_excluded_ngoai_cells": 0,
                   "n_decisions_ngoai_cells_lock": 0, "n_decisions_ngoai_cells_khac": 0}
        lock_shift_st = {}
        if args.qd01_cells.lower() != "none":
            cells = load_qd01_cells_full(args.qd01_cells)
            decisions = load_qd01a_decisions(args.qd01a_decisions)
            built_pages = {(b, pg) for b, pg, _png, _r in page_states}
            seg_backend_of = {(b, pg): r.get("seg_backend", "") for b, pg, _png, r in page_states}
            qd01_st, qd01_pending = apply_qd01_lock(records, cells, decisions, colmap, built_pages,
                                                    seg_backend_of)
            n_cells_b = qd01_st["n_cells_in_build"]
            n_acc = (qd01_st["n_locked"] + qd01_st["n_pending"] + qd01_st["n_bo"] + qd01_st["n_khac"])
            print(f"  [QĐ-01] ô khoá trong {len(built_pages)} trang build: {n_cells_b:,} / "
                  f"{len(cells):,} tệp | locked {qd01_st['n_locked']:,} | pending "
                  f"{qd01_st['n_pending']:,} (khe {qd01_st['n_khe']}, không khớp "
                  f"{qd01_st['n_khong_khop']}, âm khác {qd01_st['n_pending'] - qd01_st['n_khe'] - qd01_st['n_khong_khop']}) "
                  f"| bo {qd01_st['n_bo']} | khac {qd01_st['n_khac']} | QĐ-01a ngoài khoá: excluded "
                  f"{qd01_st['n_excluded_ngoai_cells']}, lock {qd01_st['n_decisions_ngoai_cells_lock']}, "
                  f"khac {qd01_st['n_decisions_ngoai_cells_khac']}", flush=True)
            if n_acc != n_cells_b:
                # không dừng build (spec N5g) — ghi summary + in đỏ để đối soát
                print(f"  [QĐ-01] 🔴 locked + pending + bo + khac = {n_acc} ≠ {n_cells_b} ô khoá "
                      "trong build", flush=True)
            # B-5 (--lock-scope cell): hàng xóm trùng bbox_cu ô khoá -> midpoint (shifted_by_lock)
            if args.lock_scope != "col":
                lock_shift_st = apply_lock_shift(records, colmap)
                print(f"  [B-5 khoá ô] ô khoá {lock_shift_st['n_locked']:,} | hàng xóm dời về midpoint "
                      f"{lock_shift_st['n_shifted']:,} (cùng cột {lock_shift_st['n_shifted_same_col']}, "
                      f"cột kề {lock_shift_st['n_shifted_adj_col']}; trùng đúng byte {lock_shift_st['n_exact_eq']}) "
                      f"| khoá–khoá trùng {lock_shift_st['n_locked_locked']} | pending trùng "
                      f"{lock_shift_st['n_pending_overlap']} | không có midpoint {lock_shift_st['n_no_mid']} "
                      f"| sau dời vẫn trùng {lock_shift_st['n_still_overlap']}", flush=True)
                if args.lock_scope == "cell_fallback":
                    fb = apply_lock_fallback(records, colmap)
                    lock_shift_st.update({f"fallback_{k}": v for k, v in fb.items()})
                    print(f"  [B-5 lùi cột] cột khoá {fb['n_cols_lock']:,} | lùi về legacy "
                          f"{fb['n_cols_fallback']:,} cột (ô khoá lệch {fb['n_cols_fallback_lech']}, "
                          f"có hàng xóm dời {fb['n_cols_fallback_doi']}) | ô đổi hộp {fb['n_cells_fallback']:,} "
                          f"(trong đó đã dời {fb['n_cells_was_shifted']}) | không có legacy "
                          f"{fb['n_cols_no_legacy']} cột", flush=True)
        else:
            print("  [QĐ-01] TẮT (--qd01-cells none): 100% tự động theo tier_v3, không áp đặt người", flush=True)
        # N5h: 'người' ngoài khoá — chỉ hạ ô nhãn v3 == 㝵; không chốt AM_DA_QUYET bao trùm
        # A-8: lop_nham đọc từ decisions.yaml (mục da_ky); --decisions none -> không hạ ô nào
        n_lop_nham, n_nguoi_ngoai = apply_nguoi_ngoai_khoa(
            records, lop_nham=(qdl.ap_lop_nham() if qdl is not None else []))
        print(f"  [người ngoài khoá] {n_nguoi_ngoai:,} ô (am_da_quyet_ngoai_khoa=1), trong đó "
              f"lop_nham -> REVIEW: {n_lop_nham:,} "
              f"({', '.join(x['id'] for x in (qdl.ap_lop_nham() if qdl is not None else [])) or 'không mục nào'})",
              flush=True)
        # N5k
        compute_flank_gold(records)
    else:
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

    # N5i (A-8): label_canonical = label cho MỌI ô (cả hai đường chạy); mục di_the da_ky
    # (chuan người ký) ghi mã chuẩn vào cột này — KHÔNG đổi `label` (hình quan sát)
    n_dt = dcs.apply_di_the(records, qdl)
    print(f"  [di_the] label_canonical ≠ label: {sum(n_dt.values()):,} ô"
          + (" (" + ", ".join(f"{k} {v}" for k, v in sorted(n_dt.items())) + ")" if n_dt else
             " (chưa có mục di_the da_ky)"), flush=True)

    # ---------- SPLIT: [BỎ 16/09 — A-10 giai đoạn 2] ----------------------------
    # Không chia train/val/test nữa (ràng buộc bộ giao nộp 16/09). Ba cột cũ là hàm thuần
    # của (book,page): split_group == book|page, split == int(md5(book|page),16)%100
    # (<80 train, <90 val, còn lại test), label_in_train = lớp có trong train — ai cần thì
    # tự tính lại từ công thức (README ghi). Lý do chia theo TRANG (không theo cột) và lý do
    # bỏ luật "ép nhóm singleton vào train" xem git log -S "RỜI NHAU THEO TRANG".
    # S3 đã TẮT nên proto-index GOLD∧train cũng không cần (rebuild_proto_index ngoài flow).

    # ---------- PASS 2: materialize crops + quality columns [#3,#5] ----------
    # A-7/N7: crop_tiers = GOLD + SYLLABLE (+ SILVER: 0 ô ở đường v3 vì S3 không quyết)
    # + MỌI ô QĐ-01 pending (người phải nhìn được cả hộp mới lẫn bbox_cu).
    crop_tiers = {"GOLD", "SILVER", "SYLLABLE"} | ({"REVIEW"} if args.crop_review else set())
    if not args.no_crops:
        # dọn thư mục đích TRƯỚC khi cắt (chỉ trong $out): tránh tệp mồ côi của lần chạy
        # trước lẫn vào bộ giao nộp (25/08: silver/ 4.490 + syllable/ 1.674 tệp mồ côi)
        for d in ("gold", "silver", "syllable", "review"):
            if (out / d).is_dir():
                shutil.rmtree(out / d)
        if args.crops_v2 and (out / "crops_v2").is_dir():
            shutil.rmtree(out / "crops_v2")
    crops_v2_rows: list[dict] = []          # D-1 sidecar (chỉ khi --crops-v2)
    by_page = defaultdict(list)
    for r in records:
        by_page[r["page_png"]].append(r)
    labels = []

    def _can_cat(r) -> bool:
        return r["tier"] in crop_tiers or bool(r.get("_qd01_pending"))

    for png, recs in by_page.items():
        need = (not args.no_crops) and png and any(_can_cat(r) for r in recs)
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
        if args.crops_v2 and img is not None:
            # D-1: pitch cột = trung vị khoảng cách tâm bản ghi cùng cột (như _reseg_column), rơi về
            # trung vị trang khi cột <2 bản ghi/lệch — gán tạm, không ghi ra labels.csv
            _cys = {c: [(r["bbox"][1] + r["bbox"][3]) / 2.0 for r in rs] for c, rs in by_col.items()}
            _pp = rf3g.page_pitch(_cys)
            for c, rs in by_col.items():
                _pc = rf3g.column_pitch(_cys[c], fallback=_pp)
                for r in rs:
                    r["_pitch"] = _pc

        for r in recs:
            img_rel = q = None
            if img is not None and _can_cat(r):
                fn = f"{r['book']}_{r['page']}_c{r['column']:02d}_{r['idx']:03d}.png"
                # N7: ô khoá QĐ-01 cắt bằng bbox_cu + prev/next_bbox_cu (md5 phụ thuộc cả
                # hộp hàng xóm qua carve) -> md5 tất định, không phụ thuộc hộp mới của láng giềng
                if r.get("_has_prev_next_cu"):
                    pv, nx = r.get("_prev_bbox_cu"), r.get("_next_bbox_cu")
                else:
                    pv, nx = r.get("_prev_bbox"), r.get("_next_bbox")
                q = save_crop(img, gray_full, r.get("bbox"), args.pad, out / r["tier"].lower() / fn,
                              tighten=not args.no_tighten, prev_bbox=pv, next_bbox=nx)
                if q:
                    img_rel = f"{r['tier'].lower()}/{fn}"
                if q and args.crops_v2:
                    # D-1: crop v2 song song, cùng tên tệp dưới crops_v2/<tier>/; gray_full có thể None
                    # khi --no-carve -> recenter_f3g tự rơi về cửa sổ cũ (reason no_data)
                    q2 = rf3g.save_crop_v2(img, gray_full, r.get("bbox"), pv, nx, r.get("_pitch"), args.pad,
                                           out / "crops_v2" / r["tier"].lower() / fn,
                                           tighten=not args.no_tighten)
                    crops_v2_rows.append({
                        "image": img_rel, "book": r["book"], "page": r["page"], "column": r["column"],
                        "nom_idx": r.get("nom_idx", ""), "syl_idx": r.get("syl_idx", ""), "tier": r["tier"],
                        "bbox": json.dumps(r.get("bbox")), "image_md5": q["md5"],
                        "bbox_v2": json.dumps(q2["bbox_v2"]) if q2 else "",
                        "bbox_v2_win": json.dumps(q2["bbox_v2_win"]) if q2 else "",
                        "crop_mode": q2["crop_mode"] if q2 else "",
                        "guard_reason": q2["meta"].get("reason", "") if q2 else "cut_fail",
                        "recenter_shift": q2["recenter_shift"] if q2 else "",
                        "pitch": q2["pitch"] if q2 else "",
                        "image_md5_v2": q2["md5"] if q2 else "", "ink_pct_v2": q2["ink"] if q2 else "",
                        "crop_w_v2": q2["w"] if q2 else "", "crop_h_v2": q2["h"] if q2 else "",
                        "crop_quality_flag_v2": q2["crop_quality_flag"] if q2 else "",
                        "stray_ink_v2": q2["stray_ink"] if q2 else "",
                        "border_ink_v2": q2["border_ink"] if q2 else "",
                    })
                if r.get("_qd01_pending") and r.get("_bbox_cu") and r["_bbox_cu"] != r.get("bbox"):
                    # ép cắt thêm crop theo bbox_cu để người đối chiếu (qd01a_pending.csv)
                    fn_cu = fn[:-4] + "_cu.png"
                    q_cu = save_crop(img, gray_full, r["_bbox_cu"], args.pad,
                                     out / r["tier"].lower() / fn_cu, tighten=not args.no_tighten,
                                     prev_bbox=r.get("_prev_bbox"), next_bbox=r.get("_next_bbox"))
                    if q_cu:
                        r["_image_cu"] = f"{r['tier'].lower()}/{fn_cu}"
            r["_image"], r["_md5"] = img_rel or "", q["md5"] if q else ""
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
                "ink_pct": q["ink"] if q else "", "crop_w": q["w"] if q else "",
                "crop_h": q["h"] if q else "", "image_md5": q["md5"] if q else "",
                "seg_flag": q["seg"] if q else "",
                "s3_cosine": r.get("s3_cosine", ""),
                "bbox": json.dumps(r.get("bbox")),
                "nom_idx": r.get("nom_idx", ""), "syl_idx": r.get("syl_idx", ""),
                "syllable_ocr": r.get("syllable_ocr", ""),
                "syllable_raw": r.get("syllable_raw", ""),
                "p_register": r.get("p_register", ""),
                "band_touched": r.get("band_touched", 0),
                "n_ocr": r.get("n_ocr", ""), "n_qn": r.get("n_qn", ""),
                "n_det": r.get("n_det", ""), "count_source": r.get("count_source", ""),
                "box_source": r.get("box_source", ""),
                # A-7 (PASS 1c): tier_v3 + chẩn đoán + cờ QĐ-01 (dày 0/1)
                **{k: r.get(k, v) for k, v in FIELDS_1C.items()},
                # A-8 (N5i): mã chuẩn theo bảng dị thể người ký; = label khi chưa ký
                "label_canonical": r.get("label_canonical", r["label"]),
                # B-2: chỉ ghi khi --visual-emission (fields thêm cột ở cuối; tắt cờ -> bỏ qua)
                **({k: r.get(k, "") for k in FIELDS_B2} if vis_em is not None else {}),
            })

    # ---------- QĐ-01: đối chiếu hộp/md5 + qd01a_pending.csv (N5g, N7) ----------
    n_bbox_eq_cu = n_md5_eq_cu = n_locked_total = 0
    if args.two_pass and cells:
        for r in records:
            if not r.get("qd01_locked"):
                continue
            key = (r["book"], r["page"], int(r["column"]), int(r["nom_idx"]))
            cell = cells.get(key) or decisions.get(key) or {}
            n_locked_total += 1
            n_bbox_eq_cu += (r.get("bbox") == _parse_bbox(cell.get("bbox_cu")))
            n_md5_eq_cu += bool(r.get("_md5")) and r.get("_md5") == cell.get("image_md5_cu", "")
        print(f"  [QĐ-01] bbox == bbox_cu: {n_bbox_eq_cu}/{n_locked_total} (bắt buộc đủ) | "
              f"md5 == md5_cu: {n_md5_eq_cu}/{n_locked_total}"
              + ("" if args.no_crops else " (kỳ vọng đủ; lệch -> QĐ-01a lý do 'carve')"), flush=True)
        if n_bbox_eq_cu != n_locked_total:
            print(f"  [QĐ-01] 🔴 {n_locked_total - n_bbox_eq_cu} ô khoá có bbox ≠ bbox_cu", flush=True)
        pend_fields = ["book", "page", "column", "nom_idx", "syl_idx", "bbox_cu", "bbox_moi",
                       "syllable_moi", "ocr_char", "tier_v3", "tier_goc", "rule_goc", "ly_do",
                       "image", "image_cu"]
        with open(out / "qd01a_pending.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pend_fields)
            w.writeheader()
            for row in sorted(qd01_pending, key=lambda x: (x["book"], x["page"], int(x["column"]), int(x["nom_idx"]))):
                rr = row.pop("_rec")
                row["image"], row["image_cu"] = rr.get("_image", ""), rr.get("_image_cu", "")
                w.writerow(row)
        print(f"  [QĐ-01] pending -> {out / 'qd01a_pending.csv'} ({len(qd01_pending)} dòng, gồm "
              f"{sum(1 for x in qd01_pending if x['ly_do'] == 'qd01a_chua_quyet')} ô QĐ-01a chờ người)",
              flush=True)

    # B-5: hộp build gán cho ô khoá TRƯỚC khi ghi đè bbox_cu (đo luật hộp trên ô người đã xem)
    if args.two_pass and cells:
        with open(out / "qd01_lock_boxes.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["book", "page", "column", "nom_idx", "syl_idx", "bbox_cu",
                                              "bbox_truoc_lock", "iou", "box_source_truoc_lock",
                                              "count_source", "n_ocr", "n_qn", "n_det"])
            w.writeheader()
            for r in sorted((r for r in records if r.get("qd01_locked") and r.get("_bbox_truoc_lock") is not None),
                            key=lambda x: (x["book"], x["page"], int(x["column"]), int(x["nom_idx"]))):
                w.writerow({"book": r["book"], "page": r["page"], "column": r["column"], "nom_idx": r["nom_idx"],
                            "syl_idx": r.get("syl_idx", ""), "bbox_cu": json.dumps(r["bbox"]),
                            "bbox_truoc_lock": json.dumps(r["_bbox_truoc_lock"]),
                            "iou": f"{_iou(r['bbox'], r['_bbox_truoc_lock']):.3f}",
                            "box_source_truoc_lock": r.get("_box_source_truoc_lock", ""),
                            "count_source": r.get("count_source", ""), "n_ocr": r.get("n_ocr", ""),
                            "n_qn": r.get("n_qn", ""), "n_det": r.get("n_det", "")})
    # B-5 (--lock-scope cell*): liệt kê ô hàng xóm đã dời (đối soát, không vào labels.csv)
    if args.lock_scope != "col":
        sh = [r for r in records if r.get("box_source") == "shifted_by_lock"]
        with open(out / "lock_shift.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["book", "page", "column", "nom_idx", "syl_idx", "tier",
                                              "syllable", "bbox_truoc", "bbox_sau", "image"])
            w.writeheader()
            for r in sorted(sh, key=lambda x: (x["book"], x["page"], int(x["column"]), int(x["nom_idx"]))):
                w.writerow({"book": r["book"], "page": r["page"], "column": r["column"],
                            "nom_idx": r["nom_idx"], "syl_idx": r.get("syl_idx", ""), "tier": r["tier"],
                            "syllable": r.get("syllable", ""),
                            "bbox_truoc": json.dumps(r.get("_bbox_truoc_shift")),
                            "bbox_sau": json.dumps(r.get("bbox")), "image": r.get("_image", "")})
        print(f"  [B-5 khoá ô] shifted_by_lock -> {out / 'lock_shift.csv'} ({len(sh)} dòng)", flush=True)

    # ---------- D-1: sidecar crops_v2 (chỉ khi --crops-v2; labels.csv KHÔNG đổi) ----------
    if args.crops_v2:
        v2_fields = ["image", "book", "page", "column", "nom_idx", "syl_idx", "tier", "bbox", "bbox_v2",
                     "bbox_v2_win", "crop_mode", "guard_reason", "recenter_shift", "pitch", "image_md5",
                     "image_md5_v2", "ink_pct_v2", "crop_w_v2", "crop_h_v2", "crop_quality_flag_v2",
                     "stray_ink_v2", "border_ink_v2"]
        crops_v2_rows.sort(key=lambda r: (r["book"], r["page"], int(r["column"]), r["image"]))
        with open(out / "labels_crops_v2.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=v2_fields)
            w.writeheader()
            for r in crops_v2_rows:
                w.writerow(r)
        _n_f3 = sum(1 for r in crops_v2_rows if r["crop_mode"] == "f3g")
        _n_fb = sum(1 for r in crops_v2_rows if r["crop_mode"] == "fallback")
        _fb_eq = sum(1 for r in crops_v2_rows if r["crop_mode"] == "fallback" and r["image_md5_v2"] == r["image_md5"])
        print(f"  [D-1 crops_v2] {out / 'labels_crops_v2.csv'}: {len(crops_v2_rows):,} ô · f3g {_n_f3:,} · "
              f"fallback {_n_fb:,} (md5 == v1: {_fb_eq}/{_n_fb}) -> {out / 'crops_v2'}", flush=True)

    # ---------- write manifest + summary ----------
    fields = ["image", "book", "page", "column", "ocr_char", "syllable", "label",
              "unicode", "label_level", "tier", "rule", "s3_cosine", "ink_pct",
              "crop_w", "crop_h", "image_md5", "seg_flag", "bbox",
              "seg_backend",
              # N0b: hai cột mới ở CUỐI để giữ nguyên thứ tự cột cũ
              "nom_idx", "syl_idx",
              # N4e (A-5): âm nguyên văn / âm sau chuẩn hoá / posterior / cờ biên băng
              "syllable_ocr", "syllable_raw", "p_register", "band_touched",
              # A-6 (N4d): số đếm cột + nguồn hộp (n_*/count_source -> columns.csv ở S9)
              "n_ocr", "n_qn", "n_det", "count_source", "box_source",
              # A-7 (PASS 1c): tier_v3 + chẩn đoán + cờ QĐ-01 — mọi cờ DÀY 0/1
              # A-8 (N5i): mã chuẩn theo decisions.yaml mục di_the (= label khi chưa ký);
              # đặt TRƯỚC khối FIELDS_1C để FIELDS_1C vẫn ở cuối (selftest A-6/A-7 bám chuỗi)
              "label_canonical",
              *FIELDS_1C.keys()]
    if vis_em is not None:
        # B-2: cột sidecar ở CUỐI, chỉ khi bật cờ -> tắt cờ thì labels.csv y hệt trước. Qua
        # remediate/export (pandas/fieldnames) tới labels_final.csv rồi labels_trace.csv.
        fields += FIELDS_B2
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
    if vis_em is not None:
        # B-2: sidecar riêng cùng số dòng/thứ tự labels.csv (khoá image + book,page,column,nom_idx)
        _vf = ["image", "book", "page", "column", "nom_idx", "syl_idx", "syllable", "tier", "tier_v3", *FIELDS_B2]
        with open(out / "labels_trace_visual.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_vf, extrasaction="ignore")
            w.writeheader(); w.writerows(labels)
        print(f"  [B-2] {out / 'labels_trace_visual.csv'} ({len(labels):,} dòng; p_visual_syl có giá trị: "
              f"{sum(1 for r in labels if r.get('p_visual_syl', '') != ''):,})", flush=True)

    tiers = Counter(r["tier"] for r in records)
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
        # A-5 (flow N4a–N4c): đệ quy hai lượt
        "two_pass": bool(args.two_pass),
        "anchor_cap": anchor_cap,
        "n_anchor_pairs": n_anchor_pairs,
        # B-2: phát xạ ảnh trong DP (chỉ đo; 'unavailable' = bật cờ nhưng thiếu mô hình)
        "visual_emission": ({"enabled": True, "models_dir": str(vis_em.models_dir), **vis_params,
                             "n_cols": n_cols_vis, "n_boxes": vis_em.n_boxes, "n_cut_fail": vis_em.n_cut_fail,
                             "n_cols_changed_vs_text": n_cols_vis_changed,
                             "n_pairs_changed_vs_text": n_pairs_vis_changed,
                             "n_p_visual_syl": sum(1 for r in records if r.get("p_visual_syl", "") != "")}
                            if vis_em is not None else
                            {"enabled": False, "requested": bool(args.visual_emission)}),
        "pass1b": {"n_cols": n_cols_1b, "n_pair_keys": len(pair_pages),
                   "n_match_ops1": n_pairs_ops1, "n_match_ops2": n_pairs_ops2,
                   "n_match_changed": n_pairs_changed, "n_band_touched": n_band_touched,
                   "n_p_register_ge_0.8": sum(1 for r in records
                                              if r.get("p_register", "") != ""
                                              and r["p_register"] >= 0.8)},
        # A-6: luật hộp + phân bố nguồn đếm/nguồn hộp (theo Ô; theo CỘT đếm từ labels)
        "box_rule": args.box_rule,
        "detector_thr": (ap_mod.LEGACY_DETECTOR_THR if args.box_rule == "legacy"
                         else ap_mod.DETECTOR_THR),
        "detector_xmargin": (ap_mod.LEGACY_DETECTOR_XMARGIN if args.box_rule == "legacy"
                             else ap_mod.DETECTOR_XMARGIN),
        # tham số detector từng sách (books[].det_thr / det_xmargin ghi đè step2; chỉ syl_index)
        "detector_params_by_book": det_params_by_book,
        "n_locked_cols_qd01": sum(len(v) for v in locked_cols.values()),
        # B-5: phạm vi khoá + thống kê dời hàng xóm (chỉ khác rỗng khi --lock-scope cell)
        "lock_scope": args.lock_scope,
        "lock_shift": {k: int(v) for k, v in lock_shift_st.items()},
        "count_source": dict(Counter(r.get("count_source", "") for r in records)),
        "box_source": dict(Counter(r.get("box_source", "") for r in records)),
        "usable_char": tiers["GOLD"] + tiers["SILVER"],
        "usable_total": tiers["GOLD"] + tiers["SILVER"] + tiers["SYLLABLE"],
        # A-7 (PASS 1c): tier_v3 + L1 có cổng + khoá QĐ-01 + 'người' ngoài khoá
        "pass1c": {
            "enabled": bool(args.two_pass),
            "tier_v3": dict(cross_v3),
            "tier_v3_final": dict(Counter(r.get("tier_v3", "") for r in records)),
            "l1": {"doi": n_l1_moi + n_l1_nang, "giu_goc_thua": n_l1_giu, "hoa": n_l1_hoa},
            "qd01": {**{k: int(v) for k, v in qd01_st.items()},
                     "n_cells_file": len(cells), "n_decisions_file": len(decisions),
                     "n_pending_rows": len(qd01_pending),
                     "n_bbox_eq_cu": n_bbox_eq_cu, "n_md5_eq_cu": n_md5_eq_cu,
                     "n_locked_total": n_locked_total,
                     "dat_locked_pending": bool(
                         qd01_st["n_locked"] + qd01_st["n_pending"] + qd01_st["n_bo"]
                         + qd01_st["n_khac"] == qd01_st["n_cells_in_build"])},
            "nguoi_ngoai_khoa": {"n": n_nguoi_ngoai, "lop_nham_review": n_lop_nham},
            "flank_gold": dict(Counter(r.get("flank_gold", 0) for r in records)),
            # A-8: số ô áp theo mục da_ky + số ô label_canonical ≠ label
            "decisions": {"path": (str(args.decisions) if qdl is not None else "none"),
                          "corpus_readings_ap": dict(n_cr), "di_the_ap": dict(n_dt),
                          "lop_nham_review": n_lop_nham,
                          "n_label_canonical_ne_label": sum(
                              1 for r in records if r.get("label_canonical", r["label"]) != r["label"])},
        },
    }
    if layout_gate_stats:
        # chỉ xuất hiện khi có sách khai layout/n_columns -> summary.json STT không đổi
        summary["layout_gate"] = layout_gate_stats
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    # A-8: decisions_report.json — đếm áp/chờ ký từng mục (mục cho_ky CHỈ được báo cáo)
    if qdl is not None:
        json.dump(qdl.report(applied={"corpus_readings": dict(n_cr), "di_the": dict(n_dt),
                                      "lop_nham_review": n_lop_nham}),
                  open(out / "decisions_report.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f" DATASET -> {out}")
    print("=" * 64)
    print(f" pages {pages_done} | pairs {len(records)} | char classes {char_classes}")
    for t in ("GOLD", "SILVER", "SYLLABLE", "REVIEW"):
        print(f"   {t:9s}: {tiers.get(t, 0)}")
    if args.two_pass:
        print(f" tier_v3: " + " | ".join(f"{k} {summary['pass1c']['tier_v3_final'].get(k, 0)}"
                                        for k in ("CHAR_A", "CHAR_B", "SYL", "REVIEW")))
        print(f" QĐ-01: locked {qd01_st['n_locked']} | pending {qd01_st['n_pending']} | "
              f"bbox==cu {n_bbox_eq_cu}/{n_locked_total}")
    else:
        print(f" SYLLABLE promoted from REVIEW: {n_promoted}")
        print(f" SYLLABLE chuyển từ SILVER (L5): {n_tu_silver}")
    print(f" USABLE char-level (GOLD+SILVER): {summary['usable_char']}  | "
          f"+syllable: {summary['usable_total']}")
    print(f" manifest: {out}/labels.csv  ({len(fields)} cột)")


if __name__ == "__main__":
    main()
