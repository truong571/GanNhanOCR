"""Three-signal consensus labeling + GOLD/SILVER/REVIEW tiering.

Per Nôm crop we have up to three INDEPENDENT noisy signals:
  S1  ocr_char         — the SinoNom OCR's proposed character.
  S2  qn_to_nom[syl]   — the dictionary's valid Nôm readings R of the syllable.
  S3  visual rank      — DINOv2/FontDiffusion cosine of the crop vs the
                          candidate glyphs ({ocr_char} ∪ R), OPTIONAL.

A label is trustworthy when independent signals AGREE; we never emit a label
from one signal alone. Tiers:

  GOLD   — column anchored AND S1∩S2 agree (ocr_char is a dict reading of the
           syllable). DICTIONARY-ONLY, so it is S3-independent and is the safe
           floor any thesis can report. ~99% precise where it fires.
  SILVER — S1∩S2 do NOT agree, but S3 (vision) breaks the tie with enough
           confidence:
             (a) a dict reading r∈R wins the visual rank  -> vision CORRECTED
                 the OCR; label = r  (rule s2_inter_s3).
             (b) syllable not in dict (Ext-B variant) and ocr_char wins
                 visually -> label = ocr_char (rule s1_inter_s3).
           Needs S3; without it these go to REVIEW.
  REVIEW — diverged column, no ocr_char, or S3 below threshold. Never a label.

Signal-combination only; WHICH crop a syllable belongs to is decided upstream
by the banded anchored alignment (anchor_align.py).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from core.text.text_utils import is_plausible_qn_syllable

# --------------------------------------------------------------------------- #
# CHỐT CHẶN: ÂM ĐÃ CÓ PHÁN QUYẾT NGƯỜI
# --------------------------------------------------------------------------- #
# `pipeline/remediation/glyph_fix.py` chỉ gán được cho ô đang ở tier
# {REVIEW, SILVER, SILVER_uncalibrated} (hằng CO_THE_GAN của nó) — cố ý, để không
# lặng lẽ ghi đè một nhãn do luật khác sinh ra. Hệ quả: bất kỳ luật TỰ ĐỘNG nào
# nâng một ô âm "người" lên GOLD/SYLLABLE ở bước build đều ĐẨY ô đó RA NGOÀI tầm
# với của QĐ-01, và nhãn 𠊚 (U+2029A) mà người đã phán cho ô ấy biến mất im lặng.
# Nên: mọi luật tự động thêm mới (L1/L2/L3/L5) PHẢI tránh các âm trong tập này.
# Đây KHÔNG phải danh sách "âm khó"; nó là danh sách "âm đã có người quyết rồi".
#
# A-7 (flow N5c/N5h, 2026-09-16): trên đường --two-pass KHÔNG còn dùng làm CHỐT BAO
# TRÙM (không chặn tier theo âm): đo được chốt bao trùm làm mất 243 ô 'người' ngoài
# khoá và hạ 49 ô mốc đã ký (37 ô 𠊚 s1_inter_s2_similar + 12 khong_dung_cho). Phán
# quyết người nay là KHOÁ TỪNG Ô theo (book,page,column,nom_idx) trong build
# (build_dataset.apply_qd01_lock, config/qd01_cells.csv + qd01a_decisions.csv); ô
# 'người' ngoài khoá chỉ hạ khi nhãn v3 == 㝵 (apply_nguoi_ngoai_khoa) và mang cờ
# am_da_quyet_ngoai_khoa=1. Tập này chỉ còn: (a) L1 không đổi âm đi/đến âm trong tập;
# (b) đường --no-two-pass (tái lập bộ 64.525) giữ hành vi cũ.
AM_DA_QUYET = frozenset({"người"})


def chuan_am(syllable: str | None) -> str:
    """Chuẩn hoá âm GIỐNG HỆT glyph_fix.nrm (NFC + strip + lower).

    Nếu hai bên chuẩn hoá khác nhau thì "Người" hay dạng NFD lọt qua chốt chặn, và
    ô đó rơi ra ngoài tầm với của QĐ-01 mà không ai thấy.
    """
    return unicodedata.normalize("NFC", str(syllable or "").strip()).lower()


def am_da_quyet(syllable: str | None) -> bool:
    """True nếu âm này đã có phán quyết người -> luật tự động không được chạm."""
    return chuan_am(syllable) in AM_DA_QUYET

# Visual thresholds for the TRAINED Nôm embedder (visual_signal.NomEncoder).
# Measured on test split: same-char cosine ~0.80, different-char ~0.50 -> a
# correct match clears ~0.65 with a ~0.2+ margin. Tune on a held-out set.
TAU_SILVER = 0.62      # min visual score of the winning candidate
DELTA_SILVER = 0.06    # min (winner − runner-up) margin

# #4 head∩bank consensus gate: min ArcFace-head margin (top1−top2 logit) to accept a
# head-rescued SILVER. The boolean head_agree alone was too loose (80% GOLD-test
# precision); margin≥0.3 → 97.6% GOLD-test at ~29% coverage (validate_head_consensus.py).
# [ĐÃ BỊ BÁC — 97.6% là proxy circular tự sinh (GOLD-test tự sinh, huấn luyện vòng tròn);
#  đối chiếu audit NGƯỜI: bank_cos AUC bắt-lỗi = 0.566. Xem docs/BANG_SO_LIEU_CHINH_THUC.md]
# Still an OPTIMISTIC proxy — confirm with the human audit (eval_sample_head) + conformal.
HEAD_CONSENSUS_MARGIN = 0.3


@dataclass
class S3:
    """Optional visual signal: best candidate over {ocr_char} ∪ dict-readings."""
    top_char: str       # argmax candidate
    cosine: float       # winner score (calibrated P(match) when calibrated, else raw/remapped cosine)
    margin: float       # winner − runner-up (same scale as `cosine`)
    top_in_dict: bool   # is top_char a dict reading R of the syllable?
    p_match: float = 0.0   # Bước 2: calibrated P(match) of the winner (0 if uncalibrated)
    p_margin: float = 0.0  # calibrated winner − runner-up
    reject: bool = False   # open-set reject: winner below the calibrated operating point
    head_top: str = ""     # #4: argmax of the independent ArcFace head over the candidates
    head_agree: bool = False  # head_top == bank's top_char (two visual signals concur)
    head_margin: float = 0.0  # head top1−top2 logit gap = head confidence (gate, see HEAD_CONSENSUS_MARGIN)


@dataclass
class LabelDecision:
    label: str | None
    syllable: str
    tier: str           # GOLD | SILVER | REVIEW
    rule_id: str
    confirmed: bool      # dict-confirmed (S1∩S2)


def decide_label(ocr_char: str | None,
                 syllable: str,
                 column_count_matched: bool,
                 qn_to_nom: dict[str, list[str]],
                 similar_dict: dict[str, list[str]] | None = None,
                 s3: S3 | None = None,
                 anchored: bool = False) -> LabelDecision:
    """Combine S1/S2/(S3) into a label + tier for one aligned pair.

    `anchored` = the pair is inside a dict-confirmed run, so its local register
    is certain even if the whole column's counts diverged -> GOLD-eligible too.
    """
    syl = (syllable or "").lower()
    R = qn_to_nom.get(syl, [])
    gold_ok = column_count_matched or anchored

    # --- GOLD (direct): ocr_char IS a dict reading of the syllable. The OCR
    #     char (S1) and the QN syllable (S2) cross-confirm each other, so the
    #     char label is reliable ON ITS OWN — even in a diverged column (the
    #     banded DP only pairs a confirmed char here when it confirms). [#2]
    if ocr_char and ocr_char in R:
        return LabelDecision(ocr_char, syllable, "GOLD", "s1_inter_s2_direct", True)

    # --- GOLD (similar-bridge): ocr_char is NOT a reading of the syllable, but a
    #     visually-SIMILAR char IS. The OCR misread a lookalike; the TRUE label is
    #     that bridge char (NOT ocr_char). Emit the bridge, only when unique and
    #     the column is anchored/matched (this path is weaker than direct). [#1]
    if gold_ok and ocr_char and similar_dict:
        bridges = list(dict.fromkeys(s for s in similar_dict.get(ocr_char, [])
                                     if s in R and s != ocr_char))
        if len(bridges) == 1:
            return LabelDecision(bridges[0], syllable, "GOLD", "s1_inter_s2_similar", True)

        # --- L2 · CẦU NGƯỢC 1 BƯỚC -----------------------------------------
        # Dict/SinoNom_Similar.csv là top-20 KHÔNG đối xứng: có 33.396 chữ nguồn,
        # mỗi chữ 20 hàng xóm, nhưng "b nằm trong top-20 của a" KHÔNG kéo theo
        # "a nằm trong top-20 của b". Khi bộ OCR nhả ra một chữ HIẾM, danh sách
        # hàng xóm của nó thường không chứa chữ Nôm phổ thông đúng; chiều ngược
        # lại thì có. Nên: chỉ khi cầu XUÔI cho ĐÚNG 0 cầu mới soi chiều ngược.
        #   `for s in R` (|R| ~20) chứ không dựng chỉ mục ngược 33k chữ — rẻ hơn
        #   và không giữ trạng thái toàn cục nào.
        # KHÔNG bỏ chốt `s != ocr_char`: dù ở đây ocr_char ∉ R (nếu thuộc R thì đã
        # về GOLD trực tiếp ở khối trên), chốt này chặn luật tự khẳng định chính
        # chữ OCR đọc ra — đúng cái vòng lập luận đã giết lớp 㝵/"người".
        if not bridges and not am_da_quyet(syl):
            # CHỐT CHẶN chỉ gắn ở NHÁNH MỚI. Nhánh cầu XUÔI ở trên là luật đã có từ
            # trước bộ vá này; gắn chốt lên nó sẽ ĐỔI 326 ô âm "người" đang ở GOLD
            # (279 trong đó mang nhãn 㝵) — một thay đổi nằm ngoài phạm vi L1/L2/L3/L5
            # và làm lệch mọi số nền đang được báo cáo. Bộ vá này chỉ chịu trách nhiệm
            # cho phần nó thêm vào.
            rev = list(dict.fromkeys(
                s for s in R
                if s != ocr_char and ocr_char in (similar_dict.get(s) or ())))
            if len(rev) == 1:
                return LabelDecision(rev[0], syllable, "GOLD",
                                     "s1_inter_s2_similar_nguoc", True)
        # 0 cầu cả hai chiều -> không giống; >=2 -> nhập nhằng -> rơi xuống REVIEW

    # --- SILVER : vision breaks the tie (needs S3) ------------------------
    # The accept/reject gate now lives in visual_signal (calibrated P(match) at a
    # target precision, or the TAU/DELTA fallback when no calibration is present),
    # surfaced as s3.reject. An open-set reject -> the true char is likely not in
    # the candidate set -> REVIEW, never a forced-argmax label.
    if gold_ok and s3 is not None and not s3.reject:
        if s3.top_in_dict:
            # vision picked a valid dict reading of the syllable -> OCR corrected
            return LabelDecision(s3.top_char, syllable, "SILVER",
                                 "s2_inter_s3_corrected", False)
        if ocr_char and s3.top_char == ocr_char and not R and is_plausible_qn_syllable(syl):
            # syllable absent from dict (Ext-B variant); vision backs the OCR char.
            # GUARD: only a PLAUSIBLE syllable may confirm here. A garbage token
            # ('0'/'2017'/'ch'/'giêgiung') also has empty R, but it is not a real
            # QN reading, so it must not earn SILVER off S1∩S3 alone — it falls
            # through to REVIEW. (Legit out-of-dict readings stay: 'atina', 'giu'…)
            return LabelDecision(ocr_char, syllable, "SILVER",
                                 "s1_inter_s3_out_of_dict", False)

    # --- SILVER (head∩bank consensus) [#4] --------------------------------
    # Even when the bank-only operating point REJECTED, the independent ArcFace
    # head (a 1591-way classifier, not the prototype cosine) may AGREE with the
    # bank's top on a valid dict reading. Two independent visual signals concurring
    # on a dict reading is a strong, measured rescue (GOLD-test precision ~95.9%,
    # group1_rescue.py) — accept it rather than discard to REVIEW.
    # [ĐÃ BỊ BÁC — ~95.9% là proxy circular tự sinh; đối chiếu audit NGƯỜI:
    #  bank_cos AUC bắt-lỗi = 0.566. Xem docs/BANG_SO_LIEU_CHINH_THUC.md]
    if (gold_ok and s3 is not None and getattr(s3, "head_agree", False)
            and getattr(s3, "head_margin", 0.0) >= HEAD_CONSENSUS_MARGIN
            and s3.top_in_dict and s3.top_char in R and is_plausible_qn_syllable(syl)):
        # GUARD (3rd anchor path): an implausible syllable ('n'/'c' — vowel-less
        # stubs) must not confirm a label even when head∩bank agree on a dict
        # reading. Completes anchor-purity across ALL SILVER routes.
        return LabelDecision(s3.top_char, syllable, "SILVER", "s3_head_bank_consensus", False)

    # --- REVIEW -----------------------------------------------------------
    reason = "diverged_column" if not gold_ok else (
        "below_visual_threshold" if s3 is not None else "unconfirmed_no_s3")
    return LabelDecision(None, syllable, "REVIEW", reason, False)
