"""Bước B4' — CỔNG THEO CƠ CHẾ cho sách thạch bản (lithograph), 100 % tự động, không người.

Vị trí trong chuỗi (FACTS cli / run_pipeline.sh bước 4→6):
    labels.csv → [remediation apply] → labels_remediated.csv → [confusion_fix] → labels_final.csv
               → [mechanism_gates] → labels_gated.csv → [export_final_dataset]

Vì sao (docs/PHUONG_AN_TU_DONG_2026-09-22.md §3–§4; BAO_CAO_TONG_THE §6-A): trên thạch bản, lỗi
VỊ TRÍ HỘP (detector CenterNet học STT đếm lệch ±1, hộp midpoint/split suy từ cột kề) không đo
tự động được và proxy văn bản (khớp dị bản) MÙ với nó; chọn cổng theo proxy là vô nghĩa
(mọi cổng chỉ dịch ≤ +1,2 điểm, trong CI). Nên cổng chọn theo LÝ DO CƠ CHẾ:

    (c) crop_quality_flag ∈ {blank, truncated}            → REVIEW   (ảnh hỏng; áp cho mọi tầng có ảnh)
    (d) --cross: GOLD bất đồng dị bản KHÔNG có tham chiếu   → REVIEW   (nghi kim nhầm chữ đồng âm gần hình)
        nào khớp, và cặp (nhãn, tham chiếu) gần hình
        (SinoNom_Similar, cột `similar` của cells.csv);
        bất đồng KHÔNG gần hình → giữ tầng, cờ di_ban_khac=1
    (b) luật cầu s1_inter_s2_similar / tier_v3 CHAR_B      → SYLLABLE (kim ∉ dict, chữ cầu chưa có bằng chứng
        và s1_inter_s2_direct_am_sua_dau                               độc lập; âm QN bị sửa dấu để khớp dict)
    (a) n_det ≠ n_qn HOẶC box_source ∈ {midpoint, split}   → GOLD_text_only (giữ label/unicode/syllable;
                                                                       KHÔNG export ảnh crop)
    (a') CHẾ ĐỘ PITCH (books[].box_decoder: pitch — pipeline/align_engine/char_detector/pitch_decode.py,
         2026-09-22): hộp đã được giải mã theo bước cột nên `n_det ≠ n_qn` KHÔNG còn nghĩa "hộp lệch"
         (n_det giữ nghĩa hộp THÔ ở det_thr để I5 không thành hằng đúng); cổng (a) đổi thành theo Ô:
           box_source ∈ {ink_cut, detector_low}  → GOLD_text_only  (ô detector không tự tin: ô ảo chiếu
                                                                     mực / hộp 0,05 ≤ điểm < det_thr)
           box_source ∈ {midpoint, split}        → GOLD_text_only  (cột rơi về legacy — vẫn như (a))
           n_det ≠ n_qn                           → chỉ ghi cờ n_det_mismatch=1 (mọi tầng), KHÔNG hạ
         Đo 27 trang/sách (docs/HUONG_DAN_HUAN_LUYEN_I5_2026-09-22.md §2): ink_cut+detector_low ≈ 1,5 %
         (LVT) / 4,1 % (KVK) ô, thay vì loại 35–45 % ô GOLD theo cột như (a).
         Nhận biết chế độ pitch (detect_pitch_mode, thứ tự): --box-decoder pitch|legacy (ghi đè) >
         config books[].box_decoder == pitch > summary.json detector_params_by_book[book].box_decoder
         (cạnh --in) > labels có box_source ∈ {ink_cut, detector_low} hoặc count_source ∈ {pitch, pitch_ocr}.
         Mặc định (legacy) KHÔNG đổi: không có cột n_det_mismatch, (a) theo cột như cũ.

Thứ tự ưu tiên khi một ô trúng nhiều cổng: (c) > (d) > (b) > (a); `gate_reason` ghi cổng quyết định,
báo cáo JSON ghi số trúng THÔ từng cổng. Ô QUARANTINE/REVIEW không đụng.

Bật/tắt: đọc `books[].mechanism_gates` (bool) trong config; vắng khoá → bật khi `layout == lithograph`,
tắt với sách STT (không khai layout). Khi TẮT: --out là BẢN SAO BYTE của --in (shutil.copyfile) — STT
byte-identical. `--enable on|off` ghi đè config (thí nghiệm).

Không sửa `label` ở (a)/(c)/(d) (giữ để truy vết, như confusion_fix); ở (b) `label`/`unicode` RỖNG theo
quy ước tầng SYLLABLE (README: "SYLLABLE có `label` rỗng"), chữ gốc còn ở `label_canonical`, luật gốc
còn trong `rule` (hậu tố `|gate:<reason>` như `|quarantine_dup` của remediate).

Chạy:
  .venv/bin/python -m pipeline.remediation.mechanism_gates --in dataset_out_<BOOK>/labels_final.csv \
      --out dataset_out_<BOOK>/labels_gated.csv --config config/pipeline_<BOOK>.yaml --book <BOOK> \
      [--cross measure_out/auto_precision/cross/<BOOK>/cells.csv] [--report dataset_out_<BOOK>/mechanism_gates_report.json]
      [--box-decoder auto|legacy|pitch] [--summary dataset_out_<BOOK>/summary.json]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]


def _s(v) -> str:
    return "" if v is None else str(v).strip()


TIER_TEXT_ONLY = "GOLD_text_only"
IMAGE_TIERS = ("GOLD", "SILVER", "SYLLABLE")          # tầng có ảnh crop được export
BAD_CROP = ("blank", "truncated")
BOX_NOT_DETECTOR = ("midpoint", "split")
BOX_LOW_CONF = ("ink_cut", "detector_low")        # (a') pitch_decode: ô detector không tự tin
COUNT_SOURCE_PITCH = ("pitch", "pitch_ocr", "pitch_rule")   # count_source do assign_boxes_pitch ghi
                                                  # ("pitch_rule" 2026-09-23: N lấy từ luật 6/8)
BOX_DECODER_MODES = ("auto", "legacy", "pitch")
COL_NDET_MISMATCH = "n_det_mismatch"              # cờ (a') thay cho hạ theo cột
RULE_BRIDGE = "s1_inter_s2_similar"
TIER_V3_BRIDGE = "CHAR_B"
RULE_AM_SUA_DAU = "s1_inter_s2_direct_am_sua_dau"
LAYOUT_LITHOGRAPH = "lithograph"

# tên cổng (gate_reason) — cố định để báo cáo/tài liệu tra được
G_CROP = "crop_bad"
G_CROSS = "cross_similar"
G_BRIDGE = "bridge_similar"
G_TONE = "am_sua_dau"
G_NDET = "n_det_ne_n_qn"
G_BOX = "box_not_detector"
G_BOX_LOW = "box_low_conf"                        # (a') chỉ trong chế độ pitch
G_QNCOUNT = "qn_count_unfixed"                    # (e) cột có SỐ ĐẾM ÂM QN hỏng

# (e) 2026-09-23 — cột "số đếm âm QN không sửa được" (cờ do align_production ghi qua
# book_layout.qn_count_unfixed_columns). Đo trên NHÃN NGƯỜI 2 bộ IHR
# (docs/RA_SOAT_CAN_CHINH_2026-09-23.md §2.2): ô GOLD trong cột `n_qn != 14` chỉ ĐÚNG
# 52,2 % (n 209) / 49,0 % (n 51) — gần như tung đồng xu — trong khi phần còn lại 98,4 % /
# 98,7 %; và 85,7 % / 75,0 % toàn bộ ô "đúng chữ, sai ô" nằm trong nhóm này. Vì SAI ở chính
# NHÃN VĂN BẢN (không phải chỉ ở hộp) nên hạ xuống REVIEW chứ KHÔNG phải GOLD_text_only —
# GOLD_text_only vẫn giao nộp nhãn, tức vẫn giao nộp ~50 % nhãn sai.
COL_QN_UNFIXED = "qn_count_unfixed"
QN_COUNT_MODES = ("review", "text_only", "off")


# --------------------------------------------------------------------------- config
def load_book_cfg(config_path: Path, book: str) -> dict | None:
    """Mục `books[]` có `name` == book (không phân biệt hoa/thường: cột `book` trong labels là
    chữ thường `kimvankieu1884`, config ghi `KimVanKieu1884`). Không thấy → None."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    for b in cfg.get("books") or []:
        if isinstance(b, dict) and str(b.get("name", "")).lower() == book.lower():
            return b
    return None


def gates_enabled(book_cfg: dict | None, enable: str = "auto") -> bool:
    """`mechanism_gates: true|false` thắng; vắng → layout == lithograph; sách không có trong
    config (None) → tắt. Sai kiểu → ValueError (không rơi ngầm)."""
    if enable == "on":
        return True
    if enable == "off":
        return False
    if not book_cfg:
        return False
    if "mechanism_gates" in book_cfg:
        v = book_cfg.get("mechanism_gates")     # .get: khoá tuỳ chọn (code_facts không coi là bắt buộc)
        if not isinstance(v, bool):
            raise ValueError(f"books[{book_cfg.get('name')}].mechanism_gates = {v!r}; cần true/false")
        return v
    return book_cfg.get("layout") == LAYOUT_LITHOGRAPH


def qn_count_mode(book_cfg: dict | None, override: str | None = None) -> str:
    """(e) Hạ cấp ô thuộc cột `qn_count_unfixed` thế nào: review | text_only | off.

    Thứ tự: --qn-count-gate ghi đè > books[].qn_count_gate > mặc định theo layout
    (lithograph -> 'review'; prose/khác -> 'off'). Vì sao mặc định khác nhau: hai bộ IHR
    có NHÃN NGƯỜI chứng minh nhóm này chỉ đúng ~50 % trên sách LỤC BÁT, còn Chrestomathie
    (prose, cờ theo `dp_ratio`) KHÔNG có chuẩn độc lập nào — nên ở đó chỉ GHI CỜ."""
    if override is not None:
        if override not in QN_COUNT_MODES:
            raise ValueError(f"--qn-count-gate = {override!r}; chỉ nhận {QN_COUNT_MODES}")
        return override
    if book_cfg and "qn_count_gate" in book_cfg:
        v = book_cfg.get("qn_count_gate")
        if v not in QN_COUNT_MODES:
            raise ValueError(f"books[{book_cfg.get('name')}].qn_count_gate = {v!r}; "
                             f"chỉ nhận {QN_COUNT_MODES}")
        return str(v)
    if book_cfg and book_cfg.get("layout") == LAYOUT_LITHOGRAPH:
        return "review"
    return "off"


def detect_pitch_mode(df: pd.DataFrame | None, book_cfg: dict | None, summary_path: Path | None = None,
                      book: str | None = None, box_decoder: str = "auto") -> tuple[bool, str]:
    """(a') Chế độ pitch? Trả (bool, nguồn quyết định). Thứ tự: --box-decoder ghi đè > config
    books[].box_decoder > summary.json (detector_params_by_book[book].box_decoder, build_dataset ghi)
    > chính labels (box_source ink_cut/detector_low hoặc count_source pitch/pitch_ocr — chỉ
    assign_boxes_pitch mới ghi các giá trị này). Không thấy gì → legacy."""
    if box_decoder not in BOX_DECODER_MODES:
        raise ValueError(f"--box-decoder = {box_decoder!r}; chỉ nhận {BOX_DECODER_MODES}")
    if box_decoder != "auto":
        return box_decoder == "pitch", "cli"
    if book_cfg and "box_decoder" in book_cfg:
        v = book_cfg.get("box_decoder")
        if v not in ("legacy", "pitch"):
            raise ValueError(f"books[{book_cfg.get('name')}].box_decoder = {v!r}; chỉ nhận legacy|pitch")
        return v == "pitch", "config"
    if summary_path and Path(summary_path).exists():
        try:
            sm = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            params = sm.get("detector_params_by_book") or {}
            for k, v in params.items():
                if book is None or str(k).lower() == str(book).lower():
                    if isinstance(v, dict) and "box_decoder" in v:
                        return v["box_decoder"] == "pitch", "summary"
        except (OSError, ValueError):
            pass
    if df is not None:
        if "box_source" in df.columns and df["box_source"].map(_s).isin(BOX_LOW_CONF).any():
            return True, "labels:box_source"
        if "count_source" in df.columns and df["count_source"].map(_s).isin(COUNT_SOURCE_PITCH).any():
            return True, "labels:count_source"
    return False, "legacy"


# --------------------------------------------------------------------------- cross (d)
def load_cross(path: Path) -> dict[str, dict]:
    """cells.csv (hoặc disagreements*.csv cùng schema) → {image: {agree, similar_dis, other_dis, refs}}.

    Mỗi ô có thể có nhiều dòng (nhiều tham chiếu). Gộp theo ảnh:
      agree       = có tham chiếu nào eq == 1 (chữ khớp hẳn)   → không hạ
      similar_dis = có tham chiếu (không PUA) eq == 0 và similar == 1
      other_dis   = có tham chiếu (không PUA) eq == 0 và similar == 0
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"image", "eq", "similar"}
    thieu = need - set(df.columns)
    if thieu:
        raise ValueError(f"{path}: thiếu cột {sorted(thieu)} (cần cells.csv của auto_precision.py bước cross)")
    out: dict[str, dict] = {}
    pua = df["ref_pua"] if "ref_pua" in df.columns else pd.Series([""] * len(df), index=df.index)
    for img, eq, sim, rp in zip(df["image"], df["eq"], df["similar"], pua):
        if not img:
            continue
        d = out.setdefault(img, {"agree": False, "similar_dis": False, "other_dis": False, "n": 0})
        d["n"] += 1
        if eq == "1":
            d["agree"] = True
        elif eq == "0" and rp != "1":
            if sim == "1":
                d["similar_dis"] = True
            else:
                d["other_dis"] = True
    return out


# --------------------------------------------------------------------------- áp cổng
def apply_gates(df: pd.DataFrame, cross: dict[str, dict] | None = None,
                pitch: bool = False, qn_count: str = "off") -> tuple[pd.DataFrame, dict]:
    """Hàm THUẦN (không mutate df). Trả (df mới, báo cáo dict). `pitch=True` = luật (a') thay (a):
    hạ theo Ô box_source ink_cut/detector_low (+ midpoint/split), n_det ≠ n_qn chỉ ghi cờ.
    `qn_count` = luật (e): 'review' | 'text_only' | 'off' (xem `qn_count_mode`)."""
    if qn_count not in QN_COUNT_MODES:
        raise ValueError(f"qn_count = {qn_count!r}; chỉ nhận {QN_COUNT_MODES}")
    out = df.copy()
    n = len(out)
    for c in ("gate_reason", "di_ban_khac"):
        if c not in out.columns:
            out[c] = ""
    if pitch and COL_NDET_MISMATCH not in out.columns:
        out[COL_NDET_MISMATCH] = ""          # chỉ chế độ pitch mới có cột này (legacy không đổi schema)
    # đọc cột an toàn (thế hệ cũ có thể thiếu cột → coi như rỗng, cổng đó không trúng)
    col = lambda name: out[name].map(_s) if name in out.columns else pd.Series([""] * n, index=out.index)
    tier = col("tier")
    rule = col("rule")
    tier_v3 = col("tier_v3")
    flag = col("crop_quality_flag")
    n_det, n_qn = col("n_det"), col("n_qn")
    box = col("box_source")
    image = col("image")

    is_gold = tier == "GOLD"
    is_img_tier = tier.isin(IMAGE_TIERS)

    # trúng THÔ từng cổng (đếm trên tầng liên quan, TRƯỚC khi áp ưu tiên)
    hit_crop = is_img_tier & flag.isin(BAD_CROP)
    hit_bridge = is_gold & ((rule == RULE_BRIDGE) | (tier_v3 == TIER_V3_BRIDGE))
    hit_tone = is_gold & (rule == RULE_AM_SUA_DAU)
    ndet_mismatch = (n_det != n_qn)                     # thô, mọi tầng
    hit_ndet = is_gold & ndet_mismatch & (not pitch)    # (a) chỉ legacy; pitch → cờ, không hạ
    hit_box = is_gold & box.isin(BOX_NOT_DETECTOR)
    hit_box_low = is_gold & box.isin(BOX_LOW_CONF) & bool(pitch)   # (a') chỉ pitch
    qn_unfixed = col(COL_QN_UNFIXED) == "1"                        # (e) cờ cấp CỘT, mọi tầng
    hit_qn = is_gold & qn_unfixed & (qn_count != "off")
    if cross is not None:
        cx = image.map(lambda i: cross.get(i))
        agree = cx.map(lambda d: bool(d and d["agree"]))
        sim_dis = cx.map(lambda d: bool(d and d["similar_dis"]))
        oth_dis = cx.map(lambda d: bool(d and d["other_dis"]))
        hit_cross = is_gold & ~agree & sim_dis
        hit_khac = is_gold & ~agree & ~sim_dis & oth_dis
        n_in_cross = int((is_gold & cx.notna()).sum())
    else:
        hit_cross = pd.Series([False] * n, index=out.index)
        hit_khac = pd.Series([False] * n, index=out.index)
        agree = hit_cross
        n_in_cross = 0

    raw = {
        G_CROP: {t: int((hit_crop & (tier == t)).sum()) for t in IMAGE_TIERS},
        G_CROSS: int(hit_cross.sum()),
        "cross_di_ban_khac": int(hit_khac.sum()),
        "cross_agree_GOLD": int((is_gold & agree).sum()) if cross is not None else None,
        "cross_GOLD_in_cells": n_in_cross if cross is not None else None,
        G_BRIDGE: int(hit_bridge.sum()),
        G_TONE: int(hit_tone.sum()),
        G_NDET: int((is_gold & ndet_mismatch).sum()),   # trúng thô (pitch: chỉ ghi cờ, không hạ)
        G_BOX: int(hit_box.sum()),
        G_QNCOUNT: int((is_gold & qn_unfixed).sum()),       # thô; chỉ hạ khi qn_count != off
        "qn_count_unfixed_rows": int(qn_unfixed.sum()),
        G_BOX_LOW: int((is_gold & box.isin(BOX_LOW_CONF)).sum()),   # thô; chỉ hạ khi pitch
        "box_source_GOLD": {k: int(v) for k, v in box[is_gold].value_counts().items()},
        "n_det_blank_GOLD": int((is_gold & (n_det == "")).sum()),
    }

    # ưu tiên (c) > (d) > (b) > (a): mặt nạ quyết định = trúng và chưa bị cổng mạnh hơn lấy
    taken = pd.Series([False] * n, index=out.index)

    def take(mask):
        nonlocal taken
        m = mask & ~taken
        taken = taken | m
        return m

    m_crop = take(hit_crop)
    m_cross = take(hit_cross)
    m_qn = take(hit_qn)            # (e) TRƯỚC (b)/(a): nhãn văn bản mới là thứ đáng ngờ
    m_bridge = take(hit_bridge)
    m_tone = take(hit_tone)
    m_box = take(hit_ndet | hit_box | hit_box_low)

    # (c) → REVIEW: giữ label (truy vết), label_level rỗng như confusion_fix
    out.loc[m_crop, "gate_reason"] = G_CROP + ":" + flag[m_crop]
    out.loc[m_crop, "rule"] = rule[m_crop] + "|gate:" + G_CROP
    out.loc[m_crop, "tier"] = "REVIEW"
    if "label_level" in out.columns:
        out.loc[m_crop, "label_level"] = ""
    # (d) → REVIEW
    out.loc[m_cross, "gate_reason"] = G_CROSS
    out.loc[m_cross, "rule"] = rule[m_cross] + "|gate:" + G_CROSS
    out.loc[m_cross, "tier"] = "REVIEW"
    if "label_level" in out.columns:
        out.loc[m_cross, "label_level"] = ""
    # (e) → REVIEW (mặc định thạch bản) hoặc GOLD_text_only (--qn-count-gate text_only)
    _qn_tier = "REVIEW" if qn_count == "review" else TIER_TEXT_ONLY
    out.loc[m_qn, "gate_reason"] = G_QNCOUNT
    out.loc[m_qn, "rule"] = rule[m_qn] + "|gate:" + G_QNCOUNT
    out.loc[m_qn, "tier"] = _qn_tier
    if qn_count == "review" and "label_level" in out.columns:
        out.loc[m_qn, "label_level"] = ""
    # cờ dị bản khác (không hạ): chỉ ô GOLD còn đứng (kể cả sắp thành GOLD_text_only)
    out.loc[hit_khac & ~m_crop & ~m_cross, "di_ban_khac"] = "1"
    # (b) → SYLLABLE: label/unicode rỗng theo quy ước tầng; chữ gốc còn ở label_canonical
    for m, g in ((m_bridge, G_BRIDGE), (m_tone, G_TONE)):
        out.loc[m, "gate_reason"] = g
        out.loc[m, "rule"] = rule[m] + "|gate:" + g
        out.loc[m, "tier"] = "SYLLABLE"
        if "label_canonical" in out.columns:
            lc = out.loc[m, "label_canonical"].map(_s)
            out.loc[m, "label_canonical"] = lc.where(lc != "", out.loc[m, "label"])
        for c in ("label", "unicode"):
            if c in out.columns:
                out.loc[m, c] = ""
        if "label_level" in out.columns:
            out.loc[m, "label_level"] = "syllable"
    # (a)/(a') → GOLD_text_only: giữ nguyên nhãn/luật, chỉ đổi tầng + lý do
    reason_a = pd.Series([""] * n, index=out.index)
    reason_a[hit_ndet] = G_NDET
    reason_a[hit_box] = reason_a[hit_box].where(reason_a[hit_box] == "", reason_a[hit_box] + "+") \
        + G_BOX + ":" + box[hit_box]
    reason_a[hit_box_low] = G_BOX_LOW + ":" + box[hit_box_low]
    out.loc[m_box, "gate_reason"] = reason_a[m_box]
    out.loc[m_box, "tier"] = TIER_TEXT_ONLY
    if pitch:
        # (a') n_det ≠ n_qn: cờ trên MỌI dòng (sự kiện cấp cột), không hạ tầng
        out.loc[ndet_mismatch, COL_NDET_MISMATCH] = "1"

    before = {k: int(v) for k, v in df["tier"].value_counts().items()}
    after = {k: int(v) for k, v in out["tier"].value_counts().items()}
    decided = Counter(out.loc[out["gate_reason"] != "", "gate_reason"].map(lambda r: r.split(":")[0]))
    g0 = before.get("GOLD", 0)
    g1 = after.get("GOLD", 0)
    report = {
        "gates_raw_hits": raw,
        "gates_decided": dict(decided),
        "decided_total": int((out["gate_reason"] != "").sum()),
        "tier_before": before,
        "tier_after": after,
        "gold_image": {"before": g0, "after": g1,
                       "coverage_pct": round(100 * g1 / g0, 1) if g0 else None},
        "gold_text_only": after.get(TIER_TEXT_ONLY, 0),
        "gold_any_text": g1 + after.get(TIER_TEXT_ONLY, 0),
        "images_to_export": sum(after.get(t, 0) for t in IMAGE_TIERS),
        "images_to_export_before": sum(before.get(t, 0) for t in IMAGE_TIERS),
        "di_ban_khac": int((out["di_ban_khac"] == "1").sum()),
        "n_rows": n,
        "pitch_mode": bool(pitch),
        "qn_count_gate": qn_count,
        "qn_count_decided": int(m_qn.sum()),
        COL_NDET_MISMATCH: (int((out[COL_NDET_MISMATCH] == "1").sum()) if pitch else None),
        "n_det_mismatch_GOLD_kept": (int((is_gold & ndet_mismatch & (out["tier"] == "GOLD")).sum())
                                     if pitch else None),
    }
    # proxy khớp dị bản của tầng GOLD-ảnh trước/sau (chỉ khi có cross): tính trên ô có eq ∈ {0,1}
    if cross is not None:
        def proxy(mask):
            imgs = [i for i in image[mask] if i in cross]
            k = sum(1 for i in imgs if cross[i]["agree"])
            d = sum(1 for i in imgs if cross[i]["agree"] or cross[i]["similar_dis"] or cross[i]["other_dis"])
            return {"n": d, "agree": k, "agree_pct": round(100 * k / d, 1) if d else None}
        report["proxy_agree_any_ref"] = {
            "GOLD_before": proxy(is_gold),
            # dùng mặt nạ TRÚNG THÔ (không phải mặt nạ ưu tiên): ô vừa trúng (d) vừa trúng (a)/(b)
            # phải bị loại ở đây, nếu không tập "abc_only" còn giữ chính các ô bất đồng mà (a)/(b) sẽ hạ
            "GOLD_image_after_abc_only": proxy(is_gold & ~hit_crop & ~hit_bridge & ~hit_tone & ~hit_ndet
                                               & ~hit_box & ~hit_box_low & ~hit_qn),
            "GOLD_image_after": proxy(out["tier"] == "GOLD"),
            "GOLD_text_only_after": proxy(out["tier"] == TIER_TEXT_ONLY),
            "note": "agree = có tham chiếu nào khớp hẳn; 'after' đã trừ (d) nên tự khẳng định — "
                    "đọc 'after_abc_only' để so công bằng với 'before'",
        }
    return out, report


# --------------------------------------------------------------------------- CLI
def run(in_csv: Path, out_csv: Path, config: Path | None, book: str | None, cross: Path | None,
        enable: str, report_path: Path | None, box_decoder: str = "auto",
        summary_path: Path | None = None, qn_count_gate: str | None = None) -> dict:
    book_cfg = load_book_cfg(config, book) if (config and book) else None
    enabled = gates_enabled(book_cfg, enable)
    if summary_path is None:
        summary_path = in_csv.parent / "summary.json"      # build_dataset ghi cạnh labels.csv
    rep: dict = {"enabled": enabled, "book": book, "config": str(config) if config else None,
                 "in": str(in_csv), "out": str(out_csv), "cross": str(cross) if cross else None,
                 "layout": (book_cfg or {}).get("layout"),
                 "mechanism_gates_key": (book_cfg or {}).get("mechanism_gates"),
                 "box_decoder_arg": box_decoder}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not enabled:
        shutil.copyfile(in_csv, out_csv)          # BYTE-IDENTICAL: STT không đổi
        rep["note"] = "cổng TẮT (không lithograph / mechanism_gates: false) — out là bản sao byte của in"
        print(f"[gates] TẮT cho {book or '?'} (layout={rep['layout']}, mechanism_gates={rep['mechanism_gates_key']}) "
              f"→ sao byte {in_csv.name} → {out_csv}")
    else:
        # dtype=str + keep_default_na=False: mọi ô đi qua NGUYÊN VĂN ("nan" là âm 難, "0.10" giữ số 0)
        df = pd.read_csv(in_csv, dtype=str, keep_default_na=False)
        cx = load_cross(cross) if cross else None
        pitch, pitch_src = detect_pitch_mode(df, book_cfg, summary_path, book, box_decoder)
        rep["pitch_mode_source"] = pitch_src
        qn_mode = qn_count_mode(book_cfg, qn_count_gate)
        rep["qn_count_gate_arg"] = qn_count_gate
        out, r = apply_gates(df, cx, pitch=pitch, qn_count=qn_mode)
        out.to_csv(out_csv, index=False)
        rep.update(r)
        g = r["gold_image"]
        mode_txt = ("pitch — luật (a') theo ô ink_cut/detector_low, n_det≠N chỉ ghi cờ" if pitch
                    else "legacy — luật (a) theo cột n_det≠N")
        print(f"[gates] {book}: chế độ hộp = {mode_txt} [nguồn: {pitch_src}]")
        print(f"[gates] {book}: GOLD ảnh {g['before']:,} → {g['after']:,} ({g['coverage_pct']} %); "
              f"GOLD_text_only {r['gold_text_only']:,}; quyết định theo cổng {r['gates_decided']}")
        print(f"[gates] (e) qn_count_unfixed = {qn_mode}: cờ {r['gates_raw_hits']['qn_count_unfixed_rows']:,} dòng "
              f"(GOLD trúng thô {r['gates_raw_hits'][G_QNCOUNT]:,}) -> hạ {r['qn_count_decided']:,} ô")
        if pitch:
            print(f"[gates] (a') n_det_mismatch cờ {r[COL_NDET_MISMATCH]:,} dòng (GOLD giữ ảnh dù n_det≠N: "
                  f"{r['n_det_mismatch_GOLD_kept']:,}); box_low_conf thô {r['gates_raw_hits'][G_BOX_LOW]:,}")
        print(f"[gates] tier trước: {r['tier_before']}")
        print(f"[gates] tier sau  : {r['tier_after']}")
        print(f"[gates] ảnh sẽ export {r['images_to_export_before']:,} → {r['images_to_export']:,}; "
              f"di_ban_khac {r['di_ban_khac']:,}")
        if "proxy_agree_any_ref" in r:
            p = r["proxy_agree_any_ref"]
            print(f"[gates] khớp dị bản GOLD-ảnh: trước {p['GOLD_before']['agree_pct']} % (n {p['GOLD_before']['n']}) "
                  f"→ sau (a)(b)(c) {p['GOLD_image_after_abc_only']['agree_pct']} % (n {p['GOLD_image_after_abc_only']['n']}) "
                  f"| GOLD_text_only {p['GOLD_text_only_after']['agree_pct']} % (n {p['GOLD_text_only_after']['n']})")
        print(f"[gates] -> {out_csv}")
    if report_path is None:
        report_path = out_csv.parent / "mechanism_gates_report.json"
    report_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gates] -> {report_path}")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.remediation.mechanism_gates",
                                 description="B4' cổng theo cơ chế (lithograph): labels_final.csv -> labels_gated.csv")
    ap.add_argument("--in", dest="in_csv", required=True, help="labels_final.csv (sau confusion_fix)")
    ap.add_argument("--out", required=True, help="labels_gated.csv (đầu vào export_final_dataset)")
    ap.add_argument("--config", default=None, help="config/pipeline_<BOOK>.yaml (đọc books[].layout / mechanism_gates)")
    ap.add_argument("--book", default=None, help="tên sách trong books[] (không phân biệt hoa/thường)")
    ap.add_argument("--cross", default=None,
                    help="measure_out/auto_precision/cross/<BOOK>/cells.csv (cổng (d) bất đồng dị bản gần hình)")
    ap.add_argument("--enable", choices=("auto", "on", "off"), default="auto",
                    help="auto (mặc định: theo config) | on | off (ghi đè, thí nghiệm)")
    ap.add_argument("--report", default=None, help="JSON báo cáo (mặc định <out dir>/mechanism_gates_report.json)")
    ap.add_argument("--box-decoder", choices=BOX_DECODER_MODES, default="auto",
                    help="auto (mặc định: config books[].box_decoder > summary.json > labels) | legacy | pitch — "
                         "pitch = luật (a') theo ô box_source ink_cut/detector_low, n_det≠N chỉ ghi cờ")
    ap.add_argument("--qn-count-gate", choices=QN_COUNT_MODES, default=None,
                    help="(e) cột có SỐ ĐẾM ÂM QN hỏng (cờ qn_count_unfixed): review (mặc định thạch bản) | "
                         "text_only | off (mặc định prose/STT). Vắng = theo books[].qn_count_gate > layout")
    ap.add_argument("--summary", default=None,
                    help="summary.json của build_dataset (mặc định cạnh --in) — đọc detector_params_by_book[book].box_decoder")
    a = ap.parse_args(argv)
    if a.enable == "auto" and not (a.config and a.book):
        print("[gates] --enable auto cần --config và --book (hoặc dùng --enable on|off)", file=sys.stderr)
        return 2
    run(Path(a.in_csv), Path(a.out), Path(a.config) if a.config else None, a.book,
        Path(a.cross) if a.cross else None, a.enable, Path(a.report) if a.report else None,
        box_decoder=a.box_decoder, summary_path=Path(a.summary) if a.summary else None,
        qn_count_gate=a.qn_count_gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
