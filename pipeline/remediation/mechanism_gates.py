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

TIER_TEXT_ONLY = "GOLD_text_only"
IMAGE_TIERS = ("GOLD", "SILVER", "SYLLABLE")          # tầng có ảnh crop được export
BAD_CROP = ("blank", "truncated")
BOX_NOT_DETECTOR = ("midpoint", "split")
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
def _s(v) -> str:
    return "" if v is None else str(v).strip()


def apply_gates(df: pd.DataFrame, cross: dict[str, dict] | None = None) -> tuple[pd.DataFrame, dict]:
    """Hàm THUẦN (không mutate df). Trả (df mới, báo cáo dict)."""
    out = df.copy()
    n = len(out)
    for c in ("gate_reason", "di_ban_khac"):
        if c not in out.columns:
            out[c] = ""
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
    hit_ndet = is_gold & (n_det != n_qn)
    hit_box = is_gold & box.isin(BOX_NOT_DETECTOR)
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
        G_NDET: int(hit_ndet.sum()),
        G_BOX: int(hit_box.sum()),
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
    m_bridge = take(hit_bridge)
    m_tone = take(hit_tone)
    m_box = take(hit_ndet | hit_box)

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
    # (a) → GOLD_text_only: giữ nguyên nhãn/luật, chỉ đổi tầng + lý do
    reason_a = pd.Series([""] * n, index=out.index)
    reason_a[hit_ndet] = G_NDET
    reason_a[hit_box] = reason_a[hit_box].where(reason_a[hit_box] == "", reason_a[hit_box] + "+") \
        + G_BOX + ":" + box[hit_box]
    out.loc[m_box, "gate_reason"] = reason_a[m_box]
    out.loc[m_box, "tier"] = TIER_TEXT_ONLY

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
            "GOLD_image_after_abc_only": proxy(is_gold & ~hit_crop & ~hit_bridge & ~hit_tone & ~hit_ndet & ~hit_box),
            "GOLD_image_after": proxy(out["tier"] == "GOLD"),
            "GOLD_text_only_after": proxy(out["tier"] == TIER_TEXT_ONLY),
            "note": "agree = có tham chiếu nào khớp hẳn; 'after' đã trừ (d) nên tự khẳng định — "
                    "đọc 'after_abc_only' để so công bằng với 'before'",
        }
    return out, report


# --------------------------------------------------------------------------- CLI
def run(in_csv: Path, out_csv: Path, config: Path | None, book: str | None, cross: Path | None,
        enable: str, report_path: Path | None) -> dict:
    book_cfg = load_book_cfg(config, book) if (config and book) else None
    enabled = gates_enabled(book_cfg, enable)
    rep: dict = {"enabled": enabled, "book": book, "config": str(config) if config else None,
                 "in": str(in_csv), "out": str(out_csv), "cross": str(cross) if cross else None,
                 "layout": (book_cfg or {}).get("layout"),
                 "mechanism_gates_key": (book_cfg or {}).get("mechanism_gates")}
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
        out, r = apply_gates(df, cx)
        out.to_csv(out_csv, index=False)
        rep.update(r)
        g = r["gold_image"]
        print(f"[gates] {book}: GOLD ảnh {g['before']:,} → {g['after']:,} ({g['coverage_pct']} %); "
              f"GOLD_text_only {r['gold_text_only']:,}; quyết định theo cổng {r['gates_decided']}")
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
    a = ap.parse_args(argv)
    if a.enable == "auto" and not (a.config and a.book):
        print("[gates] --enable auto cần --config và --book (hoặc dùng --enable on|off)", file=sys.stderr)
        return 2
    run(Path(a.in_csv), Path(a.out), Path(a.config) if a.config else None, a.book,
        Path(a.cross) if a.cross else None, a.enable, Path(a.report) if a.report else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
