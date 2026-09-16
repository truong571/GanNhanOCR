"""Thi hành QUYẾT ĐỊNH KHỐI GLYPH của người — gán một chữ cho cả một lớp âm.

VÌ SAO CÓ MÔ-ĐUN NÀY
--------------------
Có những chữ mà bộ OCR Nôm mù hẳn. Rõ nhất là "người": âm phổ biến nhất cả ba cuốn
(2.281 ô) nhưng 94% bị giữ lại, trong khi tỷ lệ rớt chung là 31%. Từ điển BIẾT đáp án
(𠊚 nằm trong 20 ứng viên của "người", và 37/49 ô GOLD đều dùng nó) — chỉ là bộ OCR
không đọc nổi glyph, nên luật S1∩S2 không bao giờ khớp.

Không phép đo tự động nào cứu được lớp này: đã đo và bác Unihan kVietnamese (0 ô), khôi
phục dấu Quốc ngữ (0 ô), cầu tự dạng hai chiều (T7, giữ một chiều), mô hình thị giác
ngoài (khảo sát 16 tác nhân, 0 ứng viên sống sót). Thứ còn lại là mắt người trên ảnh.

Mô-đun này là đường để một phán quyết như thế đi vào bộ nhãn — và chỉ đi vào được khi
có KHAI XUẤT XỨ.

🔴 GIAO ƯỚC XUẤT XỨ
-------------------
Mỗi quyết định phải trỏ `xuat_xu` tới một tệp có thật khai AI quyết, NHÌN GÌ, KHI NÀO.
Thiếu tệp -> TỪ CHỐI chạy, không cảnh báo rồi chạy tiếp. Dự án này đã một lần nhầm phán
quyết MÁY thành phán quyết NGƯỜI và phải huỷ sạch precision 97,98%, Fisher p=5,4e-8,
κ=0,13 dựng trên đó (docs/KE_HOACH_TONG_THE_2026-08-22.md §0). Cái giá của việc bỏ qua
một dòng kiểm tra ở đây là cả chương số liệu của luận văn.

`chi_khi_co_anh: true` (mặc định) chỉ gán cho ô CÓ ảnh crop trên đĩa — người không nhìn
được ô không có ảnh, nên không có phán quyết nào cho nó. Đó không phải chi tiết vụn: khối
"người" có 2.144 ô nhưng chỉ 2.014 ô có crop.

HAI CHẾ ĐỘ (`--mode`)
---------------------
`am`   (mặc định, hành vi cũ): gán theo ÂM — mask `syllable == am` + tier ∈ CO_THE_GAN + có
       ảnh. Đúng cho thế hệ 64.525 nhưng KHÔNG bền: đổi ma trận/hộp (A-4…A-6) là ô trôi sang
       âm khác hoặc thành khe mà không ai hay.
`kiem` (A-2, flow N12): KHÔNG GÁN GÌ. Khoá QĐ-01 đã được build áp trong PASS 1c (N5g) theo
       `config/qd01_cells.csv` (khoá bền `book,page,column,nom_idx`) + `qd01a_decisions.csv`
       (phán quyết người cho ô trôi). Chế độ này chỉ ĐỐI CHIẾU labels với hai tệp đó và đếm
       n_khop / n_pending / n_mat / n_bbox_doi / n_md5_doi, ghi report; exit 1 nếu
       n_khop + n_pending != số ô khoá. Ô trôi thì người điền `qd01a_decisions.csv` rồi
       REBUILD — không sửa tay CSV, không gán ở đây.

    python -m pipeline.remediation.glyph_fix --in dataset_out/labels_final.csv \
        --out dataset_out/labels_final.csv --config config/quyet_dinh_glyph.yaml --apply
    python -m pipeline.remediation.glyph_fix --mode kiem --in dataset_out_v3/labels_final.csv \
        --cells config/qd01_cells.csv --decisions config/qd01a_decisions.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
NA = dict(keep_default_na=False, na_values=[""])
DTYPE = {"image_md5": str, "label_in_train": str, "crop_w": str, "crop_h": str,
         # cờ/chỉ số v3 (A-7/A-5/A-6): có thể rỗng ở thế hệ cũ -> pandas ép float "1.0"
         "nom_idx": str, "syl_idx": str, "band_touched": str, "n_ocr": str, "n_qn": str,
         "n_det": str, "l1_support": str, "l1_tie": str, "flank_gold": str,
         "qd01_locked": str, "qd01_excluded": str, "am_da_quyet_ngoai_khoa": str,
         "norm_fixed": str}

# Chỉ ô đang BỊ GIỮ LẠI mới được gán. Không bao giờ ghi đè nhãn đã ở GOLD/SYLLABLE —
# ghi đè là lặng lẽ đổi một nhãn đã qua luật khác, và ta sẽ không biết đã đổi cái gì.
CO_THE_GAN = ("REVIEW", "SILVER", "SILVER_uncalibrated")

# Chế độ kiem (A-2): khoá bền + tiền tố rule mà build (N5g) gắn cho ô QĐ-01
QD01 = "quyet_dinh_nguoi"
QD01_PENDING = QD01 + ":pending"
NGUOI = "\U0002029A"                       # 𠊚 U+2029A
KEY_NOM = ["book", "page", "column", "nom_idx"]
KEY_BBOX = ["book", "page", "column", "bbox"]
QUYET_HOP_LE = ("giu_2029A", "bo")         # hoặc "khac:<chữ>" (N5g)


def nrm(s) -> str:
    return unicodedata.normalize("NFC", str(s or "").strip()).lower()


def kiem_xuat_xu(qd: dict) -> Path:
    """Trả đường dẫn tệp khai xuất xứ, hoặc DỪNG HẲN."""
    xx = (qd.get("xuat_xu") or "").strip()
    if not xx:
        raise SystemExit(
            f"[quyết định glyph] 🔴 TỪ CHỐI: quyết định cho âm {qd.get('am')!r} không khai "
            f"`xuat_xu`.\n    Một phán quyết không truy được người chấm thì không có giá trị "
            f"hơn phán quyết máy.")
    p = REPO / xx
    if not p.exists():
        raise SystemExit(
            f"[quyết định glyph] 🔴 TỪ CHỐI: `xuat_xu: {xx}` không tồn tại trên đĩa.\n"
            f"    Tạo tệp đó và khai: ai quyết, nhìn gì, ngày nào.")
    return p


def ap_dung(df: pd.DataFrame, qds: list[dict], src_root: Path) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    for c in ("rule_goc", "tier_goc"):
        if c not in out.columns:
            out[c] = ""
    log: list[dict] = []
    for qd in qds:
        if qd.get("hanh_dong") != "gan_nhan":
            continue
        kiem_xuat_xu(qd)
        am, chu = nrm(qd["am"]), str(qd["chu"])
        if qd.get("unicode") and f"U+{ord(chu):04X}" != str(qd["unicode"]).upper():
            raise SystemExit(
                f"[quyết định glyph] 🔴 TỪ CHỐI: `chu: {chu}` là U+{ord(chu):04X} nhưng cấu "
                f"hình ghi `unicode: {qd['unicode']}`. Hai chữ 'người' 𠊚 U+2029A và 𠊛 "
                f"U+2029B chỉ khác nhau một mã điểm — lệch ở đây là gán sai cả khối.")
        mask = (out["syllable"].map(nrm) == am) & (out["tier"].isin(CO_THE_GAN))
        n_khop = int(mask.sum())
        n_khong_anh = 0
        if qd.get("chi_khi_co_anh", True):
            co_anh = out["image"].fillna("").map(
                lambda p: bool(str(p).strip()) and (src_root / str(p)).exists())
            n_khong_anh = int((mask & ~co_anh).sum())
            mask = mask & co_anh
        n = int(mask.sum())
        by_book = out.loc[mask, "book"].value_counts().to_dict() if n else {}
        truoc = out.loc[mask, "ocr_char"].value_counts().to_dict() if n else {}
        # giữ lại xuất xứ cũ TRƯỚC khi ghi đè, để bảng precision quy lỗi đúng luật đã sinh ra ô
        out.loc[mask & (out["rule_goc"] == ""), "rule_goc"] = out.loc[
            mask & (out["rule_goc"] == ""), "rule"]
        out.loc[mask & (out["tier_goc"] == ""), "tier_goc"] = out.loc[
            mask & (out["tier_goc"] == ""), "tier"]
        out.loc[mask, "label"] = chu
        out.loc[mask, "unicode"] = f"U+{ord(chu):04X}"
        out.loc[mask, "label_level"] = "char"
        out.loc[mask, "tier"] = qd.get("to_tier", "GOLD")
        out.loc[mask, "rule"] = "quyet_dinh_nguoi:" + str(qd.get("ly_do", "gan_nhan"))
        log.append({"am": am, "chu": chu, "unicode": f"U+{ord(chu):04X}",
                    "to_tier": qd.get("to_tier", "GOLD"), "khop": n_khop, "gan": n,
                    "bo_vi_khong_anh": n_khong_anh, "by_book": by_book,
                    "ocr_truoc_do": dict(sorted(truoc.items(), key=lambda kv: -kv[1])[:8]),
                    "xuat_xu": qd.get("xuat_xu")})
    return out, log


# ------------------------------------------------------------- chế độ kiem ----
def _khoa(df: pd.DataFrame, key: list[str]) -> pd.Series:
    """Khoá ghép dạng chuỗi; column/nom_idx ép về int để '01' và '1' không lệch nhau."""
    parts = []
    for c in key:
        s = df[c].fillna("").astype(str).str.strip()
        if c in ("column", "nom_idx"):
            s = s.map(lambda v: str(int(float(v))) if v else "")
        parts.append(s)
    return parts[0].str.cat(parts[1:], sep="|")


def _doc_decisions(p: Path | None) -> pd.DataFrame:
    """Đọc qd01a_decisions.csv; dòng có `quyet` phải hợp lệ VÀ có xuất xứ thật."""
    cot = ["book", "page", "column", "nom_idx", "bbox_cu", "syllable", "ocr_char",
           "label_hien_tai", "ly_do", "quyet", "nguoi_ky", "ngay", "xuat_xu"]
    if p is None or not p.exists():
        print(f"[kiem] không có tệp phán quyết QĐ-01a ({p}) — coi như 0 phán quyết")
        return pd.DataFrame(columns=cot)
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    for c in cot:
        if c not in d.columns:
            d[c] = ""
    for r in d[d["quyet"].str.strip() != ""].itertuples():
        q = r.quyet.strip()
        if q not in QUYET_HOP_LE and not q.startswith("khac:"):
            raise SystemExit(f"[kiem] 🔴 TỪ CHỐI: `quyet: {q}` ở ô {r.book}/{r.page}/c{r.column}/"
                             f"nom{r.nom_idx} không thuộc {QUYET_HOP_LE} hoặc 'khac:<chữ>'.")
        kiem_xuat_xu({"xuat_xu": r.xuat_xu, "am": f"{r.book}/{r.page}/c{r.column}/nom{r.nom_idx}"})
    return d


def kiem_khoa(df: pd.DataFrame, cells: pd.DataFrame, decisions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Đối chiếu labels với khoá QĐ-01 (không gán). Trả (report, bảng từng ô).

    Khoá chính (book,page,column,nom_idx); rơi về (book,page,column,bbox)↔bbox_cu khi
    labels chưa có nom_idx (thế hệ trước N0b). Ô đạt = khớp (label 𠊚, rule QĐ-01) hoặc
    pending (rule quyet_dinh_nguoi:pending, chờ người ở QĐ-01a).
    """
    for c in ("nom_idx", "image_md5", "bbox"):
        if c not in df.columns:
            df = df.assign(**{c: ""})
    dung_nom = bool((df["nom_idx"].fillna("").astype(str).str.strip() != "").any())
    key_lab = KEY_NOM if dung_nom else KEY_BBOX
    key_cell = KEY_NOM if dung_nom else ["book", "page", "column", "bbox_cu"]
    lab = df.assign(_k=_khoa(df, key_lab))
    n_dup = int(lab["_k"].duplicated().sum())
    lab = lab.drop_duplicates("_k", keep="first")
    lab = lab[["_k", "label", "rule", "tier", "syllable", "bbox", "image_md5"]].rename(
        columns={"label": "label_moi", "rule": "rule_moi", "tier": "tier_moi",
                 "syllable": "syllable_moi", "bbox": "bbox_moi", "image_md5": "md5_moi"})
    c = cells.assign(_k=_khoa(cells, key_cell))
    j = c.merge(lab, on="_k", how="left", indicator=True)
    thay = j["_merge"] == "both"
    rule = j["rule_moi"].fillna("")
    la_qd = rule.str.startswith(QD01)
    dk = decisions.assign(_k=_khoa(decisions, key_cell)) if len(decisions) \
        else decisions.assign(_k=pd.Series(dtype=str))
    bo_set = set(dk.loc[dk["quyet"].str.strip() == "bo", "_k"]) if len(dk) else set()
    khac_map = dict(dk.loc[dk["quyet"].str.startswith("khac:"), ["_k", "quyet"]].values) if len(dk) else {}
    da_bo = thay & j["_k"].isin(bo_set)
    da_khac = thay & j["_k"].isin(khac_map)

    pending = thay & (rule == QD01_PENDING)
    khop = thay & la_qd & ~pending & (j["label_moi"].fillna("") == NGUOI)
    lech = thay & ~khop & ~pending & ~da_bo & ~da_khac
    j["trang_thai"] = "khop"
    j.loc[pending, "trang_thai"] = "pending"
    j.loc[da_bo, "trang_thai"] = "bo"
    j.loc[da_khac, "trang_thai"] = "khac"
    j.loc[lech, "trang_thai"] = "lech"
    j.loc[~thay, "trang_thai"] = "mat"
    bbox_doi = thay & (j["bbox_moi"].fillna("") != j["bbox_cu"].fillna(""))
    md5_doi = thay & (j["md5_moi"].fillna("") != j["image_md5_cu"].fillna(""))
    syl_doi = thay & (j["syllable_moi"].fillna("").map(nrm) != "người")

    # QĐ-01a: phán quyết đã điền mà ô vẫn pending -> chưa rebuild (cùng loại khoá với cells)
    co_quyet = set(dk.loc[dk["quyet"].str.strip() != "", "_k"]) if len(dk) else set()
    pending_co_quyet = j.loc[pending & j["_k"].isin(co_quyet), "_k"].tolist()
    # A-3: GOLD âm 'người' mang mã ≠ 𠊚 ngoài khoá phải có dòng trong decisions
    am_nguoi = df["syllable"].fillna("").map(nrm) == "người"
    g = df[(df["tier"] == "GOLD") & am_nguoi & (df["label"].fillna("") != NGUOI)
           & ~df["rule"].fillna("").str.startswith(QD01)]
    g_k = _khoa(g, key_lab) if len(g) else pd.Series(dtype=str)
    khoa_dk = set(dk["_k"]) if len(dk) else set()
    gold_chua_quyet = g[~g_k.isin(khoa_dk)] if len(g) else g
    cot_bbox = key_cell + [c for c in ("bbox_cu", "bbox_moi") if c not in key_cell]

    rep = {
        "mode": "kiem", "khoa": "nom_idx" if dung_nom else "bbox", "n_cells": int(len(j)),
        "n_khop": int(khop.sum()), "n_pending": int(pending.sum()),
        "n_bo": int(da_bo.sum()), "n_khac": int(da_khac.sum()),
        "n_mat": int((~thay).sum()),
        "n_lech": int(lech.sum()), "n_bbox_doi": int(bbox_doi.sum()), "n_md5_doi": int(md5_doi.sum()),
        "n_syllable_doi": int(syl_doi.sum()), "n_labels_trung_khoa": n_dup,
        "n_decisions": int(len(dk)), "n_decisions_co_quyet": int(len(co_quyet)),
        "n_decisions_cho_nguoi": int(len(dk) - len(co_quyet)),
        "n_pending_co_quyet_chua_rebuild": len(pending_co_quyet),
        "n_gold_nguoi_chua_quyet": int(len(gold_chua_quyet)),
        "dat": bool(int(khop.sum()) + int(pending.sum()) + int(da_bo.sum()) + int(da_khac.sum()) == len(j)),
        "o_mat": j.loc[~thay, key_cell].to_dict("records"),
        "o_lech": j.loc[lech, key_cell + ["label", "label_moi", "rule_moi", "tier_moi"]].to_dict("records"),
        "o_bbox_doi": j.loc[bbox_doi, cot_bbox].to_dict("records"),
        "o_md5_doi": j.loc[md5_doi, key_cell + ["image_md5_cu", "md5_moi"]].to_dict("records"),
        "gold_nguoi_chua_quyet": gold_chua_quyet[
            [c for c in ["book", "page", "column", "nom_idx", "bbox", "label", "rule"] if c in gold_chua_quyet.columns]
        ].to_dict("records"),
        "pending_co_quyet_chua_rebuild": pending_co_quyet,
    }
    return rep, j


def run_kiem(in_csv: Path, cells_csv: Path, decisions_csv: Path | None, report: Path) -> int:
    if not cells_csv.exists():
        raise SystemExit(f"[kiem] 🔴 TỪ CHỐI: không có {cells_csv} — sinh bằng "
                         f"`python -m pipeline.tools.sinh_qd01_cells` (N0d) trước.")
    df = pd.read_csv(in_csv, dtype=DTYPE, **NA)
    cells = pd.read_csv(cells_csv, dtype=str, keep_default_na=False)
    dec = _doc_decisions(decisions_csv)
    rep, j = kiem_khoa(df, cells, dec)
    print("=" * 60)
    print(f" KIỂM KHOÁ QĐ-01 (không gán) · khoá {rep['khoa']} · {rep['n_cells']} ô")
    print(f"   khớp {rep['n_khop']} · pending {rep['n_pending']} · bỏ {rep['n_bo']} · khác {rep['n_khac']} · mất {rep['n_mat']} · lệch {rep['n_lech']}"
          f" · bbox đổi {rep['n_bbox_doi']} · md5 đổi {rep['n_md5_doi']} · âm ≠ người {rep['n_syllable_doi']}")
    print(f"   QĐ-01a: {rep['n_decisions']} dòng ({rep['n_decisions_co_quyet']} đã quyết, "
          f"{rep['n_decisions_cho_nguoi']} chờ người) · pending đã có quyết nhưng chưa rebuild "
          f"{rep['n_pending_co_quyet_chua_rebuild']} · GOLD 'người' ≠ 𠊚 chưa có phán quyết "
          f"{rep['n_gold_nguoi_chua_quyet']}")
    if rep["n_labels_trung_khoa"]:
        print(f"   ⚠ labels có {rep['n_labels_trung_khoa']} dòng trùng khoá — lấy dòng đầu")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f" -> {report}")
    if not rep["dat"]:
        print(f" 🔴 khớp + pending + bỏ + khác = {rep['n_khop'] + rep['n_pending'] + rep['n_bo'] + rep['n_khac']} != {rep['n_cells']}")
        for k in ("o_mat", "o_lech"):
            for r in rep[k]:
                print(f"    [{k[2:]}] {r}")
        return 1
    print(f" ✓ khớp ({rep['n_khop']}) + bỏ ({rep['n_bo']}) + khác ({rep['n_khac']}) + pending ({rep['n_pending']}) == {rep['n_cells']}")
    return 0


def run(in_csv: Path, out_csv: Path, cfg: Path, src_root: Path, apply: bool) -> dict:
    df = pd.read_csv(in_csv, dtype=DTYPE, **NA)
    qds = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("quyet_dinh", []) \
        if cfg.exists() else []
    if not qds:
        print(f"[quyết định glyph] không có quyết định nào trong {cfg.name} — bỏ qua.")
        return {"quyet_dinh": []}
    truoc = df["tier"].value_counts().to_dict()
    new, log = ap_dung(df, qds, src_root)
    sau = new["tier"].value_counts().to_dict()
    print("=" * 60)
    print(f" QUYẾT ĐỊNH KHỐI GLYPH: {len(log)} quyết định")
    for x in log:
        print(f"   âm {x['am']!r} -> {x['chu']} ({x['unicode']}) · gán {x['gan']:,} ô "
              f"-> {x['to_tier']}  {x['by_book']}")
        if x["bo_vi_khong_anh"]:
            print(f"     bỏ {x['bo_vi_khong_anh']} ô KHÔNG có ảnh crop — người không nhìn "
                  f"được thì không có phán quyết")
        print(f"     OCR trước đó đọc nhầm ra: {x['ocr_truoc_do']}")
        print(f"     xuất xứ: {x['xuat_xu']}")
    print(f" tier trước: {truoc}")
    print(f" tier sau  : {sau}")
    rep = {"quyet_dinh": log, "tier_truoc": truoc, "tier_sau": sau}
    if apply:
        new.to_csv(out_csv, index=False)
        print(f" -> {out_csv}")
        rp = out_csv.parent / "glyph_fix_report.json"
        rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f" -> {rp}")
    else:
        print(" (xem thử — thêm --apply để ghi)")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_csv", default="dataset_out/labels_final.csv")
    ap.add_argument("--out", default="dataset_out/labels_final.csv")
    ap.add_argument("--config", default="config/quyet_dinh_glyph.yaml")
    ap.add_argument("--src-root", default="dataset_out")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--mode", choices=("am", "kiem"), default="am",
                    help="am = gán theo âm (cũ); kiem = chỉ đối chiếu khoá QĐ-01, không gán")
    ap.add_argument("--cells", default="config/qd01_cells.csv", help="khoá bền QĐ-01 (N0d)")
    ap.add_argument("--decisions", default="config/qd01a_decisions.csv",
                    help="phán quyết người cho ô trôi / 12 ô A-3")
    ap.add_argument("--report", default=None,
                    help="mode kiem: JSON report (mặc định <thư mục --in>/glyph_fix_kiem_report.json)")
    a = ap.parse_args(argv)
    if a.mode == "kiem":
        # A-12 (N12): chế độ kiem vẫn nhận --config để KIỂM XUẤT XỨ của phán quyết âm
        # (quyet_dinh_glyph.yaml) — không gán gì, nhưng một phán quyết mất xuất xứ thì
        # bộ nhãn mất chuỗi truy nguyên, phải dừng như chế độ am.
        cfg = Path(a.config)
        if cfg.exists():
            for qd in (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("quyet_dinh", []) or []:
                if qd.get("hanh_dong") == "gan_nhan":
                    kiem_xuat_xu(qd)
        rp = Path(a.report) if a.report else Path(a.in_csv).parent / "glyph_fix_kiem_report.json"
        return run_kiem(Path(a.in_csv), Path(a.cells), Path(a.decisions), rp)
    run(Path(a.in_csv), Path(a.out), Path(a.config), Path(a.src_root), a.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
