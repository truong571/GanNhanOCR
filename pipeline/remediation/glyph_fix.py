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

    python -m pipeline.remediation.glyph_fix --in dataset_out/labels_final.csv \
        --out dataset_out/labels_final.csv --config config/quyet_dinh_glyph.yaml --apply
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
DTYPE = {"image_md5": str, "label_in_train": str, "crop_w": str, "crop_h": str}

# Chỉ ô đang BỊ GIỮ LẠI mới được gán. Không bao giờ ghi đè nhãn đã ở GOLD/SYLLABLE —
# ghi đè là lặng lẽ đổi một nhãn đã qua luật khác, và ta sẽ không biết đã đổi cái gì.
CO_THE_GAN = ("REVIEW", "SILVER", "SILVER_uncalibrated")


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
    a = ap.parse_args(argv)
    run(Path(a.in_csv), Path(a.out), Path(a.config), Path(a.src_root), a.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
