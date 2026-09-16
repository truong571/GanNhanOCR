"""Sinh khoá bền QĐ-01 (flow N0d) + khung phán quyết QĐ-01a (A-3).

VÌ SAO CÓ TỆP NÀY
-----------------
2.014 ô 'người' → 𠊚 là phán quyết NGƯỜI (docs/NGUOI_CHAM_QUYET_DINH.md). Hiện chúng
chỉ tồn tại như hệ quả của `glyph_fix --mode am` (mask theo âm + tier) — mọi thay đổi
ma trận/hộp về sau (A-4…A-6) đều có thể làm ô trôi sang âm khác, thành khe, hoặc đổi
bbox → mất ô mà không ai hay. Khoá bền theo `(book,page,column,nom_idx)` (chỉ số chữ
OCR Nôm — cache OCR đóng băng nên nom_idx bất biến qua mọi thế hệ) là thứ duy nhất
sống qua cả ba bước đó; `bbox/prev/next_cu` + `image_md5_cu` để PASS 2 tái tạo crop
đúng byte (md5 phụ thuộc cả hộp hàng xóm qua carve: build_dataset.py:568-576).

NGUỒN: bản NGƯỜI ĐÃ KÝ `dataset_out/labels_final.csv` (rule `quyet_dinh_nguoi*`) JOIN
bản tái lập HEAD `dataset_out_v3/labels_final.csv` (có nom_idx/syl_idx, N0b) theo
`(book,page,column,bbox,image_md5)`; `tier_build/rule_build` từ `dataset_out/labels.csv`
(tier/rule TRƯỚC remediation) theo `image`.

QĐ-01a (A-3): 12 ô GOLD âm 'người' mang mã ≠ 𠊚 (昆 8, 辞 2, 匕 1, 命 1) — nằm ngoài
2.014, `quyet_dinh_glyph.yaml` mục `khong_dung_cho` đã nói rõ đó là quyết định KHÁC,
cần người xem riêng → chèn vào `config/qd01a_decisions.csv` với `quyet` TRỐNG. Tệp
này người điền tay; script KHÔNG ghi đè dòng đã có (luỹ đẳng), chỉ thêm dòng thiếu.

    python -m pipeline.tools.sinh_qd01_cells \\
        --signed dataset_out/labels_final.csv --build dataset_out/labels.csv \\
        --head dataset_out_v3/labels_final.csv \\
        --out-cells config/qd01_cells.csv --out-decisions config/qd01a_decisions.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
NA = dict(dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
QD01 = "quyet_dinh_nguoi"
NGUOI = "\U0002029A"           # 𠊚 U+2029A — KHÔNG phải 𠊛 U+2029B
N_QD01 = 2014
KEY_NOM = ["book", "page", "column", "nom_idx"]
KEY_JOIN = ["book", "page", "column", "bbox", "image_md5"]
KEY_BBOX = ["book", "page", "column", "bbox"]

COT_CELLS = ["book", "page", "column", "nom_idx", "syl_idx", "syllable", "ocr_char", "label",
             "unicode", "bbox_cu", "prev_bbox_cu", "next_bbox_cu", "image_md5_cu", "image_cu",
             "tier_build", "rule_build"]
COT_DECISIONS = ["book", "page", "column", "nom_idx", "bbox_cu", "syllable", "ocr_char",
                 "label_hien_tai", "ly_do", "quyet", "nguoi_ky", "ngay", "xuat_xu"]
LY_DO_A3 = "GOLD người ngoài QĐ-01, bộ kiểm FALSE"


def _y_center(bbox: str) -> float:
    b = json.loads(bbox)
    return (b[1] + b[3]) / 2.0


def prev_next_bbox(df: pd.DataFrame) -> pd.DataFrame:
    """Hộp liền trước/sau theo tâm y trong cùng (book,page,column) — ĐÚNG cách
    build_dataset.py:568-576: MỌI record có bbox, kể cả REVIEW/QUARANTINE; sắp ổn
    định theo y tâm (thứ tự dòng làm tie-break như `list.sort` trên recs)."""
    d = df[df["bbox"].str.strip() != ""].copy()
    d["_yc"] = d["bbox"].map(_y_center)
    d["_ord"] = range(len(d))
    d = d.sort_values(["book", "page", "column", "_yc", "_ord"], kind="stable")
    g = d.groupby(["book", "page", "column"], sort=False)["bbox"]
    d["prev_bbox_cu"] = g.shift(1).fillna("")
    d["next_bbox_cu"] = g.shift(-1).fillna("")
    return d[["prev_bbox_cu", "next_bbox_cu"]].reindex(df.index).fillna("")


def sinh_cells(signed: pd.DataFrame, build: pd.DataFrame, head: pd.DataFrame) -> pd.DataFrame:
    q = signed[signed["rule"].str.startswith(QD01)].copy()
    if len(q) != N_QD01:
        sys.exit(f"[qd01_cells] bản ký có {len(q)} ô QĐ-01, kỳ vọng {N_QD01}")
    pn = prev_next_bbox(signed)
    q["prev_bbox_cu"] = pn.loc[q.index, "prev_bbox_cu"]
    q["next_bbox_cu"] = pn.loc[q.index, "next_bbox_cu"]

    # nom_idx/syl_idx từ bản tái lập HEAD: khoá 5 cột, rơi về 4 cột nếu md5 lệch
    h = head[KEY_JOIN + ["nom_idx", "syl_idx"]].drop_duplicates(KEY_JOIN)
    j = q.merge(h, on=KEY_JOIN, how="left", indicator="_m5")
    thieu = j["_m5"] != "both"
    if thieu.any():
        print(f"[qd01_cells] {int(thieu.sum())} ô không join được theo 5 khoá "
              f"(bbox+md5) — thử lại theo (book,page,column,bbox):")
        print(j.loc[thieu, KEY_JOIN].to_string(index=False))
        h4 = head[KEY_BBOX + ["nom_idx", "syl_idx"]].drop_duplicates(KEY_BBOX) \
            .rename(columns={"nom_idx": "_n4", "syl_idx": "_s4"})
        j = j.merge(h4, on=KEY_BBOX, how="left")
        j.loc[thieu, "nom_idx"] = j.loc[thieu, "_n4"]
        j.loc[thieu, "syl_idx"] = j.loc[thieu, "_s4"]
        j = j.drop(columns=["_n4", "_s4"])
    j = j.drop(columns=["_m5"])
    if j["nom_idx"].fillna("").eq("").any():
        sys.exit(f"[qd01_cells] {int(j['nom_idx'].fillna('').eq('').sum())} ô không có "
                 f"nom_idx sau cả hai khoá — bản HEAD chưa phát nom_idx (N0b)?")

    # tier/rule TRƯỚC remediation (labels.csv) — image cùng thế hệ nên join theo image
    b = build[["image", "tier", "rule"]].rename(
        columns={"tier": "tier_build", "rule": "rule_build"}).drop_duplicates("image")
    j = j.merge(b, on="image", how="left")
    if j["tier_build"].isna().any():
        sys.exit(f"[qd01_cells] {int(j['tier_build'].isna().sum())} ô không có trong labels.csv")

    out = j.rename(columns={"bbox": "bbox_cu", "image_md5": "image_md5_cu", "image": "image_cu"})
    out = out[COT_CELLS].copy()
    out["nom_idx"] = out["nom_idx"].astype(int)
    out["syl_idx"] = out["syl_idx"].astype(int)
    out["column"] = out["column"].astype(int)
    out = out.sort_values(["book", "page", "column", "nom_idx"]).reset_index(drop=True)

    # ASSERT — hỏng thì dừng, không ghi khoá sai
    assert len(out) == N_QD01, len(out)
    assert not out.duplicated(KEY_NOM).any(), "trùng khoá (book,page,column,nom_idx)"
    assert (out["label"] == NGUOI).all(), "có ô QĐ-01 không phải 𠊚"
    assert (out["bbox_cu"] != "").all() and (out["image_md5_cu"] != "").all()
    return out


def sinh_decisions_a3(signed: pd.DataFrame) -> pd.DataFrame:
    """12 ô GOLD âm 'người', nhãn ≠ 𠊚, rule không phải QĐ-01 → chờ người quyết."""
    am = signed["syllable"].str.strip().str.lower() == "người"
    return signed[(signed["tier"] == "GOLD") & am & (signed["label"] != NGUOI)
                  & ~signed["rule"].str.startswith(QD01)].copy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.sinh_qd01_cells",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--signed", default="dataset_out/labels_final.csv",
                    help="bản NGƯỜI ĐÃ KÝ (rule quyet_dinh_nguoi*)")
    ap.add_argument("--build", default="dataset_out/labels.csv",
                    help="labels.csv cùng thế hệ với --signed (tier/rule trước remediation)")
    ap.add_argument("--head", default="dataset_out_v3/labels_final.csv",
                    help="bản tái lập HEAD có nom_idx/syl_idx (N0c)")
    ap.add_argument("--out-cells", default="config/qd01_cells.csv")
    ap.add_argument("--out-decisions", default="config/qd01a_decisions.csv")
    a = ap.parse_args(argv)

    signed = pd.read_csv(a.signed, **NA)
    build = pd.read_csv(a.build, **NA)
    head = pd.read_csv(a.head, **NA)
    if "nom_idx" not in head.columns:
        sys.exit(f"[qd01_cells] {a.head} không có cột nom_idx — chạy N0b trước")

    cells = sinh_cells(signed, build, head)
    oc = Path(a.out_cells); oc.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(oc, index=False)
    print(f"[qd01_cells] {len(cells)} ô -> {oc}  | tier_build "
          f"{cells['tier_build'].value_counts().to_dict()} | 0 trùng khoá, 100% {NGUOI}")

    # ---- QĐ-01a: 12 ô A-3, KHÔNG ghi đè phán quyết đã điền ----
    g = sinh_decisions_a3(signed)
    h = head[KEY_JOIN + ["nom_idx"]].drop_duplicates(KEY_JOIN)
    g = g.merge(h, on=KEY_JOIN, how="left")
    if g["nom_idx"].isna().any():
        h4 = head[KEY_BBOX + ["nom_idx"]].drop_duplicates(KEY_BBOX).rename(columns={"nom_idx": "_n4"})
        g = g.merge(h4, on=KEY_BBOX, how="left")
        g["nom_idx"] = g["nom_idx"].fillna(g["_n4"]); g = g.drop(columns=["_n4"])
    if g["nom_idx"].isna().any():
        sys.exit("[qd01a] ô A-3 không có nom_idx trong bản HEAD")
    moi = pd.DataFrame({
        "book": g["book"], "page": g["page"], "column": g["column"].astype(int),
        "nom_idx": g["nom_idx"].astype(int), "bbox_cu": g["bbox"], "syllable": g["syllable"],
        "ocr_char": g["ocr_char"], "label_hien_tai": g["label"], "ly_do": LY_DO_A3,
        "quyet": "", "nguoi_ky": "", "ngay": "", "xuat_xu": ""})
    moi = moi.sort_values(["book", "page", "column", "nom_idx"]).reset_index(drop=True)
    od = Path(a.out_decisions)
    if od.exists():
        cu = pd.read_csv(od, **NA)
        for c in COT_DECISIONS:
            if c not in cu.columns:
                cu[c] = ""
        cu = cu[COT_DECISIONS]
        k_cu = set(zip(cu["book"], cu["page"], cu["column"].astype(int), cu["nom_idx"].astype(int)))
        them = moi[~moi.apply(lambda r: (r["book"], r["page"], int(r["column"]), int(r["nom_idx"]))
                              in k_cu, axis=1)]
        out = pd.concat([cu, them], ignore_index=True)
        print(f"[qd01a] {od} đã có {len(cu)} dòng (giữ nguyên, kể cả `quyet`) · thêm {len(them)}")
    else:
        out = moi
        print(f"[qd01a] tạo mới {od}: {len(out)} dòng A-3 "
              f"{out['label_hien_tai'].value_counts().to_dict()} · `quyet` trống, chờ người")
    out[COT_DECISIONS].to_csv(od, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
