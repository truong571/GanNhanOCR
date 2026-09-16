"""Đối soát HAI THẾ HỆ bộ nhãn — bản cũ (đã giao nộp) với bản mới (vừa dựng lại).

VÌ SAO CÓ TỆP NÀY (A-1b / flow N0c, N15)
----------------------------------------
Bộ 64.525 ô đang giao nộp chưa từng được TÁI LẬP từ HEAD: không ai biết chạy lại
`run_pipeline.sh` hôm nay có ra đúng bộ đó không (memory 04/09: mẻ giao nộp chạy tay,
không checkpoint). Mọi thay đổi hằng số/luật về sau (A-4…A-8) đều phải đo "đổi bao
nhiêu ô, ô nào" so với bộ cũ — muốn thế phải có MỘT công cụ so hai `labels*.csv` theo
khoá ô, in ra khớp/khác từng cột, ma trận tier cũ×mới, rule cũ×mới, ô mới/mất, và
riêng lớp ô phán quyết NGƯỜI (rule `quyet_dinh_nguoi*`, 2.014 ô) phải còn nguyên.

KHOÁ Ô
------
  · khoá chính  (book,page,column,bbox)      — có ở mọi thế hệ (bbox JSON nguyên văn)
  · khoá bền    (book,page,column,nom_idx)   — chỉ khi CẢ HAI bên có cột `nom_idx`
    (N0b phát thêm; bản 64.525 KHÔNG có nên lần tái lập HEAD chỉ so được theo bbox)
Khoá có thể trùng (lỗi F1 xuất đôi, memory 13/07: 2.321 dòng trùng) -> ghép thêm số thứ
tự lặp trong khoá (cumcount theo thứ tự dòng — tệp đã sắp canon T6.b) để không nhân đôi.

LỆCH ĐƯỢC PHÉP
--------------
Chỉ ô có rule chứa `s3` hoặc tier SILVER* (ở bất kỳ bên nào) — S3 phụ thuộc cache
nguyên mẫu ký bằng mtime (repro_check R2). Mọi lệch khác là lệch THẬT, phải giải thích.

    python -m pipeline.tools.doi_soat_the_he --old dataset_out/labels_final.csv \\
        --new dataset_out_v3/labels_final.csv --out dataset_out_v3/DOI_SOAT_HEAD.md
    ... [--qd01 config/qd01_cells.csv]   # báo locked/pending cho khoá QĐ-01 (N0d)
    ... [--strict]                        # exit 1 nếu có lệch ngoài S3/SILVER hoặc mất QĐ-01
    ... [--old-idx <labels_final tái lập HEAD, N0c>]   # cấp nom_idx/syl_idx cho bản cũ
                                          # (join 5 khoá book,page,column,bbox,image_md5)

KHỐI V3 (flow N15 B1–B8, A-13) — tự bật khi CẢ HAI bên có `nom_idx` và bản mới có cột
`count_source` (schema v3): B1 crosstab tier×rule cũ→mới; B2 ô đổi âm ghép (syl_idx);
B3 ô mới/mất; B4 bbox đổi theo count_source; B5 md5 đổi + hàng xóm ô khoá so prev/next_cu;
B6 QĐ-01 locked/pending/bbox/md5; B7 GOLD→REVIEW theo rule; B8 L1; phân bố
count_source/box_source; usable; khe giả đếm trực tiếp trên cột n_ocr == n_qn ≥ 10.
Danh sách dài (md5 đổi, ô mất, GOLD→REVIEW) ghi CSV vào `--chi-tiet-dir` (mặc định
`<thư mục bản mới>/doi_soat/`), báo cáo chỉ dẫn đường.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KEY_BBOX = ["book", "page", "column", "bbox"]
KEY_NOM = ["book", "page", "column", "nom_idx"]
# Cột đem so (chỉ so cột có ở CẢ HAI bên)
SO_COT = ["tier", "rule", "label", "syllable", "image_md5", "unicode", "ocr_char"]
QD01 = "quyet_dinh_nguoi"
LIET_KE_TOI_DA = 200   # số dòng liệt kê tối đa cho mỗi bảng chi tiết trong markdown


# ------------------------------------------------------------------ đọc/khoá ----
def doc(path: Path) -> pd.DataFrame:
    """Đọc labels*.csv nguyên văn (mọi cột str, ô trống = '')."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    thieu = [c for c in KEY_BBOX if c not in df.columns]
    if thieu:
        sys.exit(f"[doi_soat] {path}: thiếu cột khoá {thieu}")
    return df


def co_nom_idx(df: pd.DataFrame) -> bool:
    return "nom_idx" in df.columns and bool((df["nom_idx"].str.strip() != "").any())


def gan_khoa(df: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """Thêm cột `_k` (khoá ghép) + `_dup` (số lần lặp của khoá, theo thứ tự dòng)."""
    df = df.copy()
    df["_k"] = df[key].agg("|".join, axis=1)
    df["_dup"] = df.groupby("_k").cumcount()
    return df


def la_qd01(s: pd.Series) -> pd.Series:
    return s.str.startswith(QD01)


def lech_cho_phep(r: pd.Series) -> bool:
    """Lệch chỉ được phép khi ô dính S3/SILVER ở bất kỳ bên nào."""
    for side in ("_cu", "_moi"):
        if "s3" in str(r.get("rule" + side, "")).lower():
            return True
        if str(r.get("tier" + side, "")).upper().startswith("SILVER"):
            return True
    return False


# ------------------------------------------------------------------ so sánh ----
def doi_soat(cu: pd.DataFrame, moi: pd.DataFrame, key: list[str]) -> dict:
    """Ghép hai bên theo khoá + số lặp; trả về dict các bảng/số đo."""
    a = gan_khoa(cu, key)
    b = gan_khoa(moi, key)
    cot_so = [c for c in SO_COT if c in a.columns and c in b.columns]
    lay = ["_k", "_dup"] + key + cot_so
    m = a[lay].merge(b[lay], on=["_k", "_dup"], how="outer",
                     suffixes=("_cu", "_moi"), indicator=True)
    # cột khoá bị suffix — gộp lại một bản để in
    for c in key:
        m[c] = m[c + "_cu"].where(m["_merge"] != "right_only", m[c + "_moi"])
    ca_hai = m[m["_merge"] == "both"].copy()
    chi_cu = m[m["_merge"] == "left_only"].copy()
    chi_moi = m[m["_merge"] == "right_only"].copy()

    khac_theo_cot = {}
    for c in cot_so:
        d = ca_hai[c + "_cu"] != ca_hai[c + "_moi"]
        khac_theo_cot[c] = int(d.sum())
        ca_hai["_khac_" + c] = d
    mask_khac = ca_hai[["_khac_" + c for c in cot_so]].any(axis=1) if cot_so else \
        pd.Series(False, index=ca_hai.index)
    doi = ca_hai[mask_khac].copy()
    if len(doi):
        doi["_cho_phep"] = doi.apply(lech_cho_phep, axis=1)
    else:
        doi["_cho_phep"] = pd.Series(dtype=bool)

    return {
        "key": key, "cot_so": cot_so,
        "n_cu": len(a), "n_moi": len(b),
        "dup_cu": int((a["_dup"] > 0).sum()), "dup_moi": int((b["_dup"] > 0).sum()),
        "n_ca_hai": len(ca_hai), "chi_cu": chi_cu, "chi_moi": chi_moi,
        "khac_theo_cot": khac_theo_cot, "doi": doi, "ca_hai": ca_hai,
    }


def bang_md(df: pd.DataFrame, cols: list[str], toi_da: int = LIET_KE_TOI_DA) -> list[str]:
    """Bảng markdown gọn từ DataFrame (cắt ở `toi_da` dòng)."""
    if df.empty:
        return ["_(không có)_"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.head(toi_da).iterrows():
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "\\|") for c in cols) + " |")
    if len(df) > toi_da:
        out.append(f"\n_(… còn {len(df) - toi_da} dòng nữa, chỉ liệt kê {toi_da})_")
    return out


def crosstab_md(a: pd.Series, b: pd.Series, ten_a: str, ten_b: str) -> list[str]:
    ct = pd.crosstab(a, b, margins=True, margins_name="TỔNG")
    ct.index.name = f"{ten_a} \\ {ten_b}"
    cols = list(ct.columns)
    out = ["| " + ct.index.name + " | " + " | ".join(map(str, cols)) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for idx, row in ct.iterrows():
        out.append("| " + str(idx) + " | " + " | ".join(str(int(v)) for v in row) + " |")
    return out


# ------------------------------------------------------------------ QĐ-01 ----
def qd01_khoa(qd: pd.DataFrame, moi: pd.DataFrame) -> dict:
    """Join config/qd01_cells.csv (N0d) với bản mới: locked / pending.

    locked  = tìm thấy ô theo (book,page,column,nom_idx) [hoặc bbox_cu↔bbox khi
              không có nom_idx] VÀ nhãn mới == nhãn khoá VÀ rule mới là QĐ-01.
    pending = còn lại, chia nhỏ theo nguyên nhân để người xử lý (QĐ-01a).
    """
    dung_nom = co_nom_idx(qd) and co_nom_idx(moi)
    if dung_nom:
        key_qd, key_moi = KEY_NOM, KEY_NOM
    else:
        key_qd = ["book", "page", "column", "bbox_cu" if "bbox_cu" in qd.columns else "bbox"]
        key_moi = KEY_BBOX
    q = qd.copy(); q["_k"] = q[key_qd].agg("|".join, axis=1)
    n = moi.copy(); n["_k"] = n[key_moi].agg("|".join, axis=1)
    n = n.drop_duplicates("_k", keep="first")
    # đổi tên TRƯỚC khi merge: qd01_cells.csv không có cột rule/bbox/image_md5 (mà là
    # rule_build/bbox_cu/image_md5_cu) nên suffix của pandas sẽ không được áp
    n = n[["_k", "label", "rule", "bbox", "image_md5"]].rename(columns={
        "label": "label_moi", "rule": "rule_moi", "bbox": "bbox_moi", "image_md5": "image_md5_moi"})
    j = q.merge(n, on="_k", how="left", indicator=True)
    tim_thay = j["_merge"] == "both"
    rule_qd = tim_thay & la_qd01(j["rule_moi"].fillna(""))
    nhan_khop = tim_thay & (j["label"] == j["label_moi"]) if "label" in j.columns \
        else tim_thay
    locked = tim_thay & rule_qd & nhan_khop
    ly_do = pd.Series("locked", index=j.index)
    ly_do[~tim_thay] = "pending: không thấy ô trong bản mới"
    ly_do[tim_thay & ~rule_qd] = "pending: có ô nhưng rule mới không phải QĐ-01"
    ly_do[tim_thay & rule_qd & ~nhan_khop] = "pending: rule QĐ-01 nhưng nhãn khác"
    j["_ly_do"] = ly_do
    bbox_khop = md5_khop = None
    if "bbox_cu" in j.columns:
        bbox_khop = int((tim_thay & (j["bbox_cu"] == j["bbox_moi"])).sum())
    if "image_md5_cu" in j.columns:
        md5_khop = int((tim_thay & (j["image_md5_cu"] == j["image_md5_moi"])).sum())
    return {"khoa": "nom_idx" if dung_nom else "bbox", "n": len(j),
            "locked": int(locked.sum()), "pending": int((~locked).sum()),
            "ly_do": j["_ly_do"].value_counts().to_dict(),
            "bbox_khop": bbox_khop, "md5_khop": md5_khop,
            "pending_rows": j[~locked]}


# ------------------------------------------------------------------ báo cáo ----
def bao_cao(args, R: dict, R2: dict | None, Q: dict | None) -> tuple[list[str], set[int], bool]:
    """Dựng markdown; trả (dòng, chỉ số dòng chi tiết, dat).

    dat=False khi có lệch ngoài S3/SILVER hoặc QĐ-01 mất/đổi. Dòng "chi tiết" (bảng liệt
    kê từng ô) chỉ ghi vào markdown, KHÔNG in stdout — bảng có thể tới 200 dòng.
    """
    L: list[str] = []
    chi_tiet: set[int] = set()

    def them_bang(df: pd.DataFrame, cols: list[str]) -> None:
        for line in bang_md(df, cols):
            chi_tiet.add(len(L)); L.append(line)
    L.append(f"# Đối soát thế hệ — `{args.old}` (cũ) ↔ `{args.new}` (mới)")
    L.append("")
    L.append(f"- Khoá chính: `({','.join(R['key'])})` + số lặp; cột so: `{', '.join(R['cot_so'])}`")
    L.append(f"- Số dòng: cũ **{R['n_cu']}** · mới **{R['n_moi']}** · ghép được cả hai **{R['n_ca_hai']}**")
    L.append(f"- Khoá trùng (dòng lặp khoá): cũ {R['dup_cu']} · mới {R['dup_moi']}")
    L.append(f"- Chỉ có ở bản MỚI (ô mới): **{len(R['chi_moi'])}** · chỉ có ở bản CŨ (ô mất): **{len(R['chi_cu'])}**")
    L.append("")
    L.append("## 1. Khớp / khác theo từng cột (trên các ô ghép được)")
    L.append("")
    L.append("| cột | khớp | khác |")
    L.append("|---|---|---|")
    for c, k in R["khac_theo_cot"].items():
        L.append(f"| {c} | {R['n_ca_hai'] - k} | {k} |")
    doi = R["doi"]
    n_doi = len(doi)
    n_cp = int(doi["_cho_phep"].sum()) if n_doi else 0
    n_kcp = n_doi - n_cp
    L.append("")
    L.append(f"Dòng có ít nhất một cột khác: **{n_doi}** — trong đó lệch cho phép (rule chứa s3 / tier SILVER*): "
             f"{n_cp}; **lệch ngoài S3/SILVER: {n_kcp}**")

    ca_hai = R["ca_hai"]
    if "tier" in R["cot_so"]:
        L.append("")
        L.append("## 2. Crosstab tier cũ × mới (mọi ô ghép được)")
        L.append("")
        L += crosstab_md(ca_hai["tier_cu"], ca_hai["tier_moi"], "tier cũ", "tier mới")
    if n_doi and "rule" in R["cot_so"]:
        L.append("")
        L.append("## 3. Rule cũ × mới — CHỈ các dòng đổi")
        L.append("")
        L += crosstab_md(doi["rule_cu"], doi["rule_moi"], "rule cũ", "rule mới")
        L.append("")
        L.append("### 3a. Dòng lệch NGOÀI S3/SILVER (phải giải thích từng dòng)")
        L.append("")
        cols = R["key"] + [c + s for c in R["cot_so"] for s in ("_cu", "_moi")]
        them_bang(doi[~doi["_cho_phep"]], cols)
        L.append("")
        L.append("### 3b. Dòng lệch cho phép (S3/SILVER)")
        L.append("")
        them_bang(doi[doi["_cho_phep"]], cols)

    L.append("")
    L.append("## 4. Ô chỉ có một bên")
    L.append("")
    cot_1ben = lambda s: R["key"] + [c + s for c in R["cot_so"] if c in ("tier", "rule", "label", "syllable")]
    if len(R["chi_cu"]) and "tier" in R["cot_so"]:
        L.append("Ô MẤT theo tier cũ: " + ", ".join(
            f"{k}={v}" for k, v in R["chi_cu"]["tier_cu"].value_counts().items()))
    if len(R["chi_moi"]) and "tier" in R["cot_so"]:
        L.append("Ô MỚI theo tier mới: " + ", ".join(
            f"{k}={v}" for k, v in R["chi_moi"]["tier_moi"].value_counts().items()))
    L.append("")
    L.append("### 4a. Ô mất (chỉ có ở bản cũ)")
    L.append("")
    them_bang(R["chi_cu"], cot_1ben("_cu"))
    L.append("")
    L.append("### 4b. Ô mới (chỉ có ở bản mới)")
    L.append("")
    them_bang(R["chi_moi"], cot_1ben("_moi"))

    # ---- QĐ-01 theo rule ----
    dat = n_kcp == 0
    if "rule" in R["cot_so"]:
        q_cu = la_qd01(ca_hai["rule_cu"]); q_moi = la_qd01(ca_hai["rule_moi"])
        n_q_cu_all = int(la_qd01(R["ca_hai"]["rule_cu"]).sum() + la_qd01(R["chi_cu"]["rule_cu"]).sum())
        n_q_moi_all = int(la_qd01(R["ca_hai"]["rule_moi"]).sum() + la_qd01(R["chi_moi"]["rule_moi"]).sum())
        q_mat = int(la_qd01(R["chi_cu"]["rule_cu"]).sum())
        q_doi = ca_hai[q_cu & ca_hai.index.isin(doi.index)]
        q_roi = ca_hai[q_cu & ~q_moi]      # cũ là QĐ-01, mới không còn rule QĐ-01
        L.append("")
        L.append("## 5. Ô phán quyết NGƯỜI (rule `quyet_dinh_nguoi*`)")
        L.append("")
        L.append(f"- QĐ-01 bản cũ: **{n_q_cu_all}** · bản mới: **{n_q_moi_all}**")
        L.append(f"- QĐ-01 MẤT (không có ô trong bản mới): **{q_mat}**")
        L.append(f"- QĐ-01 có ô nhưng ĐỔI (bất kỳ cột so nào): **{len(q_doi)}** — trong đó mất rule QĐ-01: {len(q_roi)}")
        if len(q_doi):
            cols = R["key"] + [c + s for c in R["cot_so"] for s in ("_cu", "_moi")]
            L.append("")
            them_bang(q_doi, cols)
        if q_mat or len(q_doi):
            dat = False

    if R2 is not None:
        L.append("")
        L.append("## 6. Đối soát lại theo khoá bền `(book,page,column,nom_idx)`")
        L.append("")
        L.append(f"- ghép được {R2['n_ca_hai']} · ô mới {len(R2['chi_moi'])} · ô mất {len(R2['chi_cu'])} · "
                 f"dòng đổi {len(R2['doi'])} (khoá trùng cũ {R2['dup_cu']} / mới {R2['dup_moi']})")
        L.append("| cột | khớp | khác |")
        L.append("|---|---|---|")
        for c, k in R2["khac_theo_cot"].items():
            L.append(f"| {c} | {R2['n_ca_hai'] - k} | {k} |")
    else:
        L.append("")
        L.append("_(Không đối soát theo `nom_idx`: ít nhất một bên chưa có cột này — bản 64.525 sinh trước N0b.)_")

    if Q is not None:
        L.append("")
        L.append(f"## 7. Khoá QĐ-01 `{args.qd01}` ↔ bản mới (khoá {Q['khoa']})")
        L.append("")
        L.append(f"- n = {Q['n']} · **locked = {Q['locked']}** · **pending = {Q['pending']}** "
                 f"(locked + pending = {Q['locked'] + Q['pending']})")
        for k, v in Q["ly_do"].items():
            L.append(f"  - {k}: {v}")
        if Q["bbox_khop"] is not None:
            L.append(f"- bbox_cu khớp bbox mới: {Q['bbox_khop']}/{Q['n']}")
        if Q["md5_khop"] is not None:
            L.append(f"- image_md5_cu khớp md5 mới: {Q['md5_khop']}/{Q['n']}")
        if len(Q["pending_rows"]):
            L.append("")
            cols = [c for c in ["book", "page", "column", "nom_idx", "syllable", "label",
                                "label_moi", "rule_moi", "_ly_do"] if c in Q["pending_rows"].columns]
            them_bang(Q["pending_rows"], cols)

    L.append("")
    L.append("## KẾT LUẬN")
    L.append("")
    if n_doi == 0 and not len(R["chi_cu"]) and not len(R["chi_moi"]):
        L.append(f"**0 lệch** trên {R['n_ca_hai']} ô theo khoá `({','.join(R['key'])})` — bản mới tái lập đúng bản cũ.")
    else:
        L.append(f"lệch ngoài S3/SILVER: **{n_kcp}** · lệch S3/SILVER: {n_cp} · ô mới {len(R['chi_moi'])} · ô mất {len(R['chi_cu'])}")
    L.append(f"Đạt (0 lệch ngoài S3/SILVER, QĐ-01 không mất/đổi): **{'CÓ' if dat else 'KHÔNG'}**")
    return L, chi_tiet, dat


# ------------------------------------------------------------------ nom_idx cho bản cũ ----
KEY_5 = ["book", "page", "column", "bbox", "image_md5"]


def nap_old_idx(cu: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Cấp `nom_idx/syl_idx` cho bản cũ từ bản tái lập HEAD (N0c) — bộ 64.525 sinh trước
    N0b nên không có hai cột này; bản tái lập 0 lệch (DOI_SOAT_HEAD.md) mang chúng.
    Join 5 khoá (book,page,column,bbox,image_md5); đòi 100% dòng có chỉ số, không trùng.
    """
    if co_nom_idx(cu):
        return cu
    idx = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    can = KEY_5 + ["nom_idx", "syl_idx"]
    thieu = [c for c in can if c not in idx.columns]
    if thieu:
        sys.exit(f"[doi_soat] --old-idx {path}: thiếu cột {thieu}")
    if idx.duplicated(KEY_5).any() or cu.duplicated(KEY_5).any():
        sys.exit("[doi_soat] --old-idx: khoá 5 cột trùng — không cấp nom_idx được")
    m = cu.merge(idx[can], on=KEY_5, how="left", indicator=True)
    n_thieu = int((m["_merge"] != "both").sum())
    if n_thieu:
        sys.exit(f"[doi_soat] --old-idx: {n_thieu} dòng bản cũ không tìm thấy trong {path}")
    return m.drop(columns=["_merge"])


# ------------------------------------------------------------------ khối v3 (N15 B1–B8) ----
USABLE = ("GOLD", "SILVER", "SYLLABLE")
RULE_L1 = "s1_inter_s2_direct_am_sua_dau"
COT_V3 = ["count_source", "box_source", "l1_support", "l1_tie", "tier_v3", "n_ocr", "n_qn",
          "syllable_ocr", "qd01_locked", "qd01_excluded"]


def _int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _usable(t: pd.Series) -> pd.Series:
    return t.isin(USABLE)


def _vc_md(s: pd.Series, ten: str) -> list[str]:
    vc = s.value_counts()
    return [f"| {ten} | n |", "|---|---|"] + [f"| {k} | {v} |" for k, v in vc.items()] + \
           [f"| TỔNG | {int(vc.sum())} |"]


def doi_soat_v3(cu: pd.DataFrame, moi: pd.DataFrame, qd: pd.DataFrame | None,
                chi_tiet_dir: Path) -> tuple[list[str], dict]:
    """B1–B8 + count_source/box_source + usable + khe giả (flow N15). Trả (dòng md, số đo).

    Ghép theo khoá bền (book,page,column,nom_idx) + số lặp. Số đo trả về để mục
    'đối chiếu kỳ vọng' in một bảng; danh sách dài ghi CSV vào `chi_tiet_dir`.
    """
    chi_tiet_dir.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    D: dict = {}
    lay_cu = KEY_NOM + ["syl_idx", "tier", "rule", "label", "syllable", "bbox", "image_md5", "ocr_char"]
    lay_moi = lay_cu + [c for c in COT_V3 if c in moi.columns]
    a = gan_khoa(cu, KEY_NOM)[["_k", "_dup"] + lay_cu]
    b = gan_khoa(moi, KEY_NOM)[["_k", "_dup"] + lay_moi]
    m = a.merge(b, on=["_k", "_dup"], how="outer", suffixes=("_cu", "_moi"), indicator=True)
    # cột chỉ có ở bản mới (schema v3) không bị pandas gắn hậu tố -> gắn tay cho đồng nhất
    m = m.rename(columns={c: c + "_moi" for c in COT_V3 if c in m.columns and c + "_moi" not in m.columns})
    for c in KEY_NOM:
        m[c] = m[c + "_cu"].where(m["_merge"] != "right_only", m[c + "_moi"])
    both = m[m["_merge"] == "both"].copy()
    mat = m[m["_merge"] == "left_only"].copy()
    moi_ = m[m["_merge"] == "right_only"].copy()
    D["n_ghep"] = len(both); D["n_mat"] = len(mat); D["n_moi"] = len(moi_)

    # ---- B1 ----
    L.append("## B1. Crosstab tier × rule cũ → mới (khoá bền nom_idx; kể cả ô mất/mới)")
    L.append("")
    t_cu = m["tier_cu"].where(m["_merge"] != "right_only", "(mới)")
    t_moi = m["tier_moi"].where(m["_merge"] != "left_only", "(mất)")
    L += crosstab_md(t_cu, t_moi, "tier cũ", "tier mới")
    L.append("")
    L.append("### B1a. rule cũ × tier mới")
    L.append("")
    r_cu = m["rule_cu"].where(m["_merge"] != "right_only", "(mới)")
    L += crosstab_md(r_cu, t_moi, "rule cũ", "tier mới")
    L.append("")
    L.append("### B1b. rule cũ × rule mới (chỉ ô ghép được)")
    L.append("")
    L += crosstab_md(both["rule_cu"], both["rule_moi"], "rule cũ", "rule mới")
    L.append("")
    L.append("### B1c. usable cũ × usable mới")
    L.append("")
    u_cu = _usable(t_cu).map({True: "usable", False: "không"}).where(m["_merge"] != "right_only", "(mới)")
    u_moi = _usable(t_moi).map({True: "usable", False: "không"}).where(m["_merge"] != "left_only", "(mất)")
    L += crosstab_md(u_cu, u_moi, "usable cũ", "usable mới")
    D["usable_cu"] = int(_usable(cu["tier"]).sum()); D["usable_moi"] = int(_usable(moi["tier"]).sum())
    D["usable_cu_mat_usable"] = int((_usable(both["tier_cu"]) & ~_usable(both["tier_moi"])).sum()
                                    + _usable(mat["tier_cu"]).sum())
    D["usable_moi_them"] = int((~_usable(both["tier_cu"]) & _usable(both["tier_moi"])).sum()
                               + _usable(moi_["tier_moi"]).sum())
    L.append("")
    L.append(f"- usable cũ **{D['usable_cu']}** → mới **{D['usable_moi']}** (rời usable {D['usable_cu_mat_usable']}, "
             f"vào usable {D['usable_moi_them']}) · kỳ vọng §4: 70.100–70.450")

    # ---- B2 ----
    doi_ghep = both["syl_idx_cu"] != both["syl_idx_moi"]
    doi_am = both["syllable_cu"] != both["syllable_moi"]
    L.append("")
    L.append("## B2. Ô đổi âm ghép (cùng nom_idx, syl_idx khác) — kỳ vọng GOLD ≈439±20, REVIEW ≈780")
    L.append("")
    L += crosstab_md(both["tier_cu"], doi_ghep.map({True: "đổi ghép", False: "giữ ghép"}), "tier cũ", "syl_idx")
    L.append("")
    L.append("Đổi CHỮ âm (`syllable` khác) tách theo có/không đổi ghép:")
    L.append("")
    L += crosstab_md(both["tier_cu"],
                     (doi_ghep.map({True: "ghép khác", False: "ghép giữ"}) + " & âm "
                      + doi_am.map({True: "khác", False: "giữ"})), "tier cũ", "ghép/âm")
    D["b2_doi_ghep"] = both.loc[doi_ghep, "tier_cu"].value_counts().to_dict()
    D["b2_doi_ghep_tong"] = int(doi_ghep.sum())
    b2 = both[doi_ghep]
    b2[KEY_NOM + ["syl_idx_cu", "syl_idx_moi", "tier_cu", "tier_moi", "rule_cu", "rule_moi",
                  "ocr_char_cu", "syllable_cu", "syllable_moi", "label_cu", "label_moi"]] \
        .to_csv(chi_tiet_dir / "B2_doi_am_ghep.csv", index=False)
    L.append("")
    L.append(f"→ `{chi_tiet_dir / 'B2_doi_am_ghep.csv'}` ({len(b2)} dòng)")

    # ---- B3 ----
    L.append("")
    L.append("## B3. Ô mới / ô mất (khoá bền) — kỳ vọng ≈ +660 / −210, GOLD mất ≈18 (liệt kê)")
    L.append("")
    L.append("Ô MẤT theo tier cũ: " + (", ".join(f"{k}={v}" for k, v in mat["tier_cu"].value_counts().items()) or "0"))
    L.append("")
    L.append("Ô MỚI theo tier mới: " + (", ".join(f"{k}={v}" for k, v in moi_["tier_moi"].value_counts().items()) or "0"))
    D["b3_mat_tier"] = mat["tier_cu"].value_counts().to_dict()
    D["b3_moi_tier"] = moi_["tier_moi"].value_counts().to_dict()
    gold_mat = mat[mat["tier_cu"] == "GOLD"]
    D["b3_gold_mat"] = len(gold_mat)
    L.append("")
    L.append(f"### B3a. GOLD mất ({len(gold_mat)} ô — từng ô)")
    L.append("")
    L += bang_md(gold_mat, KEY_NOM + ["syl_idx_cu", "rule_cu", "ocr_char_cu", "syllable_cu", "label_cu", "bbox_cu"])
    mat[KEY_NOM + ["syl_idx_cu", "tier_cu", "rule_cu", "ocr_char_cu", "syllable_cu", "label_cu", "bbox_cu"]] \
        .to_csv(chi_tiet_dir / "B3_o_mat.csv", index=False)
    moi_[KEY_NOM + ["syl_idx_moi", "tier_moi", "rule_moi", "ocr_char_moi", "syllable_moi", "label_moi", "bbox_moi"]] \
        .to_csv(chi_tiet_dir / "B3_o_moi.csv", index=False)
    L.append("")
    L.append(f"→ `{chi_tiet_dir / 'B3_o_mat.csv'}` ({len(mat)}), `{chi_tiet_dir / 'B3_o_moi.csv'}` ({len(moi_)})")

    # ---- B4 ----
    co_cs = "count_source_moi" in both.columns
    doi_bbox = both["bbox_cu"] != both["bbox_moi"]
    L.append("")
    L.append("## B4. bbox đổi theo `count_source` (ô ghép được) — kỳ vọng ≈11,5k ô usable")
    L.append("")
    if co_cs:
        L += crosstab_md(both["count_source_moi"], doi_bbox.map({True: "bbox đổi", False: "bbox giữ"}),
                         "count_source (mới)", "bbox")
        L.append("")
        L.append("bbox đổi × tier mới:")
        L.append("")
        L += crosstab_md(both.loc[doi_bbox, "count_source_moi"], both.loc[doi_bbox, "tier_moi"],
                         "count_source (bbox đổi)", "tier mới")
    D["b4_bbox_doi"] = int(doi_bbox.sum())
    D["b4_bbox_doi_usable"] = int((doi_bbox & _usable(both["tier_moi"])).sum())
    L.append("")
    L.append(f"- bbox đổi: **{D['b4_bbox_doi']}** ô ghép, trong đó usable (tier mới) **{D['b4_bbox_doi_usable']}**")

    # ---- B5 ----
    co_md5_cu = both["image_md5_cu"] != ""
    co_md5_moi = both["image_md5_moi"] != ""
    md5_doi = co_md5_cu & co_md5_moi & (both["image_md5_cu"] != both["image_md5_moi"])
    L.append("")
    L.append("## B5. md5 crop đổi (cả hai bên có crop) — kỳ vọng ≤14.117 ô usable, QĐ-01 = 0")
    L.append("")
    L += crosstab_md(both["tier_moi"], (co_md5_cu.map({True: "cũ có", False: "cũ không"}) + "/"
                                        + co_md5_moi.map({True: "mới có", False: "mới không"})
                                        + md5_doi.map({True: " → KHÁC", False: ""})),
                     "tier mới", "crop cũ/mới")
    D["b5_md5_doi"] = int(md5_doi.sum())
    D["b5_md5_doi_usable"] = int((md5_doi & _usable(both["tier_moi"])).sum())
    D["b5_md5_doi_qd01"] = int((md5_doi & la_qd01(both["rule_moi"])).sum())
    L.append("")
    L.append(f"- md5 đổi: **{D['b5_md5_doi']}** ô, usable (tier mới) **{D['b5_md5_doi_usable']}**, "
             f"trong đó rule QĐ-01 **{D['b5_md5_doi_qd01']}**")
    if co_cs:
        L.append("")
        L += crosstab_md(both.loc[md5_doi, "count_source_moi"],
                         (both.loc[md5_doi, "bbox_cu"] != both.loc[md5_doi, "bbox_moi"])
                         .map({True: "bbox đổi", False: "bbox giữ (chỉ đổi cắt/đệm)"}),
                         "count_source (md5 đổi)", "bbox")
    b5 = both[md5_doi]
    b5[KEY_NOM + ["tier_cu", "tier_moi", "rule_moi", "syllable_moi", "label_moi", "bbox_cu", "bbox_moi",
                  "image_md5_cu", "image_md5_moi"] + (["count_source_moi", "box_source_moi"] if co_cs else [])] \
        .to_csv(chi_tiet_dir / "B5_md5_doi.csv", index=False)
    L.append("")
    L.append(f"→ `{chi_tiet_dir / 'B5_md5_doi.csv'}` ({len(b5)} dòng)")
    # hàng xóm thật của ô khoá so với prev/next_bbox_cu — ĐÚNG định nghĩa của
    # sinh_qd01_cells.prev_next_bbox: record liền trước/sau theo TÂM Y trong cùng cột (mọi
    # tier có bbox), không phải nom_idx ± 1 (ô kề theo nom_idx có thể là khe ở cả hai đời).
    if qd is not None and {"prev_bbox_cu", "next_bbox_cu"} <= set(qd.columns):
        L.append("")
        L.append("### B5a. Ô khoá QĐ-01: hàng xóm thật (record kề theo tâm y, bản mới) so với `prev/next_bbox_cu`")
        L.append("")
        import json as _json
        n = moi.loc[moi["bbox"].str.strip() != "", KEY_NOM + ["bbox"]].copy()
        n["_yc"] = n["bbox"].map(lambda b: (lambda v: (v[1] + v[3]) / 2.0)(_json.loads(b)))
        n["_ord"] = range(len(n))
        n["column"] = n["column"].map(lambda c: str(int(float(c))))
        n = n.sort_values(["book", "page", "column", "_yc", "_ord"], kind="stable")
        g = n.groupby(["book", "page", "column"], sort=False)["bbox"]
        n["_prev"] = g.shift(1).fillna(""); n["_next"] = g.shift(-1).fillna("")
        n["_k"] = n[KEY_NOM].agg("|".join, axis=1)
        nb = {k: (pv, nx) for k, pv, nx in zip(n["_k"], n["_prev"], n["_next"])}
        st = {k: 0 for k in ("prev_khop", "prev_khac", "prev_thieu_o", "next_khop", "next_khac", "next_thieu_o")}
        rows = []
        for r in qd.itertuples():
            k = "|".join([r.book, r.page, str(int(float(r.column))), str(int(float(r.nom_idx)))])
            got = nb.get(k)
            if got is None:
                st["prev_thieu_o"] += 1; st["next_thieu_o"] += 1
                rows.append({"khoa": k, "phia": "cả hai", "bbox_cu": "", "bbox_moi": "(ô khoá không có trong bản mới)"})
                continue
            for ten, kv, gv in (("prev", r.prev_bbox_cu, got[0]), ("next", r.next_bbox_cu, got[1])):
                if kv == gv:
                    st[ten + "_khop"] += 1
                else:
                    st[ten + "_khac"] += 1
                    rows.append({"khoa": k, "phia": ten, "bbox_cu": kv or "(đầu/cuối cột)", "bbox_moi": gv or "(đầu/cuối cột)"})
        D["b5a"] = st
        L.append("| phía | khớp (kể cả cùng đầu/cuối cột) | khác | ô khoá không có |")
        L.append("|---|---|---|---|")
        for ten in ("prev", "next"):
            L.append(f"| {ten} | {st[ten + '_khop']} | {st[ten + '_khac']} | {st[ten + '_thieu_o']} |")
        L.append("")
        L.append("(md5 ô khoá đã so ở B6: build dùng thẳng `prev/next_bbox_cu` khi cắt ô khoá (N5g) nên crop giữ byte "
                 "dù hàng xóm thật đổi; hàng xóm khác = hộp kề trong bản mới đã đổi — chỉ để biết, không phải lỗi khoá)")
        if rows:
            pd.DataFrame(rows).to_csv(chi_tiet_dir / "B5a_hang_xom_qd01.csv", index=False)
            L.append("")
            L.append(f"→ `{chi_tiet_dir / 'B5a_hang_xom_qd01.csv'}` ({len(rows)} dòng)")

    # ---- B7 ----
    g2r = both[(both["tier_cu"] == "GOLD") & (both["tier_moi"] == "REVIEW")]
    L.append("")
    L.append("## B7. GOLD cũ → REVIEW mới theo rule — kỳ vọng ≈1.290 (similar ≈821, ngược ≈217, direct ≈136, L1 ≈67, cột lệch ≈22)")
    L.append("")
    L += crosstab_md(g2r["rule_cu"], g2r["rule_moi"], "rule cũ", "rule mới")
    nguoi_nk = g2r[(g2r["syllable_cu"].str.lower() == "người") & ~la_qd01(g2r["rule_cu"])]
    D["b7_tong"] = len(g2r); D["b7_rule_cu"] = g2r["rule_cu"].value_counts().to_dict()
    D["b7_nguoi_ngoai_khoa"] = len(nguoi_nk)
    D["b7_qd01_cu"] = int(la_qd01(g2r["rule_cu"]).sum())
    L.append("")
    L.append(f"- tổng GOLD→REVIEW **{len(g2r)}** · 'người' ngoài khoá **{len(nguoi_nk)}** (kỳ vọng 10) · "
             f"ô QĐ-01 cũ rớt REVIEW **{D['b7_qd01_cu']}** (kỳ vọng 0 — khoá lại)")
    g2r[KEY_NOM + ["rule_cu", "rule_moi", "ocr_char_cu", "syllable_cu", "syllable_moi", "label_cu", "label_moi"]
        + (["tier_v3_moi"] if "tier_v3_moi" in g2r.columns else [])] \
        .to_csv(chi_tiet_dir / "B7_gold_sang_review.csv", index=False)
    L.append("")
    L.append(f"→ `{chi_tiet_dir / 'B7_gold_sang_review.csv'}` ({len(g2r)} dòng)")
    # GOLD cũ → không usable (mọi đường), gộp cả mất
    L.append("")
    gold_cu = m[t_cu == "GOLD"]
    L.append("GOLD cũ theo đích mới: " + ", ".join(
        f"{k}={v}" for k, v in t_moi[gold_cu.index].value_counts().items()))

    # ---- B8 ----
    L.append("")
    L.append("## B8. L1 (sửa dấu âm, cổng 2-gram ÂM–ÂM LOO) — kỳ vọng đổi ≈376 / giữ gốc (thua) ≈245 / hoà ≈134; ghi đè bản in 0")
    L.append("")
    if "l1_support" in moi.columns:
        ls = _int(moi["l1_support"]); lt = _int(moi["l1_tie"])
        la_l1 = moi["rule"].str.startswith(RULE_L1)
        kq = pd.Series("(không qua L1)", index=moi.index)
        kq[la_l1 & (ls > 0)] = "đổi (l1_support > 0)"
        kq[la_l1 & (ls <= 0)] = "ĐỔI mà support ≤ 0 = GHI ĐÈ BẢN IN"
        kq[~la_l1 & (ls < 0)] = "giữ gốc (thua)"
        kq[lt == 1] = "hoà (giữ gốc)"
        vc = kq[kq != "(không qua L1)"].value_counts()
        L += [f"| kết quả L1 (bản mới) | n |", "|---|---|"] + [f"| {k} | {v} |" for k, v in vc.items()]
        D["b8"] = vc.to_dict()
        D["b8_ghi_de"] = int((la_l1 & (ls <= 0)).sum())
        so = moi.loc[la_l1]
        if "syllable_ocr" in so.columns:
            D["b8_l1_bang_ocr"] = int((so["syllable"].str.lower() == so["syllable_ocr"].str.lower()).sum())
            L.append("")
            L.append(f"- ô L1 mà `syllable` == `syllable_ocr` (L1 không thật sự đổi so với bản in): {D['b8_l1_bang_ocr']}")
        L.append("")
        L.append("### B8a. Ô L1 cũ (755) → tier/rule mới")
        L.append("")
        l1_cu = m[r_cu == RULE_L1]
        L += crosstab_md(l1_cu["rule_cu"], l1_cu["rule_moi"].where(l1_cu["_merge"] != "left_only", "(mất)"),
                         "rule cũ", "rule mới")
        L.append("")
        L.append("### B8b. Ô L1 mới → rule cũ")
        L.append("")
        l1_moi = m[m["rule_moi"].fillna("").str.startswith(RULE_L1)]
        L += crosstab_md(l1_moi["rule_cu"].where(l1_moi["_merge"] != "right_only", "(mới)"),
                         l1_moi["rule_moi"], "rule cũ", "rule mới")
        L.append("")
        L += crosstab_md(l1_moi["syllable_cu"].where(l1_moi["_merge"] != "right_only", "(mới)") == l1_moi["syllable_moi"],
                         l1_moi["tier_cu"].where(l1_moi["_merge"] != "right_only", "(mới)"),
                         "L1 mới: âm cũ == âm mới", "tier cũ")
    else:
        L.append("_(bản mới không có cột l1_support — không phải schema v3)_")

    # ---- phân bố count_source / box_source ----
    L.append("")
    L.append("## Phân bố `count_source` / `box_source` (bản mới)")
    L.append("")
    if "count_source" in moi.columns:
        col = moi.drop_duplicates(["book", "page", "column"])
        D["count_source_cot"] = col["count_source"].value_counts().to_dict()
        D["count_source_o"] = moi["count_source"].value_counts().to_dict()
        D["box_source_o"] = moi["box_source"].value_counts().to_dict()
        n_col = len(col)
        L.append(f"Theo CỘT ({n_col} cột) — kỳ vọng equal_qn ≈75% · equal_ocr ≈17% · conflict ≈8% · legacy_locked_col (cột có QĐ-01):")
        L.append("")
        L += ["| count_source | cột | % |", "|---|---|---|"] + \
             [f"| {k} | {v} | {v / n_col:.1%} |" for k, v in col["count_source"].value_counts().items()]
        # kỳ vọng ≈75/17/8 của N3f đo trên cột chạy luật MỚI — cột khoá (luật cũ trọn gói) phải
        # loại ra khỏi mẫu số thì mới so được
        kk = col[col["count_source"] != "legacy_locked_col"]
        D["count_source_cot_khong_khoa"] = kk["count_source"].value_counts().to_dict()
        L.append("")
        L.append(f"Chỉ cột KHÔNG khoá ({len(kk)} cột — mẫu số của kỳ vọng N3f):")
        L.append("")
        L += ["| count_source | cột | % |", "|---|---|---|"] + \
             [f"| {k} | {v} | {v / max(1, len(kk)):.1%} |" for k, v in kk["count_source"].value_counts().items()]
        L.append("")
        L.append("Theo Ô:")
        L.append("")
        L += crosstab_md(moi["count_source"], moi["box_source"], "count_source", "box_source")
        L.append("")
        L += crosstab_md(moi["count_source"], moi["tier"], "count_source", "tier mới")

    # ---- khe giả ----
    L.append("")
    L.append("## Khe giả / lệch chéo — đếm trực tiếp trên cột `n_ocr == n_qn` ≥ 10 (bản mới) — kỳ vọng ≈54 cột / 0,82% cặp")
    L.append("")
    if {"n_ocr", "n_qn"} <= set(moi.columns):
        mm = moi.copy()
        mm["n_ocr"] = _int(mm["n_ocr"]); mm["n_qn"] = _int(mm["n_qn"])
        mm["nom_i"] = _int(mm["nom_idx"]); mm["syl_i"] = _int(mm["syl_idx"])
        mm["_lech"] = mm["nom_i"] != mm["syl_i"]
        g = mm.groupby(["book", "page", "column"]).agg(
            n=("nom_idx", "size"), n_ocr=("n_ocr", "first"), n_qn=("n_qn", "first"),
            lech=("_lech", "sum"), cs=("count_source", "first") if "count_source" in mm.columns else ("n_ocr", "first"))
        eq = g[(g["n_ocr"] == g["n_qn"]) & (g["n_ocr"] >= 10)]
        khe = eq[eq["n"] < eq["n_ocr"]]
        n_pairs = int(eq["n"].sum()); n_lech = int(eq["lech"].sum())
        D["khe_gia_cot"] = len(khe); D["khe_gia_tren"] = len(eq)
        D["lech_cheo"] = n_lech; D["lech_cheo_tren"] = n_pairs
        L.append(f"- cột khớp số ≥10: **{len(eq)}** · cột có khe giả (số ô ghép < n_ocr): **{len(khe)}** "
                 f"({len(khe) / max(1, len(eq)):.2%}) · cặp lệch chéo (nom_idx ≠ syl_idx): **{n_lech}/{n_pairs}** "
                 f"({n_lech / max(1, n_pairs):.2%})")
        if "count_source" in mm.columns:
            L.append("")
            L += crosstab_md(eq["cs"], (eq["n"] < eq["n_ocr"]).map({True: "có khe", False: "không"}),
                             "count_source (cột khớp số ≥10)", "khe giả")
        khe.reset_index().to_csv(chi_tiet_dir / "khe_gia_cot.csv", index=False)
        L.append("")
        L.append(f"→ `{chi_tiet_dir / 'khe_gia_cot.csv'}`")
    else:
        L.append("_(bản mới không có n_ocr/n_qn)_")

    return L, D


def bang_ky_vong(D: dict, Q: dict | None) -> list[str]:
    """Bảng 'đo được ↔ kỳ vọng' (DANH_MUC §4 / FLOW §4) — chỉ số, KHÔNG giải thích;
    giải thích viết tay dưới báo cáo (tool ghi đè phần máy, giữ nguyên phần tay)."""
    def trong(x, lo, hi):
        return "đạt" if lo <= x <= hi else "LỆCH"
    rows = []
    u = D.get("usable_moi", 0)
    rows.append(("Ô dùng được (usable)", f"{u}", "70.100–70.450", trong(u, 70100, 70450)))
    g = D.get("b2_doi_ghep", {}).get("GOLD", 0); r = D.get("b2_doi_ghep", {}).get("REVIEW", 0)
    rows.append(("B2 đổi âm ghép GOLD", f"{g}", "≈439 ± 20", trong(g, 419, 459)))
    rows.append(("B2 đổi âm ghép REVIEW", f"{r}", "≈780", trong(r, 700, 860)))
    rows.append(("B3 ô mới / ô mất", f"+{D.get('n_moi', 0)} / −{D.get('n_mat', 0)}", "≈ +660 / −210", "—"))
    rows.append(("B3 GOLD mất", f"{D.get('b3_gold_mat', 0)}", "≈18 (liệt kê)", "—"))
    rows.append(("B4 bbox đổi (usable)", f"{D.get('b4_bbox_doi_usable', 0)}", "≈11,5k", "—"))
    rows.append(("B5 md5 đổi (usable)", f"{D.get('b5_md5_doi_usable', 0)}", "≤14.117", trong(D.get('b5_md5_doi_usable', 0), 0, 14117)))
    rows.append(("B5 md5 đổi ở ô QĐ-01", f"{D.get('b5_md5_doi_qd01', 0)}", "0", trong(D.get('b5_md5_doi_qd01', 0), 0, 0)))
    if "b5a" in D:
        st = D["b5a"]; kh = st["prev_khac"] + st["next_khac"]
        rows.append(("B5a ô khoá: hàng xóm thật khác prev/next_cu", f"{kh} (prev {st['prev_khac']} / next {st['next_khac']})",
                     "liệt kê (crop khoá vẫn giữ byte — xem B6 md5)", "—"))
    if Q is not None:
        rows.append(("B6 QĐ-01 locked + pending", f"{Q['locked']} + {Q['pending']} = {Q['locked'] + Q['pending']}",
                     "2.014; pending ≤ 10", trong(Q["locked"] + Q["pending"], 2014, 2014) if Q["pending"] <= 10 else "LỆCH"))
        if Q.get("bbox_khop") is not None:
            rows.append(("B6 bbox_cu == bbox mới", f"{Q['bbox_khop']}/{Q['n']}", "2.014/2.014 (bắt buộc)",
                         trong(Q["bbox_khop"], Q["n"], Q["n"])))
        if Q.get("md5_khop") is not None:
            rows.append(("B6 md5_cu == md5 mới", f"{Q['md5_khop']}/{Q['n']}", "2.014 (kỳ vọng)", "—"))
    rows.append(("B7 GOLD→REVIEW", f"{D.get('b7_tong', 0)}", "≈1.290", trong(D.get("b7_tong", 0), 1150, 1450)))
    for ten, lo, hi, k in (("similar", 740, 900, "s1_inter_s2_similar"), ("ngược", 180, 260, "s1_inter_s2_similar_nguoc"),
                           ("direct", 100, 175, "s1_inter_s2_direct"), ("L1", 45, 90, RULE_L1),
                           ("cột lệch", 10, 35, "s1_inter_s2_similar_cot_lech")):
        v = D.get("b7_rule_cu", {}).get(k, 0)
        rows.append((f"B7 · rule cũ {ten}", f"{v}", f"≈{(lo + hi) // 2}", trong(v, lo, hi)))
    rows.append(("B7 'người' ngoài khoá", f"{D.get('b7_nguoi_ngoai_khoa', 0)}", "10", "—"))
    rows.append(("B7 QĐ-01 cũ rớt REVIEW", f"{D.get('b7_qd01_cu', 0)}", "0 (khoá lại)", trong(D.get("b7_qd01_cu", 0), 0, 0)))
    b8 = D.get("b8", {})
    rows.append(("B8 L1 đổi / giữ gốc / hoà",
                 f"{b8.get('đổi (l1_support > 0)', 0)} / {b8.get('giữ gốc (thua)', 0)} / {b8.get('hoà (giữ gốc)', 0)}",
                 "≈376 / 245 / 134", "—"))
    rows.append(("B8 ghi đè bản in", f"{D.get('b8_ghi_de', 0)}", "0", trong(D.get("b8_ghi_de", 0), 0, 0)))
    cs = D.get("count_source_cot", {}); n_col = sum(cs.values()) or 1
    rows.append(("count_source theo cột (mọi cột)",
                 " · ".join(f"{k} {v / n_col:.1%}" for k, v in cs.items()),
                 "legacy_locked_col = cột có QĐ-01", "—"))
    ck = D.get("count_source_cot_khong_khoa", {}); n_ck = sum(ck.values()) or 1
    eq = ck.get("equal_qn", 0) / n_ck
    rows.append(("count_source theo cột KHÔNG khoá",
                 " · ".join(f"{k} {v / n_ck:.1%}" for k, v in ck.items()),
                 "equal_qn ≈75% · equal_ocr ≈17% · conflict ≈8%", trong(eq, 0.70, 0.80)))
    rows.append(("Khe giả (cột n_ocr==n_qn ≥10)", f"{D.get('khe_gia_cot', 0)}/{D.get('khe_gia_tren', 0)}", "≈54", "—"))
    lc = D.get("lech_cheo", 0); lt = D.get("lech_cheo_tren", 0) or 1
    rows.append(("Cặp lệch chéo", f"{lc}/{lt} ({lc / lt:.2%})", "0,82%", "—"))
    out = ["## Đối chiếu kỳ vọng (DANH_MUC §4 / FLOW §4)", "",
           "| chỉ tiêu | đo được | kỳ vọng | |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    return out



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.doi_soat_the_he",
                                 description="So hai thế hệ labels*.csv theo khoá ô")
    ap.add_argument("--old", required=True, help="labels*.csv thế hệ cũ (đã giao nộp)")
    ap.add_argument("--new", required=True, help="labels*.csv thế hệ mới (vừa dựng)")
    ap.add_argument("--qd01", default=None,
                    help="config/qd01_cells.csv (N0d) — báo locked/pending theo bản mới")
    ap.add_argument("--out", default=None, help="ghi báo cáo markdown ra tệp này")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 nếu có lệch ngoài S3/SILVER hoặc QĐ-01 mất/đổi")
    ap.add_argument("--old-idx", default=None,
                    help="labels_final tái lập HEAD (N0c, có nom_idx/syl_idx) để cấp chỉ số cho bản cũ "
                         "qua 5 khoá (book,page,column,bbox,image_md5) — bản 64.525 không có nom_idx")
    ap.add_argument("--chi-tiet-dir", default=None,
                    help="thư mục ghi CSV liệt kê dài của khối v3 (mặc định <thư mục bản mới>/doi_soat)")
    args = ap.parse_args(argv)

    cu = doc(Path(args.old)); moi = doc(Path(args.new))
    if args.old_idx:
        cu = nap_old_idx(cu, Path(args.old_idx))
    chi_cot_cu = sorted(set(cu.columns) - set(moi.columns))
    chi_cot_moi = sorted(set(moi.columns) - set(cu.columns))

    R = doi_soat(cu, moi, KEY_BBOX)
    R2 = doi_soat(cu, moi, KEY_NOM) if (co_nom_idx(cu) and co_nom_idx(moi)) else None
    Q = None
    if args.qd01:
        qp = Path(args.qd01)
        if qp.exists():
            Q = qd01_khoa(pd.read_csv(qp, dtype=str, keep_default_na=False), moi)
        else:
            print(f"[doi_soat] không thấy {qp} — bỏ qua mục QĐ-01 khoá", file=sys.stderr)

    L, chi_tiet, dat = bao_cao(args, R, R2, Q)
    if chi_cot_cu or chi_cot_moi:
        # chèn SAU khi đã có chi_tiet -> dịch chỉ số đi 1
        L.insert(2, f"- Cột chỉ có ở bản cũ: {chi_cot_cu or '—'} · chỉ có ở bản mới: {chi_cot_moi or '—'}")
        chi_tiet = {i + 1 if i >= 2 else i for i in chi_tiet}
    # ---- khối v3 (N15 B1–B8): chỉ khi hai bên có nom_idx và bản mới mang schema v3 ----
    if R2 is not None and "count_source" in moi.columns:
        qd_df = pd.read_csv(Path(args.qd01), dtype=str, keep_default_na=False) \
            if args.qd01 and Path(args.qd01).exists() else None
        ctd = Path(args.chi_tiet_dir) if args.chi_tiet_dir else Path(args.new).resolve().parent / "doi_soat"
        L3, D = doi_soat_v3(cu, moi, qd_df, ctd)
        L += ["", "_(Tiêu chí 'Đạt' ở trên là của phép TÁI LẬP (0 lệch, N0c). Bản mới là ĐỜI MỚI (v3) — đánh giá "
              "theo khối B1–B8 và bảng 'Đối chiếu kỳ vọng' bên dưới.)_"]
        L += [""] + L3 + [""] + bang_ky_vong(D, Q)

    # in tóm tắt ra stdout (bảng liệt kê từng ô chỉ ở markdown)
    for i, line in enumerate(L):
        if i in chi_tiet or not line.strip():
            continue
        print(line)
    if args.out:
        op = Path(args.out); op.parent.mkdir(parents=True, exist_ok=True)
        # phần VIẾT TAY (giải thích lệch, ký tên) đứng sau dấu mốc — chạy lại tool chỉ ghi đè phần máy
        moc = "<!-- VIET_TAY:START -->"
        duoi = ""
        if op.exists():
            cu_txt = op.read_text(encoding="utf-8")
            if moc in cu_txt:
                duoi = "\n" + cu_txt[cu_txt.index(moc):]
        op.write_text("\n".join(L) + "\n" + duoi, encoding="utf-8")
        print(f"\n[doi_soat] báo cáo -> {op}")
    return 0 if (dat or not args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
