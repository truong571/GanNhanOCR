"""Tập công bố theo GOLD CHÍNH XÁC (28/09) — tách TẬP ẢNH khỏi TẬP VĂN BẢN/NHÃN trên bộ gộp `dataset/_ALL/`.

VÌ SAO. `labels.csv` của bộ gộp giao ảnh cho mọi dòng GOLD/SYLLABLE, nhưng bước gold_exact (chạy ngay sau gộp) chỉ chứng nhận
ẢNH + CHỮ cùng đúng cho các ô `gold_exact == "ok"` (ảnh giao = crop chuẩn v2). Người huấn luyện mô hình ảnh→chữ cần ĐÚNG tập đó;
các ô còn lại (text_only / uncertified / review, SYLLABLE, GOLD_text_only…) vẫn có giá trị VĂN BẢN (nhãn + âm + trạng thái) nhưng
không được đi kèm ảnh vào tập huấn luyện ảnh.

  images.csv  1 dòng / ô `gold_exact == ok`: `image` = `crops_chuan/…` (tương đối `dataset/_ALL/`), `image_md5` = md5 crop chuẩn,
              `image_128`, nhãn, âm, `evidence_level`, `policy_version`, `split` (page-disjoint), `lobo_group` (= book).
  text.csv    MỌI dòng còn lại của labels.csv: nhãn/âm/tầng + `gold_exact` (trống nếu không phải GOLD) + `ly_do_khong_anh`
              (vì sao không vào tập ảnh) + `split` — KHÔNG có cột ảnh.
  EXCLUSIONS.json / RELEASE.md  số đếm theo lý do loại (LOẠI TRỪ PHẢI ỒN ÀO: in ra màn hình từng lý do), theo bộ, split, LOBO.

Chia tập (giữ nguyên nguyên tắc của splits.py): PAGE-DISJOINT trên HỢP hai tập (một trang nằm trọn một split cho cả ảnh lẫn văn
bản, khoá trang = book_set|book|page), lớp singleton (< 2 ảnh trong tập ảnh) -> trang về train; bộ đóng dấu TẬP ĐÁNH GIÁ
(`evaluation_only`) -> `eval_only`, không bao giờ train/val/test; LOBO theo `book` (test = quyển giữ lại, bộ đánh giá không vào
train). Bất biến (fail loud): tập ảnh ⊂ GOLD ∧ ok; hai tập rời nhau và phủ đủ labels.csv; không trang/md5 crop nào vắt qua hai split;
0 dòng đánh giá trong train/val/test; mọi ảnh của tập ảnh tồn tại (tuỳ chọn) và md5 khớp gold_exact.csv.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import EVAL_SPLIT, _bucket, eval_only_mask

OUT_DIRNAME = "cong_bo"          # dataset/_ALL/cong_bo/ — bước gộp + gold_exact --publish dọn thư mục này (phụ thuộc gold_exact.csv)
IMG_COLS = ["cell_uid", "image", "image_md5", "image_128", "image_goc", "book_set", "book", "page", "column", "syllable", "label",
            "unicode", "tier", "rule", "gold_exact", "evidence_level", "policy_version", "evaluation_only", "split", "lobo_group"]
TXT_COLS = ["cell_uid", "book_set", "book", "page", "column", "syllable", "label", "unicode", "tier", "rule", "gold_exact",
            "gold_exact_reason", "evidence_level", "ly_do_khong_anh", "evaluation_only", "split", "lobo_group"]


class ReleaseError(RuntimeError):
    pass


def _rd(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def load(all_dir: Path):
    """labels.csv + gold_exact.csv của bộ gộp; kiểm gold_exact.csv phủ ĐÚNG tập ô GOLD (cùng nhãn) — lệch -> ReleaseError."""
    all_dir = Path(all_dir)
    lp, gp = all_dir / "labels.csv", all_dir / "gold_exact.csv"
    if not lp.exists():
        raise ReleaseError(f"thiếu {lp} — chạy ./run_pipeline.sh --merge trước")
    if not gp.exists():
        raise ReleaseError(f"thiếu {gp} — bước gold_exact chưa chạy sau lần gộp này (./run_pipeline.sh --merge, --gold-exact on)")
    L, E = _rd(lp), _rd(gp)
    G = L[L.tier == "GOLD"]
    if not E.cell_uid.is_unique or set(E.cell_uid) != set(G.cell_uid):
        raise ReleaseError(f"gold_exact.csv ({len(E)} dòng) KHÔNG phủ đúng tập ô GOLD của labels.csv ({len(G)}) — "
                           "bộ gộp đã đổi sau bước gold_exact; chạy lại ./run_pipeline.sh --merge")
    lab = dict(zip(G.cell_uid, G.label))
    bad = int(sum(lab[u] != l for u, l in zip(E.cell_uid, E.label)))
    if bad:
        raise ReleaseError(f"{bad} ô: nhãn gold_exact.csv ≠ labels.csv — bộ gộp đã đổi sau bước gold_exact")
    return L, E


def _page_key(df: pd.DataFrame) -> pd.Series:
    return df.book_set.astype(str) + "|" + df.book.astype(str) + "|" + df.page.astype(str)


def build(L: pd.DataFrame, E: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed: int = 42, eval_books=None):
    """-> (images, text, report). images = ô gold_exact ok; text = mọi dòng còn lại. Split page-disjoint trên hợp hai tập."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios phải cộng = 1")
    L = L.reset_index(drop=True).copy()
    Ex = E.set_index("cell_uid")
    ge = L.cell_uid.map(Ex.gold_exact).fillna("")
    gr = L.cell_uid.map(Ex.reason).fillna("")
    ev = eval_only_mask(L, eval_books=eval_books if eval_books is not None else set())
    is_img = (L.tier == "GOLD") & (ge == "ok")
    # lý do không vào tập ảnh (ồn ào): GOLD không ok -> "<trạng thái>:<lý do>"; không GOLD -> "tier:<tầng>"
    why = np.where(L.tier == "GOLD", ge + ":" + gr, "tier:" + L.tier.astype(str))
    why = np.where(is_img, "", why)
    # ---- split page-disjoint trên HỢP hai tập (một trang = một split cho cả ảnh lẫn văn bản)
    pk = _page_key(L)
    img_lab = L.label[is_img & ~ev]
    cnt = img_lab.value_counts()
    singleton = set(cnt[cnt < 2].index)
    force_train = set(pk[is_img & ~ev & L.label.isin(singleton)])
    tr_hi, va_hi = ratios[0], ratios[0] + ratios[1]
    page_split = {}
    for p in sorted(set(pk[~ev])):
        if p in force_train:
            page_split[p] = "train"; continue
        r = _bucket(p, seed)
        page_split[p] = "train" if r < tr_hi else ("val" if r < va_hi else "test")
    split = pk.map(page_split).where(~ev, EVAL_SPLIT).fillna(EVAL_SPLIT)
    L["split"] = split.values
    L["lobo_group"] = L.book.values
    L["gold_exact"], L["gold_exact_reason"], L["ly_do_khong_anh"] = ge.values, gr.values, why
    L["evaluation_only"] = np.where(ev, "1", "0")
    L["evidence_level"] = L.cell_uid.map(Ex.evidence_level).fillna("").values if "evidence_level" in Ex else ""
    # ---- tập ảnh: ảnh = crop chuẩn (gold_exact.csv), image_goc = đường ảnh gốc của labels.csv (tham khảo, KHÔNG phải ảnh giao)
    I = L[is_img].copy()
    I["image_goc"] = I.image.values
    for src, dst in (("crop_chuan", "image"), ("crop_chuan_md5", "image_md5"), ("crop_chuan_128", "image_128"),
                     ("policy_version", "policy_version")):
        I[dst] = I.cell_uid.map(Ex[src]).fillna("").values
    I = I[IMG_COLS].reset_index(drop=True)
    T = L[~is_img][TXT_COLS].reset_index(drop=True)
    rep = verify(L, I, T, is_img, ev, pk)
    rep.update(counts(L, I, T, why, is_img, ev))
    rep["split_params"] = dict(ratios=list(ratios), seed=seed, singleton_classes_forced_train=len(singleton))
    return I, T, rep


def verify(L, I, T, is_img, ev, pk) -> dict:
    """Bất biến; mỗi khoá True = PASS. ReleaseError nếu có khoá False (gọi ở run())."""
    inv = {}
    inv["tap_anh_la_GOLD_ok"] = bool(((L.tier == "GOLD") & (L.gold_exact == "ok"))[is_img].all())
    inv["hai_tap_roi_nhau_va_phu_du"] = (len(I) + len(T) == len(L)) and not (set(I.cell_uid) & set(T.cell_uid))
    inv["tap_anh_co_crop_chuan_va_md5"] = bool((I.image != "").all() and (I.image_md5.str.len() == 32).all()
                                               and I.image.str.startswith("crops_chuan/").all())
    inv["tap_van_ban_khong_co_cot_anh"] = not ({"image", "image_md5", "image_128"} & set(T.columns))
    sp = L.split
    inv["khong_trang_vat_hai_split"] = int((L.assign(_p=pk, _s=sp)[~ev.values].groupby("_p")._s.nunique() > 1).sum()) == 0
    md = I[I.split != EVAL_SPLIT]
    inv["khong_md5_crop_vat_hai_split"] = int((md.groupby("image_md5").split.nunique() > 1).sum()) == 0
    inv["danh_gia_khong_vao_train_val_test"] = not bool(sp[ev.values].isin(("train", "val", "test")).any())
    inv["moi_dong_co_split"] = bool((sp != "").all())
    return inv


def counts(L, I, T, why, is_img, ev) -> dict:
    ex = Counter(w for w in why if w)
    by_set = {}
    for bs, g in L.groupby("book_set", sort=False):
        m = L.book_set == bs
        by_set[bs] = dict(dong=int(m.sum()), anh=int((is_img & m).sum()), van_ban=int((~is_img & m).sum()),
                          evaluation_only=bool(ev[m].all()))
    sp_img = {k: int(v) for k, v in I.split.value_counts().items()}
    sp_txt = {k: int(v) for k, v in T.split.value_counts().items()}
    lobo = {}
    for b in sorted(set(L.book)):
        test = L.book == b
        lobo[b] = dict(test_anh=int((is_img & test).sum()),
                       train_anh=int((is_img & ~test & ~ev).sum()),
                       danh_gia_giu_ngoai_train=int((is_img & ~test & ev).sum()))
    return dict(n_labels=int(len(L)), n_images=int(len(I)), n_text=int(len(T)), n_eval_only_images=int((is_img & ev).sum()),
                loai_khoi_tap_anh=dict(sorted(ex.items(), key=lambda kv: -kv[1])), theo_bo=by_set,
                split_images=sp_img, split_text=sp_txt, lobo=lobo,
                classes_train=int(I[I.split == "train"].label.nunique()), classes_all=int(I.label.nunique()))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(all_dir: Path, out: Path | None = None, check_files: bool = True, ratios=(0.8, 0.1, 0.1), seed: int = 42,
        log=print) -> dict:
    """Dựng tập công bố vào out (mặc định <all_dir>/cong_bo/). Bất biến FAIL -> ReleaseError, KHÔNG ghi gì."""
    all_dir = Path(all_dir)
    out = Path(out) if out else all_dir / OUT_DIRNAME
    L, E = load(all_dir)
    eb = set(L.book[L.get("evaluation_only", pd.Series("0", index=L.index)) == "1"].str.lower())
    I, T, rep = build(L, E, ratios, seed, eval_books=eb)
    if check_files:
        miss = [p for p in I.image if not (all_dir / p).is_file()]
        rep["anh_ton_tai"] = dict(thieu=len(miss), vd=miss[:3])
        bad = 0
        for p, m in zip(I.image, I.image_md5):
            f = all_dir / p
            if f.is_file() and hashlib.md5(f.read_bytes()).hexdigest() != m:
                bad += 1
        rep["md5_crop_khop"] = dict(lech=bad)
        inv_files = {"moi_anh_ton_tai": not miss, "md5_crop_khop_gold_exact": bad == 0}
    else:
        inv_files = {}
    inv = {**{k: v for k, v in rep.items() if isinstance(v, bool)}, **inv_files}
    fail = [k for k, v in inv.items() if v is False]
    # LOẠI TRỪ PHẢI ỒN ÀO
    log(f"[công bố] labels.csv {rep['n_labels']:,} dòng -> TẬP ẢNH {rep['n_images']:,} (gold_exact = ok; trong đó "
        f"{rep['n_eval_only_images']:,} ô tập đánh giá, split eval_only) · TẬP VĂN BẢN {rep['n_text']:,}")
    for k, n in rep["loai_khoi_tap_anh"].items():
        log(f"[công bố] LOẠI khỏi tập ẢNH (chỉ vào tập văn bản): {k:58s} {n:>8,}")
    log(f"[công bố] split ảnh {rep['split_images']} · văn bản {rep['split_text']} · lớp (train/tất cả) "
        f"{rep['classes_train']}/{rep['classes_all']}")
    log(f"[công bố] bất biến: {sum(v is True for v in inv.values())}/{len(inv)} PASS" + (f" · FAIL {fail}" if fail else ""))
    if fail:
        raise ReleaseError(f"bất biến FAIL {fail} — KHÔNG ghi {out}")
    out.mkdir(parents=True, exist_ok=True)
    I.to_csv(out / "images.csv", index=False)
    T.to_csv(out / "text.csv", index=False)
    rec = dict(all_dir=str(all_dir), labels_sha256=_sha(all_dir / "labels.csv"), gold_exact_sha256=_sha(all_dir / "gold_exact.csv"),
               policy_version=sorted(set(E.policy_version)), invariants=inv, **{k: v for k, v in rep.items() if not isinstance(v, bool)})
    (out / "EXCLUSIONS.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "RELEASE.md").write_text(release_md(rec), encoding="utf-8")
    (out / "CHECKSUMS.txt").write_text("".join(f"{_sha(out / n)}  {n}\n" for n in ("images.csv", "text.csv", "EXCLUSIONS.json",
                                                                                "RELEASE.md")), encoding="utf-8")
    log(f"[công bố] -> {out}/ (images.csv, text.csv, EXCLUSIONS.json, RELEASE.md, CHECKSUMS.txt)")
    return rec


def release_md(r: dict) -> str:
    n = lambda x: f"{x:,}".replace(",", ".")
    L = ["# Tập công bố theo GOLD chính xác (`dataset/_ALL/cong_bo/`)", "",
         "Sinh bởi `python -m pipeline.publish gold-exact` (hoặc `./run_pipeline.sh … --publish`), 0 API, từ `labels.csv` + "
         "`gold_exact.csv` của bộ gộp. **Không sửa nhãn**; chỉ tách:", "",
         f"- `images.csv` — **tập ẢNH** {n(r['n_images'])} ô: chỉ ô `gold_exact = ok`; `image` = crop chuẩn (`crops_chuan/…`, "
         "tương đối `dataset/_ALL/`), `image_goc` = đường ảnh gốc trong labels.csv (chỉ tham khảo).",
         f"- `text.csv` — **tập VĂN BẢN/NHÃN** {n(r['n_text'])} dòng: mọi dòng còn lại, KHÔNG cột ảnh, kèm `gold_exact`, "
         "`gold_exact_reason`, `ly_do_khong_anh`.", "",
         f"policy gold_exact {', '.join(r['policy_version'])} · labels.csv sha256 `{r['labels_sha256'][:16]}…` · gold_exact.csv sha256 "
         f"`{r['gold_exact_sha256'][:16]}…`.", "",
         "## Loại khỏi tập ảnh (theo lý do)", "", "| lý do | dòng |", "|---|---:|"]
    L += [f"| `{k}` | {n(v)} |" for k, v in r["loai_khoi_tap_anh"].items()]
    L += ["", "## Theo bộ", "", "| bộ | dòng | ảnh | văn bản | tập đánh giá |", "|---|---:|---:|---:|---|"]
    L += [f"| {b} | {n(v['dong'])} | {n(v['anh'])} | {n(v['van_ban'])} | {'có' if v['evaluation_only'] else ''} |"
          for b, v in r["theo_bo"].items()]
    L += ["", "## Split", "",
          f"Page-disjoint (khoá trang = book_set|book|page, seed {r['split_params']['seed']}, tỉ lệ {r['split_params']['ratios']}) "
          f"trên HỢP hai tập; {r['split_params']['singleton_classes_forced_train']} lớp singleton -> train; bộ tập đánh giá -> "
          "`eval_only` (không bao giờ train/val/test).", "",
          f"- ảnh: {r['split_images']}", f"- văn bản: {r['split_text']}", "",
          "LOBO (`lobo_group` = book; test = quyển giữ lại, bộ đánh giá không vào train):", "",
          "| quyển giữ lại | test (ảnh) | train (ảnh) | ảnh bộ đánh giá giữ ngoài train |", "|---|---:|---:|---:|"]
    L += [f"| {b} | {n(v['test_anh'])} | {n(v['train_anh'])} | {n(v['danh_gia_giu_ngoai_train'])} |" for b, v in r["lobo"].items()]
    L += ["", "## Bất biến", ""] + [f"- `{k}` = {v}" for k, v in r["invariants"].items()]
    L += ["", "Độ chính xác của ô ok chỉ ĐO được ở L16/TK (nhãn người IHR; `GOLD_EXACT.md` §2); KVK/L83 ước lượng; STT/Chr suy đoán.", ""]
    return "\n".join(L)
