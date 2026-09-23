"""Gộp các bộ giao nộp `dataset/<Bộ>/` thành MỘT bộ chung `dataset/_ALL/`.

VÌ SAO
------
Mỗi lần chạy pipeline xuất **một thư mục một bộ** (`dataset/SachThanhTruyen/`,
`dataset/LucVanTien1883/`…). Người nhận muốn *một* `labels.csv` để nạp thẳng vào
DataLoader thì phải tự ghép 6 thư mục, và chỗ ghép tay đó là nơi **tập ĐÁNH GIÁ lọt vào
tập huấn luyện** (2 bộ IHR-NomDB có nhãn người). Công cụ này ghép sẵn, **mang theo cờ**
`evaluation_only` / `split_hint` để hạ nguồn không trộn nhầm, và **không** đụng vào các
thư mục nguồn (chỉ ĐỌC).

BỘ GỘP KHÔNG PHẢI NGUỒN SỰ THẬT. Nguồn là `dataset/<Bộ>/`; `_ALL/` dựng lại được bằng
một lệnh, nên đừng sửa tay vào đây.

BA CHỖ DỄ SAI, ĐÃ CHẶN BẰNG BẤT BIẾN
------------------------------------
1. **Khoá.** `image` của bộ nguồn chỉ duy nhất *trong* bộ đó. Gộp xong, đường dẫn mang
   tiền tố bộ (`crops/<Bộ>/gold/…`) nên **không thể** đụng nhau giữa hai bộ — nhưng bộ
   STT tự nó đã có **21 đường dẫn trùng** (hai ô khác bbox dùng chung một tệp crop, do
   `self_training_rescue` cấp lại chỉ số ô). Đó là KHUYẾT TẬT CÓ THẬT của bộ đã công bố,
   không phải lỗi của phép gộp: công cụ giữ nguyên cả hai dòng, gắn cờ `image_dup=1`,
   liệt kê ra `TRUNG_ANH.csv`, và cấp khoá chính riêng `cell_uid` (duy nhất 100 %).
2. **Ảnh không có tệp.** Tầng `GOLD_text_only` có `image` nhưng **cố ý không giao ảnh**
   (cổng cơ chế B4'). Bất biến "ảnh tồn tại 100 %" chỉ tính dòng ngoài tầng này.
3. **Cờ đánh giá.** Bộ nào có `evaluation_only.json` (pipeline/tools/mark_eval_dataset.py)
   thì MỌI dòng của nó mang `evaluation_only=1`, `split_hint=eval`.

Chạy:
    .venv/bin/python -m pipeline.tools.merge_datasets --out dataset/_ALL
    .venv/bin/python -m pipeline.tools.merge_datasets --books SachThanhTruyen TruyenKieu1872 \
        --out /tmp/thu --no-crops
    .venv/bin/python -m pipeline.tools.merge_datasets --selftest
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, OrderedDict
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipeline.export_final_dataset import GIAO_NOP, TIER_TEXT_ONLY, TRACE  # noqa: E402
from pipeline.tools.mark_eval_dataset import is_marked  # noqa: E402

# Thứ tự bộ CỐ ĐỊNH -> tệp ra tái lập được từng byte. 3 quyển STT nằm chung MỘT bộ vì
# chúng là MỘT lần chạy pipeline (cột `book` = stt2|stt4|stt11 tách chúng ra).
BO_MAC_DINH = ["SachThanhTruyen", "LucVanTien1883", "KimVanKieu1884",
               "Chrestomathie1872", "LucVanTien1916", "TruyenKieu1872"]
# Tên thư mục KHÔNG phải bộ giao nộp (bản dựng cũ / chính bộ gộp).
BO_BO_QUA = ("_",)
HAU_TO_CU = ("probe",)

# Cột thêm vào so với 12 cột giao nộp. `cell_uid` là KHOÁ CHÍNH của bộ gộp.
COT_THEM = ["cell_uid", "book_set", "evaluation_only", "split_hint", "image_dup"]
COT_RA = ["cell_uid", "image", "book_set"] + [c for c in GIAO_NOP if c != "image"] \
    + ["evaluation_only", "split_hint", "image_dup"]
CROPS = "crops"
# Đúng những gì bước gộp sinh ra — chỉ chừng này được xoá khi ghi lại.
SINH_BOI_BUOC_NAY = {CROPS, "labels.csv", "labels_trace.csv", "columns.csv", "labels.xlsx",
                     "SOURCES.json", "CHECKSUMS.txt", "README.md", "DATASHEET.md",
                     "TAP_DANH_GIA.md", "TRUNG_ANH.csv", "THOI_GIAN.txt"}


# --------------------------------------------------------------------------- #
def _doc_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _uid(bo: str, r: dict, t: dict, i: int) -> str:
    """Khoá chính: (bộ, book, page, column, nom_idx, syl_idx) — đo được là duy nhất trên
    cả 140.733 dòng của 6 bộ. Thiếu nom_idx/syl_idx (bộ đời cũ) thì lùi về số thứ tự."""
    ni, si = (t or {}).get("nom_idx", ""), (t or {}).get("syl_idx", "")
    if ni == "" and si == "":
        return f"{bo}/{r.get('book','')}/{r.get('page','')}/c{r.get('column','')}/#{i}"
    return f"{bo}/{r.get('book','')}/{r.get('page','')}/c{r.get('column','')}/n{ni}/s{si}"


def kham_pha_bo(root: Path) -> list[str]:
    """Các thư mục con của `root` trông như bộ giao nộp (có labels.csv), bỏ bản dựng cũ."""
    ra = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith(BO_BO_QUA) or d.name.endswith(HAU_TO_CU):
            continue
        if "_v" in d.name and d.name.rsplit("_v", 1)[-1][:1].isdigit():
            continue
        if (d / "labels.csv").exists():
            ra.append(d.name)
    return sorted(ra, key=lambda n: (BO_MAC_DINH.index(n) if n in BO_MAC_DINH else 99, n))


# --------------------------------------------------------------------------- #
def gop(root: Path, bo_list: list[str], out: Path, che_do: str = "copy",
        xlsx: bool = True, on_log=print) -> dict:
    """Gộp `root/<bộ>/` -> `out/`. Trả bản ghi tóm tắt (đã kiểm bất biến)."""
    root, out = Path(root), Path(out)
    nguon: "OrderedDict[str, dict]" = OrderedDict()
    rows_out: list[dict] = []
    trace_out: list[dict] = []
    cols_out: list[dict] = []
    trace_cols: list[str] = []

    for bo in bo_list:
        d = root / bo
        lp = d / "labels.csv"
        if not lp.exists():
            raise FileNotFoundError(f"không thấy {lp} — bộ '{bo}' chưa được xuất?")
        L = _doc_csv(lp)
        thieu = [c for c in GIAO_NOP if c not in (L[0] if L else {})]
        if thieu:
            raise ValueError(f"{lp}: thiếu cột giao nộp {thieu}")
        tp = d / "labels_trace.csv"
        T = _doc_csv(tp) if tp.exists() else []
        if T and len(T) != len(L):
            raise ValueError(f"{tp}: {len(T)} dòng ≠ {len(L)} dòng của labels.csv")
        for c in (T[0].keys() if T else []):
            if c not in trace_cols:
                trace_cols.append(c)
        eval_only = is_marked(d)
        for i, r in enumerate(L):
            t = T[i] if T else {}
            img = f"{CROPS}/{bo}/{r['image']}" if r.get("image") else ""
            o = {c: r.get(c, "") for c in GIAO_NOP}
            o["image"] = img
            o["cell_uid"] = _uid(bo, r, t, i)
            o["book_set"] = bo
            o["evaluation_only"] = "1" if eval_only else "0"
            o["split_hint"] = "eval" if eval_only else "train"
            o["image_dup"] = "0"
            rows_out.append(o)
            tr = {"cell_uid": o["cell_uid"], "image": img}
            tr.update({c: t.get(c, "") for c in t})
            trace_out.append(tr)
        cp = d / "columns.csv"
        for cr in (_doc_csv(cp) if cp.exists() else []):
            cols_out.append({"book_set": bo, **cr})
        nguon[bo] = dict(dir=str(d.relative_to(root.parent)) if root.parent in d.parents else str(d),
                         n_dong=len(L), n_anh=sum(1 for r in L if r.get("tier") != TIER_TEXT_ONLY),
                         evaluation_only=eval_only, labels_sha256=_sha256(lp),
                         tiers={k: v for k, v in sorted(Counter(r.get("tier", "") for r in L).items())})
        on_log(f"[gộp] {bo:20s} {len(L):>7,} dòng · {nguon[bo]['n_anh']:>7,} ảnh"
               f" · evaluation_only={eval_only}")

    # --- cờ image_dup (khuyết tật kế thừa, xem docstring §1) -----------------
    dem = Counter(r["image"] for r in rows_out if r["image"])
    trung = {k for k, v in dem.items() if v > 1}
    for r in rows_out:
        if r["image"] in trung:
            r["image_dup"] = "1"

    # --- BẤT BIẾN trước khi ghi ---------------------------------------------
    bb = kiem_bat_bien(rows_out, nguon, root, bo_list)
    fail = [k for k, v in bb.items() if v is False]
    if fail:
        raise AssertionError(f"bất biến FAIL trước khi ghi: {fail}")

    # --- ghi ----------------------------------------------------------------
    # XOÁ RỒI GHI LẠI — nhưng CHỈ thứ do chính bước này sinh ra, và CHẶN thẳng nếu `--out`
    # trỏ nhầm vào một BỘ NGUỒN. Sự cố có thật ở export_final_dataset (09/09): `rmtree` thư
    # mục đích nuốt mất việc của người khác. Một lệnh dọn không có quyền xoá thứ mình không tạo.
    out.mkdir(parents=True, exist_ok=True)
    if (out / "gold").is_dir() or (out / "syllable").is_dir():
        raise ValueError(f"{out} trông như một BỘ NGUỒN (có gold/ hoặc syllable/), không phải "
                         f"bộ gộp — từ chối ghi đè. Chọn --out khác, vd dataset/_ALL")
    for m in out.iterdir():
        if m.name in SINH_BOI_BUOC_NAY:
            shutil.rmtree(m) if m.is_dir() else m.unlink()
        else:
            on_log(f"[gộp] GIỮ LẠI (không do bước này sinh): {m.name}")

    n_copy = n_thieu = 0
    if che_do != "none":
        da = set()
        for r in rows_out:
            if not r["image"] or r["tier"] == TIER_TEXT_ONLY or r["image"] in da:
                continue
            da.add(r["image"])
            bo = r["book_set"]
            src = root / bo / r["image"][len(f"{CROPS}/{bo}/"):]
            if not src.exists():
                n_thieu += 1
                continue
            dst = out / r["image"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if che_do == "link":
                os.link(src, dst)
            elif che_do == "symlink":
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            n_copy += 1

    with open(out / "labels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COT_RA, extrasaction="ignore")
        w.writeheader(); w.writerows(rows_out)
    tcols = ["cell_uid", "image"] + [c for c in TRACE if c in trace_cols and c != "image"] \
        + [c for c in trace_cols if c not in TRACE and c not in ("cell_uid", "image")]
    with open(out / "labels_trace.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tcols, extrasaction="ignore")
        w.writeheader()
        for t in trace_out:
            w.writerow({c: t.get(c, "") for c in tcols})
    if cols_out:
        ccols = ["book_set"] + [c for c in cols_out[0] if c != "book_set"]
        with open(out / "columns.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ccols, extrasaction="ignore")
            w.writeheader(); w.writerows(cols_out)
    if trung:
        with open(out / "TRUNG_ANH.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COT_RA, extrasaction="ignore")
            w.writeheader()
            w.writerows([r for r in rows_out if r["image_dup"] == "1"])

    # ẢNH TRÙNG BYTE giữa các bộ — chỉ để BÁO, không phải lỗi (hai bản in khác nhau vẫn có thể
    # cho hai crop giống hệt). So trên 12 hex ĐẦU vì bộ STT trộn md5 12 ký tự (build) với 32 ký
    # tự (self_training_rescue); cắt về 12 là cách duy nhất so được cả hai thế hệ.
    md5_bo: dict = {}
    for r in rows_out:
        h = (r.get("image_md5") or "")[:12]
        if h and r["tier"] != TIER_TEXT_ONLY:
            md5_bo.setdefault(h, set()).add(r["book_set"])
    n_md5_cheo = sum(1 for v in md5_bo.values() if len(v) > 1)

    tom = dict(ngay=date.today().isoformat(), bo=list(nguon), n_bo=len(nguon),
               n_md5_trung_cheo_bo=n_md5_cheo,
               n_dong=len(rows_out), n_anh_dong=sum(1 for r in rows_out if r["tier"] != TIER_TEXT_ONLY),
               n_text_only=sum(1 for r in rows_out if r["tier"] == TIER_TEXT_ONLY),
               n_tep_crop=n_copy, n_tep_thieu=n_thieu, n_image_trung=len(trung),
               bo_co_image_trung=sorted({r["book_set"] for r in rows_out if r["image_dup"] == "1"}),
               che_do_anh=che_do, nguon=nguon,
               tiers={k: v for k, v in sorted(Counter(r["tier"] for r in rows_out).items())},
               eval_only_dong=sum(1 for r in rows_out if r["evaluation_only"] == "1"),
               bat_bien=bb)
    (out / "SOURCES.json").write_text(json.dumps(tom, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "README.md").write_text(_readme(tom), encoding="utf-8")
    (out / "DATASHEET.md").write_text(_datasheet(tom), encoding="utf-8")
    if tom["eval_only_dong"]:
        (out / "TAP_DANH_GIA.md").write_text(_canh_bao(tom), encoding="utf-8")

    if xlsx:
        try:
            from pipeline.tools.make_xlsx import xuat
            r = xuat(out / "labels.csv", out / "labels.xlsx")
            on_log(f"[gộp] labels.xlsx — {r['dong']:,} dòng × {r['cot']} cột · {r['mb']:.1f} MB")
        except Exception as e:                                        # noqa: BLE001
            on_log(f"[gộp] bỏ qua labels.xlsx: {type(e).__name__}: {e}")

    _checksums(out)
    on_log(f"[gộp] {out}/ — {tom['n_dong']:,} dòng · {tom['n_tep_crop']:,} tệp crop "
           f"· {tom['eval_only_dong']:,} dòng evaluation_only · bất biến {sum(1 for v in bb.values() if v is True)}"
           f"/{len(bb)} PASS")
    on_log(f"[gộp] ảnh trùng BYTE giữa hai bộ khác nhau (md5 12 hex đầu, chỉ để báo): "
           f"{tom['n_md5_trung_cheo_bo']:,} giá trị md5")
    return tom


def kiem_bat_bien(rows: list[dict], nguon: dict, root: Path, bo_list: list[str]) -> dict:
    """Bất biến của phép gộp. True = PASS, False = FAIL, str = ghi chú (không chặn)."""
    bb: dict = OrderedDict()
    bb["tong_dong_bang_tong_cac_bo"] = len(rows) == sum(v["n_dong"] for v in nguon.values())
    uid = [r["cell_uid"] for r in rows]
    bb["cell_uid_duy_nhat"] = len(set(uid)) == len(uid)
    # image chỉ được trùng TRONG một bộ (khuyết tật kế thừa), KHÔNG được trùng chéo bộ
    theo_img: dict = {}
    for r in rows:
        if r["image"]:
            theo_img.setdefault(r["image"], set()).add(r["book_set"])
    bb["image_khong_trung_cheo_bo"] = all(len(v) == 1 for v in theo_img.values())
    bb["moi_dong_co_book_va_page"] = all(r.get("book") and r.get("page") for r in rows)
    bb["moi_dong_co_book_set"] = all(r["book_set"] in nguon for r in rows)
    bb["co_du_bo"] = list(nguon) == list(bo_list)
    for bo, v in nguon.items():
        rs = [r for r in rows if r["book_set"] == bo]
        if v["evaluation_only"]:
            bb[f"eval_co_co_{bo}"] = all(r["evaluation_only"] == "1" and r["split_hint"] == "eval"
                                         for r in rs)
        else:
            bb[f"khong_co_co_eval_{bo}"] = all(r["evaluation_only"] == "0" and r["split_hint"] == "train"
                                               for r in rs)
        bb[f"so_dong_{bo}"] = len(rs) == v["n_dong"]
    bb["text_only_khong_giao_anh"] = True      # kiểm ở bước copy (bỏ qua tier này)
    bb["split_hint_chi_train_hoac_eval"] = {r["split_hint"] for r in rows} <= {"train", "eval"}
    return bb


def kiem_tren_dia(out: Path) -> dict:
    """Bất biến ĐỌC LẠI TỪ ĐĨA (dùng cho `--check` và cho `run_pipeline.sh --verify`)."""
    out = Path(out)
    bb: dict = OrderedDict()
    L = _doc_csv(out / "labels.csv")
    T = _doc_csv(out / "labels_trace.csv")
    src = json.loads((out / "SOURCES.json").read_text(encoding="utf-8"))
    bb["tong_dong_bang_SOURCES"] = len(L) == src["n_dong"]
    bb["tong_dong_bang_tong_cac_bo"] = len(L) == sum(v["n_dong"] for v in src["nguon"].values())
    uid = [r["cell_uid"] for r in L]
    bb["cell_uid_duy_nhat"] = len(set(uid)) == len(uid)
    bb["trace_cung_so_dong_cung_thu_tu"] = len(T) == len(L) and all(
        a["cell_uid"] == b["cell_uid"] for a, b in zip(L, T))
    thieu = [r["image"] for r in L
             if r["image"] and r["tier"] != TIER_TEXT_ONLY and not (out / r["image"]).exists()]
    bb["anh_ton_tai_100pc"] = not thieu
    bb["so_tep_crop_dung"] = sum(1 for _ in (out / CROPS).rglob("*.png")) == src["n_tep_crop"] \
        if (out / CROPS).exists() else src["n_tep_crop"] == 0
    for bo, v in src["nguon"].items():
        rs = [r for r in L if r["book_set"] == bo]
        bb[f"so_dong_{bo}"] = len(rs) == v["n_dong"]
        if v["evaluation_only"]:
            bb[f"co_eval_{bo}"] = bool(rs) and all(r["evaluation_only"] == "1"
                                                   and r["split_hint"] == "eval" for r in rs)
    bb["co_it_nhat_1_bo_danh_gia"] = any(v["evaluation_only"] for v in src["nguon"].values())
    bb["khong_dong_train_nao_mang_co_eval"] = not [
        r for r in L if r["split_hint"] == "train" and r["evaluation_only"] == "1"]
    return bb


def _checksums(out: Path) -> None:
    lines = [f"# sha256 — bộ gộp {out.name} · {date.today().isoformat()}"]
    for p in sorted(out.glob("*")):
        if p.is_file() and p.name != "CHECKSUMS.txt":
            lines.append(f"{_sha256(p)}  {p.name}")
    (out / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _n(x) -> str:
    return f"{x:,}".replace(",", ".")


def _readme(t: dict) -> str:
    b = ["| bộ | dòng | ảnh | evaluation_only | sha256 labels.csv nguồn (16) |", "|---|---|---|---|---|"]
    for k, v in t["nguon"].items():
        b.append(f"| `{k}` | {_n(v['n_dong'])} | {_n(v['n_anh'])} | "
                 f"{'**CÓ**' if v['evaluation_only'] else 'không'} | `{v['labels_sha256'][:16]}` |")
    b.append(f"| **TỔNG** | **{_n(t['n_dong'])}** | **{_n(t['n_anh_dong'])}** | "
             f"{_n(t['eval_only_dong'])} dòng | |")
    tiers = " · ".join(f"{k} {_n(v)}" for k, v in t["tiers"].items())
    bo_eval = [k for k, v in t["nguon"].items() if v["evaluation_only"]]
    khoi_eval = "" if not bo_eval else f"""
## ⚠️ {len(bo_eval)} bộ là TẬP ĐÁNH GIÁ: {', '.join(f'`{b}`' for b in bo_eval)}

{_n(t['eval_only_dong'])} dòng mang `evaluation_only = 1` và `split_hint = eval`.
**Không** đưa chúng vào train/val của bất kỳ mô hình nào — xem `TAP_DANH_GIA.md`.
Lọc đúng một dòng lệnh:

```python
import pandas as pd
df = pd.read_csv("labels.csv", keep_default_na=False)
train = df[df.split_hint == "train"]        # {_n(t['n_dong'] - t['eval_only_dong'])} dòng
eval_ = df[df.split_hint == "eval"]         # {_n(t['eval_only_dong'])} dòng
```
"""
    bo_trung = sorted({r for r in t.get("bo_co_image_trung", [])})
    khoi_trung = "" if not t["n_image_trung"] else f"""
## Khuyết tật KẾ THỪA: {t['n_image_trung']} đường dẫn ảnh dùng chung

Trong {', '.join(f'`{b}`' for b in bo_trung) or 'bộ nguồn'}, {t['n_image_trung']} tệp crop bị
**hai ô khác nhau** (khác `bbox`, khác nhãn) cùng trỏ tới — ở bộ STT là do
`self_training_rescue` cấp lại chỉ số ô. Nghĩa là **một trong hai dòng có ảnh sai**.
Công cụ gộp KHÔNG tự chọn bên nào: giữ cả hai, gắn `image_dup=1`, liệt kê ở `TRUNG_ANH.csv`.
Ai huấn luyện nên loại `image_dup == 1` ({_n(t['n_anh_dong'] - t['n_tep_crop'] + t['n_image_trung'])} dòng) cho chắc.
"""
    return f"""# Bộ gộp chung — {t['n_bo']} bộ, {_n(t['n_dong'])} dòng

Sinh {t['ngay']} bằng `python -m pipeline.tools.merge_datasets` từ các thư mục
`dataset/<Bộ>/`. **Dựng lại được** — đừng sửa tay vào đây; sửa ở bộ nguồn rồi gộp lại.

{chr(10).join(b)}

Tầng: {tiers}.
{khoi_eval}
## Cột

| cột | nghĩa |
|---|---|
| `cell_uid` | **KHOÁ CHÍNH** `<Bộ>/<book>/<page>/c<cột>/n<nom_idx>/s<syl_idx>` — duy nhất 100 % |
| `image` | `crops/<Bộ>/<gold\\|syllable>/<tệp>.png`, tương đối với thư mục này |
| `book_set` | tên thư mục bộ nguồn ({', '.join(t['bo'])}) |
| `book` | quyển trong bộ (bộ STT có 3 quyển: `stt2`/`stt4`/`stt11`) |
| `evaluation_only` | `1` = bộ đã đóng dấu tập đánh giá (`evaluation_only.json`) |
| `split_hint` | `train` \\| `eval` — gợi ý chia, KHÔNG phải split chính thức |
| `image_dup` | `1` = đường dẫn ảnh này bị **hai dòng** dùng chung (xem `TRUNG_ANH.csv`) |
| 12 cột còn lại | y hệt `labels.csv` của bộ nguồn (schema giao nộp A-9) |

`labels_trace.csv` = sidecar chẩn đoán, cùng số dòng, khoá `cell_uid`.
`columns.csv` = một dòng mỗi cột trang, thêm `book_set`.

{khoi_trung}
## Tái lập

```bash
.venv/bin/python -m pipeline.tools.merge_datasets --out dataset/_ALL
./run_pipeline.sh --verify        # kiểm lại bất biến của bộ gộp
```
"""


def _datasheet(t: dict) -> str:
    return f"""# DATASHEET — bộ gộp {t['n_bo']} bộ ({t['ngay']})

Mọi con số dưới đây ĐỌC TỪ dữ liệu thật lúc gộp, không chép tay.

## 1. Thành phần

| | |
|---|---|
| số bộ nguồn | {t['n_bo']} |
| tổng dòng | {_n(t['n_dong'])} |
| dòng có ảnh | {_n(t['n_anh_dong'])} |
| dòng `GOLD_text_only` (nhãn, KHÔNG giao ảnh) | {_n(t['n_text_only'])} |
| tệp crop trên đĩa | {_n(t['n_tep_crop'])} |
| dòng thuộc TẬP ĐÁNH GIÁ | {_n(t['eval_only_dong'])} |
| chế độ ảnh | `{t['che_do_anh']}` |

Tầng: {' · '.join(f'{k} {_n(v)}' for k, v in t['tiers'].items())}

## 2. Vì sao "tệp crop" < "dòng có ảnh"

{_n(t['n_anh_dong'])} − {_n(t['n_tep_crop'])} = {_n(t['n_anh_dong'] - t['n_tep_crop'])}:
đúng bằng số dòng dùng CHUNG đường dẫn ảnh với một dòng khác (`image_dup=1`,
{_n(t['n_image_trung'])} nhóm × 2 dòng). Xem README §khuyết tật kế thừa.

## 3. Dùng cho việc gì

- **Được**: huấn luyện / hiệu chỉnh trên phần `split_hint == "train"`.
- **Không được**: đưa phần `eval` vào train — mọi con số precision công bố sẽ vô nghĩa.
- **Không được**: coi bộ gộp là nguồn sự thật; nguồn là `dataset/<Bộ>/`.

## 4. Giới hạn

Chỉ {2} trong {t['n_bo']} bộ có **nhãn người từng chữ** (IHR-NomDB). Các bộ còn lại
chưa có chuẩn người; số precision của chúng là khớp dị bản hoặc không có chuẩn nào —
xem `DATASHEET.md` của từng bộ nguồn và `docs/CHOT_CUOI_2026-09-23.md` §6.
"""


def _canh_bao(t: dict) -> str:
    bo = [k for k, v in t["nguon"].items() if v["evaluation_only"]]
    return f"""# ⚠️ BỘ GỘP CÓ CHỨA TẬP ĐÁNH GIÁ

{_n(t['eval_only_dong'])} trong {_n(t['n_dong'])} dòng đến từ {', '.join(f'`{b}`' for b in bo)} —
bản in **đã có nhãn chữ Nôm do NGƯỜI gán** (IHR-NomDB). Pipeline chạy lên chúng là để ĐO,
không phải để lấy dữ liệu huấn luyện.

**Lọc trước khi huấn luyện:**

```python
train = df[df.split_hint == "train"]     # hoặc df[df.evaluation_only == 0]
```

Nếu trộn, mọi phép đo "precision so nhãn người" về sau đều vô nghĩa vì mô hình đã thấy
đáp án. Chi tiết: `dataset/<Bộ>/TAP_DANH_GIA.md`, `pipeline/tools/mark_eval_dataset.py`.
"""


# ============================== SELFTEST ==================================== #
def _bo_gia(d: Path, bo: str, n: int, eval_only: bool = False, trung: int = 0) -> None:
    """Dựng một bộ giao nộp giả: labels.csv + labels_trace.csv + columns.csv + crops."""
    d = d / bo
    (d / "gold").mkdir(parents=True, exist_ok=True)
    L, T, C = [], [], []
    for i in range(n):
        # `trung` dòng cuối dùng lại đường dẫn ảnh của dòng 0 -> mô phỏng khuyết tật STT
        img = "gold/c000.png" if (trung and i >= n - trung) else f"gold/c{i:03d}.png"
        tier = "GOLD_text_only" if i == 0 and n > 2 else "GOLD"
        L.append({c: "" for c in GIAO_NOP} | dict(
            image=img, book=bo.lower(), page=f"page_{i // 3:04d}", column=str(i % 3 + 1),
            ocr_char="時", syllable="thì", label="時", unicode="U+6642", tier=tier,
            rule="s1_inter_s2_direct", bbox=f"[{i},0,{i+9},9]", image_md5=f"md5{i:04d}"))
        T.append(dict(image=img, nom_idx=str(i), syl_idx="0", crop_quality_flag="ok"))
        C.append(dict(book=bo.lower(), page=f"page_{i // 3:04d}", column=str(i % 3 + 1),
                      n_ocr="3", n_qn="3", n_det="3", count_source="ocr"))
        if tier != "GOLD_text_only":
            (d / img).write_bytes(b"\x89PNG" + bytes([i % 251]) * 8)
    for name, rows, cols in (("labels.csv", L, GIAO_NOP),
                             ("labels_trace.csv", T, ["image", "nom_idx", "syl_idx", "crop_quality_flag"]),
                             ("columns.csv", C, ["book", "page", "column", "n_ocr", "n_qn", "n_det", "count_source"])):
        with open(d / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    if eval_only:
        (d / "evaluation_only.json").write_text(
            json.dumps(dict(evaluation_only=True, book=bo)), encoding="utf-8")


def selftest() -> int:
    n = ok = 0

    def chk(name, cond, detail=""):
        nonlocal n, ok
        n += 1; ok += bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'} {name}{('  ' + str(detail)) if not cond else ''}")

    with tempfile.TemporaryDirectory() as td:
        root, out = Path(td) / "dataset", Path(td) / "out"
        _bo_gia(root, "BoA", 12)
        _bo_gia(root, "BoB", 9, trung=2)                 # 2 dòng dùng chung ảnh của dòng 0
        _bo_gia(root, "BoEval", 7, eval_only=True)
        (root / "BoA_v7").mkdir(); (root / "BoA_v7" / "labels.csv").write_text("image\n")
        (root / "_ALL").mkdir(); (root / "_ALL" / "labels.csv").write_text("image\n")

        chk("kham_pha_bo bỏ bản cũ *_v7 và thư mục _*",
            kham_pha_bo(root) == ["BoA", "BoB", "BoEval"], kham_pha_bo(root))

        t = gop(root, ["BoA", "BoB", "BoEval"], out, che_do="copy", xlsx=False, on_log=lambda *a: None)
        L = _doc_csv(out / "labels.csv")

        chk("tổng dòng = tổng các bộ (12+9+7)", len(L) == 28, len(L))
        chk("cell_uid duy nhất toàn cục", len({r["cell_uid"] for r in L}) == 28)
        chk("image không trùng CHÉO bộ",
            all(len({r["book_set"] for r in L if r["image"] == i}) == 1
                for i in {r["image"] for r in L if r["image"]}))
        thieu = [r["image"] for r in L
                 if r["image"] and r["tier"] != TIER_TEXT_ONLY and not (out / r["image"]).exists()]
        chk("ảnh tồn tại 100 % (trừ GOLD_text_only)", not thieu, thieu[:3])
        chk("GOLD_text_only KHÔNG được copy ảnh",
            not (out / f"{CROPS}/BoA/gold/c000.png").exists())
        chk("bộ đóng dấu -> evaluation_only=1 + split_hint=eval trên MỌI dòng",
            all(r["evaluation_only"] == "1" and r["split_hint"] == "eval"
                for r in L if r["book_set"] == "BoEval") and
            sum(1 for r in L if r["evaluation_only"] == "1") == 7)
        chk("bộ KHÔNG đóng dấu -> 0 dòng eval",
            not [r for r in L if r["book_set"] in ("BoA", "BoB") and r["evaluation_only"] != "0"])
        chk("image_dup gắn đúng 3 dòng của BoB (1 gốc + 2 trùng)",
            sum(1 for r in L if r["image_dup"] == "1") == 3,
            sum(1 for r in L if r["image_dup"] == "1"))
        chk("TRUNG_ANH.csv liệt kê đúng các dòng trùng",
            len(_doc_csv(out / "TRUNG_ANH.csv")) == 3)
        chk("số tệp crop = số đường dẫn ảnh DUY NHẤT có tệp",
            t["n_tep_crop"] == len({r["image"] for r in L
                                    if r["image"] and r["tier"] != TIER_TEXT_ONLY}),
            t["n_tep_crop"])
        T = _doc_csv(out / "labels_trace.csv")
        chk("labels_trace.csv cùng số dòng + cùng thứ tự cell_uid",
            len(T) == len(L) and all(a["cell_uid"] == b["cell_uid"] for a, b in zip(L, T)))
        chk("12 cột giao nộp giữ nguyên (chỉ `image` đổi tiền tố)",
            all(c in L[0] for c in GIAO_NOP) and L[0]["image"].startswith(f"{CROPS}/BoA/"))
        chk("columns.csv gộp đủ + có cột book_set",
            len(_doc_csv(out / "columns.csv")) == 28 and "book_set" in _doc_csv(out / "columns.csv")[0])
        chk("CHECKSUMS.txt có sha256 của labels.csv",
            "labels.csv" in (out / "CHECKSUMS.txt").read_text(encoding="utf-8"))
        chk("README + DATASHEET + TAP_DANH_GIA sinh đủ",
            all((out / f).exists() for f in ("README.md", "DATASHEET.md", "TAP_DANH_GIA.md")))
        chk("SOURCES.json ghi sha256 labels.csv NGUỒN của từng bộ",
            all(len(v["labels_sha256"]) == 64 for v in t["nguon"].values()))
        d = kiem_tren_dia(out)
        chk("kiem_tren_dia: mọi bất biến PASS", all(v is True for v in d.values()),
            [k for k, v in d.items() if v is not True])
        chk("gộp LẠI cho labels.csv y hệt từng byte",
            _sha256(out / "labels.csv") == (
                gop(root, ["BoA", "BoB", "BoEval"], out, che_do="copy", xlsx=False,
                    on_log=lambda *a: None) and _sha256(out / "labels.csv")))
        # thiếu bộ -> báo lỗi to, không ghi âm thầm
        try:
            gop(root, ["BoA", "KhongCo"], out, che_do="none", xlsx=False, on_log=lambda *a: None)
            chk("bộ không tồn tại -> raise", False)
        except FileNotFoundError:
            chk("bộ không tồn tại -> raise FileNotFoundError", True)
        # --out trỏ nhầm vào BỘ NGUỒN -> từ chối, KHÔNG xoá gì
        try:
            gop(root, ["BoA"], root / "BoB", che_do="none", xlsx=False, on_log=lambda *a: None)
            chk("--out trỏ vào bộ nguồn -> raise", False)
        except ValueError:
            chk("--out trỏ vào bộ nguồn (có gold/) -> raise, giữ nguyên bộ đó",
                (root / "BoB" / "labels.csv").exists() and (root / "BoB" / "gold").is_dir())
        # tệp lạ trong thư mục đích được GIỮ LẠI, không bị nuốt
        (out / "VIEC_CUA_NGUOI_KHAC.txt").write_text("giữ tôi lại", encoding="utf-8")
        gop(root, ["BoA"], out, che_do="none", xlsx=False, on_log=lambda *a: None)
        chk("tệp lạ trong --out được GIỮ LẠI",
            (out / "VIEC_CUA_NGUOI_KHAC.txt").exists())
        # thiếu cột giao nộp -> raise trước khi xoá gì
        (root / "BoHong").mkdir()
        (root / "BoHong" / "labels.csv").write_text("image,book\nx,y\n", encoding="utf-8")
        try:
            gop(root, ["BoHong"], out, che_do="none", xlsx=False, on_log=lambda *a: None)
            chk("labels.csv thiếu cột giao nộp -> raise", False)
        except ValueError:
            chk("labels.csv thiếu cột giao nộp -> raise ValueError", True)

    print(f"merge_datasets selftest: {ok}/{n}")
    return 0 if ok == n else 1


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset", help="thư mục chứa các bộ (mặc định dataset/)")
    ap.add_argument("--books", nargs="*", default=None,
                    help=f"tên thư mục bộ; mặc định {' '.join(BO_MAC_DINH)}; 'auto' = tự dò")
    ap.add_argument("--out", default="dataset/_ALL")
    ap.add_argument("--mode", choices=["copy", "link", "symlink", "none"], default="copy",
                    help="copy (mặc định, bàn giao độc lập) | link (hardlink) | symlink | none")
    ap.add_argument("--no-crops", action="store_true", help="= --mode none")
    ap.add_argument("--no-xlsx", action="store_true")
    ap.add_argument("--check", action="store_true", help="chỉ KIỂM bộ gộp đã có trên đĩa")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    out = Path(a.out)
    if a.check:
        if not (out / "labels.csv").exists():
            print(f"[gộp] không thấy {out}/labels.csv — chạy merge_datasets trước", file=sys.stderr)
            return 1
        bb = kiem_tren_dia(out)
        for k, v in bb.items():
            print(f"  {'PASS' if v is True else 'FAIL'} {k}")
        n_fail = sum(1 for v in bb.values() if v is not True)
        print(f"[gộp --check] {out}: {len(bb) - n_fail}/{len(bb)} PASS")
        return 1 if n_fail else 0

    root = Path(a.root)
    bo = a.books if a.books else BO_MAC_DINH
    if bo == ["auto"] or (a.books is None and not all((root / b / "labels.csv").exists()
                                                      for b in BO_MAC_DINH)):
        bo = kham_pha_bo(root)
        print(f"[gộp] tự dò bộ trong {root}/: {' '.join(bo)}")
    if not bo:
        print(f"[gộp] không tìm thấy bộ nào trong {root}/", file=sys.stderr)
        return 1
    # CẢNH BÁO thư mục trông-như-bộ mà KHÔNG nằm trong danh sách gộp. 23/09 đã gặp:
    # dataset/SachThanhTruyen{2,4,11}/ xuất hiện (bản tách theo quyển, không do pipeline
    # sinh) — gộp nhầm chúng vào là ĐẾM HAI LẦN toàn bộ bộ STT.
    du = [b for b in kham_pha_bo(root) if b not in bo]
    if du:
        print(f"[gộp] ⚠️  BỎ QUA {len(du)} thư mục trông như bộ nhưng KHÔNG khai trong --books: "
              f"{' '.join(du)}\n      (nếu đó là bản tách theo quyển của một bộ đã gộp thì ĐÚNG là "
              f"phải bỏ; muốn gộp thì khai bằng --books)", file=sys.stderr)
    che_do = "none" if a.no_crops else a.mode
    t = gop(root, bo, out, che_do=che_do, xlsx=not a.no_xlsx)
    return 0 if all(v is not False for v in t["bat_bien"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
