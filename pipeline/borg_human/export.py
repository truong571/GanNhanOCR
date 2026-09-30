"""export.py — ghi bản giao dataset/_BORG_NHAN_NGUOI/: labels.csv, README.md, DATASHEET.md, BUILD_INFO.json, CHECKSUMS.txt."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .params import BOOKS, KEEP, KNOWN_NORMALISED, PADDLE, PROBE_PAIRS, REPO

COLS = ["cell_uid", "book", "shelfmark", "page", "folio", "source_file", "column", "idx", "sent", "char", "codepoint", "syllable",
        "kind", "bbox", "image", "image_md5", "crop_chuan", "crop_chuan_md5", "crop_chuan_flags", "one_char_ok",
        "align_conf", "post_v1", "post_v2", "agree_v2", "det_score", "keep_level", "paddle_test",
        "chuan_hoa_nguoi_phien", "khoi_trang", "nguon_nhan"]   # KHÔNG cột chia tập (A-10, 30/09)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def keep_level(r) -> str:
    if r.keep_v5 == 1:
        return "keep_v5"
    if r.keep == 1:
        return "keep"
    if r.keep_high == 1:
        return "keep_high"
    return "khong"


def norm_flag(ch: str, han: str) -> str:
    """'' | 'N~L' (chữ thuộc 4 cặp đã biết người phiên chuẩn hoá tự dạng) | 'N~L|paddle' (Paddle thấy dạng Hán L trong hộp này,
    cho 11 cặp thăm dò)."""
    if han:
        return f"{ch}~{han}|paddle"
    if ch in KNOWN_NORMALISED:
        return f"{ch}~{KNOWN_NORMALISED[ch][0]}"
    return ""


def labels_frame(A: pd.DataFrame, R: pd.DataFrame, ok1: pd.Series) -> pd.DataFrame:
    man = {}
    for b in BOOKS:
        for p in json.load(open(REPO / "prepared" / b / "manifest.json", encoding="utf-8"))["pages"]:
            man[(b, p["page_name"])] = (p["source_file"], p.get("folio", ""))
    X = A.merge(R.drop(columns=[c for c in ("char",) if c in R.columns]), on=["book", "page", "idx"], how="left")
    X["one_char_ok"] = ""
    has_cc = X.cc_status.notna()
    ok_map = dict(zip(zip(R.book, R.page, R.idx), ok1.values))
    X.loc[has_cc, "one_char_ok"] = [("1" if ok_map[(b, p, i)] else "0") for b, p, i in zip(X.book[has_cc], X.page[has_cc], X.idx[has_cc])]
    out = pd.DataFrame({
        "cell_uid": [f"BORG/{b}/{p}/i{int(i):04d}" for b, p, i in zip(X.book, X.page, X.idx)],
        "book": X.book, "shelfmark": X.book.map(lambda b: BOOKS[b]["shelfmark"]), "page": X.page,
        "folio": [man[(b, p)][1] for b, p in zip(X.book, X.page)],
        "source_file": [man[(b, p)][0] for b, p in zip(X.book, X.page)],
        "column": np.where(X.kind == "skip", "", X.col.astype(int).astype(str)),
        "idx": X.idx.astype(int), "sent": X.sent.astype(int), "char": X.char,
        "codepoint": ["U+%04X" % ord(c) for c in X.char], "syllable": X.syllable.fillna(""),
        "kind": X.kind,
        "bbox": [json.dumps([int(round(a)), int(round(b)), int(round(c)), int(round(d))]) if k != "skip" else ""
                 for a, b, c, d, k in zip(X.x1, X.y1, X.x2, X.y2, X.kind)],
        "image": X.image.fillna(""), "image_md5": X.image_md5.fillna(""),
        "crop_chuan": X.crop_chuan.fillna(""), "crop_chuan_md5": X.crop_chuan_md5.fillna(""),
        "crop_chuan_flags": X.cc_flags.fillna(""), "one_char_ok": X.one_char_ok,
        "align_conf": X.conf.round(5), "post_v1": X.post.round(4), "post_v2": X.post2.round(4),
        "agree_v2": X.agree2.astype(int), "det_score": X.det_score.round(4),
        "keep_level": [keep_level(r) for r in X.itertuples()],
        "paddle_test": X.paddle_test.fillna(""),
        "chuan_hoa_nguoi_phien": [norm_flag(c, h) for c, h in zip(X.char, X.paddle_han.fillna(""))],
        "khoi_trang": X.fold.astype(int), "nguon_nhan": "nguoi_phien",
    })
    return out[COLS]


def counts(L: pd.DataFrame) -> dict:
    lv = {b: L[L.book == b].keep_level.value_counts().to_dict() for b in BOOKS}
    tot = L.keep_level.value_counts().to_dict()
    cum = {}
    for b in list(BOOKS) + ["tong"]:
        s = L if b == "tong" else L[L.book == b]
        cum[b] = dict(cells=int(len(s)), keep_v5=int((s.keep_level == "keep_v5").sum()),
                      keep=int(s.keep_level.isin(["keep_v5", "keep"]).sum()),
                      keep_high=int(s.keep_level.isin(["keep_v5", "keep", "keep_high"]).sum()),
                      with_image=int((s.image != "").sum()), one_char_ok=int((s.one_char_ok == "1").sum()),
                      pages=int(s.groupby(["book", "page"]).ngroups))
    return dict(level_exclusive=dict(by_book=lv, total=tot), cumulative=cum)


def du(path: Path) -> dict:
    n = b = 0
    for f in path.rglob("*"):
        if f.is_file():
            n += 1
            b += f.stat().st_size
    return dict(files=n, bytes=b, mb=round(b / 1e6, 1))


def write_checksums(out: Path):
    lines = [f"# sha256 — dataset/_BORG_NHAN_NGUOI · {date.today().isoformat()} (md5 từng ảnh nằm trong labels.csv)"]
    for p in sorted(out.glob("*")):
        if p.is_file() and p.name != "CHECKSUMS.txt":
            lines.append(f"{sha256_file(p)}  {p.name}")
    (out / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(x):
    return f"{100 * x:.2f}".replace(".", ",") + " %"


def _d(x):
    return str(x).replace(".", ",")


def _n(x):
    return f"{int(x):,}".replace(",", ".")


def readme(info: dict) -> str:
    c = info["counts"]["cumulative"]
    th = info["theta"]["paddle"]["keep"]
    t1 = info["theta"].get("encoder", {}).get("T1_v2|keep", {})
    rows = "\n".join(f"| {b} ({BOOKS[b]['shelfmark']}) | {_n(c[b]['pages'])} | {_n(c[b]['cells'])} | {_n(c[b]['keep_high'])} | "
                     f"{_n(c[b]['keep'])} | {_n(c[b]['keep_v5'])} |" for b in BOOKS)
    t = c["tong"]
    return f"""# _BORG_NHAN_NGUOI — bộ crop NHÃN NGƯỜI từ 2 bản chép tay Vatican Borgiano Tonchinese

> **Nhãn ở đây là chữ Nôm do NGƯỜI PHIÊN gõ (bản phiên Excel), KHÔNG phải nhãn máy.** Máy chỉ làm một việc: GIÓNG
> chuỗi chữ người của mỗi trang với các hộp chữ trên ảnh (0 API). Bộ này TÁCH khỏi GOLD tự động (`dataset/<Bộ>/`,
> `dataset/_ALL/`) — không gộp, không dùng chung `cell_uid`.

| sách | trang | ô chữ người | keep_high | keep | keep_v5 |
|---|---:|---:|---:|---:|---:|
{rows}
| **tổng** | {_n(t['pages'])} | {_n(t['cells'])} | {_n(t['keep_high'])} | {_n(t['keep'])} | {_n(t['keep_v5'])} |

(Các cột mức là LUỸ KẾ: keep_v5 ⊂ keep ⊂ keep_high. Cột `keep_level` trong `labels.csv` ghi mức CAO NHẤT đạt được.)

**Khuyên dùng:** `keep_level == "keep_v5"` (và `one_char_ok == "1"` nếu cần ảnh đúng một chữ). Tỉ lệ trượt ±1 ô ước trên
`keep` bằng bộ đọc độc lập Paddle: **θ ≈ {_pct(th['theta'])} [CI 95 % {_pct(th['theta_ci'][0])}–{_pct(th['theta_ci'][1])}]**
(n = {_n(th['n'])} ô thử được); keep_v5 đã bỏ thêm mọi ô Paddle báo lệch nên phần trượt còn lại ≤ mức đó.
{"Đo bằng encoder v2 + glyph font (T1): θ(keep) ≈ " + _pct(t1['theta']) + " (lệch lạc quan: encoder v2 tham gia chọn keep; chỉ để đối chiếu r4)." if t1 else ""}

## Tệp

- `labels.csv` — MỌI chữ người của 2 sách ({_n(t['cells'])} dòng; ô `khong` không có ảnh). Cột chính: `cell_uid`, `book`, `page`,
  `column` (cột detector, 0 = PHẢI nhất), `idx` (vị trí chữ trong chuỗi người của trang), `char` (chữ người phiên), `syllable`
  (âm người phiên; rỗng khi câu có số chữ ≠ số âm), `bbox` [x1,y1,x2,y2] toạ độ ảnh gốc 720 px, `image` + `image_md5`
  (crop luật save_crop), `crop_chuan` + `crop_chuan_md5` (crop chuẩn v2, vuông), `align_conf` = min(hậu nghiệm căn v1, v2)
  khi hai bộ căn chọn CÙNG hộp (ngược lại 0), `keep_level` ∈ {{keep_v5, keep, keep_high, khong}}, `paddle_test`
  (`ok` / `lech±d` / rỗng = không thử được), `chuan_hoa_nguoi_phien`, `one_char_ok`, `khoi_trang`
  (5 khối trang liền nhau/sách dùng nội bộ khi dựng nguyên mẫu). Bộ này KHÔNG chia tập train/val/test.
- `crops/<book>/<page>_c<col>_<idx>.png` — crop theo luật `save_crop` của pipeline (pad 0,12, carve mực láng giềng, tighten,
  điểm ảnh ẢNH GỐC); `crops_chuan/…` — crop chuẩn v2 (`pipeline/gold_exact/crop_chuan.py`), cho mọi ô mức keep trở lên (keep_high chỉ có hộp, không kèm ảnh).
- `README.md`, `DATASHEET.md`, `BUILD_INFO.json` (tham số, số đếm, θ, kiểm tái lập), `CHECKSUMS.txt` (sha256 tệp gốc).

## Tái lập (0 API)

```bash
.venv/bin/python -m pipeline.borg_human --stage all          # ≈ 11 phút (MPS), ≤ 3 worker; bit-exact với r4/r5
.venv/bin/python scripts/measure/borg_human_eval.py          # nghiệm thu (invariants PASS/FAIL)
```
Mã: `pipeline/borg_human/` · tham số đóng băng + nguồn: `pipeline/borg_human/params.py` · tài liệu: `docs/BORG_NHAN_NGUOI_2026-09-27.md`.
"""


def datasheet(info: dict) -> str:
    th = info["theta"]["paddle"]
    c = info["counts"]["cumulative"]["tong"]
    kn = "\n".join(f"| `{n}` | `{l}` | {_pct(p)} [{_pct(ci[0])}–{_pct(ci[1])}] |" for n, (l, p, ci) in KNOWN_NORMALISED.items())
    ps = info["paddle"]
    return f"""# DATASHEET — _BORG_NHAN_NGUOI (bản {date.today().isoformat()})

## 1. Động cơ
Cần một tập crop chữ Nôm VIẾT TAY có nhãn do NGƯỜI đặt (không phải máy OCR đoán) để (a) huấn luyện/hiệu chuẩn bộ nhận dạng
và bộ kiểm ảnh↔nhãn, (b) làm nguồn nguyên mẫu viết tay độc lập với nhãn máy của các sách khác trong dự án.

## 2. Nguồn
- Ảnh: Biblioteca Apostolica Vaticana, DigiVatLib — **Borg.tonch.18** (*Sách kinh Thầy cả Bỉnh*, 529 trang văn bản) và
  **Borg.tonch.34** (*Sách Dũng Lý Hộ Thần*, 112 trang). IIIF: `{BOOKS['SachKinhThayCaBinh']['iiif']}`,
  `{BOOKS['SachDungLyHoThan']['iiif']}`. Ghi công: "Images Copyright Biblioteca Apostolica Vaticana" — chỉ dùng cho nghiên cứu,
  phân phối lại ảnh phải theo điều khoản của BAV.
- Nhãn: bản phiên Nôm + quốc ngữ theo câu (tệp Excel trong `data/<Sách>/`), chuẩn hoá bởi `pipeline/tools/ingest_borg_tonch.py`
  → `prepared/<Sách>/transcriptions/page_XXXX.json` (`nom_clean`, `qn_syllables`). Kho không ghi tên người phiên.
- Ảnh trang dùng để căn = `prepared/<Sách>/pages/` = kéo giãn xám của jpg gốc, cùng kích thước (kiểm mad = 0 trên
  {info['page_map']['n']} trang) → toạ độ hộp dùng thẳng trên ảnh gốc; crop lấy ĐIỂM ẢNH GỐC.

## 3. Cách gióng (máy chỉ căn vị trí, không đặt nhãn)
1. Hộp chữ: detector CenterNet v1 (`train_crop/detector_r34.best.pt`, huấn luyện trên Sách Thánh Truyện), thr 0,05;
   cột = cụm tâm-x hộp tin cậy (≥ 0,2), xếp PHẢI→TRÁI; ô ẢO ở khe trống lớn.
2. Chuỗi chữ người của trang (chỉ chữ Hán/Nôm) ↔ chuỗi đơn vị: DP đơn điệu (Viterbi + forward-backward) cho phép ghép hộp
   thật/ô ảo/gộp 2 hộp, bỏ hộp, bỏ chữ; phát xạ thị giác = cos giữa crop và glyph font, rồi nguyên mẫu dựng từ các KHỐI TRANG
   KHÁC (2 vòng) — làm riêng với encoder v1 (`nom-embed`) và v2 (`ArcFace`).
3. **keep**: hộp detector thật, điểm ≥ {_d(KEEP['det_min'])}, trang bỏ chữ < {_pct(KEEP['page_skip_max'])}, hậu nghiệm căn v1 ≥ {_d(KEEP['post_v1'])},
   căn v2 chọn CÙNG hộp với hậu nghiệm ≥ {_d(KEEP['post_v2'])}; **keep_high** như keep nhưng hậu nghiệm v2 ≥ {_d(KEEP['post_v2_high'])}.
4. **keep_v5** = keep − ô bộ đọc ĐỘC LẬP Paddle PP-OCRv6 (không học trên STT/Borg) thấy chữ HÀNG XÓM khớp hơn chữ nhãn
   trong cửa sổ 7 chữ ({_n(ps['keep_paddle_flagged'])} ô) − ô người phiên đã chuẩn hoá tự dạng mà Paddle thấy dạng Hán ({_n(ps['removed_normalised'])} ô:
   {', '.join(f'{k} {v}' for k, v in ps['removed_by_char'].items())}). Cửa sổ Paddle dời theo trễ đỉnh CTC đo trên nhãn người IHR
   (delay = {_d(PADDLE['delay_h'])}·h); hồ sơ Paddle đóng băng (`measure_out/_borg_human_src/`, sha256 kiểm khi chạy).

## 4. Mức tin cậy và tỉ lệ trượt ước
- θ trượt ±1 ô trên **keep** (Paddle, n = {_n(th['keep']['n'])} ô thử được): **{_pct(th['keep']['theta'])}**
  [CI 95 % bootstrap cụm trang {_pct(th['keep']['theta_ci'][0])}–{_pct(th['keep']['theta_ci'][1])}]. Trên keep_v5 phép đo này thiên lệch
  (các ô Paddle báo lệch đã bị bỏ) — đọc là "phần trượt còn lại ≤ θ(keep)"; ô không thử được (cửa sổ không đủ 7 chữ khác nhau
  hoặc chữ ngoài bảng Paddle) mang tỉ lệ chưa biết, giả định ≈ θ(keep).
- Cùng phép đo Paddle cho các mức KHÔNG kèm ảnh (chỉ có hộp, để tra cứu vị trí): keep_high {_pct(th['keep_high']['theta'])}
  [{_pct(th['keep_high']['theta_ci'][0])}–{_pct(th['keep_high']['theta_ci'][1])}] (riêng phần keep_high − keep:
  {_pct(th['keep_high_tru_keep']['theta'])}); ô thật ngoài keep_high (`khong`): {_pct(th['khong_real']['theta'])}
  [{_pct(th['khong_real']['theta_ci'][0])}–{_pct(th['khong_real']['theta_ci'][1])}] — KHÔNG dùng làm nhãn ảnh.
- Sai số khác không đo được bằng máy: người phiên gõ nhầm chữ; hộp detector cắt thiếu nét (xem `one_char_ok`).

## 5. Quy ước chuẩn hoá của người phiên (quan trọng khi dùng làm nhãn ảnh)
Người phiên đôi khi gõ chữ Nôm CHUẨN cho một hình thực tế viết theo dạng Hán/giản lược khác. Đo bằng Paddle (cận dưới π_LB
tỉ lệ hộp gõ N mà hình là L, vòng 5 q13):

| gõ (N) | hình thực tế (L) | tỉ lệ ≥ |
|---|---|---|
{kn}

Cột `chuan_hoa_nguoi_phien`: `N~L` = chữ thuộc 4 cặp trên (nhãn đúng ý người phiên, hình có thể là L); `N~L|paddle` = Paddle thấy
dạng L ngay trong hộp này (11 cặp thăm dò {''.join(PROBE_PAIRS)}); các ô này đã bị loại khỏi keep_v5.
Khi huấn luyện nhận dạng HÌNH, nên gộp N và L thành một lớp, hoặc bỏ các ô `N~L`.

## 6. Thành phần
{_n(c['cells'])} chữ người; {_n(c['with_image'])} ô có ảnh (mức keep trở lên: keep {_n(c['keep'])}, trong đó keep_v5 {_n(c['keep_v5'])}); keep_high {_n(c['keep_high'])} (chỉ hộp);
one_char_ok = 1: {_n(c['one_char_ok'])}. Không chia tập (không có cột train/val/test).

## 7. Giới hạn — ĐỌC TRƯỚC KHI DÙNG
- **Không phải nhãn máy, nhưng cũng không phải chấm từng ô**: mỗi ô là (chữ người phiên, hộp máy chọn). Sai vị trí ±1 là rủi ro
  chính (mục 4).
- **Không dùng bộ này để ĐÁNH GIÁ chính các bộ kiểm đã học trên nó**: bộ kiểm chữ viết tay LOBO-sách của `pipeline/gold_exact`
  (`models/gold_exact/hand_*_Kinh.pt` = mô hình + nguyên mẫu + hiệu chuẩn học trên SachKinhThayCaBinh; biến thể DungLy học trên
  SachDungLyHoThan) chỉ được đo trên sách KIA; mọi mô hình/nguyên mẫu dựng từ các ô này cũng vậy. Encoder v1/v2 tham gia
  CHỌN keep → mọi phép đo bằng v1/v2 trên keep lệch lạc quan.
- Hai bản chép tay Borg.tonch — không đại diện cho mộc bản/thạch bản.
- Ô `kind = virtual/merge` (ô ảo, gộp 2 hộp) không bao giờ vào keep.
"""

