"""publish.py — đưa kết quả gold_exact vào bộ giao nộp (bước SAU merge_datasets, 0 API).

Ghi vào <all_dir> (dataset/_ALL):  gold_exact.csv · GOLD_EXACT.md · crops_chuan/ · crops_chuan_128/ (chỉ ô ok)
Ghi vào <root>/<book_set>/:         gold_exact.csv lọc theo bộ (không chép crop; `crop_chuan` trỏ ../_ALL/…)
Sau cùng sinh lại <all_dir>/CHECKSUMS.txt (cùng hàm của merge_datasets).

Chốt an toàn:
  - labels.csv của bộ gộp phải còn nguyên như lúc merge: sha256 hiện tại == dòng `labels.csv` trong CHECKSUMS.txt do merge ghi;
    lệch -> PublishError, KHÔNG ghi gì. Không bao giờ ghi labels.csv.
  - chỉ xoá đúng thứ bước này sinh (OUTPUTS + bản lọc theo bộ `<root>/<Bộ>/gold_exact.csv`, N2); thư mục bộ nguồn chỉ bị
    thêm/ghi đè đúng một tệp gold_exact.csv (+ khối gold_exact có marker trong README/DATASHEET của bộ đó).
  - md5 của crop đọc từ cache phải trùng md5 đã ghi lúc dựng crop (sq_md5) -> lệch = cache hỏng -> PublishError; sau khi chép,
    đọc lại TỪNG tệp ở đích và so md5 (verify_crops).
  - N3: tập ô GOLD (cột image) trong `<root>/<Bộ>/labels.csv` phải trùng đúng các dòng của bản lọc theo bộ -> lệch = bộ nguồn đã
    đổi sau lần gộp -> PublishError (không ghi bản lọc lệch).
  - số thực ghi `%.17g` (tái lập ĐÚNG từng bit các cờ so ngưỡng từ CSV).
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd

from .common import sha256_file, uid_path

OUTPUTS = ("gold_exact.csv", "GOLD_EXACT.md", "crops_chuan", "crops_chuan_128")
CROPS_DIRS = ("crops_chuan", "crops_chuan_128")


class PublishError(RuntimeError):
    pass


def checksums_labels_sha(all_dir: Path) -> str | None:
    """sha256 của labels.csv ghi trong CHECKSUMS.txt (merge_datasets._checksums: '<sha>  <tên>')."""
    f = Path(all_dir) / "CHECKSUMS.txt"
    if not f.exists():
        return None
    for ln in f.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) == 2 and p[1] == "labels.csv":
            return p[0]
    return None


def check_labels(all_dir: Path) -> str:
    all_dir = Path(all_dir)
    cur = sha256_file(all_dir / "labels.csv")
    rec = checksums_labels_sha(all_dir)
    if rec is None:
        raise PublishError(f"{all_dir}/CHECKSUMS.txt không có dòng labels.csv — chạy merge_datasets trước (bước gộp).")
    if rec != cur:
        raise PublishError(f"labels.csv của {all_dir} đã đổi so với lúc merge (sha {cur[:12]} ≠ CHECKSUMS {rec[:12]}) — "
                           "gộp lại rồi chạy gold_exact.")
    return cur


def clean(all_dir: Path, root: Path | None = None, books=None) -> list[str]:
    """Xoá ĐÚNG đầu ra cũ của bước này trong all_dir; có root -> xoá cả bản lọc theo bộ <root>/<Bộ>/gold_exact.csv (N2)."""
    gone = []
    for n in OUTPUTS:
        p = Path(all_dir) / n
        if p.is_dir():
            shutil.rmtree(p); gone.append(n + "/")
        elif p.exists():
            p.unlink(); gone.append(n)
    if root is not None:
        from .per_book import clean_per_book
        gone += clean_per_book(Path(root), Path(all_dir), books)
    return gone


def verify_crops(P: pd.DataFrame, base: Path) -> dict:
    """Đọc lại TỪNG crop chuẩn của ô ok ở đích và so md5 với cột (thay invariant hằng số cũ crop_md5_eq_cache_record)."""
    ok = P[P.gold_exact == "ok"]
    n = miss = bad = 0
    for c, m in (("crop_chuan", "crop_chuan_md5"), ("crop_chuan_128", "crop_chuan_128_md5")):
        for rel, h in zip(ok[c], ok[m]):
            n += 1
            f = Path(base) / rel
            if not rel or not h or not f.is_file():
                miss += 1
            elif hashlib.md5(f.read_bytes()).hexdigest() != h:
                bad += 1
    return dict(n=n, missing=miss, md5_bad=bad, PASS=miss == 0 and bad == 0)


def export_crops(uids, sq_md5: dict, src_sq: Path, src_128: Path, dests: list[Path]) -> tuple[dict, dict]:
    """Đọc crop chuẩn (vuông + 128) của các ô từ cache, kiểm md5 vuông == sq_md5, ghi vào <dest>/crops_chuan{,_128}/.
    Trả (md5_vuông, md5_128) theo cell_uid. dests rỗng = chỉ tính md5."""
    m1, m2 = {}, {}
    bad = []
    for u in uids:
        rel = uid_path(u)
        b1 = (Path(src_sq) / rel).read_bytes(); b2 = (Path(src_128) / rel).read_bytes()
        h1, h2 = hashlib.md5(b1).hexdigest(), hashlib.md5(b2).hexdigest()
        if sq_md5.get(u) and sq_md5[u] != h1:
            bad.append(u)
        m1[u], m2[u] = h1, h2
        for d in dests:
            for sub, b in (("crops_chuan", b1), ("crops_chuan_128", b2)):
                f = Path(d) / sub / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b)
    if bad:
        raise PublishError(f"{len(bad)} crop chuẩn trong cache có md5 ≠ md5 lúc dựng (vd {bad[0]}) — xoá cache crop_chuan, chạy lại.")
    return m1, m2


def _write_csv(df: pd.DataFrame, f: Path):
    tmp = f.with_name(f.name + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.17g")
    tmp.replace(f)


def per_book_frame(P: pd.DataFrame, bs: str, rel_all: str) -> pd.DataFrame:
    """Bản lọc theo bộ: `image` tương đối với dataset/<Bộ>/ (bỏ tiền tố crops/<Bộ>/ của bộ gộp), crop_chuan trỏ ../_ALL/."""
    B = P[P.book_set == bs].copy()
    pre = f"crops/{bs}/"
    B["image"] = [s[len(pre):] if s.startswith(pre) else s for s in B.image]
    for c in ("crop_chuan", "crop_chuan_128"):
        B[c] = [f"{rel_all}/{s}" if s else "" for s in B[c]]
    return B


def publish(P: pd.DataFrame, all_dir: Path, md_path: Path, root: Path | None = None, log=print) -> dict:
    """P = khung gold_exact.csv đã chốt (đường dẫn tương đối all_dir). Crop đã được export_crops ghi vào all_dir.
    Ghi gold_exact.csv + GOLD_EXACT.md vào all_dir, bản lọc theo bộ vào root/<book_set>/, rồi sinh lại CHECKSUMS.txt."""
    from .doc_text import refresh_book_docs
    from .per_book import gold_images
    all_dir = Path(all_dir)
    root = Path(root) if root else all_dir.parent
    sha0 = check_labels(all_dir)
    # N3: kiểm TRƯỚC khi ghi bất cứ gì — bản lọc theo bộ phải trùng tập ô GOLD của labels.csv bộ nguồn
    frames, lech = {}, {}
    for bs in sorted(P.book_set.unique()):
        d = root / bs
        if not (d / "labels.csv").exists():
            log(f"[publish] bỏ qua {d}: không có labels.csv (không phải bộ nguồn)")
            continue
        rel_all = os.path.relpath(all_dir, d).replace(os.sep, "/")
        B = per_book_frame(P, bs, rel_all)
        gi = gold_images(d / "labels.csv")
        if sorted(B.image) != gi:
            lech[bs] = dict(gold_labels=len(gi), rows=int(len(B)), only_labels=len(set(gi) - set(B.image)),
                            only_gold_exact=len(set(B.image) - set(gi)))
        frames[bs] = (d, B)
    if lech:
        raise PublishError(f"N3: tập ô GOLD của labels.csv bộ nguồn ≠ bản lọc theo bộ {lech} — bộ nguồn đã đổi sau lần gộp; "
                           "chạy ./run_pipeline.sh --merge")
    _write_csv(P, all_dir / "gold_exact.csv")
    shutil.copyfile(md_path, all_dir / "GOLD_EXACT.md")
    per_book = {}
    for bs, (d, B) in frames.items():
        _write_csv(B, d / "gold_exact.csv")
        docs = refresh_book_docs(d, present=True)
        per_book[bs] = dict(rows=int(len(B)), ok=int((B.gold_exact == "ok").sum()), path=str(d / "gold_exact.csv"),
                            gold_set_eq_labels=True, docs_refreshed=docs)
    from pipeline.tools.merge_datasets import _checksums
    _checksums(all_dir)
    sha1 = checksums_labels_sha(all_dir)
    if sha1 != sha0 or sha256_file(all_dir / "labels.csv") != sha0:
        raise PublishError("labels.csv đổi trong lúc publish — không được xảy ra.")
    n_png = {c: sum(1 for _ in (all_dir / c).rglob("*.png")) if (all_dir / c).exists() else 0 for c in CROPS_DIRS}
    return dict(all_dir=str(all_dir), labels_sha256=sha0, rows=int(len(P)), ok=int((P.gold_exact == "ok").sum()),
                n_png=n_png, per_book=per_book, gold_exact_csv_sha256=sha256_file(all_dir / "gold_exact.csv"))
