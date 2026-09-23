"""Official group-aware splits + cross-split leakage detection.

Two configs the dataset paper reports (report §04, KMNIST/NomNaOCR pattern):

  split_page   PAGE-DISJOINT — every crop of one scanned page is in exactly one split.
               Stronger than the old column-level split (which let the F1 duplicate
               crops leak the SAME pixels across train/test).
  book_holdout LEAVE-ONE-BOOK-OUT — train on 2 books, test on the 3rd (domain shift).

Plus a cross-split near-duplicate audit: exact (image_md5) AND perceptual (dHash), so a
byte- or near-identical crop can never sit in two splits. Deterministic (seeded hash).

RÒ RỈ TẬP ĐÁNH GIÁ (2026-09-23). Hai bộ IHR-NomDB có **nhãn người từng chữ** và được
`pipeline/tools/mark_eval_dataset.py` đóng dấu `evaluation_only.json`. Chúng chỉ để ĐO,
không được nằm trong train/val/test của bất kỳ mô hình nào — nếu lọt, mọi con số
precision công bố về sau đều vô nghĩa. Mô-đun này TRƯỚC ĐÂY không biết điều đó: nó chia
mọi dòng `usable` theo trang. Nay mọi dòng thuộc bộ đã đóng dấu nhận split riêng
`EVAL_SPLIT = "eval_only"`, không bao giờ vào train/val/test, và `_verify` FAIL nếu có
một dòng nào lọt.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .hashing import dhash, hamming

__all__ = ["USABLE_TIERS", "EVAL_SPLIT", "assign_page_disjoint", "lobo_split",
           "cross_split_exact", "perceptual_duplicates", "SplitReport",
           "eval_only_books", "eval_only_mask"]

USABLE_TIERS = ("GOLD", "SILVER", "SYLLABLE")
# Nhãn split riêng cho tập ĐÁNH GIÁ. KHÔNG dùng "" (dòng không usable đã mang "") để
# phân biệt được "không chia vì REVIEW" với "không chia vì CẤM huấn luyện".
EVAL_SPLIT = "eval_only"
REPO = Path(__file__).resolve().parents[2]


def eval_only_books(dataset_root: Path | str | None = None) -> set[str]:
    """Tên quyển (chữ thường) thuộc bộ export đã đóng dấu TẬP ĐÁNH GIÁ.

    Quét `dataset/<Bộ>/evaluation_only.json` qua `mark_eval_dataset.is_marked()` — một
    nguồn sự thật duy nhất với bước đóng dấu của `run_pipeline.sh`. Trả cả tên thư mục
    lẫn trường `book` trong tệp cờ, đều hạ chữ thường, vì cột `book` của `labels.csv`
    viết thường (`lucvantien1916`) còn thư mục viết hoa (`LucVanTien1916`).
    """
    from pipeline.tools.mark_eval_dataset import MARK_JSON, is_marked
    root = Path(dataset_root) if dataset_root else REPO / "dataset"
    out: set[str] = set()
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not is_marked(d):
            continue
        out.add(d.name.lower())
        try:
            rec = json.loads((d / MARK_JSON).read_text(encoding="utf-8"))
            if rec.get("book"):
                out.add(str(rec["book"]).lower())
        except Exception:      # noqa: BLE001
            pass
    return out


def eval_only_mask(df: pd.DataFrame, eval_books: set[str] | None = None,
                   dataset_root: Path | str | None = None) -> pd.Series:
    """Dòng nào thuộc tập ĐÁNH GIÁ: cột `evaluation_only` (bộ gộp `dataset/_ALL/`) HOẶC
    cột `book`/`book_set` khớp thư mục export đã đóng dấu."""
    m = pd.Series(False, index=df.index)
    if "evaluation_only" in df.columns:
        m |= df["evaluation_only"].astype(str).str.strip().isin({"1", "True", "true"})
    books = eval_only_books(dataset_root) if eval_books is None else {b.lower() for b in eval_books}
    if books:
        for col in ("book", "book_set"):
            if col in df.columns:
                m |= df[col].fillna("").astype(str).str.lower().isin(books)
    return m


def _page_id(row) -> str:
    return f"{row['book']}|{row['page']}"


def _bucket(key: str, seed: int) -> float:
    """Deterministic uniform value in [0,1) from a group key."""
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(1 << 48)


@dataclass
class SplitReport:
    counts: dict
    pages_spanning_splits: int
    md5_spanning_splits: int
    singleton_classes_in_train: int
    perceptual_cross_split_pairs: int = 0
    violations: list = field(default_factory=list)
    eval_only_rows: int = 0

    def ok(self) -> bool:
        return (self.pages_spanning_splits == 0 and self.md5_spanning_splits == 0
                and not self.violations)

    def summary(self) -> str:
        return (f"splits={self.counts} | page-span={self.pages_spanning_splits} "
                f"md5-span={self.md5_spanning_splits} "
                f"singletons->train={self.singleton_classes_in_train} "
                f"eval_only(held out)={self.eval_only_rows} "
                f"| {'OK' if self.ok() else 'VIOLATIONS: ' + str(self.violations)}")


def assign_page_disjoint(df: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed: int = 42,
                         label_col: str = "label", eval_books: set[str] | None = None,
                         dataset_root: Path | str | None = None) -> tuple[pd.Series, SplitReport]:
    """Assign train/val/test so a whole page lands in one split.

    Singleton classes (labelled by <2 usable crops) are forced to train so val/test
    never contain a class the model has never seen. Returns (split_series, report).

    Dòng thuộc bộ đã đóng dấu TẬP ĐÁNH GIÁ nhận `EVAL_SPLIT` và bị loại khỏi mọi bước
    chia (kể cả khỏi phép đếm lớp singleton) — xem docstring đầu mô-đun.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")
    ev = eval_only_mask(df, eval_books, dataset_root)
    u = df[df["tier"].isin(USABLE_TIERS) & ~ev].copy()
    if u.empty:
        raise ValueError("no usable rows to split"
                         + (" (mọi dòng usable đều thuộc TẬP ĐÁNH GIÁ)" if ev.any() else ""))

    # classes with only one usable crop -> their page must be train
    counts = u[label_col].fillna("").value_counts()
    singleton_labels = set(counts[counts < 2].index)
    force_train_pages = set(
        u[u[label_col].isin(singleton_labels)].apply(_page_id, axis=1))

    tr_hi, va_hi = ratios[0], ratios[0] + ratios[1]
    split = pd.Series("train", index=df.index, dtype=object)
    page_split: dict[str, str] = {}
    for pid in sorted(set(u.apply(_page_id, axis=1))):
        if pid in force_train_pages:
            page_split[pid] = "train"
            continue
        r = _bucket(pid, seed)
        page_split[pid] = "train" if r < tr_hi else ("val" if r < va_hi else "test")

    pid_series = df.apply(_page_id, axis=1)
    usable_mask = df["tier"].isin(USABLE_TIERS) & ~ev
    split[usable_mask] = pid_series[usable_mask].map(page_split)
    split[~usable_mask] = ""                       # non-usable rows carry no split
    split[ev] = EVAL_SPLIT                         # TẬP ĐÁNH GIÁ: không train/val/test

    rep = _verify(df, split, singleton_labels, force_train_pages, ev)
    return split, rep


def _verify(df, split, singleton_labels, force_train_pages, ev=None) -> SplitReport:
    if ev is None:
        ev = pd.Series(False, index=df.index)
    u = df[df["tier"].isin(USABLE_TIERS) & ~ev].copy()
    u = u.assign(_s=split[u.index], _pid=u.apply(_page_id, axis=1))
    page_span = int((u.groupby("_pid")["_s"].nunique() > 1).sum())
    md5 = u["image_md5"].fillna("").astype(str)
    has = md5.str.len() > 0
    md5_span = int((u[has].assign(_m=md5[has]).groupby("_m")["_s"].nunique() > 1).sum())
    # singletons must be train
    viol = []
    bad_singleton = u[u["label"].isin(singleton_labels) & (u["_s"] != "train")]
    if len(bad_singleton):
        viol.append(f"{len(bad_singleton)} singleton-class crops not in train")
    # CỔNG RÒ RỈ: không một dòng nào của bộ ĐÁNH GIÁ được mang split huấn luyện.
    leak = int(((split[ev].isin(("train", "val", "test")))).sum()) if len(ev) else 0
    if leak:
        viol.append(f"{leak} eval-only rows leaked into train/val/test")
    counts = {k: int(v) for k, v in u["_s"].value_counts().items()}
    return SplitReport(counts=counts, pages_spanning_splits=page_span,
                       md5_spanning_splits=md5_span,
                       singleton_classes_in_train=len(singleton_labels),
                       violations=viol, eval_only_rows=int(ev.sum()))


def lobo_split(df: pd.DataFrame, holdout_book: str, eval_books: set[str] | None = None,
               dataset_root: Path | str | None = None) -> pd.Series:
    """Leave-one-book-out: test = holdout_book, train = the others (usable rows only).

    Bộ ĐÁNH GIÁ không bao giờ vào `train`; giữ holdout là chính nó thì vẫn được `test`
    (đo trên tập đánh giá là đúng mục đích của nó).
    """
    if holdout_book not in set(df["book"]):
        raise ValueError(f"unknown book {holdout_book!r}")
    ev = eval_only_mask(df, eval_books, dataset_root)
    usable = df["tier"].isin(USABLE_TIERS)
    s = pd.Series("", index=df.index, dtype=object)
    s[usable & (df["book"] == holdout_book)] = "test"
    s[usable & (df["book"] != holdout_book)] = "train"
    s[ev & (df["book"] != holdout_book)] = EVAL_SPLIT
    return s


def cross_split_exact(df: pd.DataFrame, split: pd.Series) -> int:
    """# image_md5 values that appear in >1 split (exact byte duplicates leaking)."""
    u = df[df["tier"].isin(USABLE_TIERS)].copy()
    u = u.assign(_s=split[u.index])
    md5 = u["image_md5"].fillna("").astype(str)
    has = md5.str.len() > 0
    return int((u[has].assign(_m=md5[has]).groupby("_m")["_s"].nunique() > 1).sum())


def perceptual_duplicates(paths_by_split: dict[str, list[str]], dataset_dir,
                          max_hamming: int = 4) -> list[tuple]:
    """Perceptual near-duplicate crops that straddle two splits.

    paths_by_split: {split: [relative image paths]}. Buckets dHash by 16-bit prefix and
    compares within buckets only (avoids O(n^2)). Returns list of (pathA, splitA, pathB,
    splitB, hamming). Reads images, so pass a manageable candidate set.
    """
    from pathlib import Path
    dd = Path(dataset_dir)
    entries = []
    for sp, paths in paths_by_split.items():
        for p in paths:
            h = dhash(dd / p)
            if h >= 0:                                 # -1 = unreadable (0 is valid)
                entries.append((h, p, sp))
    buckets: dict[int, list] = defaultdict(list)
    for h, p, sp in entries:
        buckets[h >> 48].append((h, p, sp))          # 16-bit prefix bucket
    out = []
    for group in buckets.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                hi, pi, si = group[i]
                hj, pj, sj = group[j]
                if si != sj and hamming(hi, hj) <= max_hamming:
                    out.append((pi, si, pj, sj, hamming(hi, hj)))
    return out
