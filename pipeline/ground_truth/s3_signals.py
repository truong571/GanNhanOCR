"""Gắn 6 tín hiệu thị giác S3 (đã chấm corpus-wide) vào khung labels.

VÌ SAO CẦN MODULE NÀY
---------------------
`build_dataset.maybe_s3()` CỐ Ý bỏ qua bước so ảnh mỗi khi `ocr_char` đã nằm trong danh
sách ứng viên từ điển (`build_dataset.py`, nhánh `if p["ocr_char"] in cands: return None`).
Đó chính là luật `s1_inter_s2_direct` — chiếm **94,2% GOLD** (47.361/50.286). Hệ quả đo
được: cột `s3_cosine` trong labels.csv chỉ phủ **5,8% GOLD**, tức phần lớn nhãn GOLD chưa
từng được đối chiếu với ảnh lần nào; "98% precision" của GOLD đứng gần như hoàn toàn trên
sự đồng thuận OCR + từ điển.

`pipeline/consensus_fusion/score_s3.py --all` đã chấm bù việc đó và ghi ra
`dataset_out/fusion/s3_corpus.csv` (6 tín hiệu, phủ **100% GOLD**), nhưng kết quả chỉ được
dùng cho tầng fuse ở bước 6 — chưa bao giờ quay ngược lại bước 3/4. Module này làm đúng
một việc: **nối phần đã chấm đó trở lại khung labels** để xếp hạng nghi ngờ và lấy mẫu
audit có thể nhìn thấy nó.

Ở đây tuyệt đối KHÔNG đổi `tier`/`label` — chỉ thêm cột chẩn đoán. Việc hạ cấp GOLD chờ
ngưỡng hiệu chỉnh trên verdict NGƯỜI (bước B), vì đặt ngưỡng bằng chính tín hiệu máy rồi
lấy máy chấm máy là đúng lỗi tuần hoàn mà audit trước đã cảnh báo.

SÁU TÍN HIỆU (xem docstring của score_s3.py)
  head_cos     cosine-logit của NHÃN tại đầu ArcFace       "crop có phải nhãn không?"
  head_prob    softmax(logits)[nhãn]
  head_margin  logit[nhãn] − max(logit lớp khác); < 0 nghĩa là nhãn KHÔNG phải argmax
  head_isarg   1.0 nếu argmax == nhãn
  bank_cos     max cosine tới ngân hàng tham chiếu của nhãn (tín hiệu "production")
  mls          max-logit-score, độc lập nhãn (tín hiệu OOD/độ-giống-glyph)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "RAW_SIGNALS", "SIGNAL_COLS", "DEFAULT_CORPUS",
    "AttachReport", "load_corpus", "attach",
]

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO / "dataset_out" / "fusion" / "s3_corpus.csv"

# tên trong s3_corpus.csv -> tên cột sau khi gắn vào labels (tránh đụng `s3_cosine` cũ)
RAW_SIGNALS = {
    "head_cos": "s3_head_cos",
    "head_prob": "s3_head_prob",
    "head_margin": "s3_head_margin",
    "head_isarg": "s3_head_isarg",
    "bank_cos": "s3_bank_cos",
    "mls": "s3_mls",
}
SIGNAL_COLS = tuple(RAW_SIGNALS.values())

# Ngưỡng cảnh báo lệch thế hệ dữ liệu: s3_corpus.csv được chấm từ MỘT bản labels cụ thể.
# Nếu labels.csv được sinh lại mà corpus không được chấm lại, `image` vẫn khớp nhưng NHÃN
# bên trong đã đổi -> mọi tín hiệu đều chấm sai chữ. Bắt lỗi này ngay lúc join.
MAX_LABEL_MISMATCH = 0.001      # 0,1%


@dataclass(frozen=True)
class AttachReport:
    """Nhật ký của một lần gắn — để log và để selftest kiểm."""
    labels_rows: int
    corpus_rows: int
    matched: int
    coverage: float                 # matched / labels_rows
    label_mismatch: int             # cùng `image` nhưng khác `label` => corpus lệch thế hệ
    coverage_by_tier: dict[str, float]

    def __str__(self) -> str:
        per_tier = ", ".join(f"{t}={c:.1%}" for t, c in sorted(self.coverage_by_tier.items()))
        return (f"gắn S3: {self.matched:,}/{self.labels_rows:,} hàng ({self.coverage:.1%}); "
                f"corpus {self.corpus_rows:,}; lệch nhãn {self.label_mismatch}; [{per_tier}]")


def load_corpus(path: str | Path | None = None) -> pd.DataFrame:
    """Đọc s3_corpus.csv. Lỗi rõ ràng nếu thiếu file hoặc thiếu cột tín hiệu."""
    p = Path(path) if path is not None else DEFAULT_CORPUS
    if not p.exists():
        raise FileNotFoundError(
            f"không thấy {p}. Sinh bằng:\n"
            f"    .venv/bin/python -m pipeline.consensus_fusion.score_s3 --all")
    df = pd.read_csv(p)
    missing = [c for c in ("image", "label", *RAW_SIGNALS) if c not in df.columns]
    if missing:
        raise ValueError(f"{p} thiếu cột: {missing}")
    if df["image"].duplicated().any():
        n = int(df["image"].duplicated().sum())
        raise ValueError(f"{p} có {n} `image` trùng — join sẽ nhân bản hàng; hãy chấm lại")
    return df


def attach(
    labels: pd.DataFrame,
    corpus: pd.DataFrame | None = None,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, AttachReport]:
    """Trả về (labels + cột tín hiệu S3, báo cáo). KHÔNG sửa khung đầu vào.

    Cột thêm vào: 6 cột trong SIGNAL_COLS, cộng
      s3_signals_present  bool — hàng này có được chấm S3 corpus-wide không
      s3_head_disagree    bool — đầu ArcFace chọn chữ KHÁC nhãn (head_isarg == 0)

    strict=True (mặc định) -> ném lỗi nếu tỷ lệ lệch nhãn vượt MAX_LABEL_MISMATCH, tức
    s3_corpus.csv được chấm từ một thế hệ labels khác và mọi tín hiệu đều vô nghĩa.
    """
    if "image" not in labels.columns:
        raise ValueError("khung labels phải có cột `image`")
    corpus = load_corpus() if corpus is None else corpus

    take = corpus[["image", "label", *RAW_SIGNALS]].rename(
        columns={**RAW_SIGNALS, "label": "_s3_label"})
    # many_to_one: chỉ ràng buộc CORPUS là duy nhất theo `image` (đã kiểm ở load_corpus).
    # Bên trái KHÔNG dùng one_to_one được vì tier REVIEW có ~12.8k hàng `image` rỗng —
    # chúng không khớp gì cả và phải đi qua join nguyên vẹn.
    out = labels.merge(take, on="image", how="left", validate="many_to_one",
                       indicator="_s3_join")

    # --- chống lệch thế hệ: cùng `image` nhưng nhãn đã đổi ---------------------- #
    both = out["_s3_label"].notna() & out.get(
        "label", pd.Series(index=out.index, dtype=object)).notna()
    mism = int((both & (out["label"].astype(str) != out["_s3_label"].astype(str))).sum())
    matched = int((out["_s3_join"] == "both").sum())
    if strict and matched and mism / matched > MAX_LABEL_MISMATCH:
        raise ValueError(
            f"s3_corpus.csv LỆCH THẾ HỆ với labels: {mism}/{matched} hàng cùng `image` "
            f"nhưng khác `label` ({mism / matched:.2%} > {MAX_LABEL_MISMATCH:.1%}). "
            f"Tín hiệu đang chấm sai chữ. Chấm lại:\n"
            f"    .venv/bin/python -m pipeline.consensus_fusion.score_s3 --all")
    # "Đã được chấm" = có mặt trong corpus, KHÔNG phải "head_cos khác NaN". Phân biệt này
    # là thật: 9 hàng mang nhãn ngoài 1591 lớp của đầu ArcFace (寡, 桁, 𠁝, 𨓐) nên toàn bộ
    # head_* là NaN, nhưng bank_cos/mls vẫn tính được. Dùng head_cos làm cờ hiện diện sẽ
    # xếp nhầm chúng vào nhóm "chưa từng soi ảnh" trong khi thực ra đã soi.
    out["s3_signals_present"] = out["_s3_join"].eq("both").to_numpy()
    out["s3_head_present"] = out["s3_head_cos"].notna()
    out = out.drop(columns=["_s3_label", "_s3_join"])

    isarg = pd.to_numeric(out["s3_head_isarg"], errors="coerce")
    # NaN (nhãn ngoài từ vựng) KHÔNG phải là "bất đồng" — chỉ đánh dấu khi có tín hiệu thật.
    out["s3_head_disagree"] = isarg.notna() & (isarg == 0)

    by_tier: dict[str, float] = {}
    if "tier" in out.columns:
        by_tier = {str(t): float(g["s3_signals_present"].mean())
                   for t, g in out.groupby("tier")}

    rep = AttachReport(
        labels_rows=len(labels),
        corpus_rows=len(corpus),
        matched=matched,
        coverage=float(matched / len(labels)) if len(labels) else 0.0,
        label_mismatch=mism,
        coverage_by_tier=by_tier,
    )
    return out, rep


def audit_priority(df: pd.DataFrame) -> pd.Series:
    """Điểm ưu tiên audit theo margin, dùng cho tầng active-learning (thấp = đáng soi).

    Dùng `s3_head_margin` = logit[nhãn] − max(logit lớp khác): âm nghĩa là đầu ArcFace
    chấm một chữ KHÁC cao điểm hơn chính nhãn đang gán. Hàng chưa chấm trả về NaN để
    không bao giờ lọt vào tầng active-learning một cách âm thầm.
    """
    if "s3_head_margin" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df["s3_head_margin"], errors="coerce")
