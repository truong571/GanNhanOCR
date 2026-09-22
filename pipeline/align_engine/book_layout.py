"""Bố cục trang theo sách (số cột Nôm, kiểu trang) — tham số hoá con số 9 của STT.

Khoá tuỳ chọn trong `books:` của config/pipeline.yaml (vắng = hành vi STT cũ):

    layout: stt | lithograph        # mặc định "stt" (1 tầng, 9 cột QN đánh số)
    n_columns: 9                    # số cột Nôm/QN kỳ vọng mỗi trang (mặc định 9)
    qn_syllables_per_column: 14     # chỉ dùng khi layout=lithograph: số âm tiết mỗi cột
                                    # (1 cột = 1 cặp lục bát 6⧺8); 0 = không kiểm
    det_xmargin: 0.05               # biên x lọc hộp thô detector theo cột (± phần bề rộng
                                    # cột), ghi đè step2.det_xmargin CHO SÁCH NÀY. Vắng:
                                    # stt -> None (= step2.det_xmargin toàn cục, 0,25);
                                    # lithograph -> LITHO_DET_XMARGIN (0,05).
    det_thr: 0.15                   # ngưỡng tin cậy CenterNet cho sách này; vắng -> None
                                    # (= step2.det_thr toàn cục). Chỉ luật --box-rule syl_index
                                    # đọc hai khoá này (legacy giữ trọn gói 0,3 / ±0,5w).

Nguyên tắc: `book_layout({})` == `book_layout(None)` == BookLayout() == hành vi STT
(n_columns 9, không cổng lithograph, det_xmargin/det_thr None = toàn cục). Không sách
STT nào khai báo khoá này nên labels.csv STT không đổi byte.

Vì sao det_xmargin theo sách (2026-09-22, LVT1883): cửa sổ lọc hộp = x_range ± m·w với
x_range = (min x1, max x2) của hộp kim trong cột. Trên thạch bản hộp kim rộng (w ≈
1,0–1,6 × bước cột 157 px) nên ±0,25w đưa tâm cột kề (cách 157 px) vào cửa sổ; NMS dọc
(IoU 0,45) rồi GIỮ hộp điểm cao hơn — thường là hộp cột kề — nên n_det không đổi mà hộp
bị TRÁO: cùng một hộp xuất cho 2 cột (census F1 cross-col 393 ô/105 trang). Đo 1.050 cột:
hộp lạ 607 (m 0,25) → 188 (0,15) → 87 (0,10) → 42 (0,05) → 24 (0,0); chọn 0,05 = gần
cửa sổ tâm ± pitch/2 nhất mà còn ~8 px đệm cho jitter x_range kim.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_N_COLUMNS = 9
LAYOUT_STT = "stt"
LAYOUT_LITHOGRAPH = "lithograph"
LAYOUTS = (LAYOUT_STT, LAYOUT_LITHOGRAPH)
LITHO_QN_PER_COLUMN = 14          # 6 (câu lục, tầng trên) + 8 (câu bát, tầng dưới)
LITHO_DET_XMARGIN = 0.05          # biên x mặc định cho thạch bản (xem docstring mô-đun)


@dataclass(frozen=True)
class BookLayout:
    """Bố cục một sách; mặc định = STT (9 cột, không cổng phụ)."""
    layout: str = LAYOUT_STT
    n_columns: int = DEFAULT_N_COLUMNS
    qn_per_column: int = 0        # 0 = không kiểm số âm tiết mỗi cột
    det_xmargin: float | None = None   # None = step2.det_xmargin toàn cục (STT: 0,25)
    det_thr: float | None = None       # None = step2.det_thr toàn cục (STT: 0,2)

    @property
    def is_lithograph(self) -> bool:
        return self.layout == LAYOUT_LITHOGRAPH


DEFAULT_LAYOUT = BookLayout()


def book_layout(book_cfg: dict | None) -> BookLayout:
    """Đọc khoá `layout` / `n_columns` / `qn_syllables_per_column` / `det_xmargin` /
    `det_thr` của một mục sách.

    Vắng khoá -> BookLayout() (STT). Giá trị sai kiểu/miền -> ValueError ngay
    (dừng trước khi tốn phút align), không rơi ngầm về 9. `det_xmargin`/`det_thr`
    ∈ [0, 1]; vắng -> None (= step2.* toàn cục), riêng lithograph vắng det_xmargin
    -> LITHO_DET_XMARGIN.
    """
    if not book_cfg:
        return DEFAULT_LAYOUT
    name = book_cfg.get("name", "?")
    layout = book_cfg.get("layout", LAYOUT_STT)
    if layout not in LAYOUTS:
        raise ValueError(f"books[{name}].layout = {layout!r}; chỉ nhận {LAYOUTS}")
    n_columns = book_cfg.get("n_columns", DEFAULT_N_COLUMNS)
    if isinstance(n_columns, bool) or not isinstance(n_columns, int) or n_columns < 1:
        raise ValueError(f"books[{name}].n_columns = {n_columns!r}; cần số nguyên >= 1")
    if layout == LAYOUT_LITHOGRAPH:
        qpc = book_cfg.get("qn_syllables_per_column", LITHO_QN_PER_COLUMN)
    else:
        qpc = book_cfg.get("qn_syllables_per_column", 0)
    if isinstance(qpc, bool) or not isinstance(qpc, int) or qpc < 0:
        raise ValueError(f"books[{name}].qn_syllables_per_column = {qpc!r}; cần số nguyên >= 0")
    det_xmargin = _float_key(book_cfg, name, "det_xmargin",
                             LITHO_DET_XMARGIN if layout == LAYOUT_LITHOGRAPH else None,
                             lo=0.0, hi=1.0)
    det_thr = _float_key(book_cfg, name, "det_thr", None, lo=0.0, hi=1.0)
    if (layout == LAYOUT_STT and n_columns == DEFAULT_N_COLUMNS and qpc == 0
            and det_xmargin is None and det_thr is None):
        return DEFAULT_LAYOUT
    return BookLayout(layout=layout, n_columns=n_columns, qn_per_column=qpc,
                      det_xmargin=det_xmargin, det_thr=det_thr)


def _float_key(book_cfg: dict, name: str, key: str, default: float | None,
               lo: float, hi: float) -> float | None:
    """Khoá số thực tuỳ chọn trong [lo, hi]; vắng -> default; sai kiểu/miền -> ValueError."""
    if key not in book_cfg:
        return default
    v = book_cfg[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= float(v) <= hi):
        raise ValueError(f"books[{name}].{key} = {v!r}; cần số thực trong [{lo}, {hi}]")
    return float(v)


def lithograph_gate(cols: list, qn_lines: dict, lay: BookLayout,
                    expected_counts: dict | None = None) -> tuple[bool, dict]:
    """Cổng page_ok cho layout=lithograph.

    PASS khi: số cột Nôm == n_columns, số cột QN == n_columns, và mỗi cột QN có
    đúng `expected_counts[line_id]` âm tiết (mặc định lay.qn_per_column = 14; nếu
    transcriptions/<page>.json ghi `num_syllables` cho cột đó thì lấy số ấy —
    câu 5/7/9 âm ghi rõ trong JSON vẫn qua cổng). qn_per_column = 0 -> bỏ kiểm âm.
    Trả (ok, chi tiết) để ghi vào bản ghi trang. align_production._detect còn đòi
    phương pháp cột hybrid* (projection_fallback luôn ép đúng n_columns nên đếm cột
    ở nhánh đó là tautology).
    """
    n_nom = len(cols)
    n_qn = len(qn_lines)
    bad_cols: list[dict] = []
    if lay.qn_per_column or expected_counts:
        for lid in sorted(qn_lines):
            want = (expected_counts or {}).get(lid, lay.qn_per_column)
            if not want:
                continue
            got = len(qn_lines[lid])
            if got != want:
                bad_cols.append({"column": lid, "n_syl": got, "want": want})
    ok = (n_nom == lay.n_columns and n_qn == lay.n_columns and not bad_cols)
    return ok, {"layout": lay.layout, "n_columns": lay.n_columns,
                "n_nom_cols": n_nom, "n_qn_cols": n_qn, "bad_syl_cols": bad_cols}


def expected_qn_counts(data_dir, page_name: str) -> dict[int, int]:
    """Đọc `num_syllables` từng cột trong transcriptions/<page>.json (nếu có).

    Trả {line_id (1-based): num_syllables}. Thiếu tệp/khoá -> {} (cổng dùng 14).
    Chỉ gọi cho layout=lithograph; đường STT không đọc tệp này (giữ nguyên).
    """
    import json
    from pathlib import Path
    p = Path(data_dir) / "transcriptions" / f"{page_name}.json"
    if not p.exists():
        return {}
    try:
        data = json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}
    out: dict[int, int] = {}
    for i, c in enumerate(data.get("columns") or []):
        if not isinstance(c, dict):
            continue
        n = c.get("num_syllables")
        lid = c.get("column", i + 1)
        if isinstance(n, int) and not isinstance(n, bool) and n > 0 and isinstance(lid, int):
            out[lid] = n
    return out
