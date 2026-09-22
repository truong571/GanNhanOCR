"""Bố cục trang theo sách (số cột Nôm, kiểu trang) — tham số hoá con số 9 của STT.

Khoá tuỳ chọn trong `books:` của config/pipeline.yaml (vắng = hành vi STT cũ):

    layout: stt | lithograph | prose  # mặc định "stt" (1 tầng, 9 cột QN đánh số)
    n_columns: 9                    # số cột Nôm/QN kỳ vọng mỗi trang (mặc định 9);
                                    # layout=prose nhận thêm "auto" (mặc định của prose):
                                    # số cột THEO TRANG = số dòng transcriptions/<page>.txt
    qn_syllables_per_column: 14     # chỉ dùng khi layout=lithograph: số âm tiết mỗi cột
                                    # (1 cột = 1 cặp lục bát 6⧺8); 0 = không kiểm
    det_xmargin: 0.05               # biên x lọc hộp thô detector theo cột (± phần bề rộng
                                    # cột), ghi đè step2.det_xmargin CHO SÁCH NÀY. Vắng:
                                    # stt -> None (= step2.det_xmargin toàn cục, 0,25);
                                    # lithograph -> LITHO_DET_XMARGIN (0,05).
    det_thr: 0.15                   # ngưỡng tin cậy CenterNet cho sách này; vắng -> None
                                    # (= step2.det_thr toàn cục). Chỉ luật --box-rule syl_index
                                    # đọc hai khoá này (legacy giữ trọn gói 0,3 / ±0,5w).
    box_decoder: pitch              # (2026-09-22, tuỳ chọn) "legacy" (mặc định) = hộp thô ở det_thr
                                    # + assign_boxes 3 nhánh; "pitch" = pitch_decode.decode_column:
                                    # ứng viên detector ở 0,05 + ô ảo chiếu mực, quy hoạch động chọn
                                    # ĐÚNG n_qn hộp theo bước cột (chỉ với --box-rule syl_index và
                                    # --reseg detector). n_det trong labels.csv VẪN là số hộp thô ở
                                    # det_thr (I5 không thành hằng đúng); box_source ghi
                                    # detector | detector_low | ink_cut; count_source = 'pitch'.

Nguyên tắc: `book_layout({})` == `book_layout(None)` == BookLayout() == hành vi STT
(n_columns 9, không cổng lithograph, det_xmargin/det_thr None = toàn cục). Không sách
STT nào khai báo khoá này nên labels.csv STT không đổi byte.

layout=prose (2026-09-22, Chrestomathie1872 — văn xuôi, số cột mỗi trang biến thiên 4–7,
cột ~21 chữ, không có số âm cố định): n_columns mặc định "auto" = số cột kỳ vọng của
trang lấy từ số dòng QN adapter ghi (mỗi dòng .txt = chuỗi âm tiết đã ghép cho 1 cột Nôm,
xem pipeline/tools/ingest_prose_book.py). Cổng page_ok (prose_gate): số cột Nôm dò được ==
số dòng QN (== n_columns nếu khai số nguyên), KHÔNG kiểm số âm/cột, và phương pháp cột
hybrid* như thạch bản. det_xmargin vắng -> PROSE_DET_XMARGIN (= 0,05, hộp kim rộng so với
bước cột 140 px như thạch bản).

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
LAYOUT_PROSE = "prose"
LAYOUTS = (LAYOUT_STT, LAYOUT_LITHOGRAPH, LAYOUT_PROSE)
LITHO_QN_PER_COLUMN = 14          # 6 (câu lục, tầng trên) + 8 (câu bát, tầng dưới)
LITHO_DET_XMARGIN = 0.05          # biên x mặc định cho thạch bản (xem docstring mô-đun)
N_COLUMNS_AUTO = "auto"           # chỉ layout=prose: số cột theo trang = số dòng QN
PROSE_DET_XMARGIN = LITHO_DET_XMARGIN   # văn xuôi in đá: hộp kim cũng rộng so với bước cột
BOX_DECODER_LEGACY = "legacy"
BOX_DECODER_PITCH = "pitch"
BOX_DECODERS = (BOX_DECODER_LEGACY, BOX_DECODER_PITCH)


@dataclass(frozen=True)
class BookLayout:
    """Bố cục một sách; mặc định = STT (9 cột, không cổng phụ)."""
    layout: str = LAYOUT_STT
    n_columns: int | str = DEFAULT_N_COLUMNS   # số nguyên, hoặc "auto" (chỉ layout=prose)
    qn_per_column: int = 0        # 0 = không kiểm số âm tiết mỗi cột
    det_xmargin: float | None = None   # None = step2.det_xmargin toàn cục (STT: 0,25)
    det_thr: float | None = None       # None = step2.det_thr toàn cục (STT: 0,2)
    box_decoder: str = BOX_DECODER_LEGACY   # "legacy" | "pitch" (pitch_decode, tuỳ chọn)

    @property
    def is_lithograph(self) -> bool:
        return self.layout == LAYOUT_LITHOGRAPH

    @property
    def is_prose(self) -> bool:
        return self.layout == LAYOUT_PROSE

    @property
    def n_columns_auto(self) -> bool:
        """True khi số cột kỳ vọng lấy theo trang (số dòng QN) — chỉ với layout=prose."""
        return self.n_columns == N_COLUMNS_AUTO


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
    n_columns = book_cfg.get("n_columns", N_COLUMNS_AUTO if layout == LAYOUT_PROSE else DEFAULT_N_COLUMNS)
    if n_columns == N_COLUMNS_AUTO:
        if layout != LAYOUT_PROSE:
            raise ValueError(f"books[{name}].n_columns = 'auto' chỉ hợp lệ với layout=prose (đang {layout!r})")
    elif isinstance(n_columns, bool) or not isinstance(n_columns, int) or n_columns < 1:
        raise ValueError(f"books[{name}].n_columns = {n_columns!r}; cần số nguyên >= 1"
                         + (" hoặc 'auto'" if layout == LAYOUT_PROSE else ""))
    if layout == LAYOUT_LITHOGRAPH:
        qpc = book_cfg.get("qn_syllables_per_column", LITHO_QN_PER_COLUMN)
    else:
        qpc = book_cfg.get("qn_syllables_per_column", 0)
    if isinstance(qpc, bool) or not isinstance(qpc, int) or qpc < 0:
        raise ValueError(f"books[{name}].qn_syllables_per_column = {qpc!r}; cần số nguyên >= 0")
    if layout == LAYOUT_PROSE and qpc:
        raise ValueError(f"books[{name}].qn_syllables_per_column = {qpc!r}; văn xuôi (prose) không có "
                         "số âm cố định mỗi cột — bỏ khoá hoặc đặt 0")
    det_xmargin = _float_key(book_cfg, name, "det_xmargin",
                             (LITHO_DET_XMARGIN if layout == LAYOUT_LITHOGRAPH
                              else PROSE_DET_XMARGIN if layout == LAYOUT_PROSE else None),
                             lo=0.0, hi=1.0)
    det_thr = _float_key(book_cfg, name, "det_thr", None, lo=0.0, hi=1.0)
    box_decoder = book_cfg.get("box_decoder", BOX_DECODER_LEGACY)
    if box_decoder not in BOX_DECODERS:
        raise ValueError(f"books[{name}].box_decoder = {box_decoder!r}; chỉ nhận {BOX_DECODERS}")
    if (layout == LAYOUT_STT and n_columns == DEFAULT_N_COLUMNS and qpc == 0
            and det_xmargin is None and det_thr is None and box_decoder == BOX_DECODER_LEGACY):
        return DEFAULT_LAYOUT
    return BookLayout(layout=layout, n_columns=n_columns, qn_per_column=qpc,
                      det_xmargin=det_xmargin, det_thr=det_thr, box_decoder=box_decoder)


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


def prose_gate(cols: list, qn_lines: dict, lay: BookLayout) -> tuple[bool, dict]:
    """Cổng page_ok cho layout=prose (văn xuôi, số cột biến thiên theo trang).

    PASS khi số cột Nôm dò được == số dòng QN của trang (adapter ghi 1 dòng .txt cho
    mỗi cột Nôm có chữ kim; cột không ghép được nội dung vẫn có dòng giữ chỗ). Nếu
    n_columns khai số nguyên thì cả hai còn phải == n_columns. KHÔNG kiểm số âm/cột
    (bad_syl_cols luôn []; giữ khoá để build_dataset gom thống kê cùng dạng thạch bản).
    align_production._detect còn đòi phương pháp cột hybrid* (không projection_fallback,
    không hybrid_no_image) như thạch bản.
    """
    n_nom = len(cols)
    n_qn = len(qn_lines)
    ok = (n_nom == n_qn and n_qn > 0)
    if not lay.n_columns_auto:
        ok = ok and n_nom == lay.n_columns
    return ok, {"layout": lay.layout, "n_columns": lay.n_columns,
                "n_nom_cols": n_nom, "n_qn_cols": n_qn, "bad_syl_cols": []}


def expected_tier_counts(data_dir, page_name: str) -> dict[int, list[int]]:
    """(box_decoder=pitch) Số âm QN MỖI TẦNG của từng cột từ transcriptions/<page>.json:
    thạch bản ghi `len_odd` (câu lục, tầng trên) và `num_syllables` → [len_odd, num − len_odd].
    Trả {line_id: [n_tầng_trên, n_tầng_dưới]}; thiếu tệp/khoá -> {} (decoder chia theo chữ kim /
    chiều cao tầng). Chỉ đọc khi box_decoder=pitch; đường STT không đọc tệp này."""
    import json
    from pathlib import Path
    p = Path(data_dir) / "transcriptions" / f"{page_name}.json"
    if not p.exists():
        return {}
    try:
        data = json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}
    out: dict[int, list[int]] = {}
    for i, c in enumerate(data.get("columns") or []):
        if not isinstance(c, dict):
            continue
        lid = c.get("column", i + 1)
        n = c.get("num_syllables")
        lo = c.get("len_odd")
        if lo is None and isinstance(c.get("verse_odd"), dict):
            lo = c["verse_odd"].get("n_syll")
        if (isinstance(n, int) and isinstance(lo, int) and not isinstance(n, bool) and not isinstance(lo, bool)
                and 0 < lo < n and isinstance(lid, int)):
            out[lid] = [lo, n - lo]
    return out


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
