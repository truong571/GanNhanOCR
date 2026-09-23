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
    detector_ckpt: train_crop/detector_r34_v2_litho.pt
                                    # (2026-09-22, tuỳ chọn) checkpoint CenterNet RIÊNG cho sách này
                                    # (lab/i5_detector_v2: v2 fine-tune thạch bản). Vắng = None = ckpt
                                    # toàn cục (env NOM_DETECTOR_CKPT > train_crop/detector_r34.best.pt)
                                    # -> STT không đổi. Đường dẫn tuyệt đối hoặc tương đối gốc repo;
                                    # build_dataset kiểm tệp tồn tại TRƯỚC khi align (fail fast).
                                    # Engine cache detector theo (ckpt, resize, thr); chỉ luật syl_index đọc.
    kim_lang_type: 2                # (2026-09-23, tuỳ chọn) tham số `lang_type` gửi kênh OCR Hán-Nôm
                                    # (kim / kinhhannom) khi ingest sách này: 0 Tự động · 1 Hán (MẶC ĐỊNH =
                                    # bộ cũ) · 2 Nôm. Đo 5 trang/sách (docs/CHOT_KENH_OCR_VA_QUY_HOACH_GAN
                                    # _2026-09-23.md §1): 1 -> 2 nâng "kim ∈ tập chữ của âm QN" LVT 70,0 ->
                                    # 88,4 %, KVK 72,5 -> 92,7 %, CHR 74,9 -> 85,0 % và "kim == chữ dị bản
                                    # độc lập" LVT 55,8 -> 72,7 %, KVK 64,4 -> 86,3 % (CI không chồng);
                                    # cái giá: tỉ lệ cột kim đếm đúng N giảm (LVT 84 -> 74, KVK 90 -> 78,
                                    # CHR 83 -> 51 %). CHỈ adapter ingest đọc (cache kim_raw/ tách theo
                                    # tham số); engine/step2 không đọc. STT không khai -> gửi lang_type 1.
    tier_dp: true                   # (2026-09-23, tuỳ chọn, CHỈ layout=lithograph) chạy DP chữ↔âm
                                    # RIÊNG trong từng tầng (6↔6 rồi 8↔8) thay vì cả cột 14↔14
                                    # (anchor_align.realign_column_tiered). Ranh giới câu lục/câu bát
                                    # thành ràng buộc CỨNG: khe ở câu lục không trôi sang câu bát.
                                    # Mặc định false = hành vi cũ (STT/prose không nhận khoá này).
                                    # Chỉ có hiệu lực khi số chữ kim mỗi tầng và số âm mỗi tầng đều
                                    # xác định (tier_split trong ocr_cache + luật 6/8); cột không đủ
                                    # điều kiện tự rơi về DP cả cột.
    kim_ocr_id: 1                   # (2026-09-23, tuỳ chọn) `ocr_id` của kênh kim: -1 Tự động · 1 Văn bản
                                    # thông thường (mặc định) · 2 Hành chính · 3 Ngoại cảnh · 4 Y học ·
                                    # 5 Văn bia · 6 Kinh Phật. Đã đo: 5 (+epitaph) KÉM hơn -> giữ 1.
    kim_font_type: 1                # (2026-09-23, tuỳ chọn) `font_type`: 0 Tự động · 1 In (mặc định) ·
                                    # 2 Viết tay. Đã đo: 2 KÉM hơn trên sách in -> giữ 1.
    crop_source: original           # (2026-09-23, tuỳ chọn) NGUỒN ĐIỂM ẢNH của crop GIAO NỘP:
                                    # "processed" (MẶC ĐỊNH = hành vi cũ, STT không đổi byte) cắt từ
                                    # prepared/<Book>/pages/*.png — ảnh đã qua adapter (xám L +
                                    # --contrast stretch|otsu), nền bị kéo/ép về 255; "original" cắt
                                    # ĐÚNG CÙNG khung hình đó từ ẢNH QUÉT GỐC trong data/<Book>/
                                    # (giữ màu/nền giấy). HÌNH HỌC KHÔNG ĐỔI: cửa sổ pad, seam carve
                                    # và tighten_box vẫn tính trên ảnh ĐÃ XỬ LÝ rồi áp NGUYÊN xi sang
                                    # ảnh gốc -> crop_w/crop_h/ink_pct/seg_flag giữ nguyên, chỉ điểm
                                    # ảnh đổi. Khi bật, build_dataset ghi THÊM bản đã xử lý ở
                                    # $out/crops_bin/<tier>/<cùng tên>.png cho mã đọc crop nhị phân
                                    # (pipeline.tools.enrich_crop_quality tự ưu tiên crops_bin/ nên
                                    # crop_quality_flag/stray_ink/border_ink KHÔNG đổi); crops_bin/
                                    # KHÔNG đi vào bộ giao nộp (export_final_dataset chỉ copy `image`).
                                    # Ảnh gốc tra qua prepared/<Book>/manifest.json (pages[].source_file
                                    # + scale); thiếu ảnh gốc -> lỗi ngay (fail fast), không rơi ngầm.
    detector_resize: area           # (2026-09-22, tuỳ chọn) phép thu ảnh trang cho detector: "linear"
                                    # (mặc định = cv2.resize cũ, STT không đổi) | "area" (INTER_AREA khử
                                    # răng cưa khi thu ~3×; đo v1 27 trang thạch bản: tầng n==N 77,0 →
                                    # 91,9 %, ok50 95,0 → 98,1 %, STT F1 không giảm). Độc lập với ckpt.

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
KIM_LANG_TYPES = (0, 1, 2)        # 0 Tự động · 1 Hán (mặc định = bộ cũ) · 2 Nôm
KIM_OCR_IDS = (-1, 1, 2, 3, 4, 5, 6)
KIM_FONT_TYPES = (0, 1, 2)        # 0 Tự động · 1 In (mặc định) · 2 Viết tay
KIM_LANG_TYPE_DEFAULT = 1
KIM_OCR_ID_DEFAULT = 1
KIM_FONT_TYPE_DEFAULT = 1
DETECTOR_RESIZE_LINEAR = "linear"
DETECTOR_RESIZE_AREA = "area"
DETECTOR_RESIZES = (DETECTOR_RESIZE_LINEAR, DETECTOR_RESIZE_AREA)
CROP_SOURCE_PROCESSED = "processed"   # cắt từ prepared/<Book>/pages/*.png (hành vi cũ)
CROP_SOURCE_ORIGINAL = "original"     # cắt từ ảnh quét gốc data/<Book>/… (giữ nền giấy)
CROP_SOURCES = (CROP_SOURCE_PROCESSED, CROP_SOURCE_ORIGINAL)


@dataclass(frozen=True)
class BookLayout:
    """Bố cục một sách; mặc định = STT (9 cột, không cổng phụ)."""
    layout: str = LAYOUT_STT
    n_columns: int | str = DEFAULT_N_COLUMNS   # số nguyên, hoặc "auto" (chỉ layout=prose)
    qn_per_column: int = 0        # 0 = không kiểm số âm tiết mỗi cột
    det_xmargin: float | None = None   # None = step2.det_xmargin toàn cục (STT: 0,25)
    det_thr: float | None = None       # None = step2.det_thr toàn cục (STT: 0,2)
    box_decoder: str = BOX_DECODER_LEGACY   # "legacy" | "pitch" (pitch_decode, tuỳ chọn)
    detector_ckpt: str | None = None   # None = ckpt toàn cục (v1); chuỗi = ckpt riêng sách (v2)
    detector_resize: str = DETECTOR_RESIZE_LINEAR   # "linear" (v1) | "area" (khử răng cưa)
    crop_source: str = CROP_SOURCE_PROCESSED        # "processed" (cũ) | "original" (ảnh quét gốc)
    tier_dp: bool = False           # True = DP riêng từng tầng 6/8 (chỉ lithograph)
    kim_lang_type: int = KIM_LANG_TYPE_DEFAULT      # body lang_type của kênh kim (1 = Hán = bộ cũ)
    kim_ocr_id: int = KIM_OCR_ID_DEFAULT            # body ocr_id (1 = văn bản thông thường)
    kim_font_type: int = KIM_FONT_TYPE_DEFAULT      # body font_type (1 = in)

    @property
    def kim_params(self) -> dict:
        """Tham số body cho core.ocr.ocr_api.recognize (chỉ adapter ingest dùng)."""
        return {"ocr_id": self.kim_ocr_id, "lang_type": self.kim_lang_type,
                "font_type": self.kim_font_type}

    @property
    def kim_is_default(self) -> bool:
        """True khi bộ tham số kim == bộ cũ (1, 1, 1) -> cache kim_raw/ KHÔNG đổi tên."""
        return (self.kim_lang_type == KIM_LANG_TYPE_DEFAULT
                and self.kim_ocr_id == KIM_OCR_ID_DEFAULT
                and self.kim_font_type == KIM_FONT_TYPE_DEFAULT)

    @property
    def crop_from_original(self) -> bool:
        """True khi crop giao nộp phải cắt từ ẢNH QUÉT GỐC (data/<Book>/…)."""
        return self.crop_source == CROP_SOURCE_ORIGINAL

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
    detector_ckpt = book_cfg.get("detector_ckpt")
    if detector_ckpt is not None and (not isinstance(detector_ckpt, str) or not detector_ckpt.strip()):
        raise ValueError(f"books[{name}].detector_ckpt = {detector_ckpt!r}; cần chuỗi đường dẫn .pt (hoặc bỏ khoá)")
    if isinstance(detector_ckpt, str):
        detector_ckpt = detector_ckpt.strip()
    detector_resize = book_cfg.get("detector_resize", DETECTOR_RESIZE_LINEAR)
    if detector_resize not in DETECTOR_RESIZES:
        raise ValueError(f"books[{name}].detector_resize = {detector_resize!r}; chỉ nhận {DETECTOR_RESIZES}")
    crop_source = book_cfg.get("crop_source", CROP_SOURCE_PROCESSED)
    if crop_source not in CROP_SOURCES:
        raise ValueError(f"books[{name}].crop_source = {crop_source!r}; chỉ nhận {CROP_SOURCES}")
    tier_dp = book_cfg.get("tier_dp", False)
    if not isinstance(tier_dp, bool):
        raise ValueError(f"books[{name}].tier_dp = {tier_dp!r}; cần true/false")
    if tier_dp and layout != LAYOUT_LITHOGRAPH:
        raise ValueError(f"books[{name}].tier_dp = true chỉ hợp lệ với layout=lithograph (đang {layout!r})")
    kim_lang_type = _enum_key(book_cfg, name, "kim_lang_type", KIM_LANG_TYPE_DEFAULT, KIM_LANG_TYPES)
    kim_ocr_id = _enum_key(book_cfg, name, "kim_ocr_id", KIM_OCR_ID_DEFAULT, KIM_OCR_IDS)
    kim_font_type = _enum_key(book_cfg, name, "kim_font_type", KIM_FONT_TYPE_DEFAULT, KIM_FONT_TYPES)
    if (layout == LAYOUT_STT and n_columns == DEFAULT_N_COLUMNS and qpc == 0
            and det_xmargin is None and det_thr is None and box_decoder == BOX_DECODER_LEGACY
            and detector_ckpt is None and detector_resize == DETECTOR_RESIZE_LINEAR
            and crop_source == CROP_SOURCE_PROCESSED
            and kim_lang_type == KIM_LANG_TYPE_DEFAULT and kim_ocr_id == KIM_OCR_ID_DEFAULT
            and kim_font_type == KIM_FONT_TYPE_DEFAULT and not tier_dp):
        return DEFAULT_LAYOUT
    return BookLayout(layout=layout, n_columns=n_columns, qn_per_column=qpc,
                      det_xmargin=det_xmargin, det_thr=det_thr, box_decoder=box_decoder,
                      detector_ckpt=detector_ckpt, detector_resize=detector_resize,
                      crop_source=crop_source, tier_dp=tier_dp, kim_lang_type=kim_lang_type, kim_ocr_id=kim_ocr_id,
                      kim_font_type=kim_font_type)


def resolve_detector_ckpt(detector_ckpt: str | None, repo_root) -> str | None:
    """Đường dẫn ckpt theo sách -> tuyệt đối (thử nguyên văn, rồi tương đối gốc repo). None -> None.
    Không tồn tại -> FileNotFoundError (build_dataset gọi TRƯỚC khi align: fail fast, không rơi ngầm về v1)."""
    if detector_ckpt is None:
        return None
    from pathlib import Path
    cands = [Path(detector_ckpt)]
    if not cands[0].is_absolute():
        cands.append(Path(repo_root) / detector_ckpt)
    for c in cands:
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError(f"books[].detector_ckpt = {detector_ckpt!r} không tồn tại (thử {[str(c) for c in cands]}); "
                            "bỏ khoá để dùng ckpt toàn cục v1, hoặc chạy lab/i5_detector_v2/apply_v2.sh <best.pt>")


def _enum_key(book_cfg: dict, name: str, key: str, default: int, allowed: tuple) -> int:
    """Khoá số nguyên tuỳ chọn trong tập `allowed`; vắng -> default; sai kiểu/miền -> ValueError
    (dừng TRƯỚC khi tiêu lượt gọi API, không rơi ngầm về mặc định)."""
    if key not in book_cfg:
        return default
    v = book_cfg[key]
    if isinstance(v, bool) or not isinstance(v, int) or v not in allowed:
        raise ValueError(f"books[{name}].{key} = {v!r}; chỉ nhận {allowed}")
    return int(v)


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
                    expected_counts: dict | None = None,
                    tier_rule: tuple[int, ...] | None = None) -> tuple[bool, dict]:
    """Cổng page_ok cho layout=lithograph.

    PASS khi: số cột Nôm == n_columns, số cột QN == n_columns, và mỗi cột QN có
    đúng `expected_counts[line_id]` âm tiết (mặc định lay.qn_per_column = 14; nếu
    transcriptions/<page>.json ghi `num_syllables` cho cột đó thì lấy số ấy —
    câu 5/7/9 âm ghi rõ trong JSON vẫn qua cổng). qn_per_column = 0 -> bỏ kiểm âm.
    Trả (ok, chi tiết) để ghi vào bản ghi trang. align_production._detect còn đòi
    phương pháp cột hybrid* (projection_fallback luôn ép đúng n_columns nên đếm cột
    ở nhánh đó là tautology).

    tier_rule (2026-09-23, docs/RA_SOAT_CAN_CHINH_2026-09-23.md §4 #4): khi sách có
    luật thể thơ (lithograph 14 âm -> (6, 8), xem `tier_rule_for`) thì KỲ VỌNG lấy
    LUẬT `sum(tier_rule)`, KHÔNG lấy `num_syllables` của JSON. Lý do: `num_syllables`
    do chính tesseract đếm, nên cột 13/15 âm "khớp với chính mình" và ĐI LỌT cổng —
    đo được 27/1.044 (LVT1883), 26/1.628 (KVK1884), 22/988 (LVT1916), 7/1.610 (TK1872)
    cột như vậy, và 76–86 % trôi căn chỉnh nằm đúng trong nhóm ấy. Cổng chỉ ĐẾM
    (page_ok vào summary.json["layout_gate"]), KHÔNG loại ô nào — việc hạ cấp do
    cờ `qn_count_unfixed` + mechanism_gates làm. None = hành vi cũ (STT/prose).
    """
    n_nom = len(cols)
    n_qn = len(qn_lines)
    bad_cols: list[dict] = []
    rule_n = sum(tier_rule) if tier_rule else None
    if lay.qn_per_column or expected_counts or rule_n:
        for lid in sorted(qn_lines):
            want = rule_n if rule_n else (expected_counts or {}).get(lid, lay.qn_per_column)
            if not want:
                continue
            got = len(qn_lines[lid])
            if got != want:
                bad_cols.append({"column": lid, "n_syl": got, "want": want,
                                 **({"src": "tier_rule"} if rule_n else {})})
    ok = (n_nom == lay.n_columns and n_qn == lay.n_columns and not bad_cols)
    return ok, {"layout": lay.layout, "n_columns": lay.n_columns,
                "n_nom_cols": n_nom, "n_qn_cols": n_qn, "bad_syl_cols": bad_cols,
                **({"expect_src": "tier_rule", "expect_n": rule_n} if rule_n else {})}


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


# Luật thể thơ lục bát: tầng trên (câu lục) 6 âm, tầng dưới (câu bát) 8 âm. Dùng làm
# RÀNG BUỘC CỨNG khi tổng số âm của cột đúng 14 — thay vì tin `len_odd` do tesseract
# đếm (docs/CHOT_KENH_OCR_VA_QUY_HOACH_GAN_2026-09-23.md §5.2 bước 6 và §6 #4: tách tầng
# của kim đúng 100 % khi đủ 14 chữ, còn số đếm QN mới là khâu yếu).
LITHO_TIER_RULE = (6, 8)


def tier_rule_for(lay) -> tuple[int, ...] | None:
    """Luật đếm tầng theo sách: lithograph 14 âm/cột -> (6, 8); còn lại -> None
    (prose/STT giữ nguyên hành vi cũ)."""
    if lay is None or not getattr(lay, "is_lithograph", False):
        return None
    if lay.qn_per_column != sum(LITHO_TIER_RULE):
        return None
    return LITHO_TIER_RULE


def expected_tier_counts(data_dir, page_name: str,
                         tier_rule: tuple[int, ...] | None = None) -> dict[int, list[int]]:
    """(box_decoder=pitch) Số âm QN MỖI TẦNG của từng cột từ transcriptions/<page>.json:
    thạch bản ghi `len_odd` (câu lục, tầng trên) và `num_syllables` -> [len_odd, num - len_odd].
    Trả {line_id: [n_tầng_trên, n_tầng_dưới]}; thiếu tệp/khoá -> {} (decoder chia theo chữ kim /
    chiều cao tầng). Chỉ đọc khi box_decoder=pitch; đường STT không đọc tệp này.

    tier_rule (2026-09-23, chỉ lithograph — xem `tier_rule_for`): khi tổng số âm của cột ĐÚNG
    sum(tier_rule) (14) thì trả LUẬT (6, 8) thay cho `len_odd` của QN, vì `len_odd` là số âm
    tesseract đếm được cho câu lục còn luật lục bát là bất biến của bản in; hai số chỉ khác
    nhau khi QN đếm sai. None = hành vi cũ (đường STT/prose)."""
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
    rule_n = sum(tier_rule) if tier_rule else None
    for i, c in enumerate(data.get("columns") or []):
        if not isinstance(c, dict):
            continue
        lid = c.get("column", i + 1)
        n = c.get("num_syllables")
        lo = c.get("len_odd")
        if lo is None and isinstance(c.get("verse_odd"), dict):
            lo = c["verse_odd"].get("n_syll")
        if (rule_n is not None and isinstance(n, int) and not isinstance(n, bool)
                and n == rule_n and isinstance(lid, int)):
            out[lid] = list(tier_rule)          # luật 6/8 thắng len_odd khi cột đủ 14 âm
            continue
        if (isinstance(n, int) and isinstance(lo, int) and not isinstance(n, bool) and not isinstance(lo, bool)
                and 0 < lo < n and isinstance(lid, int)):
            out[lid] = [lo, n - lo]
    return out


# --- Cờ "SỐ ĐẾM ÂM QN KHÔNG SỬA ĐƯỢC" (2026-09-23) --------------------------------
# docs/RA_SOAT_CAN_CHINH_2026-09-23.md §2.2/§4 #1. Đo trên NHÃN NGƯỜI (2 bộ IHR):
# ô GOLD nằm trong cột `n_qn != 14` chỉ đúng 52,2 % (n 209, LucVanTien1916) và 49,0 %
# (n 51, TruyenKieu1872), trong khi phần còn lại đạt 98,4 % / 98,7 %; và 85,7 % / 75,0 %
# TOÀN BỘ ô "đúng chữ, sai ô" nằm trong đúng nhóm cột ấy. Adapter đã cố sửa số đếm bằng
# luật 6/8 (`ingest_lithograph_book.repair_tier_syllables`), cột nào không sửa nổi thì
# ở đây thành CỜ THẬT trong labels.csv để cổng cơ chế B4' hạ cấp được.
#   lithograph (có tier_rule): cột có số âm QN != sum(tier_rule) (= 14)
#   prose      : cột có dp_ratio < PROSE_DP_RATIO_MIN (tỉ lệ khớp DP chữ↔âm của cột,
#                ingest_prose_book ghi vào transcriptions/<page>.json) — Chrestomathie
#                KHÔNG có chuẩn độc lập nên mặc định chỉ GHI CỜ, không hạ cấp.
#   STT        : tier_rule = None và không phải prose -> cờ LUÔN RỖNG, cột không tồn tại
#                trong labels.csv (STT byte-identical).
PROSE_DP_RATIO_MIN = 0.75
COL_QN_COUNT_UNFIXED = "qn_count_unfixed"


def qn_count_flag_on(lay, tier_rule: tuple[int, ...] | None = None) -> bool:
    """Sách này có phát cờ `qn_count_unfixed` không? (lithograph có luật, hoặc prose)."""
    return bool(tier_rule) or bool(lay is not None and getattr(lay, "is_prose", False))


def prose_dp_ratios(data_dir, page_name: str) -> dict[int, float]:
    """{line_id: dp_ratio} từ transcriptions/<page>.json (layout=prose). Thiếu -> {}."""
    import json
    from pathlib import Path
    p = Path(data_dir) / "transcriptions" / f"{page_name}.json"
    if not p.exists():
        return {}
    try:
        data = json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}
    out: dict[int, float] = {}
    for i, c in enumerate(data.get("columns") or []):
        if not isinstance(c, dict):
            continue
        lid = c.get("column", i + 1)
        r = c.get("dp_ratio")
        if isinstance(lid, int) and isinstance(r, (int, float)) and not isinstance(r, bool):
            out[lid] = float(r)
    return out


def qn_count_unfixed_columns(qn_lines: dict, lay, tier_rule: tuple[int, ...] | None = None,
                             data_dir=None, page_name: str | None = None,
                             dp_min: float | None = None) -> dict[int, str]:
    """{line_id: lý do} các cột SỐ ĐẾM ÂM QN hỏng. Rỗng với STT (xem chú thích trên)."""
    out: dict[int, str] = {}
    if tier_rule:
        want = sum(tier_rule)
        for lid, syl in (qn_lines or {}).items():
            got = len(syl or [])
            if got != want:
                out[lid] = f"n_qn={got}!={want}"
        return out
    if lay is not None and getattr(lay, "is_prose", False) and data_dir is not None and page_name:
        lo = PROSE_DP_RATIO_MIN if dp_min is None else float(dp_min)
        for lid, r in prose_dp_ratios(data_dir, page_name).items():
            if r < lo:
                out[lid] = f"dp_ratio={r:.3f}<{lo}"
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
