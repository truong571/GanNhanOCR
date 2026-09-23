"""L3 — sửa LỖI KÝ TỰ của OCR quốc ngữ trên âm NGOÀI TỪ ĐIỂN, trước khi căn chỉnh (2026-09-24).

VẤN ĐỀ
------
Kênh quốc ngữ của sách thạch bản/văn xuôi là tesseract trên bản in đá. Nó đọc hỏng một số
ký tự theo kiểu HÌNH DẠNG, lặp đi lặp lại: `eho` (cho), `ehi` (chi), `lạe` (lạc), `khúe`
(khúc), `ðã` (đã), `ñ` (n), `œ` (c), `ø` (o), `ö` (ô), và các ký tự rác `‹ › ¿`. Âm vỡ như
vậy KHÔNG có trong từ điển ⇒ tập chữ ứng viên rỗng ⇒ luật GOLD (`chữ kim ∈ tập đọc từ điển
của âm QN`) không bao giờ thoả ⇒ ô rơi xuống REVIEW dù chữ kim đọc được bình thường.

BA CHỐT AN TOÀN (giữ đúng triết lý của `syllable_normalize`)
-----------------------------------------------------------
1. CHỈ đụng âm KHÔNG phải khoá từ điển ("only-if-OOV"). Âm đã là từ tiếng Việt có thật thì
   "sai chữ" và "sai âm" là hai giả thuyết ngang nhau — sửa máy sẽ CHE mất lỗi chữ. Đo
   được: 0 ô GOLD nào của ba sách có âm ngoài từ điển (n 10.736 / 19.925 / 4.381), nên
   phép sửa này KHÔNG THỂ phá một ô GOLD nào theo cấu trúc.
2. Ứng viên phải DUY NHẤT. Nhiều ứng viên ⇒ để nguyên, ô ở lại REVIEW.
3. Quyết định hoàn toàn ở PHÍA QUỐC NGỮ, KHÔNG nhìn chữ kim. Nếu chọn âm *vì* chữ kim nằm
   trong tập đọc của nó thì hai kênh không còn độc lập và luật GOLD tự khẳng định.

THỨ TỰ: L3 CHẠY TRƯỚC tầng dời-dấu của `normalize_column`
--------------------------------------------------------
Ca bắt được khi làm ngược lại: `eó` (tesseract đọc hỏng `có`) bị tầng dấu dời thành `éo`,
mà `éo` LÀ khoá từ điển nên được nhận là "duy nhất, hợp lệ". Chữ kim của ô là 要:
要 ∈ R(éo) nhưng 要 ∉ R(có) ⇒ một âm SAI hợp thức hoá đúng chữ kim mà âm ĐÚNG sẽ loại.
Chạy L3 trước thì `eó → có`, ô ấy ở lại REVIEW (đúng).

Bật/tắt: khoá `qn_charfix: true` trong `books:` của config (mặc định FALSE = hành vi cũ,
STT không đổi một byte). Bảng ký tự nằm ở `config/lexicon/qn_charfix.json`.
"""
from __future__ import annotations

from core.text.lexicon import load_mapping

__all__ = ["CHAR_FIX", "char_variants", "fix_syllable"]

# Bảng ký tự — HỌC TỪ DỮ LIỆU THẬT (cặp âm OCR ↔ âm phiên âm chuẩn của dị bản, học CHÉO
# SÁCH để phép chấm không tự khẳng định), không đoán. Giá trị "" = xoá ký tự rác.
_DEFAULT_CHAR_FIX = {
    "e": "c",     # eho→cho, ehi→chi, eon→con, lạe→lạc, khúe→khúc (luật mạnh nhất: 44 lần)
    "ð": "đ",     # ðã→đã, ðồng→đồng (21 lần)
    "ñ": "n",     # (8 lần)
    "œ": "c",     # œáúc…
    "ø": "o",     # (7 lần)
    "ö": "ô",
    "‹": "", "›": "", "¿": "",   # ký tự rác của tesseract (‹→∅ 47 lần)
}

CHAR_FIX = load_mapping("qn_charfix.json", _DEFAULT_CHAR_FIX)


def char_variants(s: str, charmap: dict[str, str] | None = None) -> set[str]:
    """Mọi biến thể sinh bởi ĐÚNG MỘT phép thay/xoá ký tự trong bảng (không ghép nhiều phép)."""
    cm = CHAR_FIX if charmap is None else charmap
    out: set[str] = set()
    for i, ch in enumerate(s):
        rep = cm.get(ch)
        if rep is None:
            continue
        v = s[:i] + rep + s[i + 1:]
        if v and v != s:
            out.add(v)
    return out


def fix_syllable(s: str, qn_keys, charmap: dict[str, str] | None = None) -> str | None:
    """Âm OOV → âm đã sửa nếu có ĐÚNG MỘT biến thể là khoá từ điển; ngược lại None.

    Trả None ngay khi `s` đã là khoá từ điển (chốt 1) — không bao giờ đụng âm hợp lệ.
    """
    if not s or s in qn_keys:
        return None
    cand = sorted(v for v in char_variants(s, charmap) if v in qn_keys)
    return cand[0] if len(cand) == 1 else None
