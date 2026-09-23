"""Chuẩn hoá âm Quốc ngữ NGAY TRƯỚC khi căn chỉnh một cột — vá dấu phụ VietOCR rụng.

VÌ SAO ĐẶT Ở ĐÂY, KHÔNG PHẢI SAU BUILD
--------------------------------------
Đo 2026-08-23: có 115 ô mà âm QN chỉ sai dấu phụ, và sau khi sửa thì **115/115 đủ điều
kiện `s1_inter_s2_direct` (= GOLD)**. Nhưng tier hiện tại của chúng là 106
SILVER_uncalibrated + 9 REVIEW — **không ô nào nằm trong bộ giao nộp**. Nghĩa là:

    áp SAU build  -> +0 ô vào bộ giao nộp (chỉ đổi nhãn ở tier không giao nộp)
    áp TRƯỚC align -> +100 ô vào GOLD

Đây là lý do `pipeline/tools/fix_tone.py` tự viết trong docstring rằng chỗ đúng của phép
sửa là bước chuẩn hoá TRƯỚC build. Module này là chỗ đó.

VÌ SAO 100 CHỨ KHÔNG PHẢI 115
-----------------------------
`fix_tone` chạy SAU build nên biết CẶP GHÉP (ô nào ứng với chữ nào) và dùng đọc âm của
đúng chữ ấy. Ở đây cặp ghép CHƯA có — đó chính là thứ Needleman-Wunsch sắp tính. Nên ứng
viên phải lấy từ đọc âm của **mọi chữ trong cột**, rộng hơn nên dễ nhập nhằng hơn.
Đo được: 100 ô vẫn ra đúng một đáp án · 14 ô thành nhập nhằng · 1 ô mất ứng viên.
Nhập nhằng thì ĐỂ NGUYÊN — 14 ô đó đi vào mẻ chấm tay, không đoán.

BA CHỐT AN TOÀN — giữ nguyên triết lý của fix_tone
--------------------------------------------------
1. Âm là TỪ TIẾNG VIỆT CÓ THẬT -> KHÔNG đụng. Ở đó "sai dấu" và "sai chữ" là hai giả
   thuyết ngang nhau (異 "là" vs "lạ"), sửa máy sẽ CHE mất lỗi chữ.
2. Ứng viên phải là ĐỌC ÂM của một chữ đang có trong cột — không lấy bừa từ từ điển.
3. Nhiều ứng viên -> để nguyên, ghi vào log. Thà còn 14 ô chờ người xem còn hơn đoán 14 ô.

Tầng 2 (`strip_all`, bỏ cả dấu tạo chữ) CHỈ được thử khi tầng 1 (`strip_tone`) không ra
ứng viên nào — rộng hơn thì phải xếp sau.
"""
from __future__ import annotations

import collections
import re

from core.text.text_utils import strip_all, strip_tone
from pipeline.align_engine.qn_charfix import fix_syllable

__all__ = ["build_readings", "normalize_column"]

_GARBAGE = re.compile(r"^[\W\d_]+$")


_READINGS_CACHE: dict[int, dict[str, set[str]]] = {}


def build_readings(qn_to_nom: dict[str, list[str]]) -> dict[str, set[str]]:
    """{chữ Nôm: {âm QN mà từ điển công nhận}} — dựng MỘT LẦN cho cả lần chạy.

    Có cache theo `id()` của từ điển: `align_page` gọi hàm này mỗi trang, mà đảo 104k
    cặp × 445 trang là lãng phí thuần tuý. Từ điển bất biến trong một lần chạy.
    """
    key = id(qn_to_nom)
    if key in _READINGS_CACHE:
        return _READINGS_CACHE[key]
    readings: dict[str, set[str]] = collections.defaultdict(set)
    for syl, chars in qn_to_nom.items():
        for ch in chars:
            readings[ch].add(syl)
    _READINGS_CACHE[key] = readings
    return readings


def normalize_column(chars, syllables: list[str], qn_keys, readings,
                     charfix: bool = False) -> tuple[list[str], list[dict]]:
    """Trả (âm đã chuẩn hoá, nhật ký sửa). KHÔNG sửa gì thì trả về chính danh sách cũ.

    `chars`   — cluster["chars"] (dict có 'ocr_char'/'char') hoặc chuỗi.
    `qn_keys` — set khoá từ điển, để nhận ra "âm là từ có thật".
    `charfix` — (2026-09-24, khoá config `qn_charfix`) bật tầng L3 `qn_charfix.fix_syllable`
      CHẠY TRƯỚC tầng dời dấu dưới đây. Mặc định False = hành vi cũ, byte-identical.
      L3 quyết định hoàn toàn ở phía quốc ngữ (chỉ âm OOV + ứng viên DUY NHẤT trong từ
      điển), KHÔNG nhìn chữ kim — xem docstring của `pipeline.align_engine.qn_charfix`.
    """
    if not syllables:
        return syllables, []

    # --- L3: sửa lỗi ký tự OCR, TRƯỚC mọi thứ khác (không cần đọc âm của chữ trong cột) ---
    fixlog: list[dict] = []
    if charfix:
        out3 = list(syllables)
        for i, s in enumerate(out3):
            low = str(s).lower()
            if not low or _GARBAGE.match(low):
                continue
            f = fix_syllable(low, qn_keys)
            if f:
                out3[i] = f
                fixlog.append({"idx": i, "raw": low, "fixed": f,
                               "rule": "charfix_unique", "action": "fixed", "kind": "charfix"})
        if fixlog:
            syllables = out3

    pool: set[str] = set()
    for c in chars or ():
        ch = c.get("ocr_char") or c.get("char") if isinstance(c, dict) else c
        if ch:
            pool |= readings.get(ch, set())
    if not pool:
        return syllables, fixlog

    out = list(syllables)
    log: list[dict] = list(fixlog)
    for i, s in enumerate(out):
        low = str(s).lower()
        if not low or _GARBAGE.match(low):
            continue                       # rác marker: việc của parser, không phải ở đây
        if low in pool or low in qn_keys:
            continue                       # đã khớp, hoặc là từ có thật -> CHỐT 1
        cand = sorted(x for x in pool if strip_tone(x) == strip_tone(low))
        level = "tone"
        if not cand:
            cand = sorted(x for x in pool if strip_all(x) == strip_all(low))
            level = "diacritic"
        if len(cand) != 1:
            if cand:
                log.append({"idx": i, "raw": low, "candidates": cand,
                            "action": "ambiguous_kept"})   # CHỐT 3
            continue
        out[i] = cand[0]
        log.append({"idx": i, "raw": low, "fixed": cand[0],
                    "rule": f"{level}_unique", "action": "fixed", "kind": "tone_place"})
    return out, log
