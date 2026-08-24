"""CHUẨN NHIỄU LOẠN CÓ ĐÁP ÁN — đo ĐỘ BỀN của căn chỉnh mà không cần một nhãn người nào.

Ý TƯỞNG
-------
Không có ground truth thì không đo được precision. Nhưng ĐỘ BỀN thì đo được: lấy dữ liệu
thật, tiêm vào một hỏng hóc mà ta BIẾT TRƯỚC, rồi xem thuật toán có giữ nguyên những cặp
ghép chắc chắn đúng hay không.

Ô NEO (anchor) = ô có `rule = s1_inter_s2_direct`: chữ OCR nằm trong danh sách đọc âm của
âm tiết, chi phí ghép 0,0. Đây là những cặp gần chắc chắn đúng. Đo được trên corpus: trung
vị **58%** ô mỗi cột là ô neo (p10 42%, p90 73%) — quá đủ tín hiệu.

    ⚠️ KHÔNG dùng "cột sạch tuyệt đối" làm chuẩn: toàn corpus chỉ có 6 cột đạt 100%
    dict-confirmed (54 ô), quá ít. Chuẩn phải neo vào Ô NEO, không phải cột sạch.

CHỈ SỐ: `anchor_retention` = trong số ô neo CÒN SỐNG sau khi tiêm hỏng (âm/chữ của nó
không bị xoá), bao nhiêu phần trăm vẫn được ghép đúng cặp cũ.

    ⚠️ ĐỪNG tối ưu theo "số ô GOLD" hay tỉ lệ dict-confirm — đó chính là luật gán nhãn,
    cấu hình nới lỏng nhất sẽ luôn thắng trong khi sinh nhiều nhãn sai nhất. Dùng biên
    Pareto giữa `anchor_retention` và sản lượng.

BẢY LOẠI HỎNG — mô phỏng lỗi THẬT đã quan sát được trong corpus:

    drop_syl     VietOCR nuốt âm            ins_syl      VietOCR sinh âm thừa
    swap_syl     lỗi thứ tự                 tone_syl     lỗi thanh điệu (nhóm A/B)
    subst_char   S1 đọc nhầm tự dạng        drop_char    detector sót hộp
    split_char   detector cắt đôi hộp
"""
from __future__ import annotations

import collections
import csv
import hashlib
import random
import unicodedata as ud
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANCHOR_RULE = "s1_inter_s2_direct"

PERTURBATIONS = ("drop_syl", "ins_syl", "swap_syl", "tone_syl",
                 "subst_char", "drop_char", "split_char")
LEVELS = (1, 2, 3)


# --------------------------------------------------------------------------- #
def load_columns(labels_csv: Path | str, min_cells: int = 6) -> list[dict]:
    """Dựng lại (chữ Nôm, âm QN, ô neo) cho từng cột từ bộ nhãn.

    Hàng trong labels.csv là ĐẦU RA của align, nên chuỗi dựng lại đã là một phép ghép
    1-1 theo đường chéo. Đó chính là điều ta muốn: mốc xuất phát KHÔNG hỏng, rồi mới
    tiêm hỏng vào và đo mức tụt.
    """
    with open(labels_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_col: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if r.get("column") and r.get("ocr_char") and r.get("syllable"):
            by_col[(r["book"], r["page"], r["column"])].append(r)
    cols = []
    for key, rs in sorted(by_col.items()):
        chars = [r["ocr_char"] for r in rs]
        syls = [str(r["syllable"]).lower() for r in rs]
        anchors = [i for i, r in enumerate(rs)
                   if str(r.get("rule", "")).split("|")[0] == ANCHOR_RULE]
        if len(rs) >= min_cells and anchors:
            cols.append({"key": key, "chars": chars, "syllables": syls,
                         "anchors": anchors})
    return cols


# --------------------------------------------------------------------------- #
def _seed(*parts) -> int:
    """Seed TẤT ĐỊNH giữa các tiến trình.

    KHÔNG dùng `hash()`/`tuple.__hash__()`: hash của chuỗi trong Python được NGẪU NHIÊN
    HOÁ theo tiến trình (PYTHONHASHSEED), nên hai lần chạy cho hai kết quả khác nhau —
    đúng lỗi đã bị phép kiểm tất định của bàn thí nghiệm bắt được ngày 2026-08-23.
    """
    blob = "|".join(str(x) for x in parts).encode()
    return int.from_bytes(hashlib.md5(blob).digest()[:8], "big")


_TONE = "̣̀́̃̉"          # huyền sắc ngã hỏi nặng


def _retone(s: str, rng: random.Random) -> str:
    """Đổi dấu thanh của âm — mô phỏng đúng lớp lỗi VietOCR nhóm A/B."""
    d = ud.normalize("NFD", s)
    base = "".join(c for c in d if c not in _TONE)
    keep = [c for c in _TONE if c in d]
    new = rng.choice([t for t in _TONE if t not in keep] + [""])
    if not new:
        return ud.normalize("NFC", base)
    # gắn dấu vào nguyên âm đầu tiên
    for i, c in enumerate(base):
        if c.lower() in "aeiouyăâêôơư":
            return ud.normalize("NFC", base[:i + 1] + new + base[i + 1:])
    return ud.normalize("NFC", base)


def apply_perturbation(chars: list[str], syls: list[str], anchors: list[int],
                       kind: str, level: int, rng: random.Random,
                       similar: dict | None = None):
    """Trả (chars', syls', map_anchor) — map_anchor[i_gốc] = (i_char', j_syl') hoặc None.

    None = ô neo đó KHÔNG còn sống (chữ hoặc âm của nó đã bị xoá) -> loại khỏi mẫu số.
    """
    c, s = list(chars), list(syls)
    ci = list(range(len(c)))          # ci[k] = chỉ số GỐC của chữ đang ở vị trí k
    si = list(range(len(s)))
    for _ in range(level):
        if kind == "drop_syl" and len(s) > 2:
            j = rng.randrange(len(s)); s.pop(j); si.pop(j)
        elif kind == "ins_syl":
            j = rng.randrange(len(s) + 1)
            s.insert(j, rng.choice(s) if s else "xxx"); si.insert(j, -1)
        elif kind == "swap_syl" and len(s) > 2:
            j = rng.randrange(len(s) - 1)
            s[j], s[j + 1] = s[j + 1], s[j]; si[j], si[j + 1] = si[j + 1], si[j]
        elif kind == "tone_syl" and s:
            j = rng.randrange(len(s)); s[j] = _retone(s[j], rng)
        elif kind == "subst_char" and c:
            k = rng.randrange(len(c))
            alt = (similar or {}).get(c[k]) or []
            c[k] = rng.choice(alt[:5]) if alt else "口"
        elif kind == "drop_char" and len(c) > 2:
            k = rng.randrange(len(c)); c.pop(k); ci.pop(k)
        elif kind == "split_char" and c:
            k = rng.randrange(len(c))
            c.insert(k + 1, c[k]); ci.insert(k + 1, -1)   # hộp bị cắt đôi -> 2 hộp
    pos_c = {orig: k for k, orig in enumerate(ci) if orig >= 0}
    pos_s = {orig: k for k, orig in enumerate(si) if orig >= 0}
    amap = {a: ((pos_c[a], pos_s[a]) if a in pos_c and a in pos_s else None)
            for a in anchors}
    return c, s, amap


# --------------------------------------------------------------------------- #
def anchor_retention(columns: list[dict], qn_to_nom: dict, similar: dict,
                     kind: str, level: int, seed: int = 0,
                     band_slack: int | None = None, cost: dict | None = None) -> dict:
    """Tỉ lệ ô neo còn sống vẫn giữ đúng cặp ghép cũ, sau khi tiêm hỏng `kind` mức `level`.

    `cost` cho phép quét ma trận chi phí mà không phải sửa mã (T3.2): các khoá hợp lệ là
    COST_CONFIRM / COST_SIMILAR / COST_DICTMISS / COST_NODICT / COST_DEL / COST_INS.
    """
    import sys
    sys.path.insert(0, str(REPO))
    from pipeline.align_engine import anchor_align as aa

    saved = {k: getattr(aa, k) for k in
             ("COST_CONFIRM", "COST_SIMILAR", "COST_DICTMISS",
              "COST_NODICT", "COST_DEL", "COST_INS", "BAND_SLACK")}
    try:
        for k, v in (cost or {}).items():
            setattr(aa, k, v)
        if band_slack is not None:
            aa.BAND_SLACK = band_slack
        slack = aa.BAND_SLACK

        kept = alive = 0
        for ci, col in enumerate(columns):
            rng = random.Random(_seed(seed, ci, kind, level))
            c, s, amap = apply_perturbation(
                col["chars"], col["syllables"], col["anchors"], kind, level, rng, similar)
            live = {a: p for a, p in amap.items() if p is not None}
            if not live:
                continue
            ops = aa.realign_column([{"ocr_char": x} for x in c], s,
                                    qn_to_nom, similar, band_slack=slack)
            got = {(o["nom_idx"], o["syl_idx"]) for o in ops if o["op"] == "match"}
            alive += len(live)
            kept += sum(1 for p in live.values() if p in got)
        return {"kind": kind, "level": level, "anchors_alive": alive,
                "anchors_kept": kept,
                "anchor_retention": round(kept / alive, 4) if alive else None}
    finally:
        for k, v in saved.items():
            setattr(aa, k, v)


def content_anchors(col: dict, readings: dict[str, set[str]]) -> list[tuple[int, int]]:
    """Ô neo ĐỘC LẬP VỚI MỌI CẤU HÌNH — không chạy align, nên không cấu hình nào được ưu ái.

    Neo = cặp ĐƯỜNG CHÉO (i, i) trên các cột có m == n, mà `chars[i]` là đọc âm từ điển của
    `syllables[i]`. Cột m == n thì đường chéo là lời giải hiển nhiên và mọi cấu hình đều
    cho ra nó; cặp nào lại còn được từ điển xác nhận thì gần như chắc chắn đúng.

    VÌ SAO PHẢI ĐỔI: bản cũ (`extract_columns.add_anchors`) lấy neo = cặp `confirmed` của
    CẤU HÌNH MỐC. Đo được: retention của mốc ở mức nhiễu = 0 bằng đúng 1,00000 theo định
    nghĩa, và chênh lệch giữa các cấu hình ở mức nhiễu = 0 (0,622pp) đã LỚN HƠN toàn bộ
    biên độ của chỉ số sau khi tiêm nhiễu (0,577pp). Tức chỉ số cũ không đo ĐỘ BỀN mà đo
    ĐỘ GIỐNG LỜI GIẢI CỦA MỐC. Bỏ đặc quyền đó thì mốc rơi từ hạng 1/17 xuống 13/17.
    """
    c, s = col["chars"], col["syllables"]
    if len(c) != len(s):
        return []
    return [(i, i) for i in range(len(c)) if c[i] and s[i] in readings.get(c[i], ())]


def _score_content(anchor_pairs, live, orig_chars, orig_syls,
                   new_chars, new_syls, ops) -> tuple[int, int]:
    """Chấm theo NỘI DUNG (chữ, âm) chứ không theo CẶP CHỈ SỐ — dùng ĐA TẬP.

    VÌ SAO: `split_char` nhân đôi một chữ và `ins_syl` chèn bản sao một âm, nên sau khi
    tiêm hỏng có HAI ứng viên giống hệt; DP chọn bản sao thứ hai và neo ở chỉ số cũ bị đếm
    là MẤT dù nhãn sinh ra HOÀN TOÀN ĐÚNG. Đo được: `split_char` 0,9140 (chỉ số) ->
    **0,9982** (nội dung). 88,7% khối lượng "mất neo" của bản cũ là giả tạo.

    Dùng đa tập chứ không dùng tập: một cột có thể có hai cặp (chữ, âm) y hệt nhau, giữ
    được một cái thì chỉ được tính một.
    """
    want: collections.Counter = collections.Counter()
    for k, p in enumerate(live):
        if p is None:
            continue                       # neo không còn sống -> ngoài mẫu số
        i, j = anchor_pairs[k]
        want[(orig_chars[i], orig_syls[j])] += 1
    if not want:
        return 0, 0
    got = collections.Counter((new_chars[o["nom_idx"]], new_syls[o["syl_idx"]])
                              for o in ops if o["op"] == "match")
    return sum(min(n, got[key]) for key, n in want.items()), sum(want.values())


def _remap(chars, syls, anchor_pairs, kind, level, rng, similar):
    """Như apply_perturbation nhưng theo CẶP (nom_idx, syl_idx) của đầu vào THẬT.

    Trả (chars', syls', [(i',j') hoặc None]). None = cặp neo không còn sống vì chữ HOẶC
    âm của nó đã bị xoá -> loại khỏi mẫu số, không tính là mất.
    """
    c, s = list(chars), list(syls)
    ci, si = list(range(len(c))), list(range(len(s)))
    for _ in range(level):
        if kind == "drop_syl" and len(s) > 2:
            j = rng.randrange(len(s)); s.pop(j); si.pop(j)
        elif kind == "ins_syl":
            j = rng.randrange(len(s) + 1)
            s.insert(j, rng.choice(s) if s else "xxx"); si.insert(j, -1)
        elif kind == "swap_syl" and len(s) > 2:
            j = rng.randrange(len(s) - 1)
            s[j], s[j + 1] = s[j + 1], s[j]; si[j], si[j + 1] = si[j + 1], si[j]
        elif kind == "tone_syl" and s:
            j = rng.randrange(len(s)); s[j] = _retone(s[j], rng)
        elif kind == "subst_char" and c:
            k = rng.randrange(len(c))
            alt = (similar or {}).get(c[k]) or []
            c[k] = rng.choice(alt[:5]) if alt else "口"
        elif kind == "drop_char" and len(c) > 2:
            k = rng.randrange(len(c)); c.pop(k); ci.pop(k)
        elif kind == "split_char" and c:
            k = rng.randrange(len(c)); c.insert(k + 1, c[k]); ci.insert(k + 1, -1)
    pc = {o: k for k, o in enumerate(ci) if o >= 0}
    ps = {o: k for k, o in enumerate(si) if o >= 0}
    live = [((pc[i], ps[j]) if i in pc and j in ps else None) for i, j in anchor_pairs]
    return c, s, live


def sweep_config(columns: list[dict], qn_to_nom: dict, similar: dict,
                 band_slack: int | None = None, cost: dict | None = None,
                 seeds: int = 3, adaptive_band: bool = False,
                 readings: dict | None = None, demoted: set | None = None) -> dict:
    """MỘT cấu hình -> (sản lượng, độ bền) trên ĐẦU VÀO THẬT.

    BẢN NÀY ĐÃ SỬA 5 KHIẾM KHUYẾT mà đợt phản biện 2026-08-24 chỉ ra (bản trước cho kết
    luận SAI — xem docs/VIEC_CAN_LAM.md mục "T3 — thiết kế hỏng và cách sửa"):

    1. NEO ĐỘC LẬP CẤU HÌNH (`content_anchors`) thay vì neo sinh bởi cấu hình MỐC. Bản cũ
       cho mốc retention = 1,00000 theo định nghĩa; bỏ đặc quyền thì mốc rơi 1/17 -> 13/17.
    2. CHẤM THEO NỘI DUNG (`_score_content`) thay vì theo cặp chỉ số. 88,7% "mất neo" của
       bản cũ là giả tạo; `split_char` 0,914 -> 0,9997.
    3. `ret_noise0` — CỘT ĐỐI CHỨNG ở mức nhiễu = 0. Nếu một cấu hình đã lệch ngay khi
       KHÔNG có nhiễu thì phần tụt của nó không phải do "kém bền".
    4. SẢN LƯỢNG tách ba: `yield_match` (số hộp Nôm được gán âm — KHÔNG tra từ điển nên
       không tự chấm điểm), `yield_confirmed`, `yield_unconfirmed` (nguyên liệu
       SILVER/SYLLABLE mà bản cũ mù hoàn toàn). Trừ các cặp đã bị `confusion_fix` hạ cấp.
    5. `seeds` mặc định 3 (bản cũ vòng A dùng 1, mà biên độ giữa seed còn LỚN HƠN khoảng
       cách giữa nhiều cặp cấu hình -> thứ hạng là ngẫu nhiên).
    """
    import sys
    sys.path.insert(0, str(REPO))
    from pipeline.align_engine import anchor_align as aa
    if readings is None:
        from pipeline.align_engine.syllable_normalize import build_readings
        readings = build_readings(qn_to_nom)
    demoted = demoted or set()

    saved = {k: getattr(aa, k) for k in
             ("COST_CONFIRM", "COST_SIMILAR", "COST_DICTMISS",
              "COST_NODICT", "COST_DEL", "COST_INS", "BAND_SLACK")}
    try:
        for k, v in (cost or {}).items():
            setattr(aa, k, v)
        if band_slack is not None:
            aa.BAND_SLACK = band_slack
        base = aa.BAND_SLACK

        def slack(m, n):
            return base + 1 if (adaptive_band and abs(m - n) > 2) else base

        def align(c, s):
            return aa.realign_column([{"ocr_char": x} for x in c], s, qn_to_nom,
                                     similar, band_slack=slack(len(c), len(s)))

        # ---- SẢN LƯỢNG ----------------------------------------------------
        n_match = n_conf = n_unconf = n_del = n_ins = 0
        for col in columns:
            for o in align(col["chars"], col["syllables"]):
                if o["op"] != "match":
                    n_del += o["op"] == "del"; n_ins += o["op"] == "ins"; continue
                if (o.get("ocr_char"), o.get("syllable")) in demoted:
                    continue            # cặp này confusion_fix sẽ hạ về REVIEW -> không phải sản lượng
                n_match += 1
                if o.get("confirmed"):
                    n_conf += 1
                else:
                    n_unconf += 1

        # ---- ĐỘ BỀN + ĐỐI CHỨNG mức nhiễu 0 --------------------------------
        kept = alive = kept0 = alive0 = 0
        per: dict[str, list[int]] = {k: [0, 0] for k in PERTURBATIONS}
        for ci, col in enumerate(columns):
            ap = content_anchors(col, readings)
            if not ap:
                continue
            c0, s0 = col["chars"], col["syllables"]
            k0, a0 = _score_content(ap, [(i, j) for i, j in ap], c0, s0, c0, s0, align(c0, s0))
            kept0 += k0; alive0 += a0
            for kind in PERTURBATIONS:
                for level in LEVELS:
                    for sd in range(seeds):
                        rng = random.Random(_seed(sd, ci, kind, level))
                        c, s, live = _remap(c0, s0, ap, kind, level, rng, similar)
                        k, a = _score_content(ap, live, c0, s0, c, s, align(c, s))
                        kept += k; alive += a
                        per[kind][0] += k; per[kind][1] += a

        ret = kept / alive if alive else None
        ret0 = kept0 / alive0 if alive0 else None
        out = {"yield_match": n_match, "yield_confirmed": n_conf,
               "yield_unconfirmed": n_unconf, "yield_del": n_del, "yield_ins": n_ins,
               "anchors_tested": alive, "anchor_retention": round(ret, 5) if ret else None,
               "ret_noise0": round(ret0, 5) if ret0 else None,
               # phần TỤT VÌ NHIỄU — tách khỏi "lệch sẵn khi chưa có nhiễu"
               "ret_drop": round(ret0 - ret, 5) if (ret and ret0) else None}
        for k, (a, b) in per.items():
            out[f"ret_{k}"] = round(a / b, 5) if b else None
        return out
    finally:
        for k, v in saved.items():
            setattr(aa, k, v)


def yield_proxy(columns: list[dict], qn_to_nom: dict, similar: dict,
                band_slack: int | None = None, cost: dict | None = None) -> dict:
    """SẢN LƯỢNG — chạy align KHÔNG nhiễu loạn, đếm cặp ghép và cặp được từ điển xác nhận.

    Đây là trục thứ hai của biên Pareto. Nó là PROXY của số ô GOLD: `confirmed=True`
    tương ứng luật `s1_inter_s2_direct`/`_similar`, tức đúng phần sinh ra GOLD. Dùng proxy
    vì đo sản lượng thật phải build lại 445 trang cho MỖI cấu hình (~20 phút × 144).

    ⚠️ ĐỪNG cực đại hoá riêng chỉ số này — nới lỏng chi phí sẽ luôn làm nó tăng trong khi
    sinh thêm nhãn sai. Chỉ dùng cùng `anchor_retention` trên biên Pareto.
    """
    import sys
    sys.path.insert(0, str(REPO))
    from pipeline.align_engine import anchor_align as aa

    saved = {k: getattr(aa, k) for k in
             ("COST_CONFIRM", "COST_SIMILAR", "COST_DICTMISS",
              "COST_NODICT", "COST_DEL", "COST_INS", "BAND_SLACK")}
    try:
        for k, v in (cost or {}).items():
            setattr(aa, k, v)
        if band_slack is not None:
            aa.BAND_SLACK = band_slack
        slack = aa.BAND_SLACK
        n_match = n_conf = n_del = n_ins = 0
        for col in columns:
            ops = aa.realign_column([{"ocr_char": c} for c in col["chars"]],
                                    col["syllables"], qn_to_nom, similar, band_slack=slack)
            for o in ops:
                if o["op"] == "match":
                    n_match += 1
                    n_conf += bool(o.get("confirmed"))
                elif o["op"] == "del":
                    n_del += 1
                else:
                    n_ins += 1
        return {"yield_match": n_match, "yield_confirmed": n_conf,
                "yield_del": n_del, "yield_ins": n_ins}
    finally:
        for k, v in saved.items():
            setattr(aa, k, v)


def run_suite(columns: list[dict], qn_to_nom: dict, similar: dict,
              seeds: int = 3, band_slack: int | None = None,
              cost: dict | None = None) -> dict:
    """Chạy cả 7 loại × 3 mức × `seeds` lần. Trả một dòng số phẳng."""
    out: dict = {}
    tot_k = tot_a = 0
    for kind in PERTURBATIONS:
        kk = aa_ = 0
        for level in LEVELS:
            for sd in range(seeds):
                r = anchor_retention(columns, qn_to_nom, similar, kind, level,
                                     seed=sd, band_slack=band_slack, cost=cost)
                kk += r["anchors_kept"]; aa_ += r["anchors_alive"]
        out[f"ret_{kind}"] = round(kk / aa_, 4) if aa_ else None
        tot_k += kk; tot_a += aa_
    out["anchor_retention"] = round(tot_k / tot_a, 4) if tot_a else None
    out["anchors_tested"] = tot_a
    return out
