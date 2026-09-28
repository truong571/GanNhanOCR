"""signals_lt2.py — tín hiệu `lt2_agree` (28/09): LẦN ĐỌC THỨ HAI của kim ở chế độ Nôm (lang_type=2) cho STT chép tay.

Nguồn: pipeline/tools/stt_reocr_lt2.py OCR lại ĐÚNG ảnh STT, ĐÚNG quy trình của lt1, chỉ đổi lang_type=2 ->
prepared/<SachThanhTruyenN>/kim_raw_lt2/<page>_lt2.json (cùng lược đồ cache lt1 + kim_params). Mô-đun này CHỈ ĐỌC các tệp đó
và cache lt1 prepared/<Sách>/detected/<page>_ocr_cache.json — không bao giờ ghi prepared/.

Ghép ô ↔ chữ lt2 = PHƯƠNG PHÁP `line` của scripts/measure/stt_lt2_eval.py (bản chép NGUYÊN các hàm raw_lines/sub_box/locate/
best_line/align/PagePair.by_line; selftest của gói kiểm tương đương từng kết quả với bản đo trên dữ liệu ngẫu nhiên):
hộp chữ lt1 của ô = cột kim dựng lại (signals_geom.kim_vs_box, chữ tại nom_idx phải == ocr_char) -> dòng kim lt1 chứa ô ->
dòng lt2 có IoU hộp dòng ≥ 0,5 -> căn chuỗi Needleman–Wunsch có dải (khớp V1+ +2, lệch −1, khoảng −1) -> chữ lt2 cặp với ô.

Hiệu chuẩn (docs/STT_LT2_2026-09-28.md §4 — THẠCH BẢN in LVT1883/KVK1884 so dị bản, ước lượng cận dưới; với STT chép tay
chỉ là SUY ĐOÁN): P(nhãn đúng | lt1 == lt2) 89,8 % [89,3; 90,3] vs 11,3 % khi lệch, nền 81,8 %.

LUẬT (policy.decide, CHỈ HẠ, đặt CUỐI nhóm uncertified): ô sẽ là `ok` mà
  - `lt2_dis`: ghép được chữ lt2 nhưng NHÃN ≠ lt2 (V1+; nhãn STT = chữ lt1 tại nom_idx trừ ô đã sửa nhãn) -> uncertified
    `U_STT_lt2_khac_lt1`;
  - `lt2_unm`: trang CÓ lt2 nhưng không ghép được chữ lt2 cho ô (khoảng trống/không dòng/không định vị) và config
    `unmatched: demote` -> uncertified `U_STT_lt2_khong_ghep_duoc` (bảo thủ như đề xuất cổng của bản đo).
KHÔNG BAO GIỜ dùng lt2 để đổi nhãn. Ô không ok giữ nguyên trạng thái + lý do.

KÍCH HOẠT (`profiles.handwriting.second_read.mode`): auto = bật cho MỘT bộ (stt2/stt4/stt11) chỉ khi MỌI trang lt1 của bộ có
cache lt2 HỢP LỆ (lang_type 2, cùng image_hash với lt1, toạ độ fullpage); thiếu/hỏng dù một trang -> TẮT cho cả bộ (ghi rõ lý
do + số trang vào GOLD_EXACT.md/summary.json). off = tắt. `--second-read partial` (CLI, CHỈ ĐỂ ĐO, cấm với --publish) = bật trên
các trang đã có lt2. Tắt ⇒ mọi quyết định y hệt khi không có mô-đun này (invariant `lt2_chi_ha` so từng cell_uid).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import PREP_DIR, REPO, var_eq_plus

LT2_DIR = "kim_raw_lt2"
STT_SETS = ("stt2", "stt4", "stt11")      # chỉ STT có cache lt1 (detected/) + lt2 (kim_raw_lt2/)
MODES = ("auto", "off")                   # config; CLI thêm "partial" (chỉ đo)
UNMATCHED = ("demote", "keep")
X_OV, Y_OV, LINE_IOU = 0.5, 0.5, 0.5      # = stt_lt2_eval.py
REASON_DIS = "U_STT_lt2_khac_lt1"
REASON_UNM = "U_STT_lt2_khong_ghep_duoc"
MATCH_OK = "ok"


def veq(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a == b or var_eq_plus(a, b))


# ------------------------------------------------------------------------------------------------ cấu hình
def second_read_cfg(cfg: dict | None) -> dict:
    """profiles.handwriting.second_read đã kiểm. Vắng -> mode off (hành vi cũ). Sai khoá/giá trị -> ValueError."""
    hw = (((cfg or {}).get("profiles") or {}).get("handwriting") or {})
    sr = hw.get("second_read")
    if sr is None:
        return dict(mode="off", sets=[], unmatched="demote", declared=False)
    if isinstance(sr, str):
        sr = dict(mode=sr)
    mode = str(sr.get("mode", "auto"))
    if mode not in MODES:
        raise ValueError(f"profiles.handwriting.second_read.mode = {mode!r} (chỉ {list(MODES)}; 'partial' chỉ qua CLI để đo)")
    sets = [str(s) for s in (sr.get("sets") or list(STT_SETS))]
    hw_sets = [str(s) for s in (hw.get("sets") or [])]
    bad = [s for s in sets if s not in STT_SETS or s not in hw_sets]
    if bad:
        raise ValueError(f"second_read.sets chỉ nhận bộ STT thuộc profile handwriting: {bad}")
    um = str(sr.get("unmatched", "demote"))
    if um not in UNMATCHED:
        raise ValueError(f"second_read.unmatched = {um!r} (chỉ {list(UNMATCHED)})")
    return dict(mode=mode, sets=sets, unmatched=um, declared=True, calib=sr.get("calib"))


# ------------------------------------------------------------------------------------------------ cache lt1/lt2
def lt1_caches(s8: str) -> list:
    return sorted((REPO / "prepared" / PREP_DIR[s8] / "detected").glob("*_ocr_cache.json"))


def _page(p: Path) -> str:
    return p.name[: -len("_ocr_cache.json")]


def load_pages(s8: str) -> dict:
    """{lt1: [trang], lt2: {trang: (rec_lt1, rec_lt2)}, bad: [(trang, lý do)]} — trang lt2 hợp lệ = lang_type 2, cùng image_hash
    với lt1, toạ độ fullpage (cả hai). Đọc hỏng (tệp đang được ghi) = bad."""
    lt2, bad, lt1 = {}, [], []
    for cf in lt1_caches(s8):
        pg = _page(cf); lt1.append(pg)
        f = REPO / "prepared" / PREP_DIR[s8] / LT2_DIR / f"{pg}_lt2.json"
        if not f.exists():
            continue
        try:
            r2 = json.loads(f.read_text(encoding="utf-8")); r1 = json.loads(cf.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            bad.append((pg, f"đọc lỗi {type(e).__name__}")); continue
        why = [w for w, b in (("lang_type≠2", (r2.get("kim_params") or {}).get("lang_type") != 2),
                              ("ảnh khác lt1", r2.get("image_hash") != r1.get("image_hash")),
                              ("toạ độ không fullpage", r2.get("coords_space") != "fullpage" or r1.get("coords_space") != "fullpage"))
               if b]
        if why:
            bad.append((pg, ",".join(why))); continue
        lt2[pg] = (r1, r2)
    return dict(lt1=lt1, lt2=lt2, bad=bad)


def activation(present, cfg: dict | None, override: str | None = None):
    """-> (sr_cfg, {s8: record}, {s8: pages dict}) cho các bộ STT có trong G. record: mode, active, pages_lt1/lt2, bad, why."""
    sr = second_read_cfg(cfg)
    mode = override or sr["mode"]
    if mode not in MODES + ("partial",):
        raise ValueError(f"--second-read = {mode!r} (auto | off | partial)")
    recs, pages = {}, {}
    for s8 in STT_SETS:
        if s8 not in set(present):
            continue
        if s8 not in sr["sets"] and mode != "partial":
            recs[s8] = dict(mode=mode, active=False, why="bộ không khai trong second_read.sets", pages_lt1=None, pages_lt2=None)
            continue
        P = load_pages(s8)
        n1, n2, nb = len(P["lt1"]), len(P["lt2"]), len(P["bad"])
        full = n1 > 0 and n2 == n1 and nb == 0
        if mode == "off":
            act, why = False, "second_read: off"
        elif mode == "partial":
            act, why = n2 > 0, ("CHỈ ĐO: bật trên trang đã có lt2" if n2 else "chưa có trang lt2 nào")
        else:
            act = full
            why = ("đủ cache lt2" if full else
                   f"THIẾU cache lt2 ({n2}/{n1} trang hợp lệ" + (f", {nb} trang hỏng" if nb else "") + ") -> TẮT cho cả bộ")
        recs[s8] = dict(mode=mode, active=bool(act), why=why, pages_lt1=n1, pages_lt2=n2, bad=[f"{p}: {w}" for p, w in P["bad"][:5]],
                        n_bad=nb)
        if act:
            pages[s8] = P
    return sr, recs, pages


# ------------------------------------------------------------------------------------------------ ghép (bản chép stt_lt2_eval)
def raw_lines(boxes) -> list:
    out = []
    for box in boxes or []:
        valid = [ch for ch in (box.get("transcription") or "").strip() if ch.strip()]
        if not valid:
            continue
        p = box["points"]
        out.append(dict(x1=p[0][0], y1=p[0][1], x2=p[1][0], y2=p[2][1], s=valid))
    return out


def sub_box(L: dict, i: int) -> list:
    h = (L["y2"] - L["y1"]) / len(L["s"])
    return [L["x1"], int(L["y1"] + h * i), L["x2"], int(L["y1"] + h * (i + 1))]


def locate(ch: str, bbox, lines: list):
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    cy, best = (y1 + y2) / 2, None
    for li, L in enumerate(lines):
        ox = min(x2, L["x2"]) - max(x1, L["x1"])
        if ox < 0.5 * max(1.0, min(x2 - x1, L["x2"] - L["x1"])) or not (L["y1"] - 1 <= cy <= L["y2"] + 1):
            continue
        n = len(L["s"]); est = (cy - L["y1"]) / ((L["y2"] - L["y1"]) / n) - 0.5
        cands = [k for k, c in enumerate(L["s"]) if c == ch]
        if not cands:
            continue
        k = min(cands, key=lambda k: abs(k - est))
        if abs(k - est) <= 2.5 and (best is None or abs(k - est) < best[2]):
            best = (li, k, abs(k - est))
    return best[:2] if best else None


def best_line(L: dict, lines: list):
    a = (L["x2"] - L["x1"]) * (L["y2"] - L["y1"])
    bi, bv = None, 0.0
    for i, M in enumerate(lines):
        iw = min(L["x2"], M["x2"]) - max(L["x1"], M["x1"]); ih = min(L["y2"], M["y2"]) - max(L["y1"], M["y1"])
        if iw <= 0 or ih <= 0:
            continue
        inter = iw * ih
        v = inter / (a + (M["x2"] - M["x1"]) * (M["y2"] - M["y1"]) - inter)
        if v > bv:
            bi, bv = i, v
    return bi if bv >= LINE_IOU else None


def align(s1: list, s2: list, eq=veq) -> list:
    n1, n2 = len(s1), len(s2)
    if n1 == 0:
        return []
    band = abs(n1 - n2) + 2
    NEG = -1e9
    S = np.full((n1 + 1, n2 + 1), NEG)
    B = np.zeros((n1 + 1, n2 + 1), np.int8)
    S[0, 0] = 0.0
    for i in range(n1 + 1):
        c = i * n2 / n1
        for j in range(max(0, int(np.floor(c - band))), min(n2, int(np.ceil(c + band))) + 1):
            if i == 0 and j == 0:
                continue
            best, mv = NEG, 0
            if i > 0 and j > 0 and S[i - 1, j - 1] > NEG:
                v = S[i - 1, j - 1] + (2.0 if eq(s1[i - 1], s2[j - 1]) else -1.0)
                if v > best:
                    best, mv = v, 1
            if i > 0 and S[i - 1, j] > NEG and S[i - 1, j] - 1.0 > best:
                best, mv = S[i - 1, j] - 1.0, 2
            if j > 0 and S[i, j - 1] > NEG and S[i, j - 1] - 1.0 > best:
                best, mv = S[i, j - 1] - 1.0, 3
            S[i, j], B[i, j] = best, mv
    m = [None] * n1
    i, j = n1, n2
    if S[i, j] <= NEG:
        return [k if k < n2 else None for k in range(n1)]
    while i > 0 or j > 0:
        mv = B[i, j]
        if mv == 1:
            m[i - 1] = j - 1; i, j = i - 1, j - 1
        elif mv == 2:
            i -= 1
        else:
            j -= 1
    return m


class PagePair:
    """Một trang: bên CHUẨN (lt1, hộp ô đã biết) và bên KIA (lt2). Chỉ phương pháp `line` (phương pháp chính của bản đo)."""

    def __init__(self, ref_boxes, oth_boxes, eq=veq):
        self.ref, self.oth, self.eq = raw_lines(ref_boxes), raw_lines(oth_boxes), eq
        self._aln = {}

    def by_line(self, ch, bbox) -> dict:
        loc = locate(ch, bbox, self.ref)
        if loc is None:
            return dict(st_line="no_loc")
        li, k = loc
        L = self.ref[li]
        if li not in self._aln:
            oi = best_line(L, self.oth)
            self._aln[li] = (oi, align(L["s"], self.oth[oi]["s"], self.eq) if oi is not None else None)
        oi, m = self._aln[li]
        if oi is None:
            return dict(st_line="no_line", n1=len(L["s"]))
        O = self.oth[oi]
        d = dict(n1=len(L["s"]), n2=len(O["s"]))
        if m[k] is None:
            d["st_line"] = "gap"
            return d
        d.update(st_line="ok", oth_line=O["s"][m[k]], oth_line_box=sub_box(O, m[k]))
        return d


# ------------------------------------------------------------------------------------------------ tín hiệu theo ô
def signal(G: pd.DataFrame, KB: pd.DataFrame, cfg: dict | None, override: str | None = None, log=print) -> dict:
    """G: ô GOLD (cell_uid, set8, page, ocr_char, label); KB: kim_vs_box (cell_uid, kim_char, kim_box). Trả mảng theo ô
    lt2_char / lt2_match / lt2_dis / lt2_unm + bản ghi kích hoạt `record` (cho summary/GOLD_EXACT.md)."""
    n = len(G)
    S8 = G.set8.values
    sr, recs, pages = activation(sorted(set(S8)), cfg, override)
    ch = np.array([""] * n, dtype=object)
    st = np.array([""] * n, dtype=object)
    dis = np.zeros(n, bool); unm = np.zeros(n, bool)
    self_ctl = [0, 0]                                  # ô có hộp lt1 · tự ghép với chính lt1 cho đúng chữ
    kb = KB.set_index("cell_uid").reindex(G.cell_uid)
    kchar = kb["kim_char"].fillna("").values if "kim_char" in kb else np.array([""] * n, dtype=object)
    kbox = kb["kim_box"].values if "kim_box" in kb else np.array([None] * n, dtype=object)
    for s8, P in pages.items():
        idx = np.nonzero(S8 == s8)[0]
        by_page = pd.Series(idx).groupby(G.page.values[idx]).apply(list).to_dict()
        for pg, ii in by_page.items():
            pair = P["lt2"].get(pg)
            if pair is None:
                st[ii] = "no_lt2_page"; continue
            r1, r2 = pair
            pp = PagePair(r1.get("boxes_raw"), r2.get("boxes_raw"))
            me = PagePair(r1.get("boxes_raw"), r1.get("boxes_raw"))
            for i in ii:
                b = kbox[i]
                if not isinstance(b, (list, tuple)) or len(b) < 4:
                    st[i] = "no_lt1_box"; continue
                if kchar[i] != G.ocr_char.values[i]:
                    st[i] = "lt1_mismatch"; continue
                d = pp.by_line(kchar[i], [float(v) for v in b[:4]])
                self_ctl[0] += 1
                self_ctl[1] += int(me.by_line(kchar[i], [float(v) for v in b[:4]]).get("oth_line") == kchar[i])
                st[i] = d.get("st_line", "no_loc")
                if st[i] == MATCH_OK:
                    ch[i] = d["oth_line"]
    on = np.isin(S8, list(pages)) & (st != "") & (st != "no_lt2_page")
    matched = on & (st == MATCH_OK)
    lab = G.label.values
    agree = np.array([veq(a, c) if m else False for a, c, m in zip(lab, ch, matched)])
    dis = matched & ~agree
    unm = on & ~matched & (sr["unmatched"] == "demote")
    per = {}
    for s8, r in recs.items():
        m = S8 == s8
        mo = m & on
        per[s8] = dict(r, cells=int(m.sum()), cells_on_lt2_pages=int(mo.sum()), matched=int((mo & matched).sum()),
                       agree=int((mo & agree).sum()), disagree=int((mo & dis).sum()),
                       unmatched=int((mo & ~matched).sum()),
                       status={k: int(v) for k, v in pd.Series(st[mo]).value_counts().items()})
    rec = dict(mode=override or sr["mode"], config_mode=sr["mode"], declared=sr["declared"], unmatched=sr["unmatched"],
               sets=sr["sets"], active_sets=sorted(pages), per_set=per, self_control=self_ctl,
               self_control_pass=(self_ctl[0] == self_ctl[1]) if self_ctl[0] else None, calib=sr.get("calib"),
               partial=(override == "partial"))
    for s8, r in per.items():
        log(f"lt2 {s8}: {'BẬT' if r['active'] else 'TẮT'} — {r['why']}"
            + (f" · ô trên trang lt2 {r['cells_on_lt2_pages']} · ghép {r['matched']} · khác {r['disagree']} · không ghép {r['unmatched']}"
               if r["active"] else ""))
    return dict(lt2_char=ch, lt2_match=st, lt2_dis=dis, lt2_unm=unm, record=rec)
