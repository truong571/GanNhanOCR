"""signals_text.py — tín hiệu VĂN BẢN / luật nhãn của ô GOLD (CPU, 0 API).

  luật A (gate_v2 g01_inputs, chính sách v3): rescue = rule self_training_rescue; similar = s1_inter_s2_similar (cầu tự dạng);
          weak_text = s1_inter_s2_direct_am_sua_dau | s1_inter_s2_direct_lowp | corpus_reading* (văn bản yếu).
  chữ kề nb_prev/nb_next = (nhãn hoặc ocr_char) của ô nom_idx ± 1 cùng cột trong dataset/_ALL (mọi tầng) — measure/m03.
  TA (text_attested, r5): nhãn so với chữ NGƯỜI của dị bản đã căn (V1+ ∪ bảng quy ước mã IHR↔NF):
          KVK ← Kiều 1871 LVD / 1872 DMT, L83 ← LVT1916 NF (convmap L16+TK); TK ← Kiều 1871 LVD (convmap L16, giữ ngoài theo
          sách). TA CHỈ phụ thuộc văn bản người của DỊ BẢN — KHÔNG nhìn GT người IHR (sửa N6 28/09: trước đó ô TK không có GT
          IHR bị 'na' -> rò rỉ nhãn đo vào quyết định; tài sản ta_refs.pt không còn cột has_gt). Ô TK trúng cổng (d) mô phỏng
          (ref gần hình, không trùng hẳn — như cổng thạch bản) -> 'na'. Ô không có trong bảng căn (ô mới) -> 'na' (không chứng
          được -> không ok). Âm của ô đổi so với lúc căn -> bỏ tham chiếu -> unattestable.
  M-OCR t50 (gate_v2 cấu hình khuyến nghị theo bộ): M_ocr_t50 = cờ m_hom ≤ t50 và KHÔNG bị lý do đứng trước trong thứ tự
          gate_assign bắt (A0a, A1, A2a, A0c, A0b, kim, vis, A2b) — chỉ ở bộ có mocr trong cấu hình (STT, L83, KVK).
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .common import REPO, is_pua, rd, var_eq_plus

WEAK = {"s1_inter_s2_direct_am_sua_dau", "s1_inter_s2_direct_lowp"}


def rule_flags(G: pd.DataFrame) -> dict:
    r = G.rule.astype(str).values
    corpus = np.array([x.startswith("corpus_reading") for x in r])
    return dict(rescue=r == "self_training_rescue", similar=r == "s1_inter_s2_similar",
                weak_text=np.isin(r, list(WEAK)) | corpus)


def neighbours(G: pd.DataFrame, all_dir: Path):
    """nb_prev/nb_next (m03_build_table): khoá (book_set, book, page, column, nom_idx) trên MỌI dòng _ALL."""
    L = rd(Path(all_dir) / "labels.csv", usecols=["cell_uid", "book_set", "book", "page", "column", "ocr_char", "label"])
    L["ni"] = L.cell_uid.str.extract(r"/n(\d+)/")[0].astype(float)
    key = {(a, b, c, d, int(n)): (l or o) for a, b, c, d, n, o, l in
           zip(L.book_set, L.book, L.page, L.column, L.ni, L.ocr_char, L.label) if n == n}
    T = rd(Path(all_dir) / "labels_trace.csv", usecols=["cell_uid", "nom_idx"])
    ni = pd.to_numeric(G.cell_uid.map(dict(zip(T.cell_uid, T.nom_idx))), errors="coerce").fillna(-99).astype(int).values
    prv = [key.get((a, b, c, d, n - 1), "") for a, b, c, d, n in zip(G.book_set, G.book, G.page, G.column, ni)]
    nxt = [key.get((a, b, c, d, n + 1), "") for a, b, c, d, n in zip(G.book_set, G.book, G.page, G.column, ni)]
    return np.array(prv, dtype=object), np.array(nxt, dtype=object)


def old_crop_flag(G: pd.DataFrame, all_dir: Path) -> np.ndarray:
    """crop_quality_flag của crop CŨ (labels_trace; ô rescue không có -> 'NA_rescue' như g01)."""
    T = rd(Path(all_dir) / "labels_trace.csv", usecols=["cell_uid", "crop_quality_flag"])
    m = dict(zip(T.cell_uid, T.crop_quality_flag))
    return np.array([m.get(u, "") or "NA_rescue" for u in G.cell_uid], dtype=object)


# ================================================================================================ text_attested
_AP = None


def _ap():
    global _AP
    if _AP is None:
        sys.path.insert(0, str(REPO / "scripts/measure"))
        import auto_precision as AP  # chỉ đọc hàm (canon_syl, dicts)
        _AP = AP
    return _AP


def pua_like(c):
    return bool(c) and (is_pua(c) or ord(c) >= 0x30000)


def classify(label, refs, syl, conv=None):
    """== r5/text_attested/ta_lib.classify -> (cls, sub)."""
    refs = [r for r in refs if r]
    if not label:
        return "no_label", ""
    if not refs:
        return "unattestable", ""
    if any(var_eq_plus(label, r) for r in refs):
        return "attested", ""
    if conv and any((label, r) in conv for r in refs):
        return "attested", "conv"
    AP = _ap(); D = AP.dicts()
    if all(pua_like(r) for r in refs) and not pua_like(label):
        return "contradicted", "ref_pua"
    if any((label in D["sim"].get(r, []) or r in D["sim"].get(label, [])) for r in refs if not pua_like(r)):
        return "contradicted", "similar"
    R = D["R"].get(AP.canon_syl(syl or ""), set())
    if any(r in R for r in refs):
        return "contradicted", "ref_in_R"
    return "contradicted", "other"


def gate_d_hit(label, refs_exact):
    """== ta_lib.gate_d_hit (tier GOLD): không ref nào bằng hẳn ∧ có ref không-PUA gần hình với nhãn."""
    if not label:
        return False
    D = _ap().dicts()
    refs = [r for r in refs_exact if r]
    if not refs or any(label == r for r in refs):
        return False
    return any((not pua_like(r)) and (label in D["sim"].get(r, []) or r in D["sim"].get(label, [])) for r in refs)


TA_BOOKS = ("KimVanKieu1884", "LucVanTien1883", "TruyenKieu1872")


def text_attested(G: pd.DataFrame, assets, z: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(ta, ta_sub) theo ô. z = nội dung ta_refs.pt (mặc định nạp từ tài sản). Không đọc khoá GT nào (has_gt/gt_char)."""
    z = assets.load("ta_refs.pt") if z is None else z
    AP = _ap()
    conv = {tg: {tuple(k.split("\t")) for k in v} for tg, v in z["convmap"].items()}
    CONV_FW = set()
    for tg in ("L16", "TK"):
        for a, b in conv[tg]:
            CONV_FW.add((a, b)); CONV_FW.add((b, a))
    CONV_TK = set()
    for a, b in conv["L16"]:
        CONV_TK.add((a, b)); CONV_TK.add((b, a))
    idx = {u: i for i, u in enumerate(z["cell_uid"])}
    ta = np.array(["na"] * len(G), dtype=object); sub = np.array([""] * len(G), dtype=object)
    for j, (u, bs, lab, syl) in enumerate(zip(G.cell_uid, G.book_set, G.label, G.syllable)):
        if bs not in TA_BOOKS:
            continue
        i = idx.get(u)
        if i is None:
            continue
        same_syl = AP.canon_syl(syl or "") == AP.canon_syl(z["syllable"][i] or "")
        if bs == "TruyenKieu1872":
            r = z["rev_h71"][i] if same_syl else ""
            if gate_d_hit(lab, [r]):
                continue
            ta[j], sub[j] = classify(lab, [r], syl, CONV_TK)
        else:
            refs = [z["ref71"][i], z["ref72"][i]] if bs == "KimVanKieu1884" else [z["ref16"][i]]
            if not same_syl:
                refs = []
            ta[j], sub[j] = classify(lab, refs, syl, CONV_FW)
    return ta, sub


def ta_coverage(G: pd.DataFrame, z: dict) -> dict:
    """Tỉ lệ ô GOLD (theo book_set có TA) có ÍT NHẤT một chữ người dị bản cùng âm — độ phủ tham chiếu (invariant mềm)."""
    AP = _ap()
    idx = {u: i for i, u in enumerate(z["cell_uid"])}
    out = {}
    for bs in TA_BOOKS:
        m = (G.book_set == bs).values
        if not m.any():
            continue
        n = k = 0
        for u, syl in zip(G.cell_uid[m], G.syllable[m]):
            n += 1
            i = idx.get(u)
            if i is None or AP.canon_syl(syl or "") != AP.canon_syl(z["syllable"][i] or ""):
                continue
            refs = [z["rev_h71"][i]] if bs == "TruyenKieu1872" else (
                [z["ref71"][i], z["ref72"][i]] if bs == "KimVanKieu1884" else [z["ref16"][i]])
            k += any(refs)
        out[bs] = dict(n=n, with_ref=k, frac=round(k / n, 5))
    return out


# ================================================================================================ M-OCR theo thứ tự cổng
def mocr_gate(G, mocr_cons, int_foreign, rules, old_cqf, dup_bbox, ady, adx, vis_z, gate_cfg: dict) -> np.ndarray:
    """M_ocr_t50 của gate_v2 (gate_assign, cấu hình khuyến nghị theo bộ, thứ tự: A0a, A1, A2a, A0c, A0b, kim, vis, A2b, mocr)."""
    n = len(G)
    out = np.zeros(n, bool)
    bs = G.book_set.values
    for s, c in gate_cfg.items():
        if not c.get("mocr"):
            continue
        m = bs == s
        kim = c.get("kim")
        if kim == "0.5dx":
            kf = (np.nan_to_num(ady, nan=-1) > 0.5) | (np.nan_to_num(adx, nan=-1) > 0.5)
        elif kim is None:
            kf = np.zeros(n, bool)
        else:
            kf = np.nan_to_num(ady, nan=-1) > float(kim)
        v = c.get("vis")
        vf = (np.nan_to_num(vis_z, nan=np.inf) <= float(v) + 1e-12) if v is not None else np.zeros(n, bool)
        earlier = (int_foreign | rules["rescue"] | rules["similar"] | (old_cqf == "blank") | (old_cqf == "truncated")
                   | dup_bbox | kf | vf | rules["weak_text"])
        out |= m & mocr_cons & ~earlier
    return out
