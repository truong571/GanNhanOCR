"""policy.py — quyết định của bước chọn chữ (thuần, không I/O): đọc/kiểm config, phần ô (in/out/none), nâng GOLD theo nhóm +
ngưỡng, L5 sửa nhãn GOLD (viết tay), quy tắc STT bảo thủ. Đòn bẩy ghi vào rule: |chon_chu:<L1|L2|L2b|L4|L5|L3>.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVER = {"direct_crop": "L1", "direct_qn": "L2", "direct_other": "L2b", "similar": "L2b", "other": "L2b",
         "syl": "L4", "nocontext": "L4", "lowpost": "L4", "syl_cropbad": "L4"}
NEVER = ("GOLD", "textonly", "notplaus", "quarantine", "syl_bridge")
BOOK_KEYS = {"enabled", "family", "model", "verifiers", "human", "tau_in", "tau_out", "vis_model", "stt_mode", "qn_geo", "np_geo"}
GEO_LEVERS = ("qn_geo", "np_geo")          # luật TN9 sách in (measure_out/_tn9/KET_QUA.md §7): P2 / P3
VERIFIERS = ("vft_T", "vft_L", "hand_T_Kinh", "hand_L_Kinh", "hand_T_DungLy", "hand_L_DungLy")
BORG = ("SachKinhThayCaBinh", "SachDungLyHoThan")
STT = ("SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11")
STT_MODES = ("bao_thu", "manh")
# sách có NHÃN NGƯỜI -> mã sự thật; bộ kiểm ảnh↔chữ nào đã học nhãn người của sách nào (models/gold_exact/MANIFEST.json,
# r4/verifier_ft/t02_train.py: vft = Borg keep fold 0–3 CẢ HAI sách + sách IHR tune; hand_*_<X> = một sách Borg + IHR tune)
OWN = {"SachKinhThayCaBinh": "B18", "SachDungLyHoThan": "B34", "LucVanTien1916": "L16", "TruyenKieu1872": "TK"}
VERIFIER_TRAIN = {"vft_T": {"B18", "B34", "TK"}, "vft_L": {"B18", "B34", "L16"},
                  "hand_T_Kinh": {"B18", "TK"}, "hand_L_Kinh": {"B18", "L16"},
                  "hand_T_DungLy": {"B34", "TK"}, "hand_L_DungLy": {"B34", "L16"}}


def book_cfg(cfg: dict, book: str) -> dict | None:
    """Mục books[book] (không phân biệt hoa/thường) đã KIỂM; không có -> None (= tắt). Sai -> ValueError (không rơi ngầm)."""
    books = cfg.get("books") or {}
    hit = [k for k in books if str(k).lower() == str(book).lower()]
    if not hit:
        return None
    b = dict(books[hit[0]] or {})
    bad = set(b) - BOOK_KEYS
    if bad:
        raise ValueError(f"config chon_chu books[{book}]: khoá lạ {sorted(bad)}")
    if not isinstance(b.get("enabled"), bool):
        raise ValueError(f"config chon_chu books[{book}].enabled phải là true/false")
    fam = b.get("family")
    if fam not in (cfg.get("families") or {}):
        raise ValueError(f"config chon_chu books[{book}].family = {fam!r} không có trong families")
    for v in b.get("verifiers") or []:
        if v not in VERIFIERS:
            raise ValueError(f"config chon_chu books[{book}]: verifier lạ {v!r}")
    own = OWN.get(hit[0])
    for v in b.get("verifiers") or []:
        if own and own in VERIFIER_TRAIN[v]:
            raise ValueError(f"config chon_chu books[{book}]: bộ kiểm {v} đã học nhãn người của chính sách này ({own}) — phải LOBO")
    if own and own in (b.get("human") or []):
        raise ValueError(f"config chon_chu books[{book}]: nguyên mẫu người phải của SÁCH KIA (LOBO), không {own}")
    for k in ("tau_in", "tau_out"):
        v = b.get(k)
        if v is not None and not (isinstance(v, (int, float)) and 0 <= float(v) <= 1.01):
            raise ValueError(f"config chon_chu books[{book}].{k} = {v!r}")
    if hit[0] in STT or b.get("stt_mode") is not None:
        if b.get("stt_mode", "bao_thu") not in STT_MODES:
            raise ValueError(f"config chon_chu books[{book}].stt_mode phải thuộc {STT_MODES}")
    for k in GEO_LEVERS:
        v = b.get(k, False)
        if not isinstance(v, bool):
            raise ValueError(f"config chon_chu books[{book}].{k} = {v!r}; cần true/false")
        if v and fam != "print":
            raise ValueError(f"config chon_chu books[{book}].{k}: luật kim_geo TN9 chỉ cho họ in (family: print)")
        b[k] = v
    b["name"] = hit[0]
    b["promo_groups"] = list((cfg["families"][fam] or {}).get("promo_groups") or [])
    b["relabel"] = bool((cfg["families"][fam] or {}).get("relabel", False))
    b["excl_gates"] = list(cfg.get("excl_gates") or [])
    b["tau_relabel"] = float(cfg.get("tau_relabel", 0.8))
    return b


def parts(F: pd.DataFrame, grp: np.ndarray, n: int, kim_col: str = "is_kim") -> np.ndarray:
    """'in' = có ứng viên kim ∈ R(âm); 'out' = không; 'none' = not_plausible / QUARANTINE (không bao giờ quyết)."""
    kk = pd.Series(((F[kim_col].values == 1) & (F.inR.values == 1)).astype(np.int8)).groupby(F.i.values).max()
    kin = np.zeros(n, bool); kin[kk.index.values] = kk.values > 0
    part = np.where(kin, "in", "out").astype(object)
    part[np.isin(grp, ["notplaus", "quarantine"])] = "none"
    return part


def decide(tier, grp, gate, part, P, top1, label, bcfg: dict, veq) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (promote, relabel, lever) theo ô. Thuần: chỉ đọc mảng. `veq` = V1+ (gold_exact.common.var_eq_plus).
    Họ không relabel (in/khắc): chỉ nâng ô có nhãn và top-1 ≡ nhãn (V1+)."""
    n = len(tier)
    Pn = np.nan_to_num(np.asarray(P, float), nan=-1.0)
    tau_in = float(bcfg["tau_in"]) if bcfg.get("tau_in") is not None else 9.0
    tau_out = float(bcfg["tau_out"]) if bcfg.get("tau_out") is not None else 9.0
    gate0 = np.array([str(g).split(":")[0] for g in gate], dtype=object)
    ok_grp = np.isin(grp, bcfg["promo_groups"]) & ~np.isin(grp, NEVER)
    promote = (np.asarray(tier) != "GOLD") & ok_grp & ~np.isin(gate0, bcfg["excl_gates"]) & (
        ((np.asarray(part) == "in") & (Pn >= tau_in)) | ((np.asarray(part) == "out") & (Pn >= tau_out)))
    promote &= np.array([bool(t) for t in top1])
    if not bcfg.get("relabel"):
        # họ in (30/09): chỉ XÁC NHẬN nhãn kim đang có — nâng khi top-1 ảnh ≡ nhãn (V1+), không bao giờ đổi chữ
        # (IHR: 38 ô L1 có top-1 ≠ kim đều là dị thể V1+ của kim; giữ nhãn kim để luật GOLD vẫn là BỘ LỌC)
        promote &= np.array([bool(l) and bool(t) and (t == l or veq(t, l)) for t, l in zip(top1, label)])
    relabel = np.zeros(n, bool)
    if bcfg.get("relabel"):
        dis = np.array([bool(t) and bool(l) and not veq(t, l) for t, l in zip(top1, label)])
        relabel = (np.asarray(tier) == "GOLD") & (np.asarray(part) == "in") & (Pn >= bcfg["tau_relabel"]) & dis
    lever = np.array([""] * n, dtype=object)
    lever[promote] = [LEVER.get(g, "") for g in np.asarray(grp)[promote]]
    lever[relabel] = "L5"
    return promote, relabel, lever


def new_labels(top1, label, veq) -> np.ndarray:
    """Nhãn sau bước: top-1, NHƯNG giữ nhãn đang có khi top-1 ≡ nhãn (V1+) — chỉ đổi chữ khi ảnh chọn chữ KHÁC hẳn."""
    return np.array([l if (l and t and (t == l or veq(t, l))) else t for t, l in zip(top1, label)], dtype=object)


def decide_geo(tier, grp, kim, syllable, geo, already, bcfg: dict, R_of, veq) -> tuple[np.ndarray, np.ndarray]:
    """Luật TN9 sách in (measure_out/_tn9/KET_QUA.md §7), chỉ ô CHƯA được nâng (already) và kim_geo ≡ kim (V1+):
      P2 qn_geo  ô REVIEW nhóm direct_qn (rule s1_inter_s2_direct*|gate:qn_count_unfixed), kim ∈ R(âm) -> GOLD, nhãn = kim
                 (không đổi); âm QN của hàng KHÔNG tin được (vị trí âm lệch) -> cờ rule |chon_chu:qn_geo.
      P3 np_geo  ô REVIEW not_plausible (âm QN là rác OCR), có chữ kim -> GOLD, nhãn = kim, âm BỎ TRỐNG, cờ |chon_chu:np_geo.
    Trả (p2, p3) — mặt nạ theo ô."""
    n = len(tier)
    tier = np.asarray(tier); grp = np.asarray(grp)
    geo_ok = np.array([bool(g) and bool(k) and len(k) == 1 and (g == k or veq(g, k)) for g, k in zip(geo, kim)])
    free = (tier == "REVIEW") & ~np.asarray(already, bool) & geo_ok
    p2 = np.zeros(n, bool); p3 = np.zeros(n, bool)
    if bcfg.get("qn_geo"):
        kin = np.array([bool(k) and k in R_of(s) for k, s in zip(kim, syllable)])
        p2 = free & (grp == "direct_qn") & kin
    if bcfg.get("np_geo"):
        p3 = free & (grp == "notplaus")
    return p2, p3


def decide_stt_bao_thu(tier, grp, kim, lt2, syllable, vtop, R_of, veq) -> tuple[np.ndarray, np.ndarray]:
    """STT bảo thủ (TN8 §4): ô KHÔNG GOLD, lt1 ∉ R, lt2 ∈ R, top-1 CHỈ-ẢNH == lt2 -> GOLD nhãn lt2. Trả (promote, nhãn)."""
    n = len(tier)
    pro = np.zeros(n, bool)
    lab = np.array([""] * n, dtype=object)
    for i in range(n):
        if tier[i] == "GOLD" or grp[i] in ("notplaus", "quarantine", "textonly"):
            continue
        R = R_of(syllable[i])
        if not lt2[i] or (kim[i] and kim[i] in R) or lt2[i] not in R:
            continue
        if vtop[i] and veq(vtop[i], lt2[i]):
            pro[i] = True; lab[i] = lt2[i]
    return pro, lab
