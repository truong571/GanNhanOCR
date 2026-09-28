"""policy.py — chính sách gold_exact: "lai" (TN4 §5/§7) + luật A và M-OCR (chính sách v3). core_loss = cột thông tin.

Thứ tự (luật đầu tiên khớp thắng):
  1. review       luật A: ảnh của ô khác (int_foreign) · rescue (nhãn không do OCR đọc) · cầu tự dạng (similar) · văn bản yếu;
                  M-OCR t50 (sách in: review như v3; STT: v3 xếp human_check -> ở đây cũng review, lý do riêng).
  2. text_only    A0_new ∪ B0_new (crop chuẩn: trắng, cắt nét, hai chữ, mực bất thường, bleed/truncated trên crop chặt mới,
                  tall mới; hộp: một hộp hai cột, hai hộp chồng nặng) ∪ CNT ∪ BC (trượt theo hộp kim/vis, cấu hình ngoài sách:
                  L16 = bc_k06v1, bộ khác = bc_k05dxv05). core_loss CHỈ vào cổng khi config core_loss.gate = true (lượt run1);
                  mặc định false — quyết định 27/09: đăng ký trước cho kết quả âm trên IHR (bỏ 46 ô đều đúng, bắt 6/61 damaged).
  3. uncertified  dị bản người chống nhãn · bộ kiểm ảnh↔chữ (chấm crop CŨ, ngưỡng TN1 τ = 0,995 LOBO) không chứng nhận
                  · sách có dị bản người (TK, KVK, L83) mà nhãn không được văn bản người chứng (TA ≠ attested).
  4. ok           ảnh giao = crop chuẩn v2.

PROFILE (28/09, TN5 — lab/thu_nghiem_anh_chu/TN5_viet_tay, đăng ký trước prereg_handwriting.json): config `profiles.handwriting`
áp cho các bộ CHỮ VIẾT TAY liệt kê ở `sets` (⊂ common.HANDWRITTEN: stt2/stt4/stt11 + Borg B18/B34):
  - `drop_gates: [cnt]`  CNT KHÔNG hạ ô viết tay (trên Borg có nhãn người: AUC nhãn 0,49 / khe 0,49 — không phân biệt);
  - `slot_gate`          cổng khe thay cho BC: bc (= BC cũ) | none | vis0 (vis_z < 0 hoặc thiếu -> text_only `HW_vis_z_duoi_0`);
  - `hand_q`, `n_hum_min` mức q của bộ kiểm viết tay (khoá của thang hand_tables) + số nguyên mẫu người tối thiểu.
Mọi bộ KHÔNG nằm trong `sets` (6 bộ in/khắc) = profile "printed" = hành vi cũ từng ô. Vắng khoá `profiles` = hành vi cũ cho mọi bộ.

LẦN ĐỌC THỨ HAI STT (28/09, `profiles.handwriting.second_read`, signals_lt2): tín hiệu `lt2_dis` (nhãn ≠ chữ kim lt2, V1+) và
`lt2_unm` (trang có lt2 nhưng không ghép được chữ) — CHỈ HẠ ô sẽ-là-ok xuống uncertified (luật CUỐI nhóm uncertified, lý do
riêng); không đổi nhãn, không đụng ô không ok. Vắng khoá trong `sig` (hoặc bộ chưa đủ cache lt2) = mảng 0 = hành vi cũ.
"""
from __future__ import annotations

import numpy as np

from .common import BORG8, EVIDENCE, HANDWRITTEN

STATUSES = ("ok", "text_only", "uncertified", "review")
SLOT_GATES = ("bc", "none", "vis0")      # cổng khe profile handwriting cài trong gói (TN5 lưới còn ad_dn0 — không được chọn, chưa cài)
DROPPABLE = ("cnt",)


def profile_hw(cfg: dict | None) -> dict | None:
    """Profile handwriting đã kiểm (None nếu config không có). Sai khoá/giá trị -> ValueError (không chạy nửa vời)."""
    hw = ((cfg or {}).get("profiles") or {}).get("handwriting")
    if not hw:
        return None
    th = (cfg or {}).get("thresholds") or {}
    sets = [str(s) for s in (hw.get("sets") or [])]
    bad = [s for s in sets if s not in HANDWRITTEN]
    if bad:
        raise ValueError(f"profiles.handwriting.sets chứa bộ không phải chữ viết tay: {bad} (chỉ {list(HANDWRITTEN)})")
    drop = [str(g) for g in (hw.get("drop_gates") or [])]
    if set(drop) - set(DROPPABLE):
        raise ValueError(f"profiles.handwriting.drop_gates chỉ nhận {list(DROPPABLE)}: {drop}")
    slot = str(hw.get("slot_gate", "bc"))
    if slot not in SLOT_GATES:
        raise ValueError(f"profiles.handwriting.slot_gate = {slot!r} không hỗ trợ (chỉ {list(SLOT_GATES)})")
    q = float(hw.get("hand_q", th.get("stt_q", 0.00015)))
    nmin = int(hw.get("n_hum_min", th.get("stt_n_hum_min", 3)))
    return dict(sets=sets, drop=drop, slot=slot, hand_q=q, n_hum_min=nmin)


def hw_mask(S8, cfg: dict | None) -> np.ndarray:
    """Ô thuộc profile handwriting (theo set8)."""
    p = profile_hw(cfg)
    return np.isin(np.asarray(S8, dtype=object), p["sets"]) if p else np.zeros(len(S8), bool)


def hand_q_of(S8, cfg: dict | None) -> np.ndarray:
    """Mức q của bộ kiểm viết tay theo ô: profile hand_q (ô thuộc profile) / thresholds.stt_q (còn lại)."""
    p = profile_hw(cfg)
    q0 = float(((cfg or {}).get("thresholds") or {}).get("stt_q", 0.00015))
    return np.where(hw_mask(S8, cfg), p["hand_q"] if p else q0, q0).astype(float)


def n_hum_min_of(S8, cfg: dict | None, th: dict | None = None) -> np.ndarray:
    """Số nguyên mẫu người tối thiểu theo ô: profile n_hum_min (ô thuộc profile) / thresholds.stt_n_hum_min (còn lại)."""
    th = th if th is not None else ((cfg or {}).get("thresholds") or {})
    n0 = th["stt_n_hum_min"]
    p = profile_hw(cfg)
    return np.where(hw_mask(S8, cfg), p["n_hum_min"] if p else n0, n0)


def slot_gate_masks(S8, sig: dict, cfg: dict | None):
    """(cnt_eff, bc_eff, vis_eff): cờ CNT/BC sau profile + cổng vis0 (chỉ ô profile có slot_gate vis0)."""
    b = lambda k: np.asarray(sig[k], bool)
    p = profile_hw(cfg)
    HW = hw_mask(S8, cfg)
    cnt = b("cnt") & ~(HW & bool(p and "cnt" in p["drop"]))
    slot = p["slot"] if p else "bc"
    bc = b("bc") & ~(HW & (slot != "bc"))
    if slot == "vis0":
        vz = np.asarray(sig["vis_z"], float)
        vis = HW & ~(np.nan_to_num(vz, nan=-np.inf) >= 0)
    else:
        vis = np.zeros(len(S8), bool)
    return cnt, bc, vis


def bc_flags(ady, adx, vis_z, cfg):
    visf = lambda v: np.nan_to_num(vis_z, nan=np.inf) <= v + 1e-12
    T = cfg["bc_T"]; L = cfg["bc_L"]
    BC_T = (np.nan_to_num(ady, nan=-1) > T["ady"]) | visf(T["vis_z"])
    BC_L = (np.nan_to_num(ady, nan=-1) > L["ady"]) | (np.nan_to_num(adx, nan=-1) > L["adx"]) | visf(L["vis_z"])
    return BC_T, BC_L


def simg(S8, s, th, cfg: dict | None = None):
    """Bộ kiểm ảnh "lai" (công thức TN3/TN4 s06a, điểm chấm CROP CŨ, ngưỡng TN1 LOBO). cfg (tuỳ chọn): profile handwriting ->
    n_hum_min theo profile cho ô viết tay (lobo_cert đã chấm ở hand_q của profile — __main__)."""
    n = len(S8)
    C5T, C5L, C2T, C2L = th["C5_T"], th["C5_L"], th["C2_T"], th["C2_L"]
    pwT, pwL = s["p_wood_T"], s["p_wood_L"]
    vsT, vsL, vX = (np.nan_to_num(s[c], nan=0.0) for c in ("viss_T", "viss_L", "viss_X"))
    out = np.zeros(n, bool)
    m = S8 == "L16"; out[m] = ((pwT >= C5T[0]) & (vsT >= C5T[1]))[m]
    m = S8 == "TK"; out[m] = ((pwL >= C5L[0]) & (vsL >= C5L[1]))[m]
    m = np.isin(S8, ["L83", "KVK"]); out[m] = ((pwT >= C5T[0]) & (pwL >= C5L[0]) & (vX >= max(C5T[1], C5L[1])))[m]
    m = S8 == "Chr"; out[m] = ((pwT >= C2T) & (pwL >= C2L))[m]
    m = np.isin(S8, ["stt2", "stt4", "stt11"])
    hand = (np.nan_to_num(s["lobo_cert"], nan=0).astype(int) == 1) & (np.nan_to_num(s["lobo_nh"], nan=0) >= n_hum_min_of(S8, cfg, th))
    out[m] = hand[m]
    # Borg (27/09): CÙNG công thức chữ viết tay, nhưng điểm lobo_* đã chấm bằng biến thể LOBO-sách (common.HAND_VARIANT:
    # B18 -> mô hình/nguyên mẫu/hiệu chuẩn học trên DungLy, B34 -> học trên Kinh). Không dùng p_wood/viss (có crop Borg).
    m = np.isin(S8, list(BORG8))
    out[m] = hand[m]
    return out


def core_loss_gate(cfg: dict) -> bool:
    """core_loss có là cổng text_only không (config core_loss.gate; vắng = false theo quyết định 27/09)."""
    return bool((cfg.get("core_loss") or {}).get("gate", False))


def decomposition(ok, lai, aint, mocr, core_flag, gate: bool, lt2=None) -> bool:
    """ok = lai − luật A − M-OCR (− core_loss nếu là cổng) (− lt2: lt2_dis ∪ lt2_unm, 28/09), từng ô."""
    ok, lai, aint, mocr, core_flag = (np.asarray(x, bool) for x in (ok, lai, aint, mocr, core_flag))
    lt2 = np.zeros(len(ok), bool) if lt2 is None else np.asarray(lt2, bool)
    return bool((ok == (lai & ~aint & ~mocr & ~(core_flag & gate) & ~lt2)).all())


LT2_REASONS = ("U_STT_lt2_khac_lt1", "U_STT_lt2_khong_ghep_duoc")


def lt2_masks(sig: dict, n: int):
    """(lt2_dis, lt2_unm) từ sig; vắng khoá -> 0 (hành vi cũ)."""
    g = lambda k: np.asarray(sig[k], bool) if k in sig else np.zeros(n, bool)
    return g("lt2_dis"), g("lt2_unm")


def decide(S8: np.ndarray, sig: dict, cfg: dict):
    """sig: mảng theo ô (bool/float). Trả (gold_exact, reason, masks dict)."""
    n = len(S8)
    dec = np.array([""] * n, dtype=object); why = np.array([""] * n, dtype=object)
    STT = np.isin(S8, ["stt2", "stt4", "stt11"]); PRINT = ~STT
    BORG = np.isin(S8, list(BORG8))     # 27/09: tập đánh giá chữ viết tay; lý do riêng, đặt TRƯỚC dòng PRINT (không đổi 8 bộ cũ)

    def put(mask, val, reason):
        mm = np.asarray(mask, bool) & (dec == "")
        dec[mm] = val; why[mm] = reason
    b = lambda k: np.asarray(sig[k], bool)
    # ---- 1. review (luật A + M-OCR)
    put(b("int_foreign"), "review", "A0a_anh_cua_o_khac")
    put(b("rescue"), "review", "A1_rescue_nhan_khong_do_OCR_doc")
    put(b("similar"), "review", "A2a_cau_tu_dang")
    put(b("weak_text"), "review", "A2b_van_ban_yeu")
    put(PRINT & b("mocr"), "review", "M_ocr_t50")
    put(STT & b("mocr"), "review", "M_ocr_t50_STT")
    # ---- 2. text_only (A0_new ∪ B0_new ∪ CNT ∪ BC ∪ core_loss)
    put(b("f_blank"), "text_only", "A0_crop_chuan_trang")
    put(b("f_cut"), "text_only", "A0_crop_chuan_cat_net")
    put(b("dup_bbox"), "text_only", "A0b_mot_hop_hai_cot")
    put(b("ov_heavy"), "text_only", "B0_hai_hop_chong_nang")
    put(b("f_two"), "text_only", "B0_hai_chu")
    put(b("f_ink"), "text_only", "B0_muc_bat_thuong_so_voi_nhan")
    put(b("bleed_new"), "text_only", "B0_muc_la_bleed_crop_moi")
    put(b("trunc_new"), "text_only", "B0_truncated_crop_moi")
    put(b("tall_new"), "text_only", "B0_tall_crop_moi")
    # profile handwriting (28/09, TN5): CNT bỏ ở ô viết tay; BC thay bằng cổng khe của profile (bc = như cũ)
    cnt_e, bc_e, vis_e = slot_gate_masks(S8, sig, cfg)
    put(cnt_e, "text_only", "CNT_so_chu_OCR_cot_khac_so_am_QN")
    put(bc_e, "text_only", "BC_truot_theo_hop_kim_vis")
    put(vis_e, "text_only", "HW_vis_z_duoi_0")
    if core_loss_gate(cfg):
        put(b("core_loss_flag"), "text_only", "core_loss_crop_chuan_cat_vao_chu")
    # ---- 3. uncertified
    ta = np.asarray(sig["ta"], dtype=object)
    TA_SETS = np.isin(S8, ["TK", "KVK", "L83"])
    put(TA_SETS & (ta == "contradicted"), "uncertified", "U_di_ban_nguoi_chong_nhan")
    SIMG = b("simg")
    nmin = n_hum_min_of(S8, cfg)
    put(STT & ~SIMG & (np.nan_to_num(sig["lobo_nh"], nan=0) < nmin), "uncertified", "U_STT_thieu_nguyen_mau_nguoi")
    put(STT & ~SIMG, "uncertified", "U_STT_bo_kiem_viet_tay_khong_chung_nhan")
    put(BORG & ~SIMG & (np.nan_to_num(sig["lobo_nh"], nan=0) < nmin), "uncertified", "U_Borg_thieu_nguyen_mau_nguoi_LOBO")
    put(BORG & ~SIMG, "uncertified", "U_Borg_bo_kiem_viet_tay_LOBO_khong_chung_nhan")
    put(PRINT & ~SIMG, "uncertified", "U_bo_kiem_anh_duoi_nguong_TN1")
    put(TA_SETS & (ta != "attested"), "uncertified", "U_khong_co_van_ban_nguoi_chung")
    # lần đọc thứ hai STT (28/09): CHỈ HẠ ô sẽ-là-ok; đặt CUỐI nhóm uncertified -> mọi ô khác giữ nguyên trạng thái + lý do
    l2d, l2u = lt2_masks(sig, n)
    put(STT & l2d, "uncertified", LT2_REASONS[0])
    put(STT & l2u, "uncertified", LT2_REASONS[1])
    put(np.ones(n, bool), "ok", "")
    assert (dec != "").all()
    A0n = b("int_foreign") | b("dup_bbox") | b("f_blank") | b("f_cut")
    B0n = b("dup_bbox") | b("ov_heavy") | ~b("one_char_ok") | b("tall_new")
    H1n = ~(A0n | B0n | cnt_e | bc_e | vis_e)          # printed: = ~(A0 | B0 | cnt | bc) như cũ
    TA_OK = ~TA_SETS | (ta == "attested")
    masks = dict(A0_new=A0n, B0_new=B0n, H1_new=H1n, SIMG=SIMG, TA_OK=TA_OK, lai=H1n & SIMG & TA_OK,
                 AINT=b("int_foreign") | b("rescue") | b("similar") | b("weak_text"), LT2=STT & (l2d | l2u))
    return dec, why, masks


def evidence(S8):
    return np.array([EVIDENCE.get(s, "suy_doan") for s in S8], dtype=object)
