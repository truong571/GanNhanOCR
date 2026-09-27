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
"""
from __future__ import annotations

import numpy as np

from .common import BORG8, EVIDENCE

STATUSES = ("ok", "text_only", "uncertified", "review")


def bc_flags(ady, adx, vis_z, cfg):
    visf = lambda v: np.nan_to_num(vis_z, nan=np.inf) <= v + 1e-12
    T = cfg["bc_T"]; L = cfg["bc_L"]
    BC_T = (np.nan_to_num(ady, nan=-1) > T["ady"]) | visf(T["vis_z"])
    BC_L = (np.nan_to_num(ady, nan=-1) > L["ady"]) | (np.nan_to_num(adx, nan=-1) > L["adx"]) | visf(L["vis_z"])
    return BC_T, BC_L


def simg(S8, s, th):
    """Bộ kiểm ảnh "lai" (công thức TN3/TN4 s06a, điểm chấm CROP CŨ, ngưỡng TN1 LOBO)."""
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
    hand = (np.nan_to_num(s["lobo_cert"], nan=0).astype(int) == 1) & (np.nan_to_num(s["lobo_nh"], nan=0) >= th["stt_n_hum_min"])
    out[m] = hand[m]
    # Borg (27/09): CÙNG công thức chữ viết tay, nhưng điểm lobo_* đã chấm bằng biến thể LOBO-sách (common.HAND_VARIANT:
    # B18 -> mô hình/nguyên mẫu/hiệu chuẩn học trên DungLy, B34 -> học trên Kinh). Không dùng p_wood/viss (có crop Borg).
    m = np.isin(S8, list(BORG8))
    out[m] = hand[m]
    return out


def core_loss_gate(cfg: dict) -> bool:
    """core_loss có là cổng text_only không (config core_loss.gate; vắng = false theo quyết định 27/09)."""
    return bool((cfg.get("core_loss") or {}).get("gate", False))


def decomposition(ok, lai, aint, mocr, core_flag, gate: bool) -> bool:
    """ok = lai − luật A − M-OCR (− core_loss nếu là cổng), từng ô."""
    ok, lai, aint, mocr, core_flag = (np.asarray(x, bool) for x in (ok, lai, aint, mocr, core_flag))
    return bool((ok == (lai & ~aint & ~mocr & ~(core_flag & gate))).all())


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
    put(b("cnt"), "text_only", "CNT_so_chu_OCR_cot_khac_so_am_QN")
    put(b("bc"), "text_only", "BC_truot_theo_hop_kim_vis")
    if core_loss_gate(cfg):
        put(b("core_loss_flag"), "text_only", "core_loss_crop_chuan_cat_vao_chu")
    # ---- 3. uncertified
    ta = np.asarray(sig["ta"], dtype=object)
    TA_SETS = np.isin(S8, ["TK", "KVK", "L83"])
    put(TA_SETS & (ta == "contradicted"), "uncertified", "U_di_ban_nguoi_chong_nhan")
    SIMG = b("simg")
    put(STT & ~SIMG & (np.nan_to_num(sig["lobo_nh"], nan=0) < cfg["thresholds"]["stt_n_hum_min"]), "uncertified",
        "U_STT_thieu_nguyen_mau_nguoi")
    put(STT & ~SIMG, "uncertified", "U_STT_bo_kiem_viet_tay_khong_chung_nhan")
    put(BORG & ~SIMG & (np.nan_to_num(sig["lobo_nh"], nan=0) < cfg["thresholds"]["stt_n_hum_min"]), "uncertified",
        "U_Borg_thieu_nguyen_mau_nguoi_LOBO")
    put(BORG & ~SIMG, "uncertified", "U_Borg_bo_kiem_viet_tay_LOBO_khong_chung_nhan")
    put(PRINT & ~SIMG, "uncertified", "U_bo_kiem_anh_duoi_nguong_TN1")
    put(TA_SETS & (ta != "attested"), "uncertified", "U_khong_co_van_ban_nguoi_chung")
    put(np.ones(n, bool), "ok", "")
    assert (dec != "").all()
    A0n = b("int_foreign") | b("dup_bbox") | b("f_blank") | b("f_cut")
    B0n = b("dup_bbox") | b("ov_heavy") | ~b("one_char_ok") | b("tall_new")
    H1n = ~(A0n | B0n | b("cnt") | b("bc"))
    TA_OK = ~TA_SETS | (ta == "attested")
    masks = dict(A0_new=A0n, B0_new=B0n, H1_new=H1n, SIMG=SIMG, TA_OK=TA_OK, lai=H1n & SIMG & TA_OK,
                 AINT=b("int_foreign") | b("rescue") | b("similar") | b("weak_text"))
    return dec, why, masks


def evidence(S8):
    return np.array([EVIDENCE.get(s, "suy_doan") for s in S8], dtype=object)
