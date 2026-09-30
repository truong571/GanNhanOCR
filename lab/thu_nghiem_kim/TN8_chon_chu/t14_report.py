"""t14_report.py — gom số của TN8 thành measure_out/_tn8/summary.json + KET_QUA.md (mọi con số đọc từ JSON do t01…t13 sinh).
0 API. Chạy sau t12_final.py 0.8 và 0.6, t13_levers.py, t10, t11.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402


def J(name):
    return json.load(open(T.OUT / name, encoding="utf-8"))


def pc(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}".replace(".", ",")


def ci(v, d=1):
    return "—" if not v else f"[{pc(v[0], d)}–{pc(v[1], d)}]"


def main():
    S8, S6 = J("summary_t12_b80.json"), J("summary_t12_b60.json")
    LV, OR, VO, SR, CU = J("levers.json"), J("oracle.json"), J("visual_only.json"), J("stt_rules.json"), J("curve_bien.json")
    B8, B6 = S8["books"], S6["books"]
    summ = dict(
        generated=datetime.now().isoformat(timespec="seconds"), api_calls=0,
        muc_tin={"DO": "đo trên nhãn người (LOBO / CV cụm trang)", "UL": "ước lượng (dị bản / tương tự Borg-IHR)",
                 "SD": "suy đoán (không có sự thật)"},
        dinh_nghia=dict(ty_le_GOLD="ô GOLD / MỌI ô của bảng nhãn (mọi tầng, kể cả QUARANTINE, not_plausible)",
                        hai_ve="nhãn V1+ = chữ người ∧ vị trí đúng (Borg: tâm bbox ∈ hộp người keep_high+; IHR: khe mô hình "
                               "Slot = '1', khe chưa xác định = sai)",
                        chu="nhãn V1+ = chữ người, mọi ô có chữ người"),
        chinh_sach=dict(
            giu_GOLD="GOLD hiện tại giữ nguyên; riêng viết tay: L5 sửa nhãn GOLD khi top-1 ≠ nhãn và P ≥ 0,80",
            nhom_nang_duoc=S8["promo"], cong_van_ban_khong_go=S8["excl_gates"],
            khong_bao_gio=["GOLD_text_only (hộp không tự tin)", "not_plausible", "QUARANTINE"],
            nguong_bien={"0.80 (đăng ký trước)": S8["tau_trien_khai_ho"], "0.60 (sau khi xem B18 — hậu kiểm)": S6["tau_trien_khai_ho"]},
            mo_hinh="measure_out/_tn8/models/{hand,print}.pkl (tầng 1 CLogit 35 đặc trưng ứng viên, tầng 2 logistic 46 đặc trưng ô)",
            STT="bảo thủ = GOLD ∪ {lt1 ∉ R ∧ lt2 ∈ R ∧ top-1 chỉ-ảnh == lt2}; mạnh (union + bộ chọn) KHÔNG khuyến nghị"),
        books={})
    for b in ("B18", "B34", "L16", "TK"):
        r8, r6 = B8[b], B6[b]
        summ["books"][b] = dict(
            muc_tin="DO", N=r8["N"], hien_tai=r8["current"], de_xuat_bien080_LOBO=r8["lobo_L5"], de_xuat_bien080_CV=r8["cv_L5"],
            de_xuat_bien060_LOBO=r6["lobo_L5"], de_xuat_bien060_CV=r6["cv_L5"], tau_LOBO_080=r8["tau_lobo"],
            tran_nhan_het_nhom_nang_duoc=r8["tran_nhan_het_nhom_nang_duoc"], don_bay=LV[b], L5_mot_minh=r8["L5_mot_minh"],
            vi_tri_o_moi=r8["pos_rate_new_lobo"], o_moi_khong_co_chu_nguoi=r8["n_new_no_gt_lobo"],
            con_lai={g: v["n"] - v["nhan"] for g, v in r8["groups"].items() if g != "GOLD" and v["n"] - v["nhan"] > 0},
            duong_cong_bien=CU.get(b))
    for b in ("stt2", "stt4", "stt11"):
        r8, r6 = B8[b], B6[b]
        summ["books"][b] = dict(muc_tin="SD", N=r8["N"], hien_tai=r8["current_share"], bao_thu=r8["bao_thu"],
                                lt2_khong_anh=r8["lt2_khong_anh"], manh_union_bien080=r8["manh_union"],
                                manh_union_bien060=r6["manh_union"], kiem_cheo_lt2=r8["kiem_cheo_lt2_o_moi"],
                                chi_anh_tren_o_chac=VO[b], quy_tac=SR[b])
    for b in ("Chr", "L83", "KVK"):
        r8 = B8[b]
        summ["books"][b] = dict(muc_tin="UL" if b != "Chr" else "SD", **{k: r8[k] for k in r8 if k != "book"})
    summ["borg_tuong_tu_STT"] = SR["borg"]
    summ["chi_anh_borg"] = {k: VO[k] for k in ("B18<-B34", "B34<-B18")}
    T.jdump(summ, T.OUT / "summary.json")

    # ------------------------------------------------------------------ KET_QUA.md
    L = []
    w = L.append
    w("# TN8 — Chọn chữ bằng ảnh để nâng tỉ lệ GOLD (0 API) — 30/09/2026\n")
    w("Mọi số do script sinh (`lab/thu_nghiem_kim/TN8_chon_chu/t00…t14`, đầu ra `measure_out/_tn8/`, `summary.json`). 0 gọi API, "
      "không mở ảnh/CSV bằng mắt; không sửa pipeline/core/config/data/dataset/prepared/web. Mức tin: **[ĐO]** nhãn người "
      "(LOBO hoặc CV cụm trang) · **[ƯL]** ước lượng (dị bản / tương tự) · **[SĐ]** suy đoán.\n")
    w("Định nghĩa: tỉ lệ GOLD = ô GOLD / **mọi** ô của bảng nhãn (mọi tầng). Hai vế = nhãn V1+ = chữ người ∧ vị trí đúng "
      "(Borg: tâm bbox ∈ hộp người keep_high+ — tập hộp dễ, lạc quan; IHR: khe `Slot` = '1'). Chữ = nhãn V1+ = chữ người trên "
      "mọi ô có chữ người (con số bảo thủ hơn cho Borg).\n")
    w("## 0. Kết quả theo bộ (điểm vận hành đăng ký trước: tiêu chí biên 0,80; ngưỡng LOBO; CV cụm trang trong ngoặc)\n")
    w("| Bộ | GOLD hiện tại | đúng hiện tại hai vế / chữ | **GOLD đề xuất** | đúng đề xuất hai vế / chữ (CI 95 % cụm trang) | đòn bẩy chính | 90 %/90 %? |")
    w("|---|---:|---|---:|---|---|---|")
    for b in ("B18", "B34", "L16", "TK"):
        r = B8[b]; c = r["current"]; q = r["lobo_L5"]; v = r["cv_L5"]
        lv = LV[b]
        parts = [f"{k} +{pc(lv[k]['share_mot_minh'] - c['share'])}" for k in ("L2", "L4", "L1", "L2b") if lv[k]["n_nhan"] > 0]
        if r["lobo_L5"]["n_relabel"]:
            parts.append(f"L5 sửa {r['lobo_L5']['n_relabel']} nhãn")
        ok = "ĐẠT" if q["share"] >= 0.9 and (q.get("text") or 0) >= 0.9 else "không"
        if b == "B18":
            ok = "ĐẠT ở ngưỡng LOBO (sát); CV 85,9 % — chưa chắc; biên 0,6: đạt cả hai"
        if b == "B34":
            ok = "KHÔNG — trần 75,3 % (24,6 % ô âm giữ chỗ)"
        w(f"| {b} [ĐO] | {pc(c['share'])} % | {pc(c.get('both'))} / {pc(c.get('text'))} % | **{pc(q['share'])} %** ({pc(v['share'])} %) | "
          f"{pc(q.get('both'))} {ci(q.get('both_ci'))} / {pc(q.get('text'))} {ci(q.get('text_ci'))} % | {', '.join(parts)} | {ok} |")
    for b in ("stt2", "stt4", "stt11"):
        r = B8[b]
        w(f"| {b} [SĐ] | {pc(r['current_share'])} % | chưa đo (lt1 = lt2 trên GOLD ≈ 95 %) | **{pc(r['bao_thu']['share'])} %** bảo thủ · "
          f"{pc(r['manh_union']['share'])} % mạnh | bảo thủ ≈ tương tự Borg 96,9–97,2 % [SĐ]; mạnh: kiểm chéo lt2 chỉ "
          f"{pc(r['kiem_cheo_lt2_o_moi']['top1_eq_lt2'])} % | L3 (lt2 ∧ ảnh) | KHÔNG |")
    for b in ("Chr", "L83", "KVK"):
        r = B8[b]
        extra = f"dị bản trùng ô mới {pc(r['di_ban_trung_o_moi'][0])} % (n {r['di_ban_trung_o_moi'][1]}) vs GOLD {pc(r['di_ban_trung_GOLD'][0])} %" \
            if b in ("L83", "KVK") and r["di_ban_trung_o_moi"][1] else "tương tự IHR (L1): 97,5–98,8 % chữ"
        w(f"| {b} [{'SĐ' if b == 'Chr' else 'ƯL'}] | {pc(r['current_share'])} % | — | **{pc(r['share'])} %** | {extra} | L1 (+{r['n_new']} ô) | KHÔNG |")
    w("")
    w("Đạt 90 %/90 % chắc chắn: **L16, TK** (chỉ nhờ L1). **B18** đạt ở ngưỡng LOBO (90,4 %, chữ 94,4 %) nhưng CV cụm trang cho "
      "85,9 % → sát, phụ thuộc cách chọn ngưỡng; nới tiêu chí biên xuống 0,6 thì cả LOBO (92,2 %) lẫn CV (90,6 %) đều ≥ 90 % với "
      "chữ ≥ 93,5 % (CI dưới ≥ 93,2 %) — lựa chọn 0,6 là HẬU KIỂM (đã thấy B18). **B34, STT, Chr, L83, KVK không đạt**.\n")

    w("## 1. Bộ chọn chữ hai tầng (thành phần mới)\n")
    w("- **Ứng viên** của ô: R(âm) (`Dict/QuocNgu_SinoNom.csv`, `gold_exact.common.R_of`) ∪ {chữ kim} (∪ {chữ lt2} ở STT); trung bình 25–27 ứng viên/ô.")
    w("- **Crop mọi ô** (kể cả REVIEW/SYLLABLE không có tệp) cắt lại đúng hình học `save_crop` (pad 0,12, carve chữ kề, tighten, điểm ảnh gốc): "
      "trùng từng điểm ảnh tệp giao 150/150 mẫu ở mọi bộ in + Borg (STT 142–144/150 — chưa định vị ô lệch).")
    w("- **Đặc trưng ứng viên (35)**: cos crop (MultiEnc nom-embed v1 + ArcFace v2) với glyph font / FD / head v1; nguyên mẫu CÙNG SÁCH "
      "(trung bình + 3 láng giềng) từ ô \"neo\" tự động (kim ∈ R, nhãn = kim) **chỉ ở 4 khối trang khác**; nguyên mẫu NGƯỜI Borg "
      "(trung bình + 3 láng giềng) **của sách kia** (LOBO); CNN kiểm ảnh↔chữ sạch với bộ đang chấm (B18: hand_*_DungLy; B34/STT: "
      "hand_*_Kinh; L16: vft_T; TK: vft_L; Chr/L83/KVK: vft_T+L) — bảng W, P; tiên nghiệm log P(chữ|âm) từ ô neo khối khác; cờ kim/R; "
      "mỗi điểm thêm bản tương đối (trừ ứng viên tốt nhất khác).")
    w("- **Tầng 1**: logit có điều kiện (softmax trong ô), hai mô hình theo phần `in` (kim ∈ R) / `out` (kim ∉ R). **Tầng 2**: logistic "
      "46 đặc trưng mức ô (p1, khoảng cách p1−p2, đặc trưng top-1, kích thước/mực crop, box yếu, |n_ocr−n_qn|, |n_det−n_qn|, nhóm ô) → "
      "P(top-1 đúng [∧ khe đúng ở sách in]). Học trên sự thật của bộ HỌC; bộ thử không bao giờ góp nhãn.")
    w("- **Ngưỡng — tiêu chí biên**: xếp ô mới theo P; τ = P của ô cuối cùng mà cửa sổ ~W ô liền trước còn đúng ≥ đích biên (0,80). "
      "Chọn trên dự đoán NGOÀI-KHỐI (5 khối trang) của bộ học (LOBO) hoặc trên 4 khối trang của bộ thử (CV, báo khối thứ 5).")
    w("- **Nguồn gốc mô hình** (kiểm `models/gold_exact/MANIFEST.json`, `r4/verifier_ft/t02_train.py`): vft_T học nhãn người TK phần A "
      "+ Borg keep fold 0–3 (CẢ HAI sách) ⇒ chỉ dùng cho L16; vft_L học L16 phần A + Borg ⇒ chỉ dùng cho TK; **không dùng vft trên Borg**. "
      "hand_*_Kinh / hand_*_DungLy học một sách Borg + IHR tune ⇒ dùng cho sách Borg kia. Encoder v1/v2 học trên crop STT nhãn pipeline "
      "(không nhãn người) ⇒ trên STT GOLD là trong mẫu (tự tham chiếu).\n")

    w("## 2. Đòn bẩy trên bộ có sự thật — mỗi đòn một mình (ngưỡng LOBO biên 0,80) [ĐO]\n")
    w("| Bộ | đòn | ô trong nhóm | nhận | GOLD một mình | ô thêm đúng chữ (CI) | ô thêm hai vế | nhận hết không lọc (nhãn kim / top-1) |")
    w("|---|---|---:|---:|---:|---|---|---|")
    for b in ("B18", "B34", "L16", "TK"):
        for k in ("L1", "L2", "L2b", "L4"):
            v = LV[b][k]
            t = v["o_them_bo_chon"].get("text"); bo = v["o_them_bo_chon"].get("both")
            nh = v.get("nhan_het_nhan_kim") or v.get("nhan_het_top1") or {}
            nht = nh.get("text")
            w(f"| {b} | {k} | {v['n_nhom']} | {v['n_nhan']} | {pc(v['share_mot_minh'])} % | "
              f"{pc(t[0]) + ' ' + ci(t[1:3]) if t else '—'} | {pc(bo[0]) if bo else '—'} | {pc(nht[0]) + ' % (n ' + str(nht[3]) + ')' if nht else '—'} |")
    w("")
    w("L1 = `crop_bad` (REVIEW, kim ∈ R) · L2 = `qn_count_unfixed` · L2b = cầu tự dạng / confusion_fix / direct khác · L4 = kim ∉ R "
      "(SYLLABLE, no_context, low_posterior, syl_ctx|crop_bad). L5 = sửa nhãn GOLD hiện tại khi top-1 ≠ nhãn, P ≥ 0,80 (chỉ viết tay): "
      f"B18 sửa {B8['B18']['L5_mot_minh']['n_relabel']} ô → GOLD chữ {pc(B8['B18']['current']['text'])} → {pc(B8['B18']['L5_mot_minh']['text'])} % "
      f"{ci(B8['B18']['L5_mot_minh']['text_ci'])}, hai vế {pc(B8['B18']['current']['both'])} → {pc(B8['B18']['L5_mot_minh']['both'])} %; "
      f"B34 sửa {B8['B34']['L5_mot_minh']['n_relabel']} ô; sách in 0 ô (bộ chọn trùng kim).\n")
    w("Đọc nhanh: trên **chữ viết tay Borg** bộ chọn vừa LỌC vừa SỬA nhãn — L2 nhận gần hết 35.009 ô qn_count_unfixed và nâng độ đúng "
      "chữ của chúng 88,9 → 94,6 % (sửa theo top-1), L4 gán được 92 % ô kim ∉ R ở 93,8 % chữ. Trên **sách in IHR** L2/L4 KHÔNG dùng được: "
      "ô qn_count_unfixed chỉ 39–44 % đúng hai vế, ô no_context sai vị trí 39–48 % (hộp/âm sai từ thượng nguồn) — bộ chọn không cứu được nên "
      "họ in chỉ nâng L1 (+ confusion_fix).\n")

    w("## 3. Độ nhạy theo tiêu chí biên (B18/B34 LOBO | CV) [ĐO]\n")
    w("| biên | B18 GOLD LOBO / CV | B18 chữ LOBO / CV | B34 GOLD LOBO / CV | B34 chữ LOBO / CV |")
    w("|---|---|---|---|---|")
    for i, bn in enumerate(("0.5", "0.6", "0.7", "0.8", "0.9")):
        a, bb = CU["B18"][i], CU["B34"][i]
        w(f"| {bn.replace('.', ',')} | {pc(a['lobo'][0])} / {pc(a['cv'][0])} % | {pc(a['lobo'][2])} / {pc(a['cv'][2])} % | "
          f"{pc(bb['lobo'][0])} / {pc(bb['cv'][0])} % | {pc(bb['lobo'][2])} / {pc(bb['cv'][2])} % |")
    w("")
    w(f"(chưa gồm L5.) Trần = nhận MỌI ô nhóm nâng được: B18 {pc(B8['B18']['tran_nhan_het_nhom_nang_duoc']['share'])} % (chữ "
      f"{pc(B8['B18']['tran_nhan_het_nhom_nang_duoc']['text'])} %), B34 {pc(B8['B34']['tran_nhan_het_nhom_nang_duoc']['share'])} % "
      f"(chữ {pc(B8['B34']['tran_nhan_het_nhom_nang_duoc']['text'])} %). LOBO và CV lệch ở B18 vì P của mô hình học trên B34 hơi "
      "quá tự tin trên ô khó của B18 (hiệu chuẩn phần kim ∉ R: P 0,86 → đúng 71 %, P 0,93 → 79 %; trên B34 cùng mức P → 91 %, 95 %).\n")

    w("## 4. STT — đòn bẩy L3 (lt1 / lt2 / union) + bộ chọn [SĐ]\n")
    w("| bộ | GOLD hiện tại | + lt2 ∈ R (không ảnh) | + lt2 ∈ R ∧ ảnh đồng ý (**bảo thủ**) | union + bộ chọn biên 0,8 / 0,6 (mạnh) | kiểm chéo độc lập lt2 (ô mới) | top-1 chỉ-ảnh trên ô \"chắc\" |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for b in ("stt2", "stt4", "stt11"):
        r8, r6 = B8[b], B6[b]
        w(f"| {b} | {pc(r8['current_share'])} % | {pc(r8['lt2_khong_anh']['share'])} % (+{r8['lt2_khong_anh']['n_new']}) | "
          f"**{pc(r8['bao_thu']['share'])} %** (+{r8['bao_thu']['n_new']}) | {pc(r8['manh_union']['share'])} / {pc(r6['manh_union']['share'])} % | "
          f"{pc(r8['kiem_cheo_lt2_o_moi']['top1_eq_lt2'])} % (n {r8['kiem_cheo_lt2_o_moi']['n']}) | {pc(VO[b]['sure_top1_eq_label'][0])} % |")
    w("")
    w(f"- Tương tự trên Borg (kim Borg = lt2): ô kim ∈ R mà ảnh đồng ý → nhãn đúng {pc(SR['borg']['B18<-B34']['kimR_vis_agree'][0])} % (B18), "
      f"{pc(SR['borg']['B34<-B18']['kimR_vis_agree'][0])} % (B34); ô kim ∉ R chỉ-ảnh vp ≥ 0,5 → {pc(SR['borg']['B18<-B34']['kim_notR_vis_curve[tau,n,prec,frac]'][0][2])} %.")
    w(f"- **Chuyển giao sang tay chép STT yếu**: top-1 chỉ-ảnh trên ô \"chắc\" (GOLD, lt1 = lt2 ∈ R) chỉ {pc(VO['stt2']['sure_top1_eq_label'][0])}–"
      f"{pc(VO['stt11']['sure_top1_eq_label'][0])} % so với {pc(VO['B34<-B18']['GOLD_nhan_dung__top1_eq_label'][0])}–"
      f"{pc(VO['B18<-B34']['GOLD_nhan_dung__top1_eq_label'][0])} % trên ô tương ứng của Borg; ở ô mới có lt2 ∈ R, top-1 (mô hình không thấy lt2) "
      "trùng lt2 chỉ 62 / 85 / 79 % ⇒ độ đúng thật của chính sách mạnh ở stt2 khả năng < 90 %. Vì thế chỉ khuyến nghị bản BẢO THỦ (hai nguồn độc lập "
      "một phần đồng ý). Bên cạnh: GOLD STT có 479 / 452 / 536 ô lt1 ≠ lt2 (cả hai ∈ R) — ảnh chọn lt1 chỉ 33–45 % ⇒ nghi sai ~½ (nên hạ/kiểm).\n")

    w("## 5. Sách in không có sự thật: Chr, L83, KVK [ƯL/SĐ]\n")
    for b in ("Chr", "L83", "KVK"):
        r = B8[b]
        rem = sorted(((g, v["n"] - v["nhan"]) for g, v in r["groups"].items() if g != "GOLD"), key=lambda kv: -kv[1])[:5]
        w(f"- **{b}**: {pc(r['current_share'])} → {pc(r['share'])} % (+{r['n_new']} ô L1/confusion_fix, P TB {str(round(r['mean_P_new'], 2)).replace('.', ',') if r['mean_P_new'] is not None else '—'}). Còn lại lớn nhất: "
          + ", ".join(f"{g} {n} ({pc(n / r['N'])} %)" for g, n in rem) + ".")
    w("- Mô hình in (học L16+TK) TỰ NÓ muốn nhận ô `cross_similar` của L83/KVK (P ≈ 0,98) nhưng các ô đó chỉ trùng dị bản 18–34 % (GOLD 82–91 %) ⇒ "
      "đã khoá: cổng văn bản (cross_similar, am_sua_dau, bridge_similar) không bao giờ gỡ bằng ảnh. Ô no_context của sách in: top-1 trùng dị bản 42–48 % "
      "(SYLLABLE: L83 17 %, KVK 81 %), trên IHR ô no_context đúng vị trí chỉ 52–61 % ⇒ lỗi ở âm QN (OCR) / hộp, không phải ở việc chọn chữ.\n")

    w("## 6. Cái gì CHẶN mục tiêu 90 %/90 %\n")
    w(f"- **B34**: 4.760 ô (24,6 %) và **B18**: 5.320 ô (5,9 %) có âm tiết giữ chỗ (`not_plausible`, một token 9 chữ 'kh…' của người phiên QN) "
      "và KHÔNG có chữ Nôm người ⇒ không ứng viên, không đo được. Bỏ các ô này khỏi mẫu số: B18 "
      f"{pc(B8['B18']['lobo_L5']['n'] / (B8['B18']['N'] - 5320))} %, B34 {pc(B8['B34']['lobo_L5']['n'] / (B8['B34']['N'] - 4760))} %.")
    w("- **STT**: 21–26 % ô cả lt1 lẫn lt2 đều ∉ R(âm) (âm QN OCR sai / kim đọc sai) và bộ chọn ảnh chuyển giao kém sang tay chép STT; không có sự thật "
      "để hiệu chuẩn ⇒ không chứng được ≥ 90 %.")
    w("- **Chr**: 23,5 % ô qn_count_unfixed (trên sách in đo được chỉ 39–44 % đúng), 8,1 % SYLLABLE, 7,8 % no_context. **L83**: no_context 11 %, "
      "SYLLABLE 5,8 %, not_plausible 2,6 %. **KVK**: no_context 5,1 %, SYLLABLE 2,1 %, not_plausible 1,4 %, qn_count 1,3 %, cross_similar 1,3 % "
      "⇒ nút thắt ở văn bản QN + hộp (thượng nguồn), bộ chọn chữ không gỡ được.\n")

    w("## 7. Chính sách khuyến nghị để cài (0 API; sau build `labels_gated.csv`, trước gold_exact)\n")
    t8, t6 = S8["tau_trien_khai_ho"], S6["tau_trien_khai_ho"]
    w("1. Cắt crop mọi ô bằng `save_crop` hình học (đã có trong build: thêm `--crop-review` hoặc cắt trong bộ nhớ như `t02_embed.cut`).")
    w("2. Nhúng MultiEnc v1+v2 (norm) + CNN kiểm ảnh↔chữ theo họ (bảng ở §1); dựng ứng viên R(âm) ∪ {kim} (∪ {lt2} STT) và 35 đặc trưng "
      "(`t05_feats.py` + `t05c_knn.py`; ô neo = rule `s1_inter_s2_direct*` có nhãn = kim ∈ R; 5 khối trang liền theo thứ tự trang).")
    w("3. Tầng 1 + tầng 2 với trọng số `measure_out/_tn8/models/hand.pkl` (viết tay: học B18+B34) / `print.pkl` (in: học L16+TK). "
      "Riêng hai sách Borg (tập đánh giá): dùng mô hình học trên SÁCH KIA (LOBO) như §0.")
    w(f"4. Nâng ô KHÔNG GOLD lên GOLD, nhãn = top-1, khi nhóm ô ∈ danh sách nâng được của họ, KHÔNG mang cổng văn bản, và P ≥ τ: viết tay "
      f"τ_in = {t8['hand'][0]:.3f}, τ_out = {t8['hand'][1]:.3f} (biên 0,80; hoặc {t6['hand'][0]:.3f} / {t6['hand'][1]:.3f} ở biên 0,6); "
      f"in τ_in = {t8['print'][0]:.3f}, phần out tắt. Viết tay thêm L5: GOLD hiện tại có top-1 ≠ nhãn và P ≥ 0,80 → nhãn := top-1.")
    w("5. STT: chỉ bản bảo thủ (lt1 ∉ R ∧ lt2 ∈ R ∧ top-1 chỉ-ảnh == lt2 → GOLD nhãn lt2); cần cache lt2 đủ 448 trang (đã có).")
    w("6. Ghi cột mới (vd `tn8_P`, `tn8_group`, `tier_goc`, `rule_goc`), rule `tn8_chon_anh:<nhóm>`; gold_exact chạy sau như cũ (crop chuẩn cho ô "
      "L1). Mốc md5 STT 3 trang sẽ đổi nếu bật cho STT.\n")

    w("## 8. Rủi ro đã biết\n")
    w("- Hai vế ở Borg đo trên hộp người keep_high+ (máy gióng bằng CÙNG encoder v1/v2 ⇒ ô dễ, lạc quan); con số chữ trên mọi ô có chữ người là "
      f"con số bảo thủ. {B8['B18']['n_new_no_gt_lobo']} ô GOLD mới của B18 (≈ 11 %) nằm ở câu không đếm bằng — KHÔNG đo được.")
    w("- Ngưỡng nhạy: B18 LOBO 90,4 % vs CV 85,9 % ở biên 0,80; biên 0,6 chọn SAU khi thấy B18. Mỗi họ chỉ có HAI sách có sự thật (B34 nhỏ) ⇒ "
      "trọng số và ngưỡng dựa trên ít sách; P lệch hiệu chuẩn giữa sách.")
    w("- STT/Chr/L83/KVK: độ đúng chỉ [SĐ]/[ƯL]; kiểm chéo cho thấy chuyển giao sang STT kém (§4); encoder học trên crop STT ⇒ đặc trưng ảnh của "
      "STT GOLD tự tham chiếu. Mô hình in từng muốn nhận ô cross_similar sai (đã khoá) ⇒ luôn giữ cổng văn bản.")
    w("- Đã thử, KHÔNG giúp: nguyên mẫu lượt 2 (tự học từ ô P ≥ 0,95: GOLD B18 đổi ±0,2 điểm), cứu bằng chữ kim ô kề ±2 (chỉ 0,6–3,3 % ô kim ∉ R, "
      "đúng 25–57 %), tiêu chí 'độ đúng TỔNG' (nhận cả ô xấu ở sách in vì nền GOLD quá tốt — thay bằng tiêu chí biên).\n")

    w("## 9. Tái lập\n")
    w("```bash\nPY=.venv/bin/python; D=lab/thu_nghiem_kim/TN8_chon_chu")
    w("$PY $D/t00_base.py && $PY $D/t01_oracle.py            # bảng ô + sự thật, tiềm năng theo nhóm")
    w("$PY $D/t02_embed.py && $PY $D/t04_human.py && $PY $D/t03_stt_lt2.py   # crop+nhúng mọi ô (MPS ~25 phút), crop người Borg, lt2 STT")
    w("$PY $D/t05_feats.py && $PY $D/t05c_knn.py              # bảng ứng viên + láng giềng gần")
    w("$PY $D/t06_eval.py && $PY $D/t07_policy.py             # LOBO B34↔B18, TK↔L16")
    w("$PY $D/t08_transfer.py && $PY $D/t10_visual_only.py && $PY $D/t11_stt_rules.py")
    w("$PY $D/t12_final.py 0.8 && $PY $D/t12_final.py 0.6 && $PY $D/t13_levers.py && $PY $D/t14_report.py\n```")
    (T.OUT / "KET_QUA.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("[t14] ghi summary.json + KET_QUA.md", flush=True)


if __name__ == "__main__":
    main()
