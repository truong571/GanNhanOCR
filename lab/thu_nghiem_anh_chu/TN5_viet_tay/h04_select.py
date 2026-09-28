#!/usr/bin/env python3
"""TN5 h04 — CHỌN + ĐO profile "handwriting" của gold_exact theo BẢN ĐĂNG KÝ TRƯỚC (prereg_handwriting.json).

Chạy được CHỈ KHI prereg_handwriting.json tồn tại và sha256 của tệp này == code_sha256 ghi trong bản đăng ký
(đổi mã sau khi đăng ký -> dừng). Toàn bộ quy tắc chọn nằm ở đây + bản đăng ký; không có tham số dòng lệnh nào đổi kết quả.

Profile handwriting (cho mỗi ô GOLD của bộ viết tay):
  review      luật A (ảnh ô khác, rescue, cầu tự dạng, văn bản yếu) + M-OCR  (GIỮ NGUYÊN chính sách cũ)
  text_only   cờ ảnh/hộp "một chữ": f_blank, f_cut, dup_bbox, ov_heavy, f_two, f_ink, bleed_new, trunc_new, tall_new (GIỮ)
              + cổng khe tuỳ chọn SLOT ∈ {none, bc, ad_dn0 (|n_det − n_qn| > 0), vis0 (vis_z < 0 hoặc thiếu)}
              — CNT (số chữ OCR cột ≠ số âm QN) BỎ: không phân biệt trên chữ viết tay (h03: AUC 0,49 nhãn / 0,49 khe)
  uncertified bộ kiểm viết tay LOBO-sách ở mức q ∈ thang q của hand_tables (FAR hiệu chuẩn sẵn) + lobo_nh ≥ 3
  ok          còn lại.
Quy tắc chọn (trên phần HỌC): khả thi ⇔ đúng hai vế ≥ 0,995 (ô ok có gt ∧ hộp người mức keep_high+) ∧ đúng nhãn ≥ 0,990
(ô ok có gt) ∧ số ô ok đo được hai vế ≥ N_MIN (Kinh 200, DungLy 50); chọn cấu hình khả thi có NHIỀU ô ok nhất; hoà -> thứ tự
SLOT (none < bc < ad_dn0 < vis0) rồi q nhỏ hơn (chặt hơn). Không khả thi -> cấu hình có đúng hai vế cao nhất (hoà: nhiều ô ok).
Phần kiểm: (1) LOBO Kinh→DungLy, (2) LOBO DungLy→Kinh, (3) kiểm chéo 5 khối trang trong Kinh (khoi_trang của bộ nhãn người,
trang liền nhau), gộp dự đoán của 5 khối kiểm. Cấu hình GIAO (config profiles.handwriting) = chọn trên TOÀN BỘ Kinh (= (1)).
CI 95 %: bootstrap cụm trang B = 2000, seed 20260928.

Ra: measure_out/_thu_nghiem_anh_chu/TN5/select.json (+ grid.csv); in tóm tắt.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).parent)]
OUT = REPO / "measure_out/_thu_nghiem_anh_chu/TN5"
PREREG = Path(__file__).parent / "prereg_handwriting.json"
SLOTS = ("none", "bc", "ad_dn0", "vis0")
CROP_GATES = ("f_blank", "f_cut", "dup_bbox", "ov_heavy", "f_two", "f_ink", "bleed_new", "trunc_new", "tall_new")
REVIEW = ("int_foreign", "rescue", "similar", "weak_text", "mocr")
P_BOTH, P_LAB = 0.995, 0.990
N_MIN = {"B18": 200, "B34": 50}
N_HUM_MIN = 3
BOOT_B, SEED = 2000, 20260928
VARIANT = {"B18": "DungLy", "B34": "Kinh", "stt2": "Kinh", "stt4": "Kinh", "stt11": "Kinh"}   # = common.HAND_VARIANT


def check_prereg():
    if not PREREG.exists():
        raise SystemExit(f"thiếu {PREREG.name} — phải ĐĂNG KÝ TRƯỚC khi chọn/đo")
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    now = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if pre.get("code_sha256") != now:
        raise SystemExit(f"mã h04 đã đổi sau đăng ký ({now[:12]} ≠ {pre.get('code_sha256', '')[:12]}) — dừng")
    return pre


def hand_thr():
    import torch
    out = {}
    for v in ("Kinh", "DungLy"):
        for t in "TL":
            tb = torch.load(REPO / f"models/gold_exact/hand_tables_{t}_{v}.pt", weights_only=False)
            out[(v, t)] = {float(k): float(x) for k, x in tb["thr"].items()}
    qs = sorted(out[("Kinh", "T")])
    return out, qs


def decide(D: pd.DataFrame, q: float, slot: str, TH) -> np.ndarray:
    rev = np.zeros(len(D), bool)
    for c in REVIEW:
        rev |= D[c].values.astype(bool)
    crop = np.zeros(len(D), bool)
    for c in CROP_GATES:
        crop |= D[c].values.astype(bool)
    if slot == "bc":
        sg = D.bc.values.astype(bool)
    elif slot == "ad_dn0":
        sg = ~(np.nan_to_num(D.ad_dn.values, nan=99) == 0)
    elif slot == "vis0":
        sg = ~(np.nan_to_num(D.vis_z.values, nan=-np.inf) >= 0)
    else:
        sg = np.zeros(len(D), bool)
    tT = np.array([TH[(VARIANT[s], "T")][q] for s in D.set8]); tL = np.array([TH[(VARIANT[s], "L")][q] for s in D.set8])
    cert = (np.nan_to_num(D.lobo_pT.values, nan=-1) >= tT) & (np.nan_to_num(D.lobo_pL.values, nan=-1) >= tL)
    cert &= np.nan_to_num(D.lobo_nh.values, nan=0) >= N_HUM_MIN
    return ~rev & ~crop & ~sg & cert


def boot(y, pages, B=BOOT_B, seed=SEED):
    y = np.asarray(y, float)
    if not len(y):
        return (np.nan, np.nan)
    P = pd.Series(y).groupby(np.asarray(pages)).agg(["sum", "size"])
    k, n = P["sum"].values, P["size"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(k), size=(B, len(k)))
    r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return (round(float(np.quantile(r, 0.025)), 5), round(float(np.quantile(r, 0.975)), 5))


def metrics(D, ok, ci=False):
    ev_b = ok & D.has_gt.values & D.keepbox.values
    ev_l = ok & D.has_gt.values
    r = dict(n=int(len(D)), ok=int(ok.sum()), n_both=int(ev_b.sum()), n_lab=int(ev_l.sum()))
    r["both"] = float(D.y_both.values[ev_b].mean()) if ev_b.any() else np.nan
    r["lab"] = float(D.y_lab.values[ev_l].mean()) if ev_l.any() else np.nan
    r["err_both"] = int((~D.y_both.values[ev_b]).sum()); r["err_lab"] = int((~D.y_lab.values[ev_l]).sum())
    r["err_slot_only"] = int((D.y_lab.values[ev_b] & ~D.y_slot.values[ev_b]).sum())
    if ci:
        r["both_ci"] = boot(D.y_both.values[ev_b], D.page_key.values[ev_b])
        r["lab_ci"] = boot(D.y_lab.values[ev_l], D.page_key.values[ev_l])
        # phụ: "một chữ" (hộp mực crop chuẩn không phủ ≥ 50 % hộp người KHÁC) và hộp người mức 'khong' (nhiễu ≈ 19 %)
        e1 = ev_b & ~np.isnan(D.nb_cov.values)
        r["one_char"] = float((D.nb_cov.values[e1] < 0.5).mean()) if e1.any() else np.nan
        ek = ok & D.has_gt.values & (D.keep_level.values == "khong") & D.has_hbox.values.astype(bool)
        r["both_khong_noisy"] = float(D.y_both.values[ek].mean()) if ek.any() else np.nan
        r["n_khong"] = int(ek.sum())
    return r


def select(D, grid, TH, s8):
    rows = []
    for q in grid:
        for slot in SLOTS:
            m = metrics(D, decide(D, q, slot, TH))
            m.update(q=q, slot=slot, feasible=bool(m["both"] >= P_BOTH and m["lab"] >= P_LAB and m["n_both"] >= N_MIN[s8]))
            rows.append(m)
    R = pd.DataFrame(rows)
    R["slot_rank"] = R.slot.map({s: i for i, s in enumerate(SLOTS)})
    F = R[R.feasible]
    if len(F):
        best = F.sort_values(["ok", "slot_rank", "q"], ascending=[False, True, True]).iloc[0]
    else:
        best = R.sort_values(["both", "ok"], ascending=[False, False]).iloc[0]
    return dict(q=float(best.q), slot=str(best.slot), feasible=bool(best.feasible)), R


def load():
    from h03_auc import load as L
    F, B, _, _ = L()
    tr = pd.read_csv(OUT / "truth.csv", usecols=["cell_uid", "nb_cov", "own_cov"])
    B = B.merge(tr, on="cell_uid", how="left")
    B["page_key"] = B.set8 + "/" + B.page.astype(str)
    hb = pd.read_csv(REPO / "dataset/_BORG_NHAN_NGUOI/labels.csv", dtype=str, keep_default_na=False,
                     usecols=["book", "page", "khoi_trang"]).drop_duplicates(["book", "page"])
    blk = {(b, p): int(k) for b, p, k in zip(hb.book, hb.page, hb.khoi_trang)}
    B["block"] = [blk.get((b, p), -1) for b, p in zip(B.book_set, B.page)]
    for c in ("has_gt", "keepbox", "y_lab", "y_slot", "y_both"):
        B[c] = B[c].fillna(False).astype(bool)
    return F, B


def main():
    pre = check_prereg()
    TH, grid = hand_thr()
    F, B = load()
    K = B[B.set8 == "B18"].reset_index(drop=True); Dl = B[B.set8 == "B34"].reset_index(drop=True)
    res = dict(prereg=dict(registered_at=pre["registered_at"], code_sha256=pre["code_sha256"]), grid_q=grid)
    # hiện tại (bản gold_exact.csv trước profile) trên cùng ô
    for nm, D in (("B18", K), ("B34", Dl)):
        res[f"hien_tai_{nm}"] = metrics(D, (D.gold_exact.values == "ok"), ci=True)
    # (1) LOBO Kinh -> DungLy  (= cấu hình giao)
    c1, R1 = select(K, grid, TH, "B18")
    R1.assign(train="B18").to_csv(OUT / "grid_B18.csv", index=False)
    res["chon_tren_Kinh"] = c1
    res["kinh_in_sample"] = metrics(K, decide(K, c1["q"], c1["slot"], TH), ci=True)
    res["lobo_K_to_D"] = metrics(Dl, decide(Dl, c1["q"], c1["slot"], TH), ci=True)
    # (2) LOBO DungLy -> Kinh
    c2, R2 = select(Dl, grid, TH, "B34")
    R2.assign(train="B34").to_csv(OUT / "grid_B34.csv", index=False)
    res["chon_tren_DungLy"] = c2
    res["lobo_D_to_K"] = metrics(K, decide(K, c2["q"], c2["slot"], TH), ci=True)
    # (3) kiểm chéo 5 khối trang trong Kinh
    okcv = np.zeros(len(K), bool); per = {}
    for b in sorted(set(K.block) - {-1}):
        tr, te = K.block.values != b, K.block.values == b
        cb, _ = select(K[tr].reset_index(drop=True), grid, TH, "B18")
        okcv[te] = decide(K[te].reset_index(drop=True), cb["q"], cb["slot"], TH)
        per[int(b)] = dict(cb, **{k: v for k, v in metrics(K[te].reset_index(drop=True), okcv[te]).items()
                                  if k in ("ok", "n_both", "both", "lab", "err_both")})
    res["cv_khoi_Kinh"] = dict(per_block=per, pooled=metrics(K, okcv, ci=True))
    # chiếu sang STT (SUY ĐOÁN — chỉ số ô, không có sự thật): cùng quy tắc profile, cấu hình giao
    S = F[F.set8.str.startswith("stt")].reset_index(drop=True)
    for c in ("int_foreign", "rescue", "similar", "weak_text", "mocr"):
        S[c] = S[c].astype(int)
    okS = decide(S, c1["q"], c1["slot"], TH)
    res["stt_chieu"] = {s: dict(gold=int((S.set8 == s).sum()), ok_profile=int((okS & (S.set8 == s).values).sum()),
                                ok_hien_tai=int(((S.gold_exact == "ok") & (S.set8 == s)).sum())) for s in ("stt2", "stt4", "stt11")}
    json.dump(res, open(OUT / "select.json", "w"), ensure_ascii=False, indent=1, default=float)
    f = lambda r: (f"ok {r['ok']} · hai vế {100 * r['both']:.2f} % [{100 * r['both_ci'][0]:.2f}–{100 * r['both_ci'][1]:.2f}] "
                   f"(n {r['n_both']}, lỗi {r['err_both']}) · nhãn {100 * r['lab']:.2f} % [{100 * r['lab_ci'][0]:.2f}–"
                   f"{100 * r['lab_ci'][1]:.2f}] (n {r['n_lab']}) · một chữ {100 * r['one_char']:.1f} %")
    print("chọn trên Kinh:", c1, "| chọn trên DungLy:", c2)
    for k in ("hien_tai_B18", "hien_tai_B34", "kinh_in_sample", "lobo_K_to_D", "lobo_D_to_K"):
        print(f"  {k}: {f(res[k])}")
    print("  cv_khoi_Kinh:", f(res["cv_khoi_Kinh"]["pooled"]), {b: (v["q"], v["slot"]) for b, v in per.items()})
    print("  STT:", res["stt_chieu"])


if __name__ == "__main__":
    main()
