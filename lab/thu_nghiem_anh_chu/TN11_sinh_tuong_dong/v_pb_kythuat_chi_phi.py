"""v_pb_kythuat (04/10) — PHẢN BIỆN độc lập phép tính chi phí "sinh ảnh pixel cho toàn kho > 3.000 giờ GPU" (CPU, đọc pickle TN8, không sửa tệp).

Báo cáo: "275.136 ô × 3-4 ứng viên ≈ 1 triệu cặp ... > 3.000 giờ GPU". Người kiểm chứng: ứng viên thật ≈ 25/ô; glyph chỉ phụ thuộc
(chữ, ảnh phong cách) => 244.614 chữ-theo-sách (10 bộ × 1 phong cách) = 347–747 h; "hợp 10 bộ" 37.056 chữ = 52–113 h; phóng đại ≈ 4–25×.
Script này tự đếm lại từ measure_out/_tn8/cand/*.pkl và tính CHI PHÍ THEO NHIỀU THIẾT KẾ, kể cả các thiết kế bất lợi cho người kiểm chứng:
  - n_phong_cach ∈ {1,2,3} (p02 mặc định --styles 2; mỗi sách cần ≥ 1 ảnh phong cách riêng nếu "thích ứng theo sách"),
  - có/không tính chữ đã có ảnh FD thật cục bộ (1.646) — nếu đã có thì không phải sinh,
  - tốc độ s/ảnh: 5,1 (lô đầu), 8,2 (trung bình 793 ảnh, logs/tn11_p02_gen.log), 11,0 (đỉnh chạy trung bình), 1,7 (T4, README, chưa đo),
  - "hợp 10 bộ" chỉ hợp lệ nếu DÙNG CHUNG MỘT phong cách cho cả 10 bộ (mâu thuẫn với "thích ứng theo sách").
Cũng kiểm: số ô khớp 275.136 (docs/BAO_CAO_TONG_HOP_2026-10-01.md:23)? tốc độ trong log đọc lại độc lập.
Ra: measure_out/_tn11/verify/pb_kythuat/chi_phi.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_pb_kythuat_chi_phi.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CAND = REPO / "measure_out" / "_tn8" / "cand"
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_kythuat"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["stt2", "stt4", "stt11", "Chr", "L83", "KVK", "L16", "TK", "B18", "B34"]


def main():
    # tốc độ log: đọc độc lập (dòng tổng hoặc dòng chạy)
    L = (REPO / "logs" / "tn11_p02_gen.log").read_text(errors="ignore").splitlines()
    run = [float(m.group(1)) for l in L for m in [re.search(r"done \((\d+\.\d+)s/img", l)] if m]
    fin = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for l in L
           for m in [re.search(r"FontDiffusion: (\d+) images in ([\d.]+)s \(([\d.]+)s/img", l)] if m]
    speeds = {"lo_dau_5.1": run[0], "tb_793_anh_8.2": fin[0][2] if fin else None, "dinh_chay_tb_11.0": 11.0, "T4_README_1.7": 1.7}
    # kho FD: chữ có ảnh thật cục bộ
    real = set()
    for d in ("gannhanocr-fd", "ArcFace/data/glyphs"):
        for r, _, fs in os.walk(REPO / d):
            if os.sep + ".git" in r:
                continue
            for f in fs:
                if f.startswith("U+") and f.endswith(".png") and os.path.getsize(os.path.join(r, f)) >= 1024:
                    real.add(chr(int(f[2:-4], 16)))
    tot_cells = tot_rows = 0
    per_book = {}
    union = set()
    sum_distinct = sum_distinct_noreal = 0
    for b in ORDER:
        F = pd.read_pickle(CAND / f"{b}.pkl", )[["i", "c"]]
        cells = pd.read_pickle(CAND / f"{b}_cells.pkl")
        uc = set(F.c.unique())
        union |= uc
        n_cells = len(cells)
        per_book[b] = dict(o=int(n_cells), dong=int(len(F)), ung_vien_moi_o=round(len(F) / n_cells, 2), chu_khac_nhau=len(uc),
                           chu_chua_co_anh_FD_that=len(uc - real))
        tot_cells += n_cells
        tot_rows += len(F)
        sum_distinct += len(uc)
        sum_distinct_noreal += len(uc - real)
    res = dict(toc_do_s_moi_anh=speeds, n_log_chay=len(run), kho_FD_chu_co_anh_that=len(real), bo=per_book,
               tong=dict(o=int(tot_cells), dong=int(tot_rows), ung_vien_tb=round(tot_rows / tot_cells, 2),
                         tong_chu_khac_nhau_theo_bo=int(sum_distinct), chu_khac_nhau_hop_10_bo=len(union),
                         tong_chu_theo_bo_chua_co_FD_that=int(sum_distinct_noreal), hop_chua_co_FD_that=len(union - real),
                         o_BAO_CAO_TONG_HOP=275136, lech_o=int(275136 - tot_cells)))
    hrs = {}
    for sn, s in speeds.items():
        h = {}
        h["bao_cao_o_x3.5"] = round(275136 * 3.5 * s / 3600)
        h["thuc_te_moi_dong_o_x_UV(khong_dedup)"] = round(tot_rows * s / 3600)
        for k in (1, 2, 3):
            h[f"chu_theo_sach_x{k}_phong_cach"] = round(sum_distinct * k * s / 3600)
            h[f"chu_theo_sach_chua_co_FD_x{k}_phong_cach"] = round(sum_distinct_noreal * k * s / 3600)
        h["hop_10_bo_x1_phong_cach_DUNG_CHUNG"] = round(len(union) * s / 3600)
        hrs[sn] = h
    res["gio_gpu"] = hrs
    # tỉ số phóng đại của "3000 h" so với thiết kế hợp lý nhất (theo sách, 1–3 phong cách) ở từng tốc độ
    ref = 275136 * 3.5 * 11.0 / 3600
    res["bao_cao_gio_voi_11.0"] = round(ref)
    res["ti_so_phong_dai"] = {sn: {f"x{k}_phong_cach": round(ref / (sum_distinct * k * s / 3600), 1) for k in (1, 2, 3)} for sn, s in speeds.items()}
    (OUT / "chi_phi.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
