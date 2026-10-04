"""TN11 v_nguon_goc_do_nhay_nen (03/10) — KIỂM LÝ DO của HQC2 (K09): "bộ lọc lớp đầu CNN rất nhạy với độ sáng nền và tương phản; nền 179 so với 255 làm vector lệch pha".

Phép đo trực tiếp (CPU, 0 API, không sửa mã gốc): với N chữ (seed 0) lấy từ ứng viên B34, dựng 3 phiên bản glyph NomNaTong:
  A = trắng/đen chuẩn (nền 255, mực 0)                     — HQC1
  B = ánh xạ tuyến tính đồng nhất nền 179 / mực 35          — chỉ đổi độ sáng + tương phản, không kết cấu
  C = blend_on_paper(p05) lên hồ sơ giấy B34                — HQC2 đúng như p05
nhúng bằng MultiEnc v1+v2 ở hai chế độ: norm=False (cách p05 nhúng glyph) và norm=True (cách sản xuất nhúng crop: kéo giãn phân vị 2–98 %).
Đo: (i) cos trung bình A↔B, A↔C cùng chữ; (ii) cos trung bình giữa hai chữ KHÁC nhau (thang so); (iii) truy hồi: B_i/C_i làm truy vấn, thư viện = A_j
(N chữ) -> Top-1 đúng chữ. Nếu cos A↔B ≈ 1 và truy hồi ≈ 100 % thì "nền 179 làm vector lệch pha" không đúng với bộ nhúng này.
Ra: measure_out/_tn11/verify/nguon_goc/do_nhay_nen.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tn8lib as T  # noqa: E402
import v_nguon_goc_nhieu_mau as V  # noqa: E402  (hàm sao chép nguyên văn của p05: render_font_raw, extract_book_paper_model, blend_on_paper)

OUT_DIR = REPO / "measure_out/_tn11/verify/nguon_goc"


def main(N=300, book="B34"):
    torch.set_num_threads(3)
    from fontTools.ttLib import TTFont
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    enc = SI.Scorers(Assets(), "cpu", T.OUT / "emb", False, lambda m: None).enc()
    cmap = set(TTFont(V.FONT_PATH).getBestCmap())
    F = pd.read_pickle(T.OUT / "cand" / f"{book}.pkl", )
    chars = sorted(c for c in set(F.c.unique()) if c and ord(c[0]) in cmap)
    rng = np.random.default_rng(0)
    pick = [chars[i] for i in sorted(rng.choice(len(chars), size=min(N, len(chars)), replace=False))]
    paper = V.extract_book_paper_model(book)
    raw = [V.render_font_raw(c, 128) for c in pick]
    A = raw
    B = [np.clip(35 + (g.astype(np.float32) / 255.0) * (179 - 35), 0, 255).astype(np.uint8) for g in raw]
    C = [V.blend_on_paper(g, paper) for g in raw]
    out = {"book": book, "n_chu": len(pick), "ho_so_giay": {k: paper[k] for k in ("bg_median", "ink_median", "bg_std", "contrast", "sample_page")}}
    for nm in (False, True):
        EA, EB, EC = (V.nrm(np.asarray(enc.embed(X, norm=nm), np.float32)) for X in (A, B, C))
        same_AB = np.sum(EA * EB, 1); same_AC = np.sum(EA * EC, 1)
        S_AA = EA @ EA.T
        diff = S_AA[~np.eye(len(pick), dtype=bool)]
        def retr(Q):
            S = Q @ EA.T
            return float((S.argmax(1) == np.arange(len(pick))).mean() * 100)
        out[f"norm_{nm}"] = {"cos_A_B_cung_chu_tb": round(float(same_AB.mean()), 4), "cos_A_C_cung_chu_tb": round(float(same_AC.mean()), 4),
                             "cos_A_B_min": round(float(same_AB.min()), 4), "cos_A_C_min": round(float(same_AC.min()), 4),
                             "cos_A_A_hai_chu_khac_tb": round(float(diff.mean()), 4), "cos_A_A_hai_chu_khac_p95": round(float(np.percentile(diff, 95)), 4),
                             "truy_hoi_top1_B_vs_thu_vien_A_pct": round(retr(EB), 1), "truy_hoi_top1_C_vs_thu_vien_A_pct": round(retr(EC), 1), "truy_hoi_top1_A_vs_A_pct": round(retr(EA), 1)}
        print(f"[nhạy nền] norm={nm}: cos A↔B {out[f'norm_{nm}']['cos_A_B_cung_chu_tb']} (min {out[f'norm_{nm}']['cos_A_B_min']}), A↔C {out[f'norm_{nm}']['cos_A_C_cung_chu_tb']} (min {out[f'norm_{nm}']['cos_A_C_min']}), "
              f"hai chữ khác nhau tb {out[f'norm_{nm}']['cos_A_A_hai_chu_khac_tb']} (p95 {out[f'norm_{nm}']['cos_A_A_hai_chu_khac_p95']}); truy hồi B→A {out[f'norm_{nm}']['truy_hoi_top1_B_vs_thu_vien_A_pct']}% "
              f"C→A {out[f'norm_{nm}']['truy_hoi_top1_C_vs_thu_vien_A_pct']}%", flush=True)
    (OUT_DIR / "do_nhay_nen.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {OUT_DIR / 'do_nhay_nen.json'}")


if __name__ == "__main__":
    main()
