"""v_truy_nguon_05 (03/10) — TÍNH VÒNG TRÒN của phép chấm "Pipeline GOLD" (6 bộ không nhãn người: stt2, stt4, stt11, Chr, L83, KVK). 0 API, CPU, chỉ ĐỌC.

Giao thức p05 cho 6 bộ không nhãn người (p05_full_corpus_he_quy_chieu.py:183-188): sự thật := D.label của pipeline (ô có nhãn khác rỗng và
nhãn nằm trong tập ứng viên). Script này đo CÙNG tín hiệu ảnh (f_font, f_fd, f_self, f_head, f_vW; bảng ứng viên TN8, ~26 ứng viên/ô, TẤT CẢ ô,
không lọc top-3) với hai loại "sự thật":
  (A) nhãn pipeline (như p05)  — 10 bộ;
  (B) chữ người               — chỉ B18, B34, L16, TK (cùng các ô, để thấy riêng tác động của định nghĩa sự thật).
Nếu tín hiệu ảnh độc lập với quyết định của pipeline thì (A) và (B) xấp xỉ nhau trên bộ có cả hai; nếu (A) của STT cao bất thường so với (B) của Borg
cùng loại chữ viết tay thì phép chấm (A) đo "đồng ý với pipeline", không đo độ đúng.
Ra: measure_out/_tn11/verify/truy_nguon/tron_vong.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "truy_nguon"
OUT.mkdir(parents=True, exist_ok=True)
HUMAN = ("B18", "B34", "L16", "TK")
SIG = ("f_font", "f_fd", "f_self", "f_head", "f_vW")


def top1(F, col, ycol):
    x = F[["i", col, ycol]].copy(); x[col] = x[col].fillna(-9)
    return x.loc[x.groupby("i")[col].idxmax(), ["i", ycol]].set_index("i")[ycol]


def rate(ok, pages):
    m, lo, hi = T.boot_ci_pages(np.asarray(ok, float), pages, B=1000)
    return dict(p=round(100 * m, 2), lo=round(100 * lo, 2), hi=round(100 * hi, 2), n=int(len(ok)))


def main():
    res = {}
    for b in T.ORDER:
        D = T.load_base(b)
        F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y"] + list(SIG)]
        lbl = D.label.values[F.i.values]
        F["ylab"] = ((F.c.values == lbl) & (lbl != "")).astype(int)
        tier = D.tier.values
        pg = D.page.values
        hl = F.groupby("i").ylab.max(); cells_l = hl[hl == 1].index.values
        r = dict(bo=b, nhan_nguoi=b in HUMAN, o_nhan_pipeline=int(len(cells_l)),
                 tang={t: int((tier[cells_l] == t).sum()) for t in sorted(set(tier[cells_l]))})
        tg = tier[cells_l] == "GOLD"
        r["A_nhan_pipeline_tat_ca"] = {}
        r["A_nhan_pipeline_chi_GOLD"] = {}
        for m in SIG:
            t = top1(F[F.i.isin(cells_l)], m, "ylab").reindex(cells_l).values
            r["A_nhan_pipeline_tat_ca"][m] = rate(t, pg[cells_l])
            r["A_nhan_pipeline_chi_GOLD"][m] = rate(t[tg], pg[cells_l][tg])
        if b in HUMAN:
            hh = F.groupby("i").y.max(); cells_h = hh[hh == 1].index.values
            both = np.intersect1d(cells_l, cells_h)
            r["o_chung_nhan_pipeline_va_chu_nguoi"] = int(len(both))
            # độ chính xác của chính nhãn pipeline trên các ô chung (nhãn == chữ người, dị thể không tính -> chặt, theo y của TN8)
            Fy = F[F.i.isin(both)]
            ylab_is_hum = Fy[(Fy.ylab == 1)].groupby("i").y.max().reindex(both).fillna(0).values
            r["nhan_pipeline_dung_chu_nguoi_tren_o_chung_pct"] = round(100 * float(ylab_is_hum.mean()), 2)
            r["A_tren_o_chung"] = {}; r["B_chu_nguoi_tren_o_chung"] = {}; r["chenh_A_tru_B_diem"] = {}
            for m in SIG:
                ta = top1(Fy, m, "ylab").reindex(both).values
                tb = top1(Fy, m, "y").reindex(both).values
                r["A_tren_o_chung"][m] = rate(ta, pg[both]); r["B_chu_nguoi_tren_o_chung"][m] = rate(tb, pg[both])
                r["chenh_A_tru_B_diem"][m] = round(100 * float(ta.mean() - tb.mean()), 2)
        res[b] = r
        print(f"\n== {b} {'[có chữ người]' if b in HUMAN else '[chỉ nhãn pipeline]'}: ô có nhãn pipeline∈ứng viên {r['o_nhan_pipeline']}; tầng {r['tang']}")
        print("   (A) nhãn pipeline, mọi tầng :", {m: v['p'] for m, v in r["A_nhan_pipeline_tat_ca"].items()})
        print("   (A) nhãn pipeline, chỉ GOLD :", {m: v['p'] for m, v in r["A_nhan_pipeline_chi_GOLD"].items()})
        if b in HUMAN:
            print(f"   ô chung {r['o_chung_nhan_pipeline_va_chu_nguoi']}; nhãn pipeline == chữ người trên ô chung: {r['nhan_pipeline_dung_chu_nguoi_tren_o_chung_pct']} %")
            print("   (A) vs nhãn pipeline (ô chung):", {m: v['p'] for m, v in r["A_tren_o_chung"].items()})
            print("   (B) vs chữ người     (ô chung):", {m: v['p'] for m, v in r["B_chu_nguoi_tren_o_chung"].items()})
            print("   chênh A − B (điểm):", r["chenh_A_tru_B_diem"])
    (OUT / "tron_vong.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
