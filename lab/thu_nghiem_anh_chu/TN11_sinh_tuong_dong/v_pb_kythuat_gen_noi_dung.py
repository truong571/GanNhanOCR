"""v_pb_kythuat (04/10) — PHẢN BIỆN độc lập: ảnh sinh p02 (đường FST khởi tạo ngẫu nhiên) còn GIỮ NỘI DUNG (nhận ra đúng chữ) không? (CPU, 0 API, không sửa tệp).

Người kiểm chứng chứng minh đường FST ngẫu nhiên làm YẾU việc truyền phong cách (E1). Nhưng nếu nó còn phá cả NỘI DUNG chữ thì p02/p03 càng vô nghĩa; nếu nội dung vẫn
nguyên thì mức độ vô hiệu hoá nhẹ hơn. Đo "tự truy hồi": mỗi ảnh sinh của chữ c (793 ảnh p02 gen/s0, 189 ô, top-8 ứng viên) được so (cos MultiEnc v1+v2) với glyph PHÔNG
của mọi ứng viên cùng ô; đúng nếu glyph phông của CHÍNH c gần nhất. Đối chứng: cùng phép đo cho glyph FD thật trong kho (ArcFace/data/glyphs) ở các chữ có cả hai, và cho glyph phông-với-phông
(trần). Kỳ vọng ngẫu nhiên = trung bình 1/k. Kèm bản dilate ×2 (như p02/p03) và ×1 (như kho FD v2).
Ra: measure_out/_tn11/verify/pb_kythuat/gen_noi_dung.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 nice -n 10 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_pb_kythuat_gen_noi_dung.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

torch.set_num_threads(3)
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_kythuat"
P02 = REPO / "measure_out" / "_tn11" / "p02_B34"
K3 = np.ones((3, 3), np.uint8)


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def main():
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    P = pd.read_pickle(P02 / "plan.pkl"); S = P["S"]
    EMB = T.OUT / "emb"
    Sc = SI.Scorers(Assets(), "cpu", OUT / "emb_cache", False, lambda m: None)
    enc = Sc.enc()
    GL = SI.Glyphs()
    chars = sorted(set(S.c))
    emb_font = {}
    fl = [(c, GL.font(c)) for c in chars]
    fl = [(c, g) for c, g in fl if g is not None]
    E = nrm(np.asarray(enc.embed([g for _, g in fl], norm=False), np.float32))
    emb_font = dict(zip([c for c, _ in fl], E))
    variants = {}
    for name, thin in (("gen_t0", 0), ("gen_t1", 1), ("gen_t2", 2)):
        ims, cs = [], []
        for c in chars:
            f = P02 / "gen" / "s0" / f"U+{ord(c):04X}.png"
            if f.exists():
                im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if thin:
                    im = cv2.dilate(im, K3, iterations=thin)
                ims.append(im); cs.append(c)
        variants[name] = dict(zip(cs, nrm(np.asarray(enc.embed(ims, norm=False), np.float32))))
    # kho FD thật (ArcFace/data/glyphs): ảnh >= 1 KB
    fd_ims, fd_cs = [], []
    for c in chars:
        p = REPO / "ArcFace/data/glyphs" / f"U+{ord(c):04X}.png"
        if not p.exists():
            hits = [q for q in (REPO / "ArcFace/data/glyphs").glob(f"**/U+{ord(c):04X}.png") if q.stat().st_size >= 1024]
            p = hits[0] if hits else None
        if p is not None:
            fd_ims.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)); fd_cs.append(c)
    variants["FD_kho_that"] = dict(zip(fd_cs, nrm(np.asarray(enc.embed(fd_ims, norm=False), np.float32)))) if fd_ims else {}
    variants["phong_vs_phong(tran)"] = dict(emb_font)
    cov = S.groupby("i").y.max()
    Sv = S[S.i.isin(cov[cov == 1].index)]
    res = dict(o=int(Sv.i.nunique()), n_chu=len(chars), n_phong=len(emb_font), n_anh={k: len(v) for k, v in variants.items()})
    for name, V in variants.items():
        hit = tot = 0; rnd = []
        cos_self = []
        for i, g in Sv.groupby("i"):
            cs = [c for c in g.c if c in emb_font]
            for c in cs:
                if c not in V:
                    continue
                sims = [float(V[c] @ emb_font[c2]) for c2 in cs]
                tot += 1; hit += int(cs[int(np.argmax(sims))] == c); rnd.append(1.0 / len(cs)); cos_self.append(float(V[c] @ emb_font[c]))
        res[name] = dict(n_truy_van=tot, tu_truy_hoi_top1=round(hit / max(tot, 1) * 100, 1), ngau_nhien=round(float(np.mean(rnd)) * 100, 1) if rnd else None,
                         cos_voi_phong_cung_chu_tb=round(float(np.mean(cos_self)), 3) if cos_self else None)
    # chỉ trên các chữ có CẢ gen và FD thật, để so cùng tập
    both = [c for c in chars if c in variants["gen_t0"] and c in variants["FD_kho_that"] and c in emb_font]
    res["cung_tap_chu_co_ca_gen_va_FD"] = dict(n_chu=len(both))
    for name in ("gen_t0", "gen_t2", "FD_kho_that"):
        V = variants[name]; hit = tot = 0
        for i, g in Sv.groupby("i"):
            cs = [c for c in g.c if c in emb_font and c in both]
            for c in cs:
                sims = [float(V[c] @ emb_font[c2]) for c2 in cs]
                tot += 1; hit += int(cs[int(np.argmax(sims))] == c)
        res["cung_tap_chu_co_ca_gen_va_FD"][name] = dict(n_truy_van=tot, tu_truy_hoi_top1=round(hit / max(tot, 1) * 100, 1))
    (OUT / "gen_noi_dung.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
