"""v_kythuat (03/10): 3 tâm con của encoder v2 (ArcFace/checkpoints/last.pt, K = 3) có tương ứng "3 phong cách chữ viết" không? (CPU)

Báo cáo (Công nghệ 5) viết: "mỗi mã Unicode có K = 3 tâm con ... tương ứng 3 phong cách: Chân phương, Hành thư, Thảo thư".
Kiểm tra bằng số đo trên dữ liệu huấn luyện thật của encoder v2 (ArcFace/data/manifest.csv: 59.823 crop STT + 1.564 glyph FD;
KHÔNG có cột phong cách nào):
  1. mỗi lớp lấy tối đa N crop GOLD (seed 0) + glyph FD của lớp; nhúng bằng backbone của last.pt (epoch 30);
  2. gán mỗi mẫu cho tâm con có cos lớn nhất của đúng lớp của nó (đúng cách head SubCenterArcMargin chấm: max trên K);
  3. đo: phần crop thuộc tâm con trội; số lớp mà tâm con thứ hai nhận >= 20 % crop; glyph FD có cùng tâm con trội với crop không;
     cos giữa các tâm con cùng lớp vs khác lớp; thông tin tương hỗ chuẩn hoá giữa (tâm con) và (quyển STT / tầng GOLD-SILVER / nguồn crop-FD).
Ra: measure_out/_tn11/verify/kythuat/subcenter.json
    nice -n 10 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_kythuat_subcenter.py [--per-class 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "ArcFace"))
OUT = REPO / "measure_out" / "_tn11" / "verify" / "kythuat"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
DEV = "cpu"


def mi_norm(a, b):
    """thông tin tương hỗ chuẩn hoá I(a;b)/min(H(a),H(b)) (0 = độc lập, 1 = xác định hoàn toàn)."""
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    ja = Counter(zip(a, b)); ca = Counter(a); cb = Counter(b)
    mi = sum(c / n * np.log((c / n) / ((ca[x] / n) * (cb[y] / n))) for (x, y), c in ja.items())
    ha = -sum(c / n * np.log(c / n) for c in ca.values())
    hb = -sum(c / n * np.log(c / n) for c in cb.values())
    return float(mi / max(min(ha, hb), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6)
    a = ap.parse_args()
    from model import NomEmbedder
    from dataset import NomCropDataset
    ck = torch.load(REPO / "ArcFace/checkpoints/last.pt", map_location="cpu", weights_only=False)
    k, n_cls = ck["k"], len(ck["classes"])
    lab2idx = ck["classes"]
    W = F.normalize(ck["head"]["W"].float(), dim=1).view(n_cls, k, -1)
    emb = NomEmbedder(ck["embed_dim"], pretrained=False, arch=ck["arch"]); emb.load_state_dict(ck["backbone"]); emb.eval()
    man = pd.read_csv(REPO / "ArcFace/data/manifest.csv")
    man = man[man.label.isin(lab2idx)].copy()
    man["y"] = man.label.map(lab2idx)
    rng = np.random.default_rng(0)
    crops = man[(man.source == "crop") & (man.tier == "GOLD")]
    pick = []
    for y, g in crops.groupby("y"):
        idx = g.index.to_numpy()
        pick += list(rng.choice(idx, size=min(a.per_class, len(idx)), replace=False))
    S = pd.concat([man.loc[pick], man[man.source == "fd"]]).reset_index(drop=True)
    paths = [str(REPO / "ArcFace/data" / p) for p in S.path]
    ds = NomCropDataset(paths, S.y.tolist(), img=ck["img"], train=False)
    cache = OUT / f"subcenter_emb_pc{a.per_class}.npy"
    if cache.exists() and np.load(cache).shape[0] == len(ds):
        E = torch.from_numpy(np.load(cache))
        print("  dùng lại nhúng đã lưu", cache.name, flush=True)
    else:
        E = []
        with torch.no_grad():
            for s in range(0, len(ds), 128):
                x = torch.stack([ds[i][0] for i in range(s, min(s + 128, len(ds)))])
                E.append(emb(x).float())
                if (s // 128) % 10 == 0:
                    print(f"  nhúng {s}/{len(ds)}", flush=True)
        E = torch.cat(E)
        np.save(cache, E.numpy())
    y = torch.tensor(S.y.values)
    cosk = torch.einsum("nd,nkd->nk", E, W[y])             # cos tới K tâm con của đúng lớp
    sub = cosk.argmax(1).numpy()
    S["sub"] = sub
    S["cos_max"] = cosk.max(1).values.numpy()
    cr = S[S.source == "crop"]; fd = S[S.source == "fd"].set_index("y")
    per = []
    for yy, g in cr.groupby("y"):
        if len(g) < 4:
            continue
        cnt = np.bincount(g["sub"], minlength=k)
        dom = int(cnt.argmax())
        per.append(dict(y=int(yy), n=len(g), dom=dom, share=cnt.max() / len(g), second=np.sort(cnt)[-2] / len(g),
                        fd_sub=int(fd.loc[yy, "sub"]) if yy in fd.index else -1, n_trong=int((cnt == 0).sum())))
    P = pd.DataFrame(per)
    # trong cùng một lớp, đa số tâm con của từng quyển STT (>= 2 crop/quyển) có trùng nhau không?
    ag = []
    for yy, g in cr.groupby("y"):
        maj = []
        for bk, gg in g.groupby("book"):
            if len(gg) >= 2:
                maj.append(int(np.bincount(gg["sub"], minlength=k).argmax()))
        if len(maj) >= 2:
            ag.append(len(set(maj)) == 1)
    agree = dict(so_lop=len(ag), ti_le_trung=float(np.mean(ag)) if ag else None)
    # cos giữa các tâm con
    within = []
    for c in range(n_cls):
        w = W[c] @ W[c].T
        within += [float(w[0, 1]), float(w[0, 2]), float(w[1, 2])]
    perm = rng.permutation(n_cls)
    across = [float((W[c, 0] * W[perm[c], 0]).sum()) for c in range(n_cls)]
    # tâm con theo (quyển STT, tầng, nguồn)
    res = dict(
        thiet_lap=dict(ckpt="ArcFace/checkpoints/last.pt", epoch=int(ck["epoch"]), k=k, so_lop=n_cls, per_class=a.per_class,
                       n_mau=int(len(S)), n_crop=int(len(cr)), n_fd=int(len(fd)), n_lop_co_tu_4_crop=int(len(P))),
        phan_bo_tam_con_crop=[int((cr["sub"] == j).sum()) for j in range(k)],
        phan_bo_tam_con_fd=[int((S[S.source == "fd"]["sub"] == j).sum()) for j in range(k)],
        phan_tram_crop_vao_tam_con_troi=dict(tb=float(P.share.mean()), trung_vi=float(P.share.median()),
                                            p10=float(P.share.quantile(0.1)), p90=float(P.share.quantile(0.9))),
        lop_mot_tam_con_nhan_100pct_crop=float((P.share >= 0.999).mean()),
        lop_tam_con_thu_hai_nhan_ge_20pct=float((P.second >= 0.2).mean()),
        lop_co_tam_con_rong=float((P.n_trong > 0).mean()),
        glyph_fd_cung_tam_con_troi_voi_crop=float((P.fd_sub == P.dom).mean()),
        cos_tam_con_cung_lop=dict(tb=float(np.mean(within)), min=float(np.min(within)), max=float(np.max(within))),
        cos_tam_con_khac_lop_ngau_nhien=dict(tb=float(np.mean(across))),
        cos_mau_toi_tam_con_gan_nhat_crop=float(cr.cos_max.mean()),
        cos_mau_toi_tam_con_gan_nhat_fd=float(S[S.source == "fd"].cos_max.mean()),
        ghi_chu_MI="chỉ số tâm con là nhãn tuỳ ý theo từng lớp -> thông tin tương hỗ gộp mọi lớp vô nghĩa, KHÔNG dùng; xem dong_y_giua_cac_quyen",
        dong_y_giua_cac_quyen=agree,
        ti_le_tam_con_theo_quyen={b: [float((g["sub"] == j).mean()) for j in range(k)] for b, g in cr.groupby("book")},
    )
    (OUT / "subcenter.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
