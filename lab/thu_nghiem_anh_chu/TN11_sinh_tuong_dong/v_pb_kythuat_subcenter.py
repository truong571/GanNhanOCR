"""v_pb_kythuat (04/10) — PHẢN BIỆN độc lập: "K = 3 tâm con của encoder v2 ứng với 3 phong cách chữ viết?" (CPU, 0 API, không sửa tệp).

Người kiểm chứng (v_kythuat_subcenter.py): 92,5 % crop vào MỘT tâm con; 96 % lớp có tâm con rỗng; cos tâm con cùng lớp 0,105 ≈ khác lớp 0,168;
=> KHÔNG ứng với 3 kiểu thư pháp. Script này kiểm lại bằng MẪU KHÁC (seed 1, tối đa 10 crop GOLD/lớp) và thêm 4 phép đo mà họ chưa làm:
  A  bố cục W: view(n_cls, K, D) đúng với head SubCenterArcMargin (cos.view(-1, n_classes, k)) — kiểm bằng chính hàm sub_cosines của mã gốc
     (độ lệch tối đa giữa max-theo-view của tôi và sub_cosines);
  B  GIỮ RA NGOÀI: tâm con trội của mỗi lớp xác định bằng crop trang HUẤN LUYỆN (assign_splits page_disjoint, seed 42 như train.py), rồi đo crop
     trang val/test (không thấy lúc huấn luyện) vào tâm con trội bao nhiêu % — nếu tâm con mã hoá "phong cách" thì crop ngoài tập vẫn phải trải đều;
  C  tâm con KHÔNG trội hấp thụ crop "khó" hay "phong cách"? So cos_max (độ gần tâm con gần nhất), tỉ lệ mực, tỉ lệ khung của crop vào tâm con
     trội vs không trội (AUC Mann–Whitney, kèm CI bootstrap theo lớp);
  D  khẳng định "dữ liệu huấn luyện không có nhãn phong cách": cột manifest, số quyển, nguồn.
Ra: measure_out/_tn11/verify/pb_kythuat/subcenter.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 nice -n 10 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_pb_kythuat_subcenter.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "ArcFace"))
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_kythuat"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(3)


def auc(a, b):
    """P(a > b) Mann–Whitney (hoà = 0,5) — a, b mảng 1 chiều."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # xử lý hoà bằng hạng trung bình
    s = pd.Series(allv)
    ranks = s.rank(method="average").to_numpy()
    ra = ranks[: len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def main():
    from model import NomEmbedder, SubCenterArcMargin
    from dataset import NomCropDataset, assign_splits
    ck = torch.load(REPO / "ArcFace/checkpoints/last.pt", map_location="cpu", weights_only=False)
    k, n_cls, D = ck["k"], len(ck["classes"]), ck["embed_dim"]
    lab2idx = ck["classes"]
    Wraw = ck["head"]["W"].float()
    Wn = F.normalize(Wraw, dim=1)
    W3 = Wn.view(n_cls, k, D)
    res = dict(thiet_lap=dict(k=k, n_lop=n_cls, embed_dim=D, epoch=int(ck["epoch"]), val_best=float(ck["best"]), W_shape=list(Wraw.shape)))

    # A — bố cục W: so với sub_cosines của mã gốc
    head = SubCenterArcMargin(D, n_cls, k=k)
    head.load_state_dict(ck["head"])
    emb = NomEmbedder(D, pretrained=False, arch=ck["arch"]); emb.load_state_dict(ck["backbone"]); emb.eval()
    man = pd.read_csv(REPO / "ArcFace/data/manifest.csv")
    res["D_manifest"] = dict(cot=list(man.columns), nguon=man.source.value_counts().to_dict(), quyen=man.book.value_counts().to_dict(),
                             co_cot_phong_cach=any(c.lower() in ("style", "phong_cach", "script", "font", "thu_phap") for c in man.columns))
    man = man[man.label.isin(lab2idx)].copy()
    man["y"] = man.label.map(lab2idx)
    man["split"] = assign_splits(man.reset_index(drop=True), mode="page_disjoint", val_frac=0.1, test_frac=0.1, seed=42)
    crops = man[(man.source == "crop") & (man.tier == "GOLD")]
    rng = np.random.default_rng(1)
    pick = []
    for y, g in crops.groupby("y"):
        tr = g[g.split == "train"].index.to_numpy(); ho = g[g.split != "train"].index.to_numpy()
        pick += list(rng.choice(tr, size=min(10, len(tr)), replace=False)) if len(tr) else []
        pick += list(rng.choice(ho, size=min(10, len(ho)), replace=False)) if len(ho) else []
    S = man.loc[pick].reset_index(drop=True)
    paths = [str(REPO / "ArcFace/data" / p) for p in S.path]
    ds = NomCropDataset(paths, S.y.tolist(), img=ck["img"], train=False)
    E = []
    ink = []; asp = []
    with torch.no_grad():
        for s in range(0, len(ds), 128):
            xs = [ds[i][0] for i in range(s, min(s + 128, len(ds)))]
            E.append(emb(torch.stack(xs)).float())
            if (s // 128) % 10 == 0:
                print(f"  nhúng {s}/{len(ds)}", flush=True)
    E = torch.cat(E)
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        ink.append(float((g < 128).mean())); asp.append(g.shape[1] / g.shape[0])
    S["ink"] = ink; S["asp"] = asp
    # A — đối chiếu bố cục
    with torch.no_grad():
        sub_ref = head.sub_cosines(E[:64])                      # (64, n_cls) từ mã gốc (max theo K)
        cos_all = torch.einsum("nd,ckd->nck", E[:64], W3)       # (64, n_cls, k) theo view của tôi
        mine = cos_all.max(2).values
    res["A_bo_cuc_W"] = dict(do_lech_toi_da_so_voi_sub_cosines_goc=float((sub_ref - mine).abs().max()), ghi_chu="0 => view(n_cls,k,D) đúng bố cục head")
    y = torch.tensor(S.y.values)
    cosk = torch.einsum("nd,nkd->nk", E, W3[y])
    S["sub"] = cosk.argmax(1).numpy(); S["cos_max"] = cosk.max(1).values.numpy()
    tr = S[S.split == "train"]; ho = S[S.split != "train"]
    # B — tâm con trội theo crop HUẤN LUYỆN, đo trên crop NGOÀI tập
    per = []
    for yy, g in tr.groupby("y"):
        if len(g) < 4:
            continue
        cnt = np.bincount(g["sub"], minlength=k)
        dom = int(cnt.argmax())
        h = ho[ho.y == yy]
        per.append(dict(y=int(yy), n_tr=len(g), dom=dom, share_tr=cnt.max() / len(g), n_ho=len(h),
                        share_ho=float((h["sub"] == dom).mean()) if len(h) else np.nan, n_empty=int((cnt == 0).sum()),
                        n_active=int((cnt > 0).sum())))
    P = pd.DataFrame(per)
    Pho = P[P.n_ho >= 2]
    res["B_giu_ra_ngoai"] = dict(
        n_lop_tr_ge4=int(len(P)), n_lop_co_ho_ge2=int(len(Pho)), n_crop_tr=int(len(tr)), n_crop_ho=int(len(ho)),
        share_tam_con_troi_tren_crop_HUAN_LUYEN=dict(tb=round(float(P.share_tr.mean()), 4), trung_vi=float(P.share_tr.median())),
        share_tam_con_troi_tren_crop_NGOAI_TAP=dict(tb=round(float(np.nanmean(Pho.share_ho)), 4), trung_vi=float(np.nanmedian(Pho.share_ho))),
        so_tam_con_hoat_dong_moi_lop=dict(tb=round(float(P.n_active.mean()), 3), phan_bo={int(a): int((P.n_active == a).sum()) for a in sorted(P.n_active.unique())}),
        ti_le_lop_co_tam_con_rong=round(float((P.n_empty > 0).mean()), 4),
        ngau_nhien_neu_3_tam_con_deu=round(1 / k, 3),
    )
    # C — tâm con không trội hấp thụ crop thế nào
    dom_of = dict(zip(P.y, P.dom))
    S2 = S[S.y.isin(dom_of)].copy()
    S2["is_dom"] = [int(s == dom_of[yy]) for s, yy in zip(S2["sub"], S2.y)]
    d1, d0 = S2[S2.is_dom == 1], S2[S2.is_dom == 0]
    cl = {}
    for col in ("cos_max", "ink", "asp"):
        cl[col] = dict(tb_trong=round(float(d1[col].mean()), 4), tb_khong_trong=round(float(d0[col].mean()), 4) if len(d0) else None,
                       AUC_trong_hon_khong_trong=round(auc(d1[col], d0[col]), 3) if len(d0) else None)
    # CI bootstrap theo lớp cho AUC của cos_max
    ys = np.array(sorted(S2.y.unique())); rb = np.random.default_rng(0); vals = []
    grp = {yy: g for yy, g in S2.groupby("y")}
    for _ in range(300):
        pick_y = rb.choice(ys, len(ys))
        gg = pd.concat([grp[yy] for yy in pick_y])
        a1 = gg[gg.is_dom == 1].cos_max; a0 = gg[gg.is_dom == 0].cos_max
        if len(a0) > 5:
            vals.append(auc(a1, a0))
    cl["AUC_cos_max_CI95_boot_lop"] = [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)] if vals else None
    res["C_tam_con_khong_troi"] = dict(n_crop_trong=int(len(d1)), n_crop_khong_trong=int(len(d0)), ti_le_khong_trong=round(len(d0) / max(1, len(S2)), 4), **cl)
    # cos tâm con
    within = []
    for c in range(n_cls):
        w = W3[c] @ W3[c].T
        within += [float(w[0, 1]), float(w[0, 2]), float(w[1, 2])]
    perm = rng.permutation(n_cls)
    across = [float((W3[c, 0] * W3[perm[c], 0]).sum()) for c in range(n_cls)]
    across_pairs = [float((W3[c, 0] * W3[perm[c], 1]).sum()) for c in range(n_cls)]
    res["cos_tam_con"] = dict(cung_lop_tb=round(float(np.mean(within)), 4), khac_lop_tb=round(float(np.mean(across)), 4), khac_lop_cap_khac_chi_so=round(float(np.mean(across_pairs)), 4))
    (OUT / "subcenter.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
