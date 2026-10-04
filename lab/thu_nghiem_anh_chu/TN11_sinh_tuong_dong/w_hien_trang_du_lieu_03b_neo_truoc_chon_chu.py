"""w_hien_trang_du_lieu_03b_neo_truoc_chon_chu.py — Ô NEO "THUẦN TỰ ĐỘNG" (trước bước chon_chu 4b) so với ô neo trên nhãn HIỆN TẠI, 7 bộ không phải STT.

chon_chu (config/chon_chu.yaml) dùng bộ chọn ảnh học trên nhãn NGƯỜI của sách kia (LOBO) để (a) nâng ô REVIEW lên GOLD và (b) với họ chữ viết tay (B18/B34), SỬA nhãn ô GOLD
khi top-1 ≠ nhãn (L5). Ô neo tính trên nhãn hiện tại vì vậy mất các ô bị L5 đổi nhãn. Kiểm: ô neo trên labels_gated_truoc_chon_chu.csv (cùng định nghĩa) so với
(i) ô neo của bảng TN8 (cand/<b>_cells.pkl) — nếu bằng nhau thì bảng TN8 dựng TRƯỚC chon_chu; (ii) ô neo hiện tại; (iii) số phận của ô chỉ-có-trước (tầng/nhãn/crop hiện tại).
Ra: measure_out/_tn11/full/hien_trang_du_lieu/neo_truoc_chon_chu.json. 0 API/GPU.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_03b_neo_truoc_chon_chu.py
"""
from __future__ import annotations

import collections
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import w_hien_trang_du_lieu_lib as L  # noqa: E402


def key(df):
    return (df.page + "|" + df.column.astype(int).astype(str) + "|" + df.nom_idx.astype(str) + "|" + df.syl_idx.astype(str)).to_numpy()


def main():
    out = {}
    for ma in ["Chr", "L83", "KVK", "L16", "TK", "B18", "B34"]:
        C = L.CFG[ma]
        root = L.REPO / C["all_root"]
        pre = L.rd(root / "labels_gated_truoc_chon_chu.csv")
        cur = L.load_all(ma)
        base = pd.read_pickle(L.TN8 / "base" / f"{ma}.pkl")
        cells = pd.read_pickle(L.TN8 / "cand" / f"{ma}_cells.pkl")
        inv = L.Inv()
        Ap, Ac = L.anchor_mask(pre), L.anchor_mask(cur)
        kp, kc, kb = key(pre), key(cur), key(base)
        a8 = set(kb[cells["anchor"].to_numpy(bool)])
        sp, sc = set(kp[Ap]), set(kc[Ac])
        R = dict(ma=ma, n_dong_truoc=int(len(pre)), n_dong_hien_tai=int(len(cur)), tang_truoc=pre.tier.value_counts().to_dict(), tang_hien_tai=cur.tier.value_counts().to_dict(),
                 mtime_truoc=L.mt(root / "labels_gated_truoc_chon_chu.csv"), mtime_hien_tai=L.mt(root / "labels_gated.csv"))
        R["neo_truoc_chon_chu"] = len(sp); R["neo_hien_tai"] = len(sc); R["neo_tn8"] = len(a8)
        R["neo_truoc_bang_neo_tn8"] = bool(sp == a8)
        R["chi_truoc"] = len(sp - sc); R["chi_hien_tai"] = len(sc - sp)
        inv.check("neo_hien_tai_la_tap_con_cua_neo_truoc", sc <= sp, f"{len(sc - sp)} ô chỉ có ở hiện tại")
        # số phận của ô chỉ-có-trước
        cc = cur.copy(); cc["_k"] = kc; cc = cc.set_index("_k")
        pp = pre.copy(); pp["_k"] = kp; pp = pp.set_index("_k")
        lost = sorted(sp - sc)
        if lost:
            X = pd.DataFrame(dict(tier_nay=cc.loc[lost, "tier"].to_numpy(), label_nay=cc.loc[lost, "label"].to_numpy(), kim=cc.loc[lost, "ocr_char"].to_numpy(),
                                  label_truoc=pp.loc[lost, "label"].to_numpy(), rule_nay=cc.loc[lost, "rule"].to_numpy(), image_nay=cc.loc[lost, "image"].to_numpy()))
            X["doi_nhan"] = (X.label_nay != X.label_truoc) & (X.label_nay != "")
            X["nhan_rong"] = X.label_nay == ""
            R["so_phan_o_chi_truoc"] = dict(tang_nay=X.tier_nay.value_counts().to_dict(), doi_nhan_khac_rong=int(X.doi_nhan.sum()), nhan_rong=int(X.nhan_rong.sum()),
                                            hau_to_luat_nay=dict(collections.Counter(t.split(":")[0] + ":" + t.split(":")[1] if ":" in t else t for r in X.rule_nay for t in r.split("|")[1:]).most_common(6)),
                                            co_tep_crop_nay=int(sum(os.path.isfile(str(root / im)) for im in X.image_nay if im)))
        # ĐO bằng nhãn người (CHỈ chấm; B18/B34): ô neo bị chon_chu L5 đổi nhãn — chữ kim (nhãn trước) và nhãn mới đúng bao nhiêu?
        if ma in ("B18", "B34") and lost:
            from pipeline.gold_exact import common as C0
            sys.path.insert(0, str(L.REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
            import tn8lib as T
            Cx = T.lex()
            gtm = pd.Series(base.gt_char.to_numpy(), index=kb)
            g = gtm.reindex(lost).fillna("").to_numpy()
            has = g != ""
            kim_ok = np.array([bool(x) and (a == x or Cx.var_eq_plus(x, a)) for a, x in zip(X.label_truoc, g)])
            new_ok = np.array([bool(x) and (a == x or Cx.var_eq_plus(x, a)) for a, x in zip(X.label_nay, g)])
            R["do_bang_nhan_nguoi_o_chi_truoc"] = dict(n=int(len(lost)), co_gt=int(has.sum()), nhan_truoc_dung_pct=round(float(kim_ok[has].mean()) * 100, 2) if has.any() else None,
                                                       nhan_moi_dung_pct=round(float(new_ok[has].mean()) * 100, 2) if has.any() else None,
                                                       ghi_chu="CHỈ để chấm (nhãn người); nhãn trước = chữ kim == ô neo thuần tự động; nhãn mới = chon_chu L5 (bộ chọn ảnh học trên nhãn người sách kia)")
        if ma in ("B18", "B34", "L16", "TK"):
            sys.path.insert(0, str(L.REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
            import tn8lib as T
            Cx = T.lex()
            gtm = pd.Series(base.gt_char.to_numpy(), index=kb)
            def prec(keys, labs):
                g = gtm.reindex(keys).fillna("").to_numpy()
                has = g != ""
                ok = np.array([bool(x) and (a == x or Cx.var_eq_plus(x, a)) for a, x in zip(labs, g)])
                return dict(n=int(len(keys)), co_gt=int(has.sum()), dung_chu_pct=round(float(ok[has].mean()) * 100, 2) if has.any() else None)
            kp_l = sorted(sp); kc_l = sorted(sc)
            R["do_bang_nhan_nguoi_CHI_CHAM"] = dict(
                neo_truoc_chon_chu=prec(kp_l, pp.loc[kp_l, "label"].to_numpy()), neo_hien_tai=prec(kc_l, cc.loc[kc_l, "label"].to_numpy()),
                neo_tn8=prec(sorted(a8), base.set_index(kb).loc[sorted(a8), "label"].to_numpy()))
        # tầng trước của ô neo hiện tại
        kk = sorted(sc)
        R["tang_truoc_cua_o_neo_hien_tai"] = pp.loc[kk, "tier"].value_counts().to_dict()
        R["tang_nay_cua_o_neo_hien_tai"] = cc.loc[kk, "tier"].value_counts().to_dict()
        # ô neo trước chon_chu có tệp crop (hiện tại)?
        kp_all = sorted(sp)
        have = np.array([bool(cc.loc[k, "image"]) and os.path.isfile(str(root / cc.loc[k, "image"])) for k in kp_all])
        R["neo_truoc_co_tep_crop_o_nhan_hien_tai"] = int(have.sum())
        # nhãn của ô neo trước tại bảng trước vs chữ kim
        R["chu_neo_truoc"] = int(pre[Ap].label.nunique()); R["chu_neo_hien_tai"] = int(cur[Ac].label.nunique())
        R["bat_bien"] = inv.summary(); R["bat_bien_chi_tiet"] = inv.rows
        out[ma] = R
        print(f"[03b] {ma}: neo trước chon_chu {len(sp)} = neo TN8 {len(a8)} ({R['neo_truoc_bang_neo_tn8']}) | neo hiện tại {len(sc)} | chỉ-trước {len(sp - sc)} "
              f"{R.get('so_phan_o_chi_truoc', {}).get('tang_nay', '')} đổi nhãn {R.get('so_phan_o_chi_truoc', {}).get('doi_nhan_khac_rong', 0)} | crop có {R['neo_truoc_co_tep_crop_o_nhan_hien_tai']}", flush=True)
    L.jdump(out, L.OUT / "neo_truoc_chon_chu.json")
    print("->", L.rel(L.OUT / "neo_truoc_chon_chu.json"))


if __name__ == "__main__":
    main()
