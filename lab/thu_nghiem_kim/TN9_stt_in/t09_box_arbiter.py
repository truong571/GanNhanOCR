"""t09_box_arbiter.py — giả thuyết 2 (hộp STT): ở ô GOLD mà hai bộ giải hộp LỆCH NHAU (IoU < 0,5), hộp nào đúng? 0 API.

Hai trọng tài KHÔNG dùng encoder MultiEnc (encoder học crop STT theo hộp legacy ⇒ thiên vị):
  (1) CNN kiểm ảnh↔chữ sạch với STT (hand_T/L_Kinh + hand_T/L_DungLy, học Borg): s(c) = trung bình z·W[c] qua 4 mạng;
      lề m = s(nhãn) − max s(nhãn ô kề ±1, ±2 cùng cột). Hộp đúng ⇒ nhãn thắng láng giềng.
  (2) kim_geo: tâm crop nằm trong hộp chia đều của chính chữ kim đã ghép (t06).
Kiểm độ tin trọng tài (1) trên Borg B18 (hộp người; mạng LOBO = hand_*_DungLy): hai bản TN6 legacy vs visual_dp, ô GOLD cả
hai, IoU < 0,5, P(trọng tài chọn đúng hộp | một trong hai đúng khe).
Crop cắt lại đúng hình học save_crop (TN8 t02.cut): STT ảnh đã xử lý; Borg ảnh gốc.
Ra measure_out/_tn9/box_arbiter.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402
import t02_embed as E2  # noqa: E402  (TN8: cut, page_crops)
from pipeline.gold_exact import signals_img as SI  # noqa: E402
from pipeline.gold_exact.common import Assets  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def nets_tables(names):
    A = Assets()
    U = torch.load(T.REPO / "models/gold_exact/vft_universe.pt", weights_only=False)["U"]
    uid = {c: j for j, c in enumerate(U)}
    out = []
    for nm in names:
        net = SI.load_net(A.load(f"{nm.replace('hand_', 'hand_model_')}.pt"), DEV)
        tb = torch.load(T.REPO / "models/gold_exact" / f"{nm.replace('hand_', 'hand_tables_')}.pt", map_location="cpu", weights_only=False)
        out.append((net, np.asarray(tb["W"], np.float32)))
    return out, uid


def crops_for(L: pd.DataFrame, sel_idx, prep: Path, orig: bool):
    """Crop cho các hàng sel_idx của L (láng giềng carve theo MỌI hộp cùng trang+cột của L)."""
    res = {}
    for pg, g in L.groupby("page"):
        want = [i for i in g.index if i in sel_idx]
        if not want:
            continue
        recs = [(i, c, T8b(bb)) for i, c, bb in zip(g.index, g.column, g.bbox)]
        for i, im, _ in E2.page_crops(("x", prep / "pages" / f"{pg}.png", prep, orig, recs)):
            if i in sel_idx and im is not None and im.size:
                res[i] = im
    return res


def T8b(s):
    try:
        v = json.loads(s)
        return [float(x) for x in v[:4]]
    except Exception:  # noqa: BLE001
        return None


def scores(imgs: dict, chars_by_i: dict, nets, uid):
    """-> {i: {c: s}} trung bình z·W[c] qua các mạng."""
    ids = sorted(imgs)
    if not ids:
        return {}
    X = np.stack([SI.prep64(imgs[i]) for i in ids])
    Zs = [SI.net_embed(net, X, DEV) for net, _ in nets]
    out = {}
    for r, i in enumerate(ids):
        d = {}
        for c in chars_by_i[i]:
            j = uid.get(c)
            if j is None:
                continue
            d[c] = float(np.mean([Z[r] @ W[j] for Z, (_, W) in zip(Zs, nets)]))
        out[i] = d
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    return I / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I)


def neighbors(L: pd.DataFrame):
    """(page, column, syl) -> nhãn; trả hàm lấy nhãn ô kề ±1, ±2."""
    lab = {(p, str(c), int(float(s))): l for p, c, s, l in zip(L.page, L.column, L.syl_idx, L.label) if l}

    def nb(p, c, s):
        s = int(float(s))
        return {lab.get((p, str(c), s + d)) for d in (-2, -1, 1, 2)} - {None, ""}
    return nb


def compare(A: pd.DataFrame, B: pd.DataFrame, prepA: Path, orig: bool, nets, uid, truth_cols=None, max_n=4000, seed=0):
    C = T.lex()
    K = ["page", "column", "syl_idx"]
    for X in (A, B):
        X["key"] = X.page + "/" + X.column.astype(str) + "/" + X.syl_idx.astype(str)
    Ag = A[A.tier == "GOLD"].drop_duplicates("key").set_index("key")
    Bg = B[B.tier == "GOLD"].drop_duplicates("key").set_index("key")
    common = Ag.index.intersection(Bg.index)
    same_lab = Ag.label.reindex(common).values == Bg.label.reindex(common).values
    common = common[same_lab]
    ious = np.array([iou(T8b(a), T8b(b)) if T8b(a) and T8b(b) else 1.0 for a, b in zip(Ag.bbox.reindex(common), Bg.bbox.reindex(common))])
    dif = common[ious < 0.5]
    rng = np.random.default_rng(seed)
    if len(dif) > max_n:
        dif = np.sort(rng.choice(dif, max_n, replace=False))
    nbA, nbB = neighbors(A), neighbors(B)
    kA = dict(zip(A.key, A.index)); kB = dict(zip(B.key, B.index))
    iA = {kA[k] for k in dif}; iB = {kB[k] for k in dif}
    cA = crops_for(A, iA, prepA, orig); cB = crops_for(B, iB, prepA, orig)
    rows = []
    chA, chB = {}, {}
    for k in dif:
        a, b = kA[k], kB[k]
        lab = A.at[a, "label"]
        p, c, s = A.at[a, "page"], A.at[a, "column"], A.at[a, "syl_idx"]
        chA[a] = {lab} | nbA(p, c, s); chB[b] = {lab} | nbB(p, c, s)
    sA = scores(cA, chA, nets, uid); sB = scores(cB, chB, nets, uid)
    for k in dif:
        a, b = kA[k], kB[k]
        lab = A.at[a, "label"]
        da, db = sA.get(a, {}), sB.get(b, {})
        if lab not in da or lab not in db:
            continue
        ma = da[lab] - max([v for cc, v in da.items() if cc != lab] or [-9])
        mb = db[lab] - max([v for cc, v in db.items() if cc != lab] or [-9])
        r = dict(key=k, page=A.at[a, "page"], sA=da[lab], sB=db[lab], mA=ma, mB=mb)
        if truth_cols is not None:
            r.update(okA=truth_cols[0].get(k), okB=truth_cols[1].get(k))
        rows.append(r)
    return pd.DataFrame(rows), int(len(common)), int((ious < 0.5).sum())


def main():
    out = {}
    # ---------------- Borg B18: kiểm trọng tài (mạng LOBO DungLy)
    import tn6lib as TL
    nets_b, uid = nets_tables(["hand_T_DungLy", "hand_L_DungLy"])
    book = "SachKinhThayCaBinh"
    A = T.rd(T.REPO / f"measure_out/_tn6/{book}/legacy/dataset_out/labels_gated.csv")
    B = T.rd(T.REPO / f"measure_out/_tn6/{book}/visual_dp/dataset_out/labels_gated.csv")
    JA, JB = TL.join(book, A), TL.join(book, B)
    okA = {f"{p}/{c}/{s}": (bool(o) if (h and kl in TL.KEEP_OK) else None) for p, c, s, o, h, kl in
           zip(A.page, A.column, A.syl_idx, JA.slot_ok, JA.has_h, JA.keep_level)}
    okB = {f"{p}/{c}/{s}": (bool(o) if (h and kl in TL.KEEP_OK) else None) for p, c, s, o, h, kl in
           zip(B.page, B.column, B.syl_idx, JB.slot_ok, JB.has_h, JB.keep_level)}
    R, n_common, n_dif = compare(A, B, T.REPO / "prepared/_auto" / book, True, nets_b, uid, (okA, okB))
    R = R[R.okA.notna() & R.okB.notna()]
    one = R[R.okA.astype(bool) != R.okB.astype(bool)]
    pickB = one.sB > one.sA
    corr = np.where(one.okB.astype(bool), pickB, ~pickB)
    out["borg_B18_legacy_vs_vdp"] = dict(n_common_gold=n_common, n_iou_lt05=n_dif, n_eval=int(len(R)),
                                         n_one_right=int(len(one)), arbiter_acc=round(float(corr.mean()), 4) if len(one) else None,
                                         frac_vdp_right_when_one=round(float(one.okB.astype(bool).mean()), 4) if len(one) else None,
                                         margin_pos_A=round(float((R.mA > 0).mean()), 4), margin_pos_B=round(float((R.mB > 0).mean()), 4),
                                         slotA=round(float(R.okA.astype(bool).mean()), 4), slotB=round(float(R.okB.astype(bool).mean()), 4))
    print(f"[t09] Borg B18 legacy vs visual_dp: {out['borg_B18_legacy_vs_vdp']}", flush=True)
    # ---------------- STT (mạng sạch: Kinh + DungLy)
    nets_s, uid = nets_tables(["hand_T_Kinh", "hand_L_Kinh", "hand_T_DungLy", "hand_L_DungLy"])
    for read in ("l1skel_l2", "lt1"):
        LA = T.rd(T.OUT / f"stt_builds/{read}__legacy/dataset_out/labels_final.csv")
        LB = T.rd(T.OUT / f"stt_builds/{read}__visual_dp/dataset_out/labels_final.csv")
        geoA = pd.read_pickle(T.OUT / "kimgeo" / f"{read}__legacy.pkl")
        geoB = pd.read_pickle(T.OUT / "kimgeo" / f"{read}__visual_dp.pkl")
        for bk in T.STT:
            A = LA[LA.book == bk].reset_index(drop=True); B = LB[LB.book == bk].reset_index(drop=True)
            R, n_common, n_dif = compare(A, B, T.REPO / "prepared" / T.STT_FULL[bk], False, nets_s, uid, None, max_n=2500)
            gA = geoA[geoA.book == bk]; gB = geoB[geoB.book == bk]
            gA = dict(zip(gA.page + "/" + gA.column.astype(str) + "/" + gA.syl_idx.astype(str), gA.geo_own == gA.ocr_char))
            gB = dict(zip(gB.page + "/" + gB.column.astype(str) + "/" + gB.syl_idx.astype(str), gB.geo_own == gB.ocr_char))
            R["geoA"] = [bool(gA.get(k)) for k in R.key]; R["geoB"] = [bool(gB.get(k)) for k in R.key]
            r = dict(n_common_gold_same_label=n_common, n_iou_lt05=n_dif, n_eval=int(len(R)),
                     vdp_higher_s=round(float((R.sB > R.sA).mean()), 4),
                     margin_pos_legacy=round(float((R.mA > 0).mean()), 4), margin_pos_vdp=round(float((R.mB > 0).mean()), 4),
                     geo_legacy=round(float(R.geoA.mean()), 4), geo_vdp=round(float(R.geoB.mean()), 4),
                     both_signals_vdp=round(float(((R.mB > 0) & ~(R.mA > 0) & R.geoB & ~R.geoA).mean()), 4),
                     both_signals_legacy=round(float(((R.mA > 0) & ~(R.mB > 0) & R.geoA & ~R.geoB).mean()), 4))
            out[f"stt_{read}_{bk}"] = r
            print(f"[t09] STT {read} {bk}: {r}", flush=True)
    T.jdump(out, T.OUT / "box_arbiter.json")


if __name__ == "__main__":
    main()
