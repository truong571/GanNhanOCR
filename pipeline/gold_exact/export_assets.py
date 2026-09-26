"""export_assets.py — dựng models/gold_exact/ từ thư mục thử nghiệm (chạy MỘT lần; 0 API; vài phút CPU).

Tài sản (đều là .pt — .gitignore đã loại *.pt; chỉ MANIFEST.json được git theo dõi):
  vft_model_{T,L}.pt        CNN verifier_ft (chép nguyên r4/verifier_ft/out/model_{T,L}.pt)
  vft_tables_{T,L}.pt       W (nguyên mẫu glyph, f16), P + nh (nguyên mẫu crop NGƯỜI: Borg keep fold 0–3 + IHR sách tune
                            phần A — y hệt s01_features/s03_vft), logistic 'wood' (mu, sd, b) học lại đúng s02_eval
  vft_universe.pt           vũ trụ chữ U + simtop5 (r4/verifier_ft/out)
  hand_model_{T,L}_Kinh.pt  CNN chữ viết tay LOBO-sách (chép nguyên r5/hand_lobo/out/model_{T,L}_Kinh.pt)
  hand_tables_{T,L}_Kinh.pt W, P + nh (Borg Kinh keep fold 0–3 đơn chữ + IHR tune phần A — y hệt h02), logistic 'hand'
                            đầy đủ + ngưỡng lượng tử cross-fit KHÔNG làm tròn (y hệt h03, mọi mức q)
  viss_params.pt            logit có điều kiện 'viss' (w, mu, sd) 3 hướng: tune TK / L16 / IHR2 (học lại đúng r03)
  lexicon.pt                biến thể Unihan (5 trường + kJapanese) + OpenCC t2s (= harness_lib, vlib)
  ta_refs.pt                chữ NGƯỜI của dị bản đã căn theo ô (r5/text_attested/out/ta_cells.csv) + convmap L16/TK
  mocr_refs.pt              chữ dị bản (measure_out/auto_precision/cross) cho KVK/L83 (= m_ocr h01_inputs ref_chars)
Nguyên mẫu người (Borg + IHR) lưu dạng MA TRẬN nhúng P (không chép crop). Mỗi bước tự kiểm tái lập số đã lưu
(p_wood, cert STT, p_kim viss, ref M-OCR); FAIL -> dừng, không ghi MANIFEST.
Chạy: .venv/bin/python -m pipeline.gold_exact.export_assets --sp <scratchpad>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import MODELS, REPO, dir_digest, rd, sha256_file

T0 = time.time()
INV = {}
TUNE = {"T": "TruyenKieu1872", "L": "LucVanTien1916"}
QS = [0.02, 0.01, 0.0075, 0.005, 0.0035, 0.0025, 0.0015, 0.001, 0.0007, 0.0005, 0.00035, 0.00025, 0.00015, 0.0001]


def log(s):
    print(f"[export {time.time() - T0:5.0f}s] {s}", flush=True)


def pages_split(pages):
    ps = sorted(set(pages)); rank = {p: i for i, p in enumerate(ps)}
    return np.array([rank[p] % 4 == 3 for p in pages])


def half_of(keys):
    u = sorted(set(keys)); h = {k: i % 2 for i, k in enumerate(u)}
    return np.array([h[k] for k in keys])


def wprior(y):
    return np.where(y == 1, 1.0, 0.05 / 0.95 * (y == 1).sum() / max((y == 0).sum(), 1))


def lp(m):
    return dict(mu=np.asarray(m.mu, np.float64), sd=np.asarray(m.sd, np.float64), b=np.asarray(m.b, np.float64))


def human_protos(W_dim, U, rows_emb):
    """rows_emb: list (char, emb). Trả P (chuẩn hoá, float32), nh — y hệt vòng acc của s01/h02."""
    uid = {c: i for i, c in enumerate(U)}
    acc = np.zeros((len(U), W_dim), np.float32); nh = np.zeros(len(U), np.int32)
    for c, e in rows_emb:
        k = uid.get(c)
        if k is not None:
            acc[k] += e; nh[k] += 1
    P = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-9)
    return P, nh


def ihr_proto_rows(MI, EI, tune):
    out = []
    for r, row in enumerate(MI.itertuples()):
        if row.book != tune or row.partB:
            continue
        t = row.gt_char if (row.slot_ok == "1" and row.gt_char) else (row.gt_img if row.slot_ok == "0" else "")
        if t:
            out.append((t, EI[r]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp", required=True, help="thư mục thử nghiệm (scratchpad) chứa r4/, r5/, kim_bottleneck/, gold_img_audit/")
    ap.add_argument("--out", default=str(MODELS))
    a = ap.parse_args()
    SP = Path(a.sp); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    VF = SP / "r4/verifier_ft/out"; HL = SP / "r5/hand_lobo/out"; TAO = SP / "r5/text_attested/out"
    RR = SP / "kim_bottleneck/rerank"
    sys.path.insert(0, str(SP / "kim_bottleneck/harness"))
    import harness_lib as HLIB  # chỉ dùng để KIỂM bản chép (var_eq) — không dùng lúc chạy
    from . import common as CM
    from .signals_img import Logit, Universe, X_of, features
    files = {}

    def add(name, source, desc, kind="derived"):
        p = OUT / name
        files[name] = dict(sha256=sha256_file(p), bytes=p.stat().st_size, kind=kind,
                           source=[str(s).replace(str(SP), "<SP>") for s in (source if isinstance(source, list) else [source])],
                           desc=desc)

    def copy(src, name, desc):
        shutil.copyfile(src, OUT / name)
        add(name, src, desc, kind="copy")
        assert files[name]["sha256"] == sha256_file(src)

    # ------------------------------------------------------------------ lexicon
    VAR_FIELDS = ("kTraditionalVariant", "kSimplifiedVariant", "kZVariant", "kSemanticVariant", "kSpecializedSemanticVariant")
    uv = SP / "kim_bottleneck/harness/ext/Unihan_Variants.txt"
    t2s = REPO / "lab/ocr_compare/ket_hop/t2s_map.json"
    V, SIMP, JP = {}, {}, set()
    cp = lambda tok: chr(int(tok.split("<")[0][2:], 16))
    for line in open(uv, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        c0, fld, val = line.rstrip("\n").split("\t")[:3]
        if fld in VAR_FIELDS:
            x = cp(c0)
            for t in val.split():
                y = cp(t)
                if x == y:
                    continue
                V.setdefault(x, set()).add(y); V.setdefault(y, set()).add(x)
                if fld == "kSimplifiedVariant":
                    SIMP.setdefault(x, y)
        if fld in ("kJapaneseOldVariant", "kJapaneseVariant"):
            x = cp(c0)
            for t in val.split():
                y = cp(t); JP.add((x, y)); JP.add((y, x))
    for x, y in json.loads(t2s.read_text(encoding="utf-8")).items():
        if len(x) == 1 and len(y) == 1 and x != y:
            SIMP.setdefault(x, y); V.setdefault(x, set()).add(y); V.setdefault(y, set()).add(x)
    torch.save(dict(V={k: sorted(v) for k, v in V.items()}, SIMP=SIMP, JP=sorted(JP)), OUT / "lexicon.pt")
    add("lexicon.pt", [uv, t2s], "biến thể Unihan (5 trường, cạnh trực tiếp) + OpenCC t2s + kJapanese(Old)Variant (V1+)")
    CM._LEX.clear(); CM._LEX.update(V=V, SIMP=SIMP, JP=JP)
    keys = list(V)[:20000]
    INV["lexicon_var_eq_harness"] = all(CM.variants_of(c) == HLIB.variants_of(c) and CM.simp(c) == HLIB.simp(c) for c in keys)
    log(f"lexicon {len(V)} chữ; khớp harness_lib: {INV['lexicon_var_eq_harness']}")

    # ------------------------------------------------------------------ verifier_ft (T, L)
    U = json.load(open(VF / "glyph_chars.json", encoding="utf-8")); SIM5 = json.load(open(VF / "simtop5.json", encoding="utf-8"))
    torch.save(dict(U=U, SIM5=SIM5), OUT / "vft_universe.pt")
    add("vft_universe.pt", [VF / "glyph_chars.json", VF / "simtop5.json"], "vũ trụ chữ ứng viên U + top-5 gần hình (verifier_ft)")
    uni = Universe(U, SIM5)
    MB = pd.read_pickle(VF / "meta_borg.pkl"); MI = pd.read_pickle(VF / "meta_ihr.pkl"); MI["partB"] = pages_split(MI.page.values)
    MG = pd.read_pickle(VF / "meta_gold.pkl")
    GS = pd.read_csv(SP / "r4/verifier_ft/gold_scores.csv", usecols=["cell_uid", "p_wood_T", "p_wood_L"])
    MI["cand"] = np.where(MI.label != "", MI.label, MI.ocr_char)
    y_both = np.array([HLIB.var_eq(x, g) for x, g in zip(MI.cand, MI.gt_char)]) & (MI.slot_ok == "1").values
    slot_det = MI.slot_ok.isin(["0", "1"]).values
    rng = np.random.default_rng(0)
    samp = np.sort(rng.choice(len(MG), 6000, replace=False))
    for t in "TL":
        copy(VF / f"model_{t}.pt", f"vft_model_{t}.pt", f"CNN verifier_ft mô hình {t} (tune {TUNE[t]})")
        W = np.load(VF / f"W_{t}.f16.npy")
        EB = np.load(VF / f"emb_{t}_borg.f16.npy").astype(np.float32); EI = np.load(VF / f"emb_{t}_ihr.f16.npy").astype(np.float32)
        rows = [(MB.char.iat[r], EB[r]) for r in np.nonzero((MB.keep.values == 1) & (MB.fold.values != 4))[0]]
        rows += ihr_proto_rows(MI, EI, TUNE[t])
        P, nh = human_protos(W.shape[1], U, rows)
        FIo = pd.read_pickle(VF / f"feat_{t}_ihr.pkl"); FGo = pd.read_pickle(VF / f"feat_{t}_gold.pkl")
        cal = ((MI.book == TUNE[t]) & MI.partB & slot_det & (MI.gt_char != "") & (MI.cand.str.len() == 1)).values
        m = Logit().fit(X_of(FIo)[cal], y_both[cal].astype(float))
        pg = np.round(m.p(X_of(FGo)), 5)
        INV[f"vft_{t}_p_wood_reproduce_maxabs"] = float(np.abs(pg - GS[f"p_wood_{t}"].values).max())
        EGo = np.load(VF / f"emb_{t}_gold.f16.npy").astype(np.float32)
        Fs = features(uni, EGo, W.astype(np.float32), P, nh, samp, MG.label.values[samp], MG.syllable.values[samp],
                      MG.nb_prev.values[samp], MG.nb_next.values[samp])
        d = {c: float(np.nanmax(np.abs(Fs[c].values - FGo[c].values[samp]))) for c in ("sw", "mG", "mRSN", "se", "meRSN")}
        INV[f"vft_{t}_features_port_maxabs_6000"] = d
        torch.save(dict(W=W, P=P.astype(np.float32), nh=nh, wood=lp(m), tune=TUNE[t]), OUT / f"vft_tables_{t}.pt")
        add(f"vft_tables_{t}.pt", [VF / f"W_{t}.f16.npy", VF / f"emb_{t}_borg.f16.npy", VF / f"emb_{t}_ihr.f16.npy",
                                   VF / "meta_borg.pkl", VF / "meta_ihr.pkl", VF / f"feat_{t}_ihr.pkl"],
            f"verifier_ft {t}: W glyph (f16), nguyên mẫu người P/nh (Borg keep fold0-3 + {TUNE[t]} phần A), logistic wood")
        log(f"vft {t}: p_wood tái lập maxabs {INV[f'vft_{t}_p_wood_reproduce_maxabs']}, port đặc trưng {d}")
        del EB, EI, EGo

    # ------------------------------------------------------------------ hand LOBO (Kinh)
    MA = pd.read_pickle(HL / "meta_all.pkl"); MA["pkey"] = MA.book + "/" + MA.page
    SC = pd.read_pickle(HL / "stt_cert_Kinh.pkl")
    single = MA.single.values == 1
    inb = MA.book.isin({"SachKinhThayCaBinh"}).values
    for t in "TL":
        FT = f"{t}_Kinh"
        copy(HL / f"model_{FT}.pt", f"hand_model_{FT}.pt", f"CNN chữ viết tay LOBO-sách {FT} (chỉ học Borg Kinh + IHR tune)")
        W = np.load(HL / f"W_{FT}.f16.npy")
        EA = np.load(HL / f"emb_{FT}_all.f16.npy").astype(np.float32); EI = np.load(HL / f"emb_{FT}_ihr.f16.npy").astype(np.float32)
        rows = [(MA.char.iat[r], EA[r]) for r in np.nonzero(inb & (MA.keep.values == 1) & (MA.fold.values != 4) & single)[0]]
        rows += ihr_proto_rows(MI, EI, TUNE[t])
        P, nh = human_protos(W.shape[1], U, rows)
        Fc = pd.read_pickle(HL / f"feat_{t}_Kinh_cal.pkl")
        keep_c = Fc.in_univ.values == 1; yc = (Fc.kind.values == "pos").astype(float)
        chalf = half_of(MA.pkey.values[Fc.row.values])
        Xc = X_of(Fc); w = wprior(yc); pcf = np.zeros(len(Xc))
        for h in (0, 1):
            tr = (chalf != h) & keep_c
            mh = Logit(clip=True).fit(Xc[tr], yc[tr], w=w[tr]); pcf[chalf == h] = mh.p(Xc[chalf == h])
        mfull = Logit(clip=True).fit(Xc[keep_c], yc[keep_c], w=w[keep_c])
        negc = (yc == 0) & keep_c
        thr = {str(q): float(np.quantile(pcf[negc], 1 - q)) for q in QS}
        Fs = pd.read_pickle(HL / f"feat_{t}_Kinh_stt.pkl")
        ps = mfull.p(X_of(Fs))
        INV[f"hand_{t}_p_reproduce_maxabs"] = float(np.abs(ps - SC[f"p_{t}"].values).max())
        INV[f"hand_{t}_thr_00015"] = thr["0.00015"]
        torch.save(dict(W=W, P=P.astype(np.float32), nh=nh, hand=lp(mfull), thr=thr, q_list=QS), OUT / f"hand_tables_{FT}.pt")
        add(f"hand_tables_{FT}.pt", [HL / f"W_{FT}.f16.npy", HL / f"emb_{FT}_all.f16.npy", HL / f"emb_{FT}_ihr.f16.npy",
                                    HL / "meta_all.pkl", HL / f"feat_{t}_Kinh_cal.pkl"],
            f"hand {FT}: W, nguyên mẫu người P/nh (Borg Kinh keep fold0-3 đơn chữ + {TUNE[t]} phần A), logistic hand, ngưỡng q")
        log(f"hand {t}: p tái lập maxabs {INV[f'hand_{t}_p_reproduce_maxabs']}, t(0,00015) = {thr['0.00015']:.6f}")
        del EA, EI
    # cert tái lập
    tbT = torch.load(OUT / "hand_tables_T_Kinh.pt", weights_only=False); tbL = torch.load(OUT / "hand_tables_L_Kinh.pt", weights_only=False)
    cert = (SC.p_T.values >= tbT["thr"]["0.00015"]) & (SC.p_L.values >= tbL["thr"]["0.00015"])
    INV["hand_cert_00015_reproduce"] = float((cert.astype(int) == SC["cert_0.00015"].values).mean())

    # ------------------------------------------------------------------ viss (r03 học lại, 3 hướng)
    sys.path.insert(0, str(RR))
    import r03_model as R3
    d = pd.read_csv(SP / "kim_bottleneck/harness/cells_eval.csv", dtype=str, keep_default_na=False)
    EAv = np.load(RR / "out/E_A.f16.npy").astype(np.float32)
    cand = pd.read_pickle(RR / "out/cand.pkl")
    cs_ = pd.read_pickle(RR / "out/cand_self.pkl"); cand["s_self"] = cs_.s_self.values; cand["n_self"] = cs_.n_self.values
    c = R3.build(cand, d, EAv)
    c["book"] = d.book.values[c.i.values]
    from .signals_img import VISS_FEATS, viss_features, viss_pkim, viss_predict
    assert list(R3.VARIANTS["viss"]) == VISS_FEATS
    # kiểm bản chép viss_features trên cùng hàng ứng viên
    raw = cand[cand.diag == 0][["i", "ch", "inR", "is_kim", "s_font", "s_faug", "s_fontB", "s_faugB", "s_oth", "n_oth", "s_head",
                               "gmd5", "s_self", "n_self"]].reset_index(drop=True)
    mine = viss_features(raw)
    INV["viss_features_port_maxabs"] = {f: float(np.nanmax(np.abs(mine[f].values.astype(float) - c[f].values.astype(float))))
                                        for f in VISS_FEATS}
    params = {}
    kmd5 = c[c.is_kim == 1].set_index("i").gmd5.to_dict()
    for tune, test, tag in (("TruyenKieu1872", "LucVanTien1916", "Tru2Luc"), ("LucVanTien1916", "TruyenKieu1872", "Luc2Tru"),
                            ("IHR2", "LITHO", "LITHO")):
        ctr = c[c.book.isin(["LucVanTien1916", "TruyenKieu1872"])] if tune == "IHR2" else c[c.book == tune]
        cte = c[c.book.isin(["LucVanTien1883", "KimVanKieu1884"])] if tune == "IHR2" else c[c.book == test]
        m = R3.fit(ctr, VISS_FEATS)
        prm = dict(w=m["w"].numpy().copy(), mu=m["mu"].numpy().copy(), sd=m["sd"].numpy().copy(), feats=list(VISS_FEATS),
                   n_cells=int(m["n_cells"]))
        params[tune] = prm
        p = viss_predict(prm, cte)
        pk = viss_pkim(cte, p, kmd5)
        ref = pd.read_pickle(RR / f"out/pred_viss_{tag}.pkl")
        ref = ref[ref.role == "test"].set_index("i").p_kim
        INV[f"viss_{tag}_pkim_reproduce_maxabs"] = float(np.abs(pk.reindex(ref.index).values - ref.values).max())
        log(f"viss {tag}: p_kim tái lập maxabs {INV[f'viss_{tag}_pkim_reproduce_maxabs']:.2e}")
    torch.save(params, OUT / "viss_params.pt")
    add("viss_params.pt", [RR / "out/cand.pkl", RR / "out/cand_self.pkl", SP / "kim_bottleneck/harness/cells_eval.csv"],
        "logit có điều kiện 'viss' (r03_model.fit) — tune TruyenKieu1872 (chấm L16), LucVanTien1916 (chấm TK), IHR2 (chấm L83/KVK)")
    del cand, c, raw, mine

    # ------------------------------------------------------------------ text_attested
    T = rd(TAO / "ta_cells.csv")
    T = T[T.book.isin(["KimVanKieu1884", "LucVanTien1883", "TruyenKieu1872"])]
    conv = {tg: json.load(open(TAO / f"convmap_{tg}.json", encoding="utf-8")) for tg in ("L16", "TK")}
    torch.save(dict(cell_uid=T.cell_uid.tolist(), book=T.book.tolist(), syllable=T.syllable.tolist(), ref71=T.ref71.tolist(),
                    ref72=T.ref72.tolist(), ref16=T.ref16.tolist(), rev_h71=T.rev_h71.tolist(),
                    has_gt=(T.gt_char != "").astype(int).tolist(), convmap=conv),
               OUT / "ta_refs.pt")
    add("ta_refs.pt", [TAO / "ta_cells.csv", TAO / "convmap_L16.json", TAO / "convmap_TK.json"],
        "chữ người dị bản đã căn theo ô: KVK←Kiều1871 LVD/1872 DMT, L83←LVT1916 NF, TK←Kiều1871 LVD; TK: có GT người IHR")

    # ------------------------------------------------------------------ M-OCR refs (KVK/L83)
    L = rd(REPO / "dataset/_ALL/labels.csv"); Lg = L[L.tier == "GOLD"]
    refs = {}
    for b in ("KimVanKieu1884", "LucVanTien1883"):
        cc = rd(REPO / f"measure_out/auto_precision/cross/{b}/cells.csv")
        cc = cc[cc.image != ""]
        for im, ref, rn, pua in zip(cc.image, cc.ref, cc.ref_nom, cc.ref_pua):
            refs.setdefault(f"crops/{b}/{im}", {})[ref] = (rn, int(pua))
    sub = Lg[Lg.book_set.isin(["KimVanKieu1884", "LucVanTien1883"])]
    rc = {u: sorted({v[0] for v in refs.get(im, {}).values() if len(v[0]) == 1}) for u, im in zip(sub.cell_uid, sub.image)}
    rc = {u: v for u, v in rc.items() if v}
    img_of = dict(zip(sub.cell_uid, sub.image))
    MO = pd.read_pickle(SP / "gold_img_audit/m_ocr/out/inputs.pkl")
    MO = MO[MO.book_set.isin(["KimVanKieu1884", "LucVanTien1883"])]
    INV["mocr_refs_eq_m_ocr_inputs"] = bool(all(sorted(r) == rc.get(u, []) for u, r in zip(MO.cell_uid, MO.ref_chars)))
    torch.save(dict(refs=rc, image={u: img_of[u] for u in rc}), OUT / "mocr_refs.pt")
    add("mocr_refs.pt", [REPO / "measure_out/auto_precision/cross"], "chữ dị bản cho đối thủ M-OCR (KVK: Kiều1871/1872, L83: LVT1916 NF)")
    log(f"ta_refs {len(T)} ô; mocr_refs {len(rc)} ô (khớp m_ocr: {INV['mocr_refs_eq_m_ocr_inputs']})")

    # ------------------------------------------------------------------ tệp ngoài (repo, không chép) + MANIFEST
    ext = {}
    for rel, desc in (("nom-embed/best.pt", "encoder v1 (ResNet18 ArcFace 1.591 lớp) — score_visual/viss/M-OCR"),
                      ("ArcFace/checkpoints/best.pt", "encoder v2 (ArcFace 1.564 lớp, page-disjoint)"),
                      ("fonts/NomNaTong-Regular.ttf", "font glyph tham chiếu 1"), ("fonts/PlangothicP1-Regular.ttf", "font glyph 2"),
                      ("fonts/PlangothicP2-Regular.ttf", "font glyph 3")):
        p = REPO / rel
        ext[rel] = dict(sha256=sha256_file(p), bytes=p.stat().st_size, kind="external_file", desc=desc)
    for rel in ("ArcFace/data/glyphs", "gannhanocr-fd"):
        ext[rel] = dict(digest=dir_digest(REPO / rel), kind="dir_digest", pattern="U+*.png", min_bytes=1024,
                        desc="ảnh glyph FD (score_visual s_fd); digest = sha256(danh sách (đường dẫn, kích thước))")
    bad = [k for k, v in INV.items() if (isinstance(v, bool) and not v)]
    bad += [k for k, v in INV.items() if k.endswith("reproduce_maxabs") and v > 1e-4]
    bad += [k for k, v in INV.items() if k == "hand_cert_00015_reproduce" and v < 0.9995]
    bad += [k for k, v in INV.items() if k.endswith("port_maxabs_6000") or k == "viss_features_port_maxabs"
            if max(v.values()) > 1e-5]
    man = dict(version="2026-09-27", created=time.strftime("%Y-%m-%d %H:%M"),
               note="Tài sản gold_exact. Tệp .pt không vào git (.gitignore *.pt); dựng lại: python -m pipeline.gold_exact.export_assets --sp <SP>",
               sp="<SP> = thư mục thử nghiệm phiên 2026-09-25..27 (scratchpad)", files=files, external=ext, invariants=INV,
               invariants_fail=bad)
    if bad:
        print(json.dumps(INV, ensure_ascii=False, indent=1, default=float))
        raise SystemExit(f"FAIL tái lập: {bad} — KHÔNG ghi MANIFEST")
    json.dump(man, open(OUT / "MANIFEST.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    log(f"MANIFEST: {len(files)} tài sản + {len(ext)} tệp ngoài; invariants PASS")
    print(json.dumps(INV, ensure_ascii=False, default=float)[:3000])


if __name__ == "__main__":
    main()
