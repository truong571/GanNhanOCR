"""export_assets.py — dựng models/gold_exact/ từ thư mục thử nghiệm (chạy MỘT lần; 0 API; vài phút CPU).

Tài sản (đều là .pt — .gitignore đã loại *.pt; chỉ MANIFEST.json được git theo dõi):
  vft_model_{T,L}.pt        CNN verifier_ft (chép nguyên r4/verifier_ft/out/model_{T,L}.pt)
  vft_tables_{T,L}.pt       W (nguyên mẫu glyph, f16), P + nh (nguyên mẫu crop NGƯỜI: Borg keep fold 0–3 + IHR sách tune
                            phần A — y hệt s01_features/s03_vft), logistic 'wood' (mu, sd, b) học lại đúng s02_eval
  vft_universe.pt           vũ trụ chữ U + simtop5 (r4/verifier_ft/out)
  hand_model_{T,L}_Kinh.pt  CNN chữ viết tay LOBO-sách (chép nguyên r5/hand_lobo/out/model_{T,L}_Kinh.pt)
  hand_tables_{T,L}_Kinh.pt W, P + nh (Borg Kinh keep fold 0–3 đơn chữ + IHR tune phần A — y hệt h02), logistic 'hand'
                            đầy đủ + ngưỡng lượng tử cross-fit KHÔNG làm tròn (y hệt h03, mọi mức q)
  hand_{model,tables}_{T,L}_DungLy.pt  (27/09) cùng công thức, học/nguyên mẫu/hiệu chuẩn CHỈ Borg DungLy — chấm Borg Kinh (B18)
                            theo LOBO-sách; dừng nếu nguyên mẫu/hiệu chuẩn lấy từ sách khác sách học (INV *_proto_books/_cal_books)
  viss_params.pt            logit có điều kiện 'viss' (w, mu, sd) 3 hướng: tune TK / L16 / IHR2 (học lại đúng r03)
  lexicon.pt                biến thể Unihan (5 trường + kJapanese) + OpenCC t2s (= harness_lib, vlib)
  ta_refs.pt                chữ NGƯỜI của dị bản đã căn theo ô (r5/text_attested/out/ta_cells.csv) + convmap L16/TK
  mocr_refs.pt              chữ dị bản (measure_out/auto_precision/cross) cho KVK/L83 (= m_ocr h01_inputs ref_chars)
Nguyên mẫu người (Borg + IHR) lưu dạng MA TRẬN nhúng P (không chép crop). Mỗi bước tự kiểm tái lập số đã lưu
(p_wood, cert STT, p_kim viss, ref M-OCR); FAIL -> dừng, không ghi MANIFEST.

NGUỒN BỀN (28/09, N1): mọi tệp nguồn cần để dựng lại nằm ở kho `measure_out/_gold_exact_assets_src/` (gitignored, ≈ 1,3 GB,
bố cục y như thư mục thử nghiệm <SP>: kim_bottleneck/, r4/, r5/, gold_img_audit/ + measure_out_auto_precision_cross/) kèm
`SHA256SUMS` + `SOURCE.json`. Kho được kiểm sha256 TỪNG tệp trước khi dựng (lệch/thiếu -> dừng). Sao lưu: chép nguyên thư mục
kho sang đĩa ngoài / kho lưu trữ riêng (KHÔNG đẩy lên mạng công khai: có mô hình + nhãn người IHR); khôi phục = chép về đúng
đường dẫn rồi chạy lệnh dưới (tự kiểm SHA256SUMS).
  .venv/bin/python -m pipeline.gold_exact.export_assets                      # dựng từ kho (mặc định) -> models/gold_exact
  .venv/bin/python -m pipeline.gold_exact.export_assets --snapshot-from <SP>  # (một lần) chép nguồn từ thư mục thử nghiệm vào kho
  .venv/bin/python -m pipeline.gold_exact.export_assets --out <tmp> --check-against models/gold_exact   # dựng thử + so sha
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
SRC_DEFAULT = REPO / "measure_out" / "_gold_exact_assets_src"
# Tệp nguồn (đường tương đối <SP>/kho) mà main() đọc — kho bền chép đúng danh sách này (+ mã nguồn tạo ra chúng để tra cứu).
_VF = "r4/verifier_ft/out/"
_HL = "r5/hand_lobo/out/"
_RR = "kim_bottleneck/rerank/"
SOURCE_FILES = (
    ["kim_bottleneck/harness/ext/Unihan_Variants.txt", "kim_bottleneck/harness/harness_lib.py",
     "kim_bottleneck/harness/cells_eval.csv", "kim_bottleneck/harness/h01_build_cells.py",
     _RR + "r03_model.py", _RR + "out/E_A.f16.npy", _RR + "out/cand.pkl", _RR + "out/cand_self.pkl"]
    + [_RR + f"out/pred_viss_{t}.pkl" for t in ("Tru2Luc", "Luc2Tru", "LITHO")]
    + ["r4/verifier_ft/gold_scores.csv"]
    + [_VF + f for f in ("glyph_chars.json", "simtop5.json", "meta_borg.pkl", "meta_ihr.pkl", "meta_gold.pkl")]
    + [_VF + f.format(t=t) for t in "TL" for f in ("model_{t}.pt", "W_{t}.f16.npy", "emb_{t}_borg.f16.npy", "emb_{t}_ihr.f16.npy",
                                                    "emb_{t}_gold.f16.npy", "feat_{t}_ihr.pkl", "feat_{t}_gold.pkl")]
    + [_HL + "meta_all.pkl"] + [_HL + f"stt_cert_{bk}.pkl" for bk in ("Kinh", "DungLy")]
    + [_HL + f.format(t=t, bk=bk) for bk in ("Kinh", "DungLy") for t in "TL"
       for f in ("model_{t}_{bk}.pt", "W_{t}_{bk}.f16.npy", "emb_{t}_{bk}_all.f16.npy", "emb_{t}_{bk}_ihr.f16.npy",
                 "feat_{t}_{bk}_cal.pkl", "feat_{t}_{bk}_stt.pkl")]
    + ["r5/text_attested/out/ta_cells.csv", "r5/text_attested/out/convmap_L16.json", "r5/text_attested/out/convmap_TK.json",
       "r5/text_attested/out/t01_build.json", "r5/text_attested/t01_build.py", "r5/text_attested/t015_convmap.py",
       "r5/text_attested/ta_lib.py", "gold_img_audit/m_ocr/out/inputs.pkl"])
# Nguồn nằm trong repo nhưng ở measure_out/ (không commit, bị bộ đo sinh lại) -> chép vào kho để tài sản không trôi theo.
REPO_SOURCES = {f"measure_out_auto_precision_cross/{b}/cells.csv": REPO / f"measure_out/auto_precision/cross/{b}/cells.csv"
                for b in ("KimVanKieu1884", "LucVanTien1883")}
BACKUP_NOTE = ("Kho nguồn bền của tài sản gold_exact (không vào git). Sao lưu: chép NGUYÊN thư mục này (kèm SHA256SUMS) sang đĩa "
               "ngoài / kho lưu trữ riêng; KHÔNG đẩy lên mạng công khai (có mô hình + nhãn người IHR-NomDB). Khôi phục: chép về "
               "measure_out/_gold_exact_assets_src/ rồi `.venv/bin/python -m pipeline.gold_exact.export_assets` (tự kiểm SHA256SUMS).")


class StoreError(RuntimeError):
    pass


def snapshot(sp: Path, store: Path) -> dict:
    """Chép SOURCE_FILES từ thư mục thử nghiệm <sp> (+ REPO_SOURCES từ repo) vào kho bền; ghi SHA256SUMS + SOURCE.json."""
    store.mkdir(parents=True, exist_ok=True)
    miss = [r for r in SOURCE_FILES if not (sp / r).is_file()] + [k for k, v in REPO_SOURCES.items() if not v.is_file()]
    if miss:
        raise StoreError(f"thiếu {len(miss)} tệp nguồn trong {sp}: {miss[:5]}")
    rows, nbytes = [], 0
    for rel, src in [(r, sp / r) for r in SOURCE_FILES] + list(REPO_SOURCES.items()):
        dst = store / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not (dst.exists() and dst.stat().st_size == src.stat().st_size and sha256_file(dst) == sha256_file(src)):
            shutil.copy2(src, dst)
        h = sha256_file(dst)
        if h != sha256_file(src):
            raise StoreError(f"chép lệch sha: {rel}")
        rows.append(f"{h}  {rel}"); nbytes += dst.stat().st_size
    (store / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M"), origin=str(sp), n_files=len(rows), bytes=nbytes,
                sha256sums_sha256=sha256_file(store / "SHA256SUMS"), backup=BACKUP_NOTE)
    (store / "SOURCE.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def verify_store(store: Path) -> dict:
    """Kiểm sha256 TỪNG tệp trong SHA256SUMS của kho + đủ mọi tệp main() cần. Lệch/thiếu -> StoreError (trước khi dựng)."""
    f = store / "SHA256SUMS"
    if not f.exists():
        raise StoreError(f"{store} không có SHA256SUMS — chép nguồn trước: export_assets --snapshot-from <SP>")
    want = {}
    for ln in f.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            h, rel = ln.split("  ", 1)
            want[rel] = h
    need = set(SOURCE_FILES) | set(REPO_SOURCES)
    bad = sorted(need - set(want))
    nbytes = 0
    for rel, h in want.items():
        p = store / rel
        if not p.is_file():
            bad.append(f"thiếu {rel}"); continue
        nbytes += p.stat().st_size
        if sha256_file(p) != h:
            bad.append(f"sha lệch {rel}")
    if bad:
        raise StoreError(f"kho nguồn {store} hỏng ({len(bad)}): {bad[:5]}")
    return dict(path=str(store.relative_to(REPO)) if REPO in store.parents else str(store), n_files=len(want), bytes=nbytes,
                sha256sums_sha256=sha256_file(f), backup=BACKUP_NOTE)
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
    ap.add_argument("--src", default=str(SRC_DEFAULT),
                    help="kho nguồn bền (bố cục như thư mục thử nghiệm: r4/, r5/, kim_bottleneck/, gold_img_audit/ + SHA256SUMS)")
    ap.add_argument("--sp", default=None, help="(cũ) = --src; đọc thẳng thư mục thử nghiệm, KHÔNG kiểm SHA256SUMS")
    ap.add_argument("--snapshot-from", default=None, help="chép nguồn từ thư mục thử nghiệm này vào --src (+ SHA256SUMS) rồi dừng")
    ap.add_argument("--out", default=str(MODELS))
    ap.add_argument("--check-against", default=None,
                    help="so sha256 mọi tài sản vừa dựng với MANIFEST.json của thư mục này (khác -> liệt kê, mã thoát 1)")
    a = ap.parse_args()
    if a.snapshot_from:
        meta = snapshot(Path(a.snapshot_from), Path(a.src))
        log(f"kho nguồn {a.src}: {meta['n_files']} tệp, {meta['bytes'] / 1e9:.2f} GB, SHA256SUMS {meta['sha256sums_sha256'][:12]}")
        return
    if a.sp:
        SP = Path(a.sp); store = None
    else:
        SP = Path(a.src); store = verify_store(SP)
        log(f"kho nguồn {SP}: {store['n_files']} tệp sha256 khớp SHA256SUMS")
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    VF = SP / "r4/verifier_ft/out"; HL = SP / "r5/hand_lobo/out"; TAO = SP / "r5/text_attested/out"
    RR = SP / "kim_bottleneck/rerank"
    sys.path.insert(0, str(SP / "kim_bottleneck/harness"))
    import harness_lib as HLIB  # chỉ dùng để KIỂM bản chép (var_eq) — không dùng lúc chạy
    from . import common as CM
    from .signals_img import Logit, Universe, X_of, features
    files = {}

    src_tag = "<SP>" if store is None else "<SRC>"

    def add(name, source, desc, kind="derived"):
        p = OUT / name
        files[name] = dict(sha256=sha256_file(p), bytes=p.stat().st_size, kind=kind,
                           source=[str(s).replace(str(SP), src_tag).replace(str(REPO) + "/", "")
                                   for s in (source if isinstance(source, list) else [source])],
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

    # ------------------------------------------------------------------ hand LOBO (Kinh; DungLy từ 27/09 cho Borg LOBO-sách)
    # Kinh  : học + nguyên mẫu + hiệu chuẩn CHỈ Borg Kinh (SachKinhThayCaBinh) -> chấm STT và SachDungLyHoThan.
    # DungLy: học + nguyên mẫu + hiệu chuẩn CHỈ Borg DungLy (SachDungLyHoThan) -> chấm SachKinhThayCaBinh (LOBO theo sách).
    MA = pd.read_pickle(HL / "meta_all.pkl"); MA["pkey"] = MA.book + "/" + MA.page
    single = MA.single.values == 1
    HAND_BOOK = {"Kinh": "SachKinhThayCaBinh", "DungLy": "SachDungLyHoThan"}
    for bk_, t in [(b, t) for b in ("Kinh", "DungLy") for t in "TL"]:
        SC = pd.read_pickle(HL / f"stt_cert_{bk_}.pkl")
        inb = MA.book.isin({HAND_BOOK[bk_]}).values
        FT = f"{t}_{bk_}"
        copy(HL / f"model_{FT}.pt", f"hand_model_{FT}.pt", f"CNN chữ viết tay LOBO-sách {FT} (chỉ học Borg {bk_} + IHR tune)")
        W = np.load(HL / f"W_{FT}.f16.npy")
        EA = np.load(HL / f"emb_{FT}_all.f16.npy").astype(np.float32); EI = np.load(HL / f"emb_{FT}_ihr.f16.npy").astype(np.float32)
        rows = [(MA.char.iat[r], EA[r]) for r in np.nonzero(inb & (MA.keep.values == 1) & (MA.fold.values != 4) & single)[0]]
        rows += ihr_proto_rows(MI, EI, TUNE[t])
        P, nh = human_protos(W.shape[1], U, rows)
        used = sorted(set(MA.book.values[inb & (MA.keep.values == 1) & (MA.fold.values != 4) & single]))
        INV[f"hand_{FT}_proto_books"] = used
        if used != [HAND_BOOK[bk_]]:
            raise SystemExit(f"LOBO hỏng: nguyên mẫu hand_{FT} lấy từ {used} ≠ [{HAND_BOOK[bk_]}]")
        Fc = pd.read_pickle(HL / f"feat_{t}_{bk_}_cal.pkl")
        cal_books = sorted(set(MA.book.values[Fc.row.values]))
        INV[f"hand_{FT}_cal_books"] = cal_books
        if cal_books != [HAND_BOOK[bk_]]:
            raise SystemExit(f"LOBO hỏng: hiệu chuẩn hand_{FT} lấy từ {cal_books} ≠ [{HAND_BOOK[bk_]}]")
        keep_c = Fc.in_univ.values == 1; yc = (Fc.kind.values == "pos").astype(float)
        chalf = half_of(MA.pkey.values[Fc.row.values])
        Xc = X_of(Fc); w = wprior(yc); pcf = np.zeros(len(Xc))
        for h in (0, 1):
            tr = (chalf != h) & keep_c
            mh = Logit(clip=True).fit(Xc[tr], yc[tr], w=w[tr]); pcf[chalf == h] = mh.p(Xc[chalf == h])
        mfull = Logit(clip=True).fit(Xc[keep_c], yc[keep_c], w=w[keep_c])
        negc = (yc == 0) & keep_c
        thr = {str(q): float(np.quantile(pcf[negc], 1 - q)) for q in QS}
        Fs = pd.read_pickle(HL / f"feat_{t}_{bk_}_stt.pkl")
        ps = mfull.p(X_of(Fs))
        sfx = "" if bk_ == "Kinh" else f"_{bk_}"          # khoá INV của Kinh giữ tên cũ
        INV[f"hand_{t}{sfx}_p_reproduce_maxabs"] = float(np.abs(ps - SC[f"p_{t}"].values).max())
        INV[f"hand_{t}{sfx}_thr_00015"] = thr["0.00015"]
        torch.save(dict(W=W, P=P.astype(np.float32), nh=nh, hand=lp(mfull), thr=thr, q_list=QS), OUT / f"hand_tables_{FT}.pt")
        add(f"hand_tables_{FT}.pt", [HL / f"W_{FT}.f16.npy", HL / f"emb_{FT}_all.f16.npy", HL / f"emb_{FT}_ihr.f16.npy",
                                    HL / "meta_all.pkl", HL / f"feat_{t}_{bk_}_cal.pkl"],
            f"hand {FT}: W, nguyên mẫu người P/nh (Borg {bk_} keep fold0-3 đơn chữ + {TUNE[t]} phần A), logistic hand, ngưỡng q")
        log(f"hand {FT}: p tái lập maxabs {INV[f'hand_{t}{sfx}_p_reproduce_maxabs']}, t(0,00015) = {thr['0.00015']:.6f}")
        del EA, EI
    # cert tái lập (mỗi biến thể so với stt_cert_<bk>.pkl của h03)
    for bk_ in ("Kinh", "DungLy"):
        SC = pd.read_pickle(HL / f"stt_cert_{bk_}.pkl")
        tbT = torch.load(OUT / f"hand_tables_T_{bk_}.pt", weights_only=False)
        tbL = torch.load(OUT / f"hand_tables_L_{bk_}.pt", weights_only=False)
        cert = (SC.p_T.values >= tbT["thr"]["0.00015"]) & (SC.p_L.values >= tbL["thr"]["0.00015"])
        INV["hand_cert_00015_reproduce" + ("" if bk_ == "Kinh" else f"_{bk_}")] = float(
            (cert.astype(int) == SC["cert_0.00015"].values).mean())

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
    # N6 (28/09): KHÔNG lưu cột GT người IHR (gt_char/has_gt) — quyết định TA chỉ được phụ thuộc văn bản người của DỊ BẢN;
    # GT IHR chỉ dùng trong ĐO (eval_ihr đọc manifest riêng). Căn TK↔Kiều 1871 (t01_build.cells_verses) dùng trang/cột/syl_idx/âm
    # của bộ dữ liệu, không dùng GT -> rev_h71 có cho mọi ô TK căn được, kể cả trang không có GT.
    torch.save(dict(cell_uid=T.cell_uid.tolist(), book=T.book.tolist(), syllable=T.syllable.tolist(), ref71=T.ref71.tolist(),
                    ref72=T.ref72.tolist(), ref16=T.ref16.tolist(), rev_h71=T.rev_h71.tolist(), convmap=conv),
               OUT / "ta_refs.pt")
    add("ta_refs.pt", [TAO / "ta_cells.csv", TAO / "convmap_L16.json", TAO / "convmap_TK.json"],
        "chữ người dị bản đã căn theo ô: KVK←Kiều1871 LVD/1872 DMT, L83←LVT1916 NF, TK←Kiều1871 LVD (mọi ô căn được; "
        "KHÔNG chứa GT người IHR)")
    tk = T.book == "TruyenKieu1872"
    INV["ta_refs_no_ihr_gt"] = True
    INV["ta_refs_tk_rev_h71_frac"] = round(float((T.rev_h71[tk] != "").mean()), 5)

    # ------------------------------------------------------------------ M-OCR refs (KVK/L83)
    L = rd(REPO / "dataset/_ALL/labels.csv"); Lg = L[L.tier == "GOLD"]
    refs = {}
    XR = (SP / "measure_out_auto_precision_cross") if store is not None else (REPO / "measure_out/auto_precision/cross")
    for b in ("KimVanKieu1884", "LucVanTien1883"):
        cc = rd(XR / f"{b}/cells.csv")
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
    add("mocr_refs.pt", [XR / f"{b}/cells.csv" for b in ("KimVanKieu1884", "LucVanTien1883")] + [REPO / "dataset/_ALL/labels.csv"],
        "chữ dị bản cho đối thủ M-OCR (KVK: Kiều1871/1872, L83: LVT1916 NF)")
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
    bad += [k for k, v in INV.items() if k.startswith("hand_cert_00015_reproduce") and v < 0.9995]
    bad += [k for k, v in INV.items() if k.endswith("port_maxabs_6000") or k == "viss_features_port_maxabs"
            if max(v.values()) > 1e-5]
    man = dict(version="2026-09-28", created=time.strftime("%Y-%m-%d %H:%M"),
               note="Tài sản gold_exact. Tệp .pt không vào git (.gitignore *.pt); dựng lại (0 API, ≈ 30 s): "
                    "python -m pipeline.gold_exact.export_assets (đọc kho nguồn bền source_store, tự kiểm SHA256SUMS)",
               sp=("<SRC> = source_store.path (bản chép bền của thư mục thử nghiệm 2026-09-25..27)" if store is not None
                   else "<SP> = thư mục thử nghiệm phiên 2026-09-25..27 (scratchpad)"),
               source_store=store, files=files, external=ext, invariants=INV, invariants_fail=bad)
    if bad:
        print(json.dumps(INV, ensure_ascii=False, indent=1, default=float))
        raise SystemExit(f"FAIL tái lập: {bad} — KHÔNG ghi MANIFEST")
    json.dump(man, open(OUT / "MANIFEST.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    log(f"MANIFEST: {len(files)} tài sản + {len(ext)} tệp ngoài; invariants PASS")
    print(json.dumps(INV, ensure_ascii=False, default=float)[:3000])
    if a.check_against:
        ref = json.load(open(Path(a.check_against) / "MANIFEST.json", encoding="utf-8"))
        diff = {k: (v["sha256"][:12], ref["files"].get(k, {}).get("sha256", "")[:12]) for k, v in files.items()
                if v["sha256"] != ref["files"].get(k, {}).get("sha256")}
        log(f"so với {a.check_against}: {len(files) - len(diff)}/{len(files)} tài sản trùng sha256" +
            (f"; khác: {diff}" if diff else ""))
        if diff:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
