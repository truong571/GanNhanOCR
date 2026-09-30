"""export_assets.py — dựng models/chon_chu/ (tham số bộ chọn + nguyên mẫu người Borg) + MANIFEST.json (sha256, cách dựng lại).

Nguồn (0 API; đều dựng lại được, không commit): bảng đặc trưng TN8 của 4 bộ có sự thật
  measure_out/_tn8/{base,cand}/<B18|B34|L16|TK>{.pkl,_cells.pkl}   <- lab/thu_nghiem_kim/TN8_chon_chu t00 -> t05 -> t05c
(sự thật: Borg = chữ Nôm người Excel qua borg_endtoend_eval + hộp người dataset/_BORG_NHAN_NGUOI; IHR = eval_ihr.Slot).
Học = CHÉP NGUYÊN t06_eval.fit_models (hạt giống 0, ≤ 40.000 ô/bộ/phần) và t10_visual_only.fit (≤ 60.000 ô).

Mô hình (tên -> học trên):
  hand_B34 (B34; dùng cho B18) · hand_B18 (B18; dùng cho B34) · hand_all (B18+B34; STT) ·
  print_TK (TK; dùng cho L16) · print_L16 (L16; dùng cho TK) · print_all (L16+TK; Chr, L83, KVK) · vis_hand_all (chỉ ảnh; STT bảo thủ)
Nguyên mẫu người: hum_B18.pt / hum_B34.pt = nhúng MultiEnc crop NGƯỜI keep_high+ (real) cắt lại bằng hình học save_crop từ
prepared/_auto/<Sách>/pages (+ ảnh gốc) — sao từ measure_out/_tn8/emb (t04_human.py) hoặc --rebuild-human (MPS ~7 phút).

  .venv/bin/python -m pipeline.chon_chu.export_assets --out models/chon_chu            # dựng + MANIFEST
  .venv/bin/python -m pipeline.chon_chu.export_assets --check models/chon_chu          # kiểm sha256 theo MANIFEST
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.chon_chu import model as M  # noqa: E402

TN8 = REPO / "measure_out" / "_tn8"
MODELS = {"hand_B34": ["B34"], "hand_B18": ["B18"], "hand_all": ["B18", "B34"],
          "print_TK": ["TK"], "print_L16": ["L16"], "print_all": ["L16", "TK"]}
POS_BOOKS = ("L16", "TK")                 # tầng 2 học đúng HAI VẾ (chữ ∧ khe) ở sách in IHR; Borg học đúng chữ
MAX_TRAIN_CELLS = 40000
HUMAN = {"B18": "SachKinhThayCaBinh", "B34": "SachDungLyHoThan"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_tn8(b: str) -> dict:
    D = pd.read_pickle(TN8 / "base" / f"{b}.pkl")
    F = pd.read_pickle(TN8 / "cand" / f"{b}.pkl").sort_values(["i", "c"], kind="stable").reset_index(drop=True)
    cells = pd.read_pickle(TN8 / "cand" / f"{b}_cells.pkl")
    Xc, names = M.cand_matrix(F, "is_kim")
    grp = M.group_of(D)
    kk = pd.Series(((F.is_kim.values == 1) & (F.inR.values == 1)).astype(np.int8)).groupby(F.i.values).max()
    kim_inR = np.zeros(len(D), bool); kim_inR[kk.index.values] = kk.values > 0
    part = np.where(kim_inR, "in", "out").astype(object)
    part[np.isin(grp, ["notplaus", "quarantine"])] = "none"
    return dict(b=b, D=D, F=F, Xc=Xc, names=names, cells=cells, grp=grp, part=part)


def fit_models(train: list[dict], part: str, seed=0):
    """== t06_eval.fit_models (không loại khối)."""
    rng = np.random.default_rng(seed)
    X1s, i1s, y1s = [], [], []
    off = 0
    for Bk in train:
        D, F = Bk["D"], Bk["F"]
        ok = (Bk["part"] == part) & (D.gt_char.values != "")
        anyy = F.groupby("i").y.max().reindex(np.arange(len(D))).fillna(0).values > 0
        ncand = F.groupby("i").size().reindex(np.arange(len(D))).fillna(0).values
        cand_cells = np.nonzero(ok & anyy & (ncand >= 2))[0]
        if len(cand_cells) > MAX_TRAIN_CELLS:
            cand_cells = np.sort(rng.choice(cand_cells, MAX_TRAIN_CELLS, replace=False))
        sel = np.isin(F.i.values, cand_cells)
        X1s.append(Bk["Xc"][sel]); i1s.append(F.i.values[sel] + off); y1s.append(F.y.values[sel])
        off += len(D) + 1
    m1 = M.CLogit().fit(np.concatenate(X1s), np.concatenate(i1s), np.concatenate(y1s))
    X2s, y2s = [], []
    for Bk in train:
        D, F = Bk["D"], Bk["F"]
        p = m1.predict(Bk["Xc"], F.i.values)
        Tt = M.top_table(F, p, Bk["Xc"], Bk["names"])
        G = pd.DataFrame({"i": F.i.values, "p": p, "y": F.y.values}).sort_values(["i", "p"], ascending=[True, False])
        y1 = G.groupby("i").head(1).set_index("i").y.reindex(Tt.index)
        ok = (Bk["part"] == part) & (D.gt_char.values != "")
        idx = np.nonzero(ok & np.isin(np.arange(len(D)), Tt.index.values))[0]
        X2, _ = M.cell_X(D, Bk["cells"], Bk["grp"], Tt, idx)
        y = y1.reindex(idx).fillna(0).values.astype(bool)
        if Bk["b"] in POS_BOOKS:
            y = y & D.pos_ok.values[idx]
        X2s.append(X2); y2s.append(y)
    m2 = M.Logit().fit(np.concatenate(X2s), np.concatenate(y2s))
    return m1, m2


def fit_visual(train: list[dict], max_cells=60000, seed=0):
    """== t10_visual_only.fit: tầng 1 KHÔNG 3 cột kim, mọi ô có gt ∈ ứng viên."""
    rng = np.random.default_rng(seed)
    keep = [j for j, n in enumerate(train[0]["names"]) if n not in M.KIM_COLS]
    Xs, Is, Ys = [], [], []
    off = 0
    for Bk in train:
        D, F = Bk["D"], Bk["F"]
        anyy = F.groupby("i").y.max().reindex(np.arange(len(D))).fillna(0).values > 0
        nc = F.groupby("i").size().reindex(np.arange(len(D))).fillna(0).values
        cells = np.nonzero(anyy & (nc >= 2) & (D.gt_char.values != ""))[0]
        if len(cells) > max_cells:
            cells = np.sort(rng.choice(cells, max_cells, replace=False))
        sel = np.isin(F.i.values, cells)
        Xs.append(Bk["Xc"][:, keep][sel]); Is.append(F.i.values[sel] + off); Ys.append(F.y.values[sel])
        off += len(D) + 1
    return M.CLogit().fit(np.concatenate(Xs), np.concatenate(Is), np.concatenate(Ys))


def build_models(out: Path, log=print) -> dict:
    books = {b: load_tn8(b) for b in ("B18", "B34", "L16", "TK")}
    files = {}
    for name, tr in MODELS.items():
        t0 = time.time()
        train = [books[b] for b in tr]
        mods = {part: fit_models(train, part) for part in ("in", "out")}
        meta = dict(name=name, train_books=tr, n_feat_stage1=len(train[0]["names"]), feat_stage1=train[0]["names"],
                    fit="t06_eval.fit_models (seed 0, ≤ 40.000 ô/bộ/phần)", created=datetime.now().isoformat(timespec="seconds"))
        f = out / f"chooser_{name}.npz"
        np.savez(f, **M.pack(mods, meta))
        files[f.name] = dict(desc=f"bộ chọn 2 tầng (in/out) học trên sự thật {'+'.join(tr)}", train_books=tr)
        log(f"[export] {f.name} ({time.time() - t0:.0f}s)")
    mv = fit_visual([books["B18"], books["B34"]])
    f = out / "chooser_vis_hand_all.npz"
    np.savez(f, **M.pack({"vis": mv}, dict(name="vis_hand_all", train_books=["B18", "B34"],
                                         feat_stage1=[n for n in books["B18"]["names"] if n not in M.KIM_COLS],
                                         fit="t10_visual_only.fit (seed 0, ≤ 60.000 ô)")))
    files[f.name] = dict(desc="bộ chọn CHỈ ẢNH (tầng 1, không cột kim) học B18+B34 — quy tắc STT bảo thủ", train_books=["B18", "B34"])
    return files


def build_human(out: Path, rebuild: bool, log=print) -> dict:
    import torch
    files = {}
    for code, book in HUMAN.items():
        E = M_ = None
        src = TN8 / "emb" / f"hum_{code}_enc.npy"
        if not rebuild and src.exists():
            E = np.load(src).astype(np.float16)
            M_ = pd.read_pickle(TN8 / "emb" / f"hum_{code}_meta.pkl")
        else:
            E, M_ = rebuild_human(book, log)
        f = out / f"hum_{code}.pt"
        torch.save(dict(E=torch.from_numpy(np.asarray(E, np.float16)), chars=list(M_.char.values),
                        page=list(M_.page.values), book=book), f)
        files[f.name] = dict(desc=f"nhúng MultiEnc crop NGƯỜI {book} (keep_high+, real) — nguyên mẫu người LOBO", n=int(len(M_)))
        log(f"[export] {f.name}: {len(M_):,} crop người")
    return files


def rebuild_human(book: str, log=print):
    """== t04_human.py: crop hộp người keep_high+ (real) cắt lại bằng save_crop từ prepared/_auto/<Sách>/pages -> MultiEnc."""
    import cv2
    import torch
    from collections import defaultdict
    from pipeline.align_engine.build_dataset import load_original_page
    from pipeline.chon_chu import crops as CR
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets, rd
    H = rd(REPO / "dataset/_BORG_NHAN_NGUOI/labels.csv")
    h = H[H.book == book].copy().reset_index(drop=True)

    def bb(s):
        try:
            v = json.loads(s)
            return [float(x) for x in v[:4]] if v is not None and len(v) >= 4 else None
        except Exception:  # noqa: BLE001
            return None
    h["bb"] = h.bbox.map(bb)
    h["use"] = h.keep_level.isin(("keep_v5", "keep", "keep_high")) & (h.kind == "real") & h.bb.notna() & (h.char.str.len() == 1)
    prep = REPO / "prepared/_auto" / book
    idx, grays = [], []
    for pg, g in h[h.bb.notna()].groupby("page"):
        img = cv2.imread(str(prep / "pages" / f"{pg}.png"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orig = load_original_page(prep, pg, shape_like=img.shape[:2])
        by_col = defaultdict(list)
        for i, c, b_, u in zip(g.index, g.column, g.bb, g.use):
            by_col[c].append((i, b_, u))
        for L in by_col.values():
            L.sort(key=lambda t: (t[1][1] + t[1][3]) / 2.0)
            for k, (i, b_, u) in enumerate(L):
                if not u:
                    continue
                r = CR.cut(img, gray_full, b_, L[k - 1][1] if k > 0 else None, L[k + 1][1] if k < len(L) - 1 else None, orig)
                if r is None:
                    continue
                _, gg = CR.encode(r["out"])
                idx.append(i); grays.append(gg)
    A = Assets()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = SI.MultiEnc([A.ext(SI.ENC_FILES["v1"]), A.ext(SI.ENC_FILES["v2"])], dev, True)
    E = np.concatenate([enc.embed(grays[s:s + 4096]) for s in range(0, len(grays), 4096)]).astype(np.float16)
    log(f"[export] dựng lại người {book}: {len(idx):,} crop")
    return E, h.loc[idx, ["page", "char"]].reset_index(drop=True)


def write_manifest(out: Path, files: dict) -> dict:
    man = dict(version="2026-09-30", created=datetime.now().isoformat(timespec="seconds"),
               note="Tài sản bước pipeline.chon_chu (TN8). *.pt (nguyên mẫu người, ~74 MB) bị gitignore — dựng lại bằng lệnh dưới; "
                    "chooser_*.npz nhỏ (commit được).",
               rebuild=["cd lab/thu_nghiem_kim/TN8_chon_chu && ../../../.venv/bin/python t00_base.py B18 B34 L16 TK && "
                        "../../../.venv/bin/python t02_embed.py B18 B34 L16 TK && ../../../.venv/bin/python t04_human.py && "
                        "../../../.venv/bin/python t05_feats.py B18 B34 L16 TK && ../../../.venv/bin/python t05c_knn.py B18 B34 L16 TK",
                        ".venv/bin/python -m pipeline.chon_chu.export_assets --out models/chon_chu"],
               thresholds_source="measure_out/_tn8/summary_t12_b80.json (t12_final.py 0.8: tiêu chí biên 0,80 trên dự đoán ngoài-khối "
                                 "của bộ học — LOBO) -> config/chon_chu.yaml",
               files={})
    for name, d in files.items():
        p = out / name
        man["files"][name] = dict(bytes=p.stat().st_size, sha256=sha256(p), **d)
    (out / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def check(out: Path) -> int:
    man = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [n for n, d in man["files"].items() if not (out / n).exists() or sha256(out / n) != d["sha256"]]
    print(f"[export] kiểm {len(man['files'])} tài sản: {'OK' if not bad else 'LỖI ' + str(bad)}")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.chon_chu.export_assets")
    ap.add_argument("--out", default=str(REPO / "models/chon_chu"))
    ap.add_argument("--check", default=None, help="chỉ kiểm sha256 theo MANIFEST của thư mục này")
    ap.add_argument("--rebuild-human", action="store_true", help="dựng lại nhúng người từ dataset/_BORG_NHAN_NGUOI (MPS)")
    ap.add_argument("--skip-models", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        return check(Path(a.check))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    old = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))["files"] if (out / "MANIFEST.json").exists() else {}
    files = {k: {kk: vv for kk, vv in v.items() if kk not in ("bytes", "sha256")} for k, v in old.items()}
    if not a.skip_models:
        files.update(build_models(out))
    files.update(build_human(out, a.rebuild_human))
    man = write_manifest(out, files)
    print(f"[export] MANIFEST: {len(man['files'])} tệp -> {out / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
