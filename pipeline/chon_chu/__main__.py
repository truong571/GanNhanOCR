"""python -m pipeline.chon_chu — bước CHỌN CHỮ BẰNG ẢNH (TN8, 30/09/2026), 0 API.

Chạy sau B4 gates (labels_gated.csv), trước B5 export/gold_exact. Sửa labels_gated.csv TẠI CHỖ (bản trước: <tên>_truoc_chon_chu.csv;
chạy lại tự khôi phục từ bản đó nên kết quả không cộng dồn). Sách TẮT trong config -> không đọc/ghi gì (STT mặc định TẮT).

  .venv/bin/python -m pipeline.chon_chu --labels prepared/_auto/SachKinhThayCaBinh/dataset_out/labels_gated.csv \
      --book SachKinhThayCaBinh [--config config/chon_chu.yaml] [--out <khác>] [--report <json>]
  .venv/bin/python -m pipeline.chon_chu --selftest

Ô được nâng: tier -> GOLD, label/unicode/label_canonical -> top-1, label_level char, rule += '|chon_chu:<đòn>', ô chưa có ảnh ->
cắt crop gold/chon_chu/<sách>_<trang>_c<cột>_n<nom>_s<syl>.png (hình học save_crop). L5: chỉ đổi nhãn + rule. Cột vết thêm:
chon_chu (đòn), chon_chu_p (P tầng 2), chon_chu_truoc ('<tier>|<nhãn>' trước bước) — labels.csv giao nộp vẫn 12 cột.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.chon_chu import model as M  # noqa: E402
from pipeline.chon_chu import policy as POL  # noqa: E402

TRACE_COLS = ("chon_chu", "chon_chu_p", "chon_chu_truoc")
MARK = "|chon_chu:"
CROP_DIR = "gold/chon_chu"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _int(v, d=-1):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def _bb(s):
    try:
        v = json.loads(s) if isinstance(s, str) else s
        return [float(x) for x in v[:4]] if v is not None and len(v) >= 4 else None
    except Exception:  # noqa: BLE001
        return None


def verify_models(mdir: Path, need: list[str]) -> dict:
    man = json.loads((mdir / "MANIFEST.json").read_text(encoding="utf-8"))
    for n in need:
        ent = man["files"].get(n)
        if ent is None:
            raise FileNotFoundError(f"{mdir}/MANIFEST.json không có {n} — dựng: python -m pipeline.chon_chu.export_assets")
        p = mdir / n
        if not p.exists():
            raise FileNotFoundError(f"thiếu {p} (gitignore *.pt) — dựng lại: python -m pipeline.chon_chu.export_assets")
        if sha256(p) != ent["sha256"]:
            raise RuntimeError(f"{p} sai sha256 so với MANIFEST — dựng lại: python -m pipeline.chon_chu.export_assets")
    return man


def load_human(mdir: Path, codes: list[str]):
    import torch
    out = []
    for c in codes:
        z = torch.load(mdir / f"hum_{c}.pt", map_location="cpu", weights_only=False)
        out.append((z["E"].numpy().astype(np.float32), np.asarray(z["chars"], dtype=object)))
    return out


def backup_path(labels: Path) -> Path:
    return Path(labels).with_name(Path(labels).stem + "_truoc_chon_chu.csv")


def load_input(labels: Path, out: Path, log=print) -> tuple[pd.DataFrame, bool]:
    """Bảng nhãn ĐẦU VÀO của bước (không cộng dồn): tệp đã mang dấu chon_chu -> đọc bản trước `<tên>_truoc_chon_chu.csv`
    (thiếu -> lỗi); chưa mang dấu và sửa tại chỗ -> chép byte sang bản trước. Bỏ cột vết cũ. Trả (bảng, đã_khôi_phục)."""
    backup = backup_path(labels)
    L = pd.read_csv(labels, dtype=str, keep_default_na=False)
    restored = False
    if L.rule.str.contains(MARK, regex=False).any():
        if not backup.exists():
            raise RuntimeError(f"{labels} đã mang dấu chon_chu nhưng không có {backup.name} để khôi phục")
        log(f"[chon_chu] {labels.name} đã qua bước này -> đọc lại bản trước {backup.name} (không cộng dồn)")
        L = pd.read_csv(backup, dtype=str, keep_default_na=False)
        restored = True
    elif Path(out) == Path(labels):
        shutil.copyfile(labels, backup)
    for c in TRACE_COLS:
        if c in L.columns:
            L = L.drop(columns=c)
    return L, restored


def process_book(D: pd.DataFrame, book: str, bcfg: dict, cfg: dict, crop_root: Path, log=print, dump_pred: Path | None = None) -> dict:
    """Một sách (D = các dòng của sách, index 0..N-1): crop + nhúng + đặc trưng + 2 tầng + quyết định + cắt ảnh cho ô nâng
    chưa có ảnh (ghi vào crop_root). Trả dict mảng quyết định + số liệu báo cáo. KHÔNG sửa D."""
    t0 = time.time()
    rep: dict = {}
    from pipeline.tools.duong_chay import book_cfg as pipe_book_cfg
    from pipeline.align_engine.book_layout import book_layout
    pcfg_path, bdict = pipe_book_cfg(bcfg["name"])
    pcfg = yaml.safe_load(pcfg_path.read_text(encoding="utf-8")) or {}
    prep = REPO / (pcfg.get("paths") or {}).get("data_dir", "prepared") / bdict["name"]
    orig = book_layout(bdict).crop_source == "original"
    rep.update(pipeline_config=str(pcfg_path.relative_to(REPO)), prepared=str(prep.relative_to(REPO)), crop_source_original=orig)
    mdir = REPO / cfg.get("models_dir", "models/chon_chu")
    is_stt = bcfg["name"] in POL.STT
    stt_mode = bcfg.get("stt_mode", "bao_thu") if is_stt else None
    need = [f"chooser_{bcfg['model']}.npz"] + [f"hum_{c}.pt" for c in bcfg.get("human") or []]
    if stt_mode == "bao_thu":
        need.append(f"chooser_{bcfg.get('vis_model', 'vis_hand_all')}.npz")
    man = verify_models(mdir, need)
    mods, mmeta = M.unpack(np.load(mdir / need[0]))
    own = POL.OWN.get(bcfg["name"])
    if own and own in set(mmeta.get("train_books", [])):
        raise ValueError(f"[chon_chu] {book}: mô hình {bcfg['model']} đã học nhãn người của chính sách ({own}) — phải LOBO")
    rep.update(model=bcfg["model"], model_train_books=mmeta.get("train_books"), models_manifest=man.get("version"),
               verifiers=bcfg.get("verifiers"), human=bcfg.get("human"), tau_in=bcfg.get("tau_in"), tau_out=bcfg.get("tau_out"),
               tau_relabel=bcfg["tau_relabel"], promo_groups=bcfg["promo_groups"], relabel=bcfg["relabel"], stt_mode=stt_mode)
    D = D.copy()
    D["column_i"] = D.column.map(_int)
    D["nom_i"] = D.nom_idx.map(_int) if "nom_idx" in D.columns else -1
    D["bbox4"] = D.bbox.map(_bb)
    from pipeline.chon_chu import crops as CR
    from pipeline.chon_chu.embed import Embedder
    from pipeline.gold_exact.common import Assets, f16, set_lexicon
    from pipeline.gold_exact import common as C
    from pipeline.gold_exact import signals_img as SI
    A = Assets()
    set_lexicon(A)
    N = len(D)
    keys = [""] * N
    grays: dict = {}
    meta = np.zeros((N, 4), np.float32)
    for res in CR.iter_pages(D, prep, orig):
        for i, b, g, m in res:
            if b is None:
                continue
            k = CR.md5(b)
            keys[i] = k; grays.setdefault(k, g); meta[i] = [m[0], m[1], m[2], 1]
    ok_c = np.array([bool(k) for k in keys])
    log(f"[chon_chu] {book}: crop {int(ok_c.sum()):,}/{N:,} ô ({time.time() - t0:.0f}s)")
    cache = REPO / cfg.get("cache_dir", "prepared/_chon_chu") / bcfg["name"] / "emb"
    emb = Embedder(A, cfg.get("device", "mps"), cache, log)
    Eall = emb.embed([k for k in keys if k], grays, list(bcfg.get("verifiers") or []))
    E = np.zeros((N, 512), np.float32); E[ok_c] = f16(Eall["enc"])
    Z = {}
    for v in bcfg.get("verifiers") or []:
        z = np.zeros((N, 256), np.float32); z[ok_c] = f16(Eall[v]); Z[v] = z
    del grays, Eall
    lt2 = np.array([""] * N, dtype=object)
    if is_stt:
        from pipeline.chon_chu.stt import lt2_reads
        lt2 = lt2_reads(D, bcfg["name"], bdict, log)
    from pipeline.chon_chu import features as FT
    scorers = SI.Scorers(A, str(emb.dev.type), REPO / cfg.get("cache_dir", "prepared/_chon_chu") / "_glyph", False, lambda m: None)
    F, cells = FT.build(D, E, Z, meta, lt2, C=C, scorers=scorers, assets=A, verifiers=list(bcfg.get("verifiers") or []),
                        human=load_human(mdir, list(bcfg.get("human") or [])), log=log)
    grp = M.group_of(D)
    if stt_mode == "manh":                 # union lt1 ∪ lt2 đóng vai kim (TN8 §4 — KHÔNG khuyến nghị)
        F = F.copy(); F["is_kim"] = np.maximum(F.is_kim.values, F.is_lt2.values); F["is_lt2"] = 0
    Xc, names = M.cand_matrix(F, "is_kim")
    part = POL.parts(F, grp, N, "is_kim")
    P = np.full(N, np.nan); top1 = np.array([""] * N, dtype=object)
    for pt in ("in", "out"):
        m1, m2 = mods[pt]
        p = m1.predict(Xc, F.i.values)
        Tt = M.top_table(F, p, Xc, names)
        idx = np.nonzero(part == pt)[0]
        idx = idx[np.isin(idx, Tt.index.values)]
        if not len(idx):
            continue
        X2, _ = M.cell_X(D, cells, grp, Tt, idx)
        P[idx] = m2.predict(X2)
        top1[idx] = Tt.top1.reindex(idx).values
    tier = D.tier.values.astype(object)
    gate = (D.gate_reason if "gate_reason" in D.columns else pd.Series([""] * N)).values.astype(object)
    if stt_mode == "bao_thu":
        mv, _ = M.unpack(np.load(mdir / f"chooser_{bcfg.get('vis_model', 'vis_hand_all')}.npz"))
        Xv, _ = M.cand_matrix(F, "is_kim", with_kim=False)
        pv = mv["vis"].predict(Xv, F.i.values)
        G = pd.DataFrame({"i": F.i.values, "c": F.c.values, "p": pv}).sort_values(["i", "p"], ascending=[True, False])
        vtop = G.groupby("i").head(1).set_index("i").c.reindex(np.arange(N)).fillna("").values
        promote, lab_new = POL.decide_stt_bao_thu(tier, grp, D.ocr_char.values, lt2, D.syllable.values, vtop, C.R_of,
                                                  C.var_eq_plus)
        relabel = np.zeros(N, bool)
        lever = np.where(promote, "L3", "").astype(object)
        top1 = np.where(promote, lab_new, top1)
    else:
        promote, relabel, lever = POL.decide(tier, grp, gate, part, P, top1, D.label.values, bcfg, C.var_eq_plus)
    blank_syl = np.zeros(N, bool)
    if bcfg.get("qn_geo") or bcfg.get("np_geo"):
        # luật TN9 sách in: kim_geo (chữ kim có hộp chia đều chứa tâm crop) ≡ kim -> P2 qn_geo / P3 np_geo (policy.decide_geo)
        from pipeline.chon_chu.geo import kim_geo
        geo = kim_geo(D, prep)
        p2, p3 = POL.decide_geo(tier, grp, D.ocr_char.values, D.syllable.values, geo, promote | relabel, bcfg, C.R_of,
                                C.var_eq_plus)
        promote = promote | p2 | p3
        lever = np.where(p2, "qn_geo", np.where(p3, "np_geo", lever)).astype(object)
        top1 = np.where(p2 | p3, D.ocr_char.values, top1).astype(object)
        blank_syl = p3.copy()
        geo_ok = np.array([bool(g) and bool(k) and (g == k or C.var_eq_plus(g, k)) for g, k in zip(geo, D.ocr_char.values)])
        rep["kim_geo"] = dict(co_hop=int((geo != "").sum()), trung_kim=int(geo_ok.sum()),
                              direct_qn=int((grp == "direct_qn").sum()), direct_qn_geo=int(((grp == "direct_qn") & geo_ok).sum()),
                              notplaus=int((grp == "notplaus").sum()), notplaus_geo=int(((grp == "notplaus") & geo_ok).sum()),
                              qn_geo=int(p2.sum()), np_geo=int(p3.sum()))
    if dump_pred:
        pd.DataFrame(dict(P=P, top1=top1, part=part, grp=grp, promote=promote, relabel=relabel, lever=lever)).to_pickle(dump_pred)
    # ---- cắt crop (ảnh giao + đo chất lượng trên bản ĐÃ XỬ LÝ như enrich_crop_quality) cho ô nâng chưa có ảnh
    from pipeline.align_engine import crop_quality as CQ
    need_img = promote & (D.image.values == "")
    wrote = {}
    if need_img.any():
        want = set(np.nonzero(need_img)[0])
        for res in CR.iter_pages(D, prep, orig, want=want):
            for i, b, g, m in res:
                if b is None:
                    continue
                r = D.loc[i]
                rel = f"{CROP_DIR}/{r.book}_{r.page}_c{int(r.column_i):02d}_n{r.nom_idx}_s{r.syl_idx}.png"
                p = crop_root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b)
                wrote[i] = dict(image=rel, image_md5=CR.md5(b)[:12], crop_w=str(int(m[0])), crop_h=str(int(m[1])),
                                ink_pct=f"{m[2]:.3f}")
        for res in CR.iter_pages(D, prep, False, want=set(wrote)):      # orig=False -> bản ĐÃ XỬ LÝ
            for i, b, g, m in res:
                if g is not None and i in wrote:
                    q = CQ.measure(g)
                    wrote[i].update(crop_quality_flag=q["crop_quality_flag"], stray_ink=str(q["stray_ink"]),
                                    border_ink=str(q["border_ink"]))
    no_img = need_img & ~np.isin(np.arange(N), list(wrote))
    promote = promote & ~no_img               # không cắt được ảnh -> không nâng (N5: ô GOLD phải có ảnh)
    lever = np.where(no_img, "", lever).astype(object)
    blank_syl = blank_syl & promote
    rep.update(n_rows=N, crop_ok=int(ok_c.sum()), promote=int(promote.sum()), relabel=int(relabel.sum()),
               khong_cat_duoc_anh=int(no_img.sum()), crop_moi=len(wrote),
               theo_don_bay={k: int(v) for k, v in pd.Series(lever[promote | relabel]).value_counts().items()},
               theo_nhom={k: int(v) for k, v in pd.Series(grp[promote]).value_counts().items()},
               seconds=round(time.time() - t0, 1))
    return dict(promote=promote, relabel=relabel, lever=lever, P=P, top1=top1, wrote=wrote, rep=rep, blank_syl=blank_syl)


def book_code(L: pd.DataFrame, book: str) -> str | None:
    """Giá trị cột `book` của sách trong tệp nhãn dùng chung (STT: stt2/stt4/stt11); tệp một sách -> None (lấy hết)."""
    from pipeline.chon_chu.stt import S8
    if book in S8 and "book" in L.columns and L.book.nunique() > 1:
        return S8[book]
    return None


def run(labels: Path, books, cfg_path: Path, out: Path | None = None, report: Path | None = None, log=print,
        dump_pred: Path | None = None) -> dict:
    """books: một tên hoặc danh sách (STT: 3 sách chung một labels_final.csv -> xử lý từng phần, ghi MỘT lần)."""
    t0 = time.time()
    books = [books] if isinstance(books, str) else list(books)
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
    rep: dict = dict(books=books, config=str(cfg_path), config_version=cfg.get("version"), labels=str(labels), api_calls=0,
                     per_book={})
    bcfgs = {b: POL.book_cfg(cfg, b) for b in books}
    on = [b for b in books if bcfgs[b] is not None and bcfgs[b]["enabled"]]
    for b in books:
        if b not in on:
            log(f"[chon_chu] {b}: TẮT ({'không có trong' if bcfgs[b] is None else 'enabled: false ở'} {cfg_path}) — không đọc/ghi gì")
    if not on:
        rep["enabled"] = False
        return rep
    rep["enabled"] = True
    out = Path(out) if out else Path(labels)
    report = Path(report) if report else Path(labels).parent / "chon_chu_report.json"
    backup = backup_path(labels)
    L, restored = load_input(Path(labels), out, log)
    if restored:
        rep["khoi_phuc_tu"] = str(backup)
    rep["in_sha256"] = sha256(backup if backup.exists() and out == Path(labels) else Path(labels))
    crop_root = Path(labels).parent if out == Path(labels) else out.parent   # chạy thử (--out khác): không ghi vào dataset_out
    Lo = L.copy().reset_index(drop=True)
    for c in TRACE_COLS:
        Lo[c] = ""
    before = Lo.tier.value_counts().to_dict()
    for b in on:
        code = book_code(Lo, b)
        rows = np.arange(len(Lo)) if code is None else np.nonzero(Lo.book.values == code)[0]
        D = Lo.iloc[rows].drop(columns=list(TRACE_COLS)).reset_index(drop=True)
        r = process_book(D, b, bcfgs[b], cfg, crop_root, log,
                         dump_pred=(dump_pred if len(on) == 1 else (Path(str(dump_pred) + f".{b}") if dump_pred else None)))
        pro, rel, lev, P, top1 = r["promote"], r["relabel"], r["lever"], r["P"], r["top1"]
        from pipeline.gold_exact.common import var_eq_plus
        top1 = POL.new_labels(top1, Lo.label.values[rows], var_eq_plus)     # giữ nhãn cũ khi top-1 ≡ nhãn (V1+)
        ch = pro | rel
        g = rows[ch]
        Lo.loc[g, "chon_chu_truoc"] = (Lo.tier.values[g] + "|" + Lo.label.values[g])
        bs_ = rows[r["blank_syl"]]
        if len(bs_):                          # P3 np_geo: âm QN rác -> để trống; âm gốc giữ trong vết
            Lo.loc[bs_, "chon_chu_truoc"] = Lo.tier.values[bs_] + "|" + Lo.label.values[bs_] + "|âm:" + Lo.syllable.values[bs_]
            Lo.loc[bs_, "syllable"] = ""
        Lo.loc[g, "chon_chu"] = lev[ch]
        Lo.loc[g, "chon_chu_p"] = [f"{x:.4f}" if x == x else "" for x in P[ch]]
        Lo.loc[g, "label"] = top1[ch]
        Lo.loc[g, "unicode"] = [f"U+{ord(x):04X}" if len(x) == 1 else "" for x in top1[ch]]
        if "label_canonical" in Lo.columns:
            Lo.loc[g, "label_canonical"] = top1[ch]
        gp = rows[pro]
        Lo.loc[gp, "tier"] = "GOLD"
        if "label_level" in Lo.columns:
            Lo.loc[gp, "label_level"] = "char"
        Lo.loc[g, "rule"] = Lo.rule.values[g] + MARK + lev[ch]
        for i, d in r["wrote"].items():
            if not pro[i]:
                continue
            for col, val in d.items():
                if col in Lo.columns:
                    Lo.at[rows[i], col] = val
        rep["per_book"][b] = r["rep"]
        log(f"[chon_chu] {b}: nâng {r['rep']['promote']:,} {r['rep']['theo_don_bay']}; L5 sửa nhãn {r['rep']['relabel']:,}; "
            f"crop mới {r['rep']['crop_moi']:,} [{r['rep']['seconds']}s]")
    tmp = out.with_suffix(out.suffix + ".tmp")
    Lo.to_csv(tmp, index=False)
    tmp.replace(out)
    after = Lo.tier.value_counts().to_dict()
    N = len(Lo)
    rep.update(n_rows=N, tier_before=before, tier_after=after, gold_share_before=round(before.get("GOLD", 0) / N, 4),
               gold_share_after=round(after.get("GOLD", 0) / N, 4), out=str(out), out_sha256=sha256(out),
               backup=str(backup) if out == Path(labels) else None, seconds=round(time.time() - t0, 1))
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"[chon_chu] GOLD {before.get('GOLD', 0):,} -> {after.get('GOLD', 0):,} ({rep['gold_share_before']:.3f} -> "
        f"{rep['gold_share_after']:.3f}) · -> {out} · báo cáo {report} [{rep['seconds']}s]")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.chon_chu", description=__doc__.split("\n")[0])
    ap.add_argument("--labels", help="labels_gated.csv của sách (sửa tại chỗ trừ khi --out)")
    ap.add_argument("--book", nargs="+", help="tên sách (config/pipeline_<Book>.yaml); STT: SachThanhTruyen2 SachThanhTruyen4 "
                                               "SachThanhTruyen11 (chung một labels_final.csv)")
    ap.add_argument("--config", default=str(REPO / "config/chon_chu.yaml"))
    ap.add_argument("--out", default=None, help="ghi ra tệp khác (thử nghiệm) thay vì sửa tại chỗ")
    ap.add_argument("--report", default=None)
    ap.add_argument("--dump-pred", default=None, help="(kiểm) ghi P/top1/phần/nhóm của mọi ô ra pickle")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        from pipeline.chon_chu.selftest import main as st
        return st()
    if not a.labels or not a.book:
        ap.error("cần --labels và --book")
    run(Path(a.labels), a.book, Path(a.config), Path(a.out) if a.out else None, Path(a.report) if a.report else None,
        dump_pred=Path(a.dump_pred) if a.dump_pred else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
