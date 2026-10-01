"""route.py — ĐƯỜNG STT HAI LƯỢT (TN9, 01/10): hợp theo ô hai bản dựng + cổng chặt R4 + cổng hộp visual_dp. 0 API.

Bản dựng CHÍNH  = config/pipeline.yaml như khai (books[].kim_read = l1skel_l2, box_decoder = visual_dp) -> dataset_out/.
Bản dựng PHỤ    = cùng config, chỉ đổi books[].kim_read = stt_hai_luot.aux_read (lt1) -> stt_hai_luot.aux_dir.
Mỗi bản dựng đi trọn build -> enrich -> remediation census/apply -> confusion_fix (như run_pipeline.sh). Sau đó:

HỢP (union, ĐÚNG phép đo TN9 t13/t17): khoá ô = (book, page, column, syl_idx). Hàng chọn: bản chính nếu GOLD ở bản
chính; ngược lại bản phụ nếu GOLD ở bản phụ; ngược lại bản chính (nếu có) rồi bản phụ. Crop của hàng lấy từ bản phụ được
chép sang dataset_out/<tầng>/hl_lt1__<tên>. Hai lần đọc của ô: l1 = ocr_char của bản phụ (lt1) tại ô; l2 = chữ lt2 ghép về
hộp chữ lt1 của ô (phương pháp `line` của signals_lt2 / chon_chu.stt — cột kim dựng lại bằng _detect trên cache lt1);
ô chỉ có ở bản chính: l1 = '', l2 = ocr_char bản chính.
LỚP (theo nhãn + luật của hàng chọn): rescue (luật self_training_rescue) · agree (l1 ≡ l2 ∈ R(âm), nhãn ≡ l1) ·
conflict (l1, l2 ∈ R, khác) · lt2only (l2 ∈ R, l1 ∉ R, nhãn ≡ l2) · lt1_nolt2 (l1 ∈ R, không có l2, nhãn ≡ l1) ·
lt1_l2out (l1 ∈ R, l2 có mà ∉ R) · similar (s1_inter_s2_similar) · other.
CỔNG (sau bước rescue):
  strict (R4) ô GOLD chỉ giữ khi lớp ∈ {agree, lt2only, lt1_nolt2}; ô rescue giữ khi nhãn ≡ l1 ≡ l2 ∈ R, ngược lại trả về
              hàng TRƯỚC rescue; ô khác -> REVIEW. union (R2) không hạ. off: không hạ.
  vdp         ô GOLD còn lại có box_source ∈ {vdp_low, vdp_virtual, vdp_fallback} và IoU(hộp, hộp tham chiếu syl_index
              (boxes_syl.csv của bản dựng nguồn)) < 0,5 -> REVIEW.
Nguồn gốc: luật += "|hai_luot:lt2" (nhãn từ lt2) / "|hai_luot:lt1" (GOLD lấy từ bản phụ lt1) / "|hai_luot:ha_<lớp>" /
"|hai_luot:vdp_<nguồn hộp>"; tier_goc/rule_goc ghi tầng/luật trước khi hạ; cột vết `hai_luot` (nguồn:lớp) và
`hai_luot_truoc` (tầng|nhãn trước cổng) — export đưa vào labels_trace.csv, labels.csv giữ 12 cột.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
STT_S8 = {"SachThanhTruyen2": "stt2", "SachThanhTruyen4": "stt4", "SachThanhTruyen11": "stt11"}
S8_BOOK = {v: k for k, v in STT_S8.items()}
GATES = ("strict", "union", "off")
KEEP_STRICT = ("agree", "lt2only", "lt1_nolt2")
WEAK_BOX = ("vdp_low", "vdp_virtual", "vdp_fallback")
MARK = "|hai_luot:"
AUX_PREFIX = "hl_lt1__"
TRACE = ("hai_luot", "hai_luot_truoc")
SIDE = "o_hai_luot.csv"
UNION_SNAP = "labels_final_union.csv"


# ------------------------------------------------------------------------------------------------ cấu hình
def cfg_of(config: dict | None) -> dict:
    """stt_hai_luot đã kiểm (vắng -> enabled false = đường cũ một bản dựng). Sai khoá/giá trị -> ValueError."""
    from pipeline.align_engine.book_layout import KIM_READS
    h = (config or {}).get("stt_hai_luot")
    if h is None:
        return dict(enabled=False, aux_read="lt1", aux_dir="prepared/_stt_hai_luot/doc_lt1", gate="off",
                    vdp_gate=False, declared=False)
    if not isinstance(h, dict):
        raise ValueError(f"stt_hai_luot phải là mapping (đang {type(h).__name__})")
    unknown = set(h) - {"enabled", "aux_read", "aux_dir", "gate", "vdp_gate"}
    if unknown:
        raise ValueError(f"stt_hai_luot: khoá lạ {sorted(unknown)}")
    out = dict(enabled=h.get("enabled", False), aux_read=h.get("aux_read", "lt1"),
               aux_dir=h.get("aux_dir", "prepared/_stt_hai_luot/doc_lt1"), gate=h.get("gate", "strict"),
               vdp_gate=h.get("vdp_gate", False), declared=True)
    if not isinstance(out["enabled"], bool) or not isinstance(out["vdp_gate"], bool):
        raise ValueError("stt_hai_luot.enabled / vdp_gate cần true/false")
    if out["aux_read"] not in KIM_READS:
        raise ValueError(f"stt_hai_luot.aux_read = {out['aux_read']!r}; chỉ nhận {KIM_READS}")
    if out["gate"] not in GATES:
        raise ValueError(f"stt_hai_luot.gate = {out['gate']!r}; chỉ nhận {GATES}")
    if not isinstance(out["aux_dir"], str) or not out["aux_dir"].strip():
        raise ValueError("stt_hai_luot.aux_dir cần đường dẫn")
    return out


def aux_config(config_path: Path, out_path: Path) -> dict:
    """Config bản dựng phụ: books[].kim_read := aux_read (lt1 -> bỏ khoá), run.dataset_out := aux_dir."""
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    h = cfg_of(cfg)
    if not h["enabled"]:
        raise SystemExit("stt_hai_luot.enabled = false — không có bản dựng phụ")
    for b in cfg.get("books") or []:
        if h["aux_read"] == "lt1":
            b.pop("kim_read", None)
        else:
            b.update({"kim_read": h["aux_read"]})          # (update: không gán bằng chỉ số — code_facts coi b["k"] là khoá BẮT BUỘC)
    aux = h["aux_dir"]
    cfg.setdefault("run", {})["dataset_out"] = aux
    cfg.setdefault("paths", {})["output_dir"] = str(Path(aux) / "export")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return h


# ------------------------------------------------------------------------------------------------ hai lần đọc tại ô
_KG = {}


def _init(qn_path):
    from pipeline.gold_exact import signals_geom as SG
    SG._kim_init(qn_path)


def _cols_worker(args):
    book, page, bdict = args
    from pipeline.gold_exact import signals_geom as SG
    return SG._kim_page((book, page, bdict))


def kim_cols_lt1(book: str, pages: list, cache_dir: Path, workers: int = 4, log=print) -> dict:
    """{page: {line_id: [(chữ, bbox[, lt1])]}} — cột kim dựng lại bằng _detect trên cache lt1 (books[].kim_read = lt1)."""
    from pipeline.step0_setup import load_config
    base = load_config(str(REPO / "config/pipeline.yaml"))
    bdict = dict(next(b for b in base["books"] if b["name"] == book))
    bdict.pop("kim_read", None)                                   # lt1 = detected/
    root = Path(cache_dir) / "kim_cols_lt1" / book
    root.mkdir(parents=True, exist_ok=True)
    res, jobs = {}, []
    for pg in pages:
        f = REPO / "prepared" / book / "detected" / f"{pg}_ocr_cache.json"
        png = REPO / "prepared" / book / "pages" / f"{pg}.png"
        key = hashlib.sha256((f.read_bytes() if f.exists() else b"") + str(png.stat().st_size if png.exists() else 0).encode()
                             + json.dumps(bdict, sort_keys=True, default=str).encode()).hexdigest()
        pf = root / f"{pg}.pkl"
        if pf.exists():
            try:
                z = pickle.load(open(pf, "rb"))
                if z.get("key") == key:
                    res[pg] = z["out"] or {}; continue
            except Exception:  # noqa: BLE001
                pass
        jobs.append((pg, key, pf))
    if jobs:
        qn = str(REPO / base["paths"]["qn_to_nom_dict"])
        with Pool(workers, initializer=_init, initargs=(qn,)) as pool:
            for (pg, key, pf), out in zip(jobs, pool.imap(_cols_worker, [(book, pg, bdict) for pg, _, _ in jobs], chunksize=4)):
                tmp = pf.with_suffix(".tmp")
                pickle.dump(dict(key=key, out=out), open(tmp, "wb"))
                tmp.replace(pf)
                res[pg] = out or {}
    log(f"[hai_luot] cột kim lt1 {book}: {len(pages)} trang ({len(jobs)} dựng lại, {len(pages) - len(jobs)} từ cache)")
    return res


def lt2_at_cells(A: pd.DataFrame, book: str, cache_dir: Path, workers: int = 4, log=print) -> np.ndarray:
    """Chữ lt2 ghép về hộp chữ lt1 của từng hàng A (bản dựng lt1) — = TN8 t03 / chon_chu.stt.lt2_reads. '' = không ghép."""
    from pipeline.gold_exact import signals_lt2 as SL
    from pipeline.stt_hai_luot.doc import _lexicon
    _lexicon()
    pages = SL.load_pages(STT_S8[book])
    cols = kim_cols_lt1(book, sorted(set(A.page)), cache_dir, workers, log)
    out = np.array([""] * len(A), dtype=object)
    n_ok = 0
    pos = {i: k for k, i in enumerate(A.index)}
    for pg, g in A.groupby("page"):
        pair = pages["lt2"].get(pg)
        if pair is None:
            continue
        r1, r2 = pair
        pp = SL.PagePair(r1.get("boxes_raw"), r2.get("boxes_raw"))
        C = cols.get(pg) or {}
        for i, col, ni, oc in zip(g.index, g.column, g.nom_idx, g.ocr_char):
            try:
                col, ni = int(col), int(ni)
            except (TypeError, ValueError):
                continue
            chars = C.get(col)
            if not chars or ni < 0 or ni >= len(chars) or not chars[ni][1] or chars[ni][0] != oc:
                continue
            d = pp.by_line(chars[ni][0], [float(v) for v in chars[ni][1][:4]])
            if d.get("st_line") == "ok":
                out[pos[i]] = d.get("oth_line", "")
                n_ok += 1
    log(f"[hai_luot] lt2 tại ô {book}: ghép được {n_ok:,}/{len(A):,} hàng bản lt1")
    return out


def classify(l1, l2, syl, lab, rule, R_of, veq) -> np.ndarray:
    """Lớp bằng chứng của từng ô (TN9 t13/t17)."""
    out = []
    Rc = {}
    for x, y, s, lb, ru in zip(l1, l2, syl, lab, rule):
        if s not in Rc:
            Rc[s] = R_of(s)
        R = Rc[s]
        xi, yi = bool(x) and x in R, bool(y) and y in R
        if str(ru).startswith("self_training_rescue"):
            c = "rescue"
        elif x and y and veq(x, y) and xi and veq(lb, x):
            c = "agree"
        elif xi and yi and not veq(x, y):
            c = "conflict"
        elif yi and not xi and veq(lb, y):
            c = "lt2only"
        elif xi and not y and veq(lb, x):
            c = "lt1_nolt2"
        elif xi and y and not yi and veq(lb, x):
            c = "lt1_l2out"
        elif str(ru).startswith("s1_inter_s2_similar"):
            c = "similar"
        else:
            c = "other"
        out.append(c)
    return np.array(out, dtype=object)


def _lex():
    from pipeline.gold_exact import common as C
    from pipeline.stt_hai_luot.doc import _lexicon
    _lexicon()
    return C.R_of, (lambda a, b: bool(a) and bool(b) and (a == b or C.var_eq_plus(a, b)))


def rd(p) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def keys_of(L: pd.DataFrame) -> pd.Series:
    return L.book + "/" + L.page + "/" + L.column.astype(str) + "/" + L.syl_idx.astype(str)


# ------------------------------------------------------------------------------------------------ hợp
def union(primary_dir: Path, aux_dir: Path, work_dir: Path, workers: int = 4, log=print, l2_fn=None) -> dict:
    """labels_final.csv của bản CHÍNH := hợp theo ô với bản PHỤ. Ghi work_dir/o_hai_luot.csv (hai lần đọc + lớp) và ảnh chụp
    work_dir/labels_final_union.csv (TRƯỚC rescue, để cổng trả lại hàng rescue bị hạ)."""
    primary_dir, aux_dir, work_dir = Path(primary_dir), Path(aux_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    P = rd(primary_dir / "labels_final.csv"); A = rd(aux_dir / "labels_final.csv")
    shutil.copy2(primary_dir / "labels_final.csv", work_dir / "labels_final_primary.csv")
    for nm, X in (("chính", P), ("phụ", A)):
        k = keys_of(X)
        if k.duplicated().any():
            raise SystemExit(f"[hai_luot] bản {nm}: {int(k.duplicated().sum())} khoá ô trùng (book,page,column,syl_idx)")
    if list(P.columns) != list(A.columns):
        raise SystemExit(f"[hai_luot] hai bản dựng khác cột: {sorted(set(P.columns) ^ set(A.columns))}")
    P["_key"], A["_key"] = keys_of(P), keys_of(A)
    Pi, Ai = P.set_index("_key"), A.set_index("_key")
    keys = Pi.index.union(Ai.index)
    inP, inA = keys.isin(Pi.index), keys.isin(Ai.index)
    gP = Pi.tier.reindex(keys).fillna("").values == "GOLD"
    gA = Ai.tier.reindex(keys).fillna("").values == "GOLD"
    takeP = gP | (~gA & inP)
    R_of, veq = _lex()
    # hai lần đọc: l1 = ocr_char bản lt1; l2 = lt2 ghép về hộp lt1 (ô có ở bản lt1) | ocr_char bản chính (ô chỉ ở bản chính)
    l1 = Ai.ocr_char.reindex(keys).fillna("").values
    l2 = np.array([""] * len(keys), dtype=object)
    l2_at_A = pd.Series("", index=Ai.index, dtype=object)
    for book in sorted(set(A.book)):
        bk = S8_BOOK.get(book, book)
        sub = A[A.book == book]
        if not len(sub):
            continue
        v = (l2_fn or lt2_at_cells)(sub, bk, work_dir, workers, log)
        l2_at_A.loc[sub["_key"].values] = v
    l2[inA] = l2_at_A.reindex(keys[inA]).values
    l2 = np.where(~inA, Pi.ocr_char.reindex(keys).fillna("").values, l2)
    syl = np.where(inA, Ai.syllable.reindex(keys).fillna("").values, Pi.syllable.reindex(keys).fillna("").values)
    # hàng chọn
    rows = pd.concat([Pi.reindex(keys[takeP]), Ai.reindex(keys[~takeP])]).reindex(keys)
    doc = np.where(takeP, "P", "A")
    cls = classify(l1, l2, syl, rows.label.values, rows.rule.values, R_of, veq)
    # crop của hàng bản phụ -> dataset_out/<tầng>/hl_lt1__<tên>
    n_copy = 0
    imgs = rows.image.values.copy()
    for k in np.nonzero(~takeP)[0]:
        im = imgs[k]
        if not im:
            continue
        src = aux_dir / im
        rel = str(Path(im).parent / (AUX_PREFIX + Path(im).name))
        dst = primary_dir / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.read_bytes() != src.read_bytes():
                shutil.copy2(src, dst); n_copy += 1
            imgs[k] = rel
    rows["image"] = imgs
    rows["hai_luot"] = [f"{d}:{c}" for d, c in zip(doc, cls)]
    rows["hai_luot_truoc"] = ""
    U = rows.reset_index(drop=True)
    cols = list(P.columns.drop("_key")) + [c for c in TRACE if c not in P.columns]
    U = U[cols]
    # thứ tự hàng: như bản chính (sách, trang, cột, chỉ số trong cột) — sắp theo khoá số
    U["_s"] = list(zip(U.book, U.page, pd.to_numeric(U.column, errors="coerce").fillna(-1).astype(int),
                       pd.to_numeric(U.syl_idx, errors="coerce").fillna(-1).astype(int)))
    U = U.sort_values("_s", kind="stable").drop(columns="_s").reset_index(drop=True)
    side = pd.DataFrame(dict(key=keys, doc=doc, l1=l1, l2=l2, syl=syl, cls=cls, gP=gP.astype(int), gA=gA.astype(int)))
    side.to_csv(work_dir / SIDE, index=False)
    U.to_csv(work_dir / UNION_SNAP, index=False)
    tmp = primary_dir / "labels_final.csv.tmp"
    U.to_csv(tmp, index=False); tmp.replace(primary_dir / "labels_final.csv")
    rep = {}
    for book in sorted(set(U.book)):
        m = U.book.values == book
        km = np.array([k.split("/")[0] == book for k in keys])
        rep[book] = dict(o=int(m.sum()), chi_ban_chinh=int((km & inP & ~inA).sum()), chi_ban_phu=int((km & inA & ~inP).sum()),
                         gold_chinh=int((km & gP).sum()), gold_phu=int((km & gA).sum()), gold_hop=int((km & (gP | gA)).sum()),
                         lay_ban_phu_gold=int((km & gA & ~gP).sum()),
                         xung_dot_hai_ban_gold_khac_nhan=int((km & gP & gA & (Pi.label.reindex(keys).fillna("").values
                                                                                 != Ai.label.reindex(keys).fillna("").values)).sum()),
                         lop_gold_hop={c: int(v) for c, v in pd.Series(cls[km & (gP | gA)]).value_counts().items()})
    rep["_crop_chep_tu_ban_phu"] = n_copy
    log(f"[hai_luot] hợp theo ô: {json.dumps(rep, ensure_ascii=False)}")
    return rep


# ------------------------------------------------------------------------------------------------ cổng
def _iou(a, b):
    try:
        a, b = json.loads(a), json.loads(b)
        if not a or not b:
            return np.nan
    except Exception:  # noqa: BLE001
        return np.nan
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    return I / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I)


def _syl_boxes(d: Path) -> dict:
    f = Path(d) / "boxes_syl.csv"
    if not f.exists():
        return {}
    B = rd(f)
    return dict(zip(B.book + "/" + B.page + "/" + B.column + "/" + B.syl_idx, B.bbox_syl))


def gate(labels: Path, work_dir: Path, primary_dir: Path, aux_dir: Path, mode: str = "strict", vdp_gate: bool = True,
         log=print) -> dict:
    """Cổng sau rescue: ghi đè `labels` (labels_final.csv). Trả báo cáo (cũng ghi primary_dir/hai_luot_report.json)."""
    if mode not in GATES:
        raise ValueError(mode)
    L = rd(labels)
    S = rd(Path(work_dir) / SIDE).set_index("key")
    U = rd(Path(work_dir) / UNION_SNAP)
    L["_key"], U["_key"] = keys_of(L), keys_of(U)
    Ui = U.set_index("_key")
    for c in TRACE + ("tier_goc", "rule_goc"):
        if c not in L.columns:
            L[c] = ""
    R_of, veq = _lex()
    k = L["_key"].values
    l1 = S.l1.reindex(k).fillna("").values; l2 = S.l2.reindex(k).fillna("").values
    syl = S.syl.reindex(k).fillna("").values; doc = S.doc.reindex(k).fillna("P").values
    cls = classify(l1, l2, syl, L.label.values, L.rule.values, R_of, veq)
    gold = L.tier.values == "GOLD"
    rescue = gold & (cls == "rescue")
    resc_ok = np.array([bool(a) and bool(b) and veq(a, b) and veq(lb, a) and a in R_of(s)
                        for a, b, lb, s in zip(l1, l2, L.label.values, syl)])
    keep = np.zeros(len(L), bool)
    if mode == "strict":
        keep = gold & (np.isin(cls, KEEP_STRICT) | (rescue & resc_ok))
    else:
        keep = gold.copy()
    dem = gold & ~keep
    # vdp lệch legacy
    vdem = np.zeros(len(L), bool)
    if vdp_gate:
        SB = {"P": _syl_boxes(primary_dir), "A": _syl_boxes(aux_dir)}
        if not SB["P"] and not SB["A"]:
            raise SystemExit("[hai_luot] vdp_gate bật nhưng không có boxes_syl.csv ở bản dựng nào (build thiếu stt_hai_luot.vdp_gate?)")
        bs = L.box_source.values
        for i in np.nonzero(keep & np.isin(bs, WEAK_BOX))[0]:
            ref = SB.get(doc[i], {}).get(k[i])
            if ref is None:
                continue
            v = _iou(L.bbox.values[i], ref)
            if v == v and v < 0.5:
                vdem[i] = True
    keep &= ~vdem
    # ghi
    tr = L.tier.values.copy(); ru = L.rule.values.copy(); tg = L.tier_goc.values.copy(); rg = L.rule_goc.values.copy()
    hl = L.hai_luot.values.copy(); ht = L.hai_luot_truoc.values.copy()
    restore = {}
    for i in np.nonzero(dem | vdem)[0]:
        ht[i] = f"{tr[i]}|{L.label.values[i]}"
        why = (f"vdp_{L.box_source.values[i]}" if vdem[i] else f"ha_{cls[i]}")
        if rescue[i] and not resc_ok[i] and k[i] in Ui.index:          # trả lại hàng TRƯỚC rescue
            u = Ui.loc[k[i]]
            restore[i] = u
            tr[i], ru[i] = u.tier, u.rule + MARK + why
        else:
            if not tg[i]:
                tg[i], rg[i] = tr[i], ru[i]
            tr[i], ru[i] = "REVIEW", ru[i] + MARK + why
    for i in np.nonzero(keep)[0]:
        if cls[i] == "lt2only" and MARK not in ru[i]:
            ru[i] = ru[i] + MARK + "lt2"
        elif doc[i] == "A" and MARK not in ru[i]:
            ru[i] = ru[i] + MARK + "lt1"
    hl = np.array([f"{d}:{c}" for d, c in zip(doc, cls)], dtype=object)
    L["tier"], L["rule"], L["tier_goc"], L["rule_goc"] = tr, ru, tg, rg
    L["hai_luot"], L["hai_luot_truoc"] = hl, ht
    for i, u in restore.items():
        for c in ("label", "unicode", "image", "image_md5", "label_level", "label_canonical"):
            if c in L.columns and c in u.index:
                L.at[i, c] = u[c]
        L.at[i, "tier_goc"], L.at[i, "rule_goc"] = "", ""
    # AE-1 trên đầu ra: hai ô DÙNG ĐƯỢC (GOLD/SILVER/SYLLABLE có ảnh) chung MỘT hộp cùng cột = ít nhất một ô sai ảnh.
    # visual_dp (ô ảo/dự phòng) và việc hợp hai bản dựng có thể sinh trùng hộp mà census của từng bản không thấy -> hạ CẢ
    # nhóm về REVIEW (không biết ô nào đúng; ưu tiên độ đúng). Đóng lại bất biến dup_bbox == 0 của ground_truth/suspicion.
    use = L.tier.isin(["GOLD", "SILVER", "SYLLABLE"]).values & (L.image.astype(str).str.len() > 0).values
    bb = L.bbox.astype(str).str.replace(" ", "", regex=False)
    gk = (L.book + "/" + L.page + "/" + L.column.astype(str) + "/" + bb).where(use, "")
    cnt = gk.map(gk[use].value_counts()).fillna(0).values
    dup = use & (cnt > 1)
    for i in np.nonzero(dup)[0]:
        if not L.at[i, "hai_luot_truoc"]:
            L.at[i, "hai_luot_truoc"] = f"{L.at[i, 'tier']}|{L.at[i, 'label']}"
        if not L.at[i, "tier_goc"]:
            L.at[i, "tier_goc"], L.at[i, "rule_goc"] = L.at[i, "tier"], L.at[i, "rule"]
        L.at[i, "tier"] = "REVIEW"
        L.at[i, "rule"] = L.at[i, "rule"] + MARK + "trung_hop"
    L = L.drop(columns="_key")
    tmp = Path(str(labels) + ".tmp")
    L.to_csv(tmp, index=False); tmp.replace(labels)
    rep = dict(mode=mode, vdp_gate=vdp_gate, per_book={})
    for book in sorted(set(L.book)):
        m = L.book.values == book
        rep["per_book"][book] = dict(
            o=int(m.sum()), gold_truoc=int((m & gold).sum()), gold_sau=int((m & (L.tier.values == "GOLD")).sum()),
            ty_le_gold_sau=round(float((m & (L.tier.values == "GOLD")).sum() / max(1, m.sum())), 4),
            ha_theo_lop={c: int(v) for c, v in pd.Series(cls[m & dem]).value_counts().items()},
            rescue_tra_lai=int((m & rescue & ~resc_ok).sum()), rescue_giu=int((m & rescue & resc_ok).sum()),
            ha_vdp=int((m & vdem).sum()), ha_trung_hop=int((m & dup & gold).sum()),
            ha_trung_hop_syllable=int((m & dup & ~gold).sum()),
            lop_gold_sau={c: int(v) for c, v in pd.Series(cls[m & (L.tier.values == "GOLD")]).value_counts().items()})
    (Path(primary_dir) / "hai_luot_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[hai_luot] cổng {mode} + vdp {vdp_gate}: {json.dumps(rep['per_book'], ensure_ascii=False)}")
    return rep


def write_csv(path: Path, rows: list, fields: list):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
