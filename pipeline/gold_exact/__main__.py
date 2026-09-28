"""CLI gold_exact — đánh giá tự động mọi ô GOLD (ảnh + chữ), cấu hình "lai" + luật A/M-OCR (core_loss = thông tin). 0 API.

    .venv/bin/python -m pipeline.gold_exact --all-dir dataset/_ALL --out measure_out/_gold_exact \\
        [--publish] [--device mps] [--cache-dir prepared/_gold_exact] [--recompute] [--workers 4] [--selftest]
        [--ref-base <base_v2.pkl> --ref-masks <masks_new.pkl> --ref-tn4 <gold_tn4.pkl> --ref-cov-core <cov_core.pkl>]

Ra (<out>/, thư mục làm việc — không được nằm trong dataset/): gold_exact.csv (1 dòng/ô GOLD, cùng định dạng bản giao),
GOLD_EXACT.md (bảng theo bộ), summary.json (số + invariant), gold_exact_full.pkl, ihr_cells.csv, reasons.csv;
không --publish: crops_chuan/ + crops_chuan_128/ của ô ok vào <out>/ (trừ --no-crops).
--publish (bước của run_pipeline.sh sau merge): crops_chuan{,_128}/ + gold_exact.csv + GOLD_EXACT.md vào <all-dir>, bản lọc
theo bộ vào <dataset-root>/<Bộ>/gold_exact.csv, sinh lại <all-dir>/CHECKSUMS.txt — KHÔNG ghi labels.csv (kiểm sha256 trước/sau
và so với CHECKSUMS.txt của bước gộp). Không ghi pipeline/align_engine/, core/, data/.
Chốt (rà soát 28/09): kiểm sha256 TOÀN BỘ MANIFEST ngay khi khởi động (N1); thiếu/rỗng ảnh của bất kỳ ô GOLD nào hoặc trang
không dựng được crop chuẩn -> dừng (N5); thiết bị mặc định = config `device` (mps), ghi cột `device`; TA không nhìn GT IHR (N6);
publish dọn cả bản lọc theo bộ cũ (N2) và kiểm tập GOLD bộ nguồn == bản lọc (N3); crop ở đích đọc lại so md5.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import policy as POL
from .doc_text import GIOI_HAN
from . import signals_geom as SG
from . import signals_img as SI
from . import signals_text as ST
from .common import (BORG8, CONFIG, HAND_VARIANT, MODELS, REPO, SETS8, Assets, Log, is_inside, load_cfg, load_gold,
                     set_lexicon, sets_in, sha256_file, uid_path)

# Số ô H4 cấu hình "lai" của TN4 (lab/thu_nghiem_anh_chu/TN4_crop_chuan/KET_QUA.md §5, cột lai) — chỉ để đối chiếu trong báo cáo
TN4_LAI = {"stt2": 2455, "stt4": 1567, "stt11": 2760, "Chr": 325, "L83": 641, "KVK": 3647, "L16": 2760, "TK": 9747}
TN4_LAI_ACC = {"L16": "99,46 [99,09–99,74]", "TK": "99,87 [99,80–99,93]"}


def core_loss_record(cfg: dict) -> dict:
    """core_loss: bản đăng ký gốc (config core_loss.prereg, lượt run1) + sha mã hiện tại + cổng bật/tắt.
    Đổi mã core_loss_of -> code_sha256_now ≠ prereg -> invariant core_loss_def_eq_prereg = false (cột thông tin đổi nghĩa)."""
    cl = cfg.get("core_loss") or {}
    pre = cl.get("prereg") or {}
    now = hashlib.sha256(inspect.getsource(SG.core_loss_of).encode()).hexdigest()
    return dict(registered_at=pre.get("registered_at", "?"), code_sha256=pre.get("code_sha256", ""), code_sha256_now=now,
                same_def=now == pre.get("code_sha256"), gate=POL.core_loss_gate(cfg),
                band_pitch=cl.get("band_pitch"), max_loss=cl.get("max_loss"), definition=SG.core_loss_of.__doc__)


# Cột bản giao gold_exact.csv (đường dẫn tương đối <all-dir>; bool ghi 0/1). Bản lọc theo bộ: publish.per_book_frame.
CSV_COLS = ["cell_uid", "book_set", "book", "set8", "label", "gold_exact", "reason", "evidence_level", "policy_version",
            "config_sha16", "device", "image", "image_file_md5", "crop_chuan", "crop_chuan_md5", "crop_chuan_128", "crop_chuan_128_md5",
            "crop_status", "ink_cx", "ink_cy", "one_char_ok", "core_loss", "core_loss_flag", "crop_flags", "f_blank", "f_cut",
            "f_two", "f_ink", "bleed_new", "trunc_new", "tall_new", "ink_ratio", "p_wood_T", "p_wood_L", "viss_T", "viss_L",
            "viss_X", "lobo_pT", "lobo_pL", "lobo_nh", "lobo_cert", "vis_z", "vis_m_win_glyph", "ady", "adx", "m_hom",
            "mocr_cons", "mocr", "ta", "ta_sub", "simg", "H1_new", "lai", "AINT", "cnt", "bc", "dup_bbox", "ov_heavy",
            "int_foreign", "rescue", "similar", "weak_text"]


def csv_frame(X: pd.DataFrame, G: pd.DataFrame, md5_sq: dict, md5_128: dict, policy_version: str, cfg_sha: str,
              device: str = "") -> pd.DataFrame:
    P = X.copy()
    okm = (P.gold_exact == "ok").values
    rel = [uid_path(u) for u in P.cell_uid]
    P["policy_version"] = policy_version
    P["config_sha16"] = cfg_sha[:16]
    P["device"] = device
    P["image"] = G.image.values
    P["image_file_md5"] = G.img_md5.values
    P["crop_chuan"] = np.where(okm, ["crops_chuan/" + r for r in rel], "")
    P["crop_chuan_128"] = np.where(okm, ["crops_chuan_128/" + r for r in rel], "")
    P["crop_chuan_md5"] = [md5_sq.get(u, "") if o else "" for u, o in zip(P.cell_uid, okm)]
    P["crop_chuan_128_md5"] = [md5_128.get(u, "") if o else "" for u, o in zip(P.cell_uid, okm)]
    P["crop_status"] = P["status"].fillna("MISSING").values
    P = P[CSV_COLS].copy()
    for c in P.columns:
        if P[c].dtype == bool:
            P[c] = P[c].astype(np.int8)
    return P


def md5_tn4_verdict(res: dict) -> dict:
    """PASS/N/A của phép so md5 crop chuẩn ↔ TN4 v2. Bộ KHÔNG có tham chiếu TN4 (n = 0 — Borg B18/B34, không có trong TN4)
    là N/A: không làm PASS = False. PASS = mọi bộ CÓ tham chiếu đạt n ≥ 300 và md5 trùng hết; không bộ nào có -> None (N/A)."""
    per = {k: v for k, v in res.items() if isinstance(v, dict)}
    have = {k: v for k, v in per.items() if v["n"] > 0}
    out = dict(res)
    out["na"] = sorted(k for k in per if k not in have)
    out["PASS"] = None if not have else all(v["n"] >= 300 and v["eq"] == v["n"] for v in have.values())
    return out


def md5_tn4_text(r: dict) -> str:
    """Dòng log cho md5 ↔ TN4 v2: 'PASS (8 bộ, 400/400 mỗi bộ; N/A: B18, B34)' / 'FAIL (…)' / 'N/A (…)'."""
    if "skipped" in r:
        return f"N/A ({r['skipped']})"
    per = {k: v for k, v in r.items() if isinstance(v, dict) and v["n"] > 0}
    v = r.get("PASS")
    head = "N/A" if v is None else ("PASS" if v else "FAIL")
    bad = [f"{k} {x['eq']}/{x['n']}" for k, x in per.items() if not (x["n"] >= 300 and x["eq"] == x["n"])]
    ns = sorted({x["n"] for x in per.values()})
    body = f"{len(per)} bộ, {'/'.join(map(str, ns))} ô mỗi bộ" + (f"; lệch: {', '.join(bad)}" if bad else "")
    return f"{head} ({body}" + (f"; N/A không có tham chiếu TN4: {', '.join(r['na'])}" if r.get("na") else "") + ")"


def md5_check_tn4(D: pd.DataFrame, G: pd.DataFrame, per_set=400, seed=20260927) -> dict:
    """md5 byte của crop chuẩn (cache) == TN4/v2/crops trên mẫu ≥ 300 ô/bộ (bộ không có tham chiếu TN4 -> N/A)."""
    root = REPO / "measure_out/_thu_nghiem_anh_chu/TN4/v2/crops"
    if not root.exists():
        return dict(skipped="không có measure_out/_thu_nghiem_anh_chu/TN4/v2/crops", PASS=None)
    X = G[["cell_uid", "set8"]].merge(D[["cell_uid", "sq_md5"]], on="cell_uid", how="left")
    X = X[X.sq_md5.notna()]
    rng = np.random.default_rng(seed)
    res = {}
    for s, g in X.groupby("set8"):
        pick = g.iloc[rng.permutation(len(g))[:per_set]]
        n = eq = 0
        for u, m in zip(pick.cell_uid, pick.sq_md5):
            f = root / uid_path(u)
            if f.exists():
                n += 1; eq += hashlib.md5(f.read_bytes()).hexdigest() == m
        res[s] = dict(n=n, eq=eq)
    return md5_tn4_verdict(res)


def hand_thr_at(A, t: str, var: str, q: float) -> float:
    """Ngưỡng bộ kiểm viết tay (biến thể var, nhánh t ∈ T/L) ở mức q — khoá str(q) của thang hand_tables (sai khoá -> lỗi rõ)."""
    tb = A.load(f"hand_tables_{t}_{var}.pt")["thr"]
    if str(q) not in tb:
        raise SystemExit(f"hand_q = {q} không có trong thang hand_tables_{t}_{var}.pt ({sorted(tb, key=float)})")
    return float(tb[str(q)])


def profile_record(cfg: dict, S8) -> dict | None:
    """Profile handwriting đã áp (cho summary/GOLD_EXACT.md): tập bộ, cổng bỏ, cổng khe, q, n_min, bản đăng ký."""
    p = POL.profile_hw(cfg)
    if not p:
        return None
    hw = cfg["profiles"]["handwriting"]
    have = set(np.asarray(S8, dtype=object))
    return dict(p, sets_present=[s for s in p["sets"] if s in have], prereg=hw.get("prereg"), source=hw.get("source"),
                printed="mọi bộ khác: hành vi cũ từng ô")


def resolve_device(req: str | None, log=print) -> str:
    """--device > config `device` (mps) > tự chọn. Yêu cầu mps mà máy không có -> cpu + CẢNH BÁO (kết quả có thể lệch vài ô
    sát ngưỡng; thiết bị thật ghi vào cột `device`)."""
    import torch
    if req in (None, "", "auto"):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if req == "mps" and not torch.backends.mps.is_available():
        log("CẢNH BÁO: config/--device = mps nhưng máy không có MPS -> chạy cpu (số không bảo đảm trùng byte bản MPS)")
        return "cpu"
    return req


def coverage_invariants(G: pd.DataFrame, z_ta: dict, mrefs: dict, cfg: dict) -> dict:
    """Độ phủ tham chiếu dị bản (invariant MỀM): tỉ lệ ô GOLD có tham chiếu ≥ sàn config ref_coverage_min."""
    mins = cfg.get("ref_coverage_min") or {}
    ta = ST.ta_coverage(G, z_ta)
    out = {"ta": {}, "mocr": {}}
    for bs, r in ta.items():
        lo = (mins.get("ta") or {}).get(bs)
        out["ta"][bs] = dict(r, min=lo, PASS=None if lo is None else r["frac"] >= lo)
    ms = set(mrefs)
    for bs, lo in (mins.get("mocr") or {}).items():
        m = (G.book_set == bs).values
        if not m.any():
            continue
        k = int(G.cell_uid[m].isin(ms).sum()); n = int(m.sum())
        out["mocr"][bs] = dict(n=n, with_ref=k, frac=round(k / n, 5), min=lo, PASS=k / n >= lo)
    out["PASS"] = all(v["PASS"] is not False for d in (out["ta"], out["mocr"]) for v in d.values())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline.gold_exact")
    ap.add_argument("--all-dir", default=str(REPO / "dataset/_ALL"))
    ap.add_argument("--out", default=str(REPO / "measure_out/_gold_exact"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default=str(REPO / "prepared/_gold_exact"))
    ap.add_argument("--recompute", action="store_true", help="bỏ qua cache đặc trưng")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--models", default=str(MODELS))
    ap.add_argument("--selftest", action="store_true", help="chạy selftest rồi thoát")
    ap.add_argument("--no-crops", action="store_true", help="không chép crop chuẩn của ô ok ra --out")
    ap.add_argument("--publish", action="store_true",
                    help="ghi bản giao vào --all-dir (gold_exact.csv, GOLD_EXACT.md, crops_chuan{,_128}/) + <dataset-root>/<Bộ>/gold_exact.csv")
    ap.add_argument("--dataset-root", default=None, help="thư mục chứa các bộ nguồn (mặc định: cha của --all-dir)")
    ap.add_argument("--ref-base"); ap.add_argument("--ref-masks"); ap.add_argument("--ref-tn4"); ap.add_argument("--ref-cov-core")
    ap.add_argument("--ref-harness", help="cells_eval.csv của harness: kiểm gt_char/khe cũ của bản chép eval_ihr")
    a = ap.parse_args(argv)
    if a.selftest:
        from .selftest import run
        return run()
    from . import publish as PUB
    all_dir, out, cache = Path(a.all_dir).resolve(), Path(a.out).resolve(), Path(a.cache_dir).resolve()
    if is_inside(out, REPO / "dataset"):   # so theo thành phần đường dẫn: dataset_out/ KHÔNG bị chặn nhầm
        raise SystemExit("--out không được nằm trong dataset/ (thư mục làm việc; bản giao vào dataset/ đi bằng --publish).")
    if a.publish:
        PUB.check_labels(all_dir)          # fail fast: labels.csv phải còn nguyên như lúc merge
    log = Log()
    T = {}
    t0 = time.time()
    # N1: kiểm sha256 TOÀN BỘ MANIFEST (tài sản + tệp ngoài) NGAY khi khởi động — hỏng/thiếu -> dừng trước mọi phép tính
    A = Assets(Path(a.models))
    asset_check = A.verify_all()
    log(f"tài sản: {asset_check['n_files']} tệp + {asset_check['n_external']} tệp ngoài khớp MANIFEST ({asset_check['seconds']} s)")
    set_lexicon(A)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(a.config)
    cfg_sha = sha256_file(Path(a.config))
    policy_version = str(cfg["version"])
    dev = resolve_device(a.device or cfg.get("device"), log)
    workers = min(a.workers, int(cfg.get("workers_max", 4)))
    lab_sha_before = sha256_file(all_dir / "labels.csv")
    prereg = core_loss_record(cfg)
    log(f"policy {policy_version} · config sha {cfg_sha[:12]} · core_loss: cổng={prereg['gate']}, định nghĩa "
        f"{'= bản đăng ký ' + prereg['registered_at'] if prereg['same_def'] else 'ĐÃ ĐỔI so với bản đăng ký'}")
    G = load_gold(all_dir)
    log(f"GOLD {len(G)} ô; thiết bị {dev}; cache {cache}")
    prof = profile_record(cfg, G.set8.values)      # ValueError nếu config profile sai
    if prof:
        for var in ("Kinh", "DungLy"):
            hand_thr_at(A, "T", var, prof["hand_q"])
        log(f"profile handwriting: bộ {prof['sets_present']} · bỏ cổng {prof['drop']} · cổng khe {prof['slot']} · "
            f"hand_q {prof['hand_q']} · n_hum_min {prof['n_hum_min']}; bộ khác = printed (hành vi cũ)")
    # N5: thiếu/rỗng tệp ảnh của BẤT KỲ ô GOLD nào -> dừng (không bao giờ nhúng ảnh trắng thay thế)
    miss_img = SI.check_images([all_dir / p for p in G.image])
    if miss_img:
        raise SystemExit(f"N5: {len(miss_img)} ô GOLD thiếu/rỗng tệp ảnh (vd {miss_img[:3]}) — dựng lại bộ gộp "
                         "(./run_pipeline.sh --merge); không thay bằng ảnh trắng.")

    # ------------------------------------------------------------------ hình học: crop chuẩn + core_loss
    t = time.time()
    D = SG.run_crop_chuan(G, cache, workers, a.recompute, cfg["core_loss"]["band_pitch"], log)
    CF = SG.crop_flags(G, D, cfg["core_loss"]["max_loss"])
    st_crop = CF.status.fillna("MISSING").astype(str)
    no_page = int(st_crop.isin(["NO_PAGE", "MISSING"]).sum())
    if no_page:
        ex = list(G.cell_uid[st_crop.isin(["NO_PAGE", "MISSING"]).values][:3])
        raise SystemExit(f"N5: {no_page} ô GOLD không dựng được crop chuẩn vì thiếu trang ảnh (NO_PAGE/MISSING; vd {ex}).")
    dsk = D.set_index("cell_uid").reindex(G.cell_uid).src_kind.fillna("").values if "src_kind" in D else np.array([""] * len(G))
    crop_src_fallback = int((np.isin(G.book_set.values, list(SG.ORIGINAL)) & (dsk != "original")).sum())
    md5chk = md5_check_tn4(D, G)
    T["crop_chuan_s"] = round(time.time() - t)
    log(f"crop chuẩn xong; md5 ↔ TN4 v2: {md5_tn4_text(md5chk)}")
    t = time.time()
    present = sorted(set(G.book_set))
    raw = SG.load_raw(present)          # chỉ bộ có trong bộ gộp (vắng Borg -> y hệt 6 bộ cũ)
    raw = raw.drop_duplicates("key")
    geo = SG.column_geometry(raw)
    dup = G.cell_uid.map(geo.f_dup_bbox).fillna(0).astype(int).values == 1
    ovh = G.cell_uid.map(geo.f_ov_heavy).fillna(0).astype(int).values == 1
    cnt = SG.cnt_flag(G, all_dir)
    intf = SG.int_foreign(G, all_dir)
    kim_cfg = {k: v for k, v in cfg["kim_cfg"].items() if k in present}   # vắng Borg -> đúng tập cấu hình cũ (khoá cache giữ nguyên)
    KB = SG.kim_vs_box(G, raw, geo, kim_cfg, cache, workers, a.recompute, log)
    ady = np.abs(pd.to_numeric(KB.dy_kim, errors="coerce").values); adx = np.abs(pd.to_numeric(KB.dx_kim, errors="coerce").values)
    T["geom_other_s"] = round(time.time() - t)

    # ------------------------------------------------------------------ văn bản
    t = time.time()
    rules = ST.rule_flags(G)
    G["nb_prev"], G["nb_next"] = ST.neighbours(G, all_dir)
    old_cqf = ST.old_crop_flag(G, all_dir)
    z_ta = A.load("ta_refs.pt")
    ta, ta_sub = ST.text_attested(G, A, z_ta)
    mrefs = A.load("mocr_refs.pt")["refs"]
    cov = coverage_invariants(G, z_ta, mrefs, cfg)
    if not cov["PASS"]:
        log("CẢNH BÁO độ phủ tham chiếu dưới sàn config ref_coverage_min: "
            + json.dumps({k: {b: v['frac'] for b, v in d.items() if v['PASS'] is False} for k, d in cov.items() if k != 'PASS'},
                         ensure_ascii=False))
    T["text_s"] = round(time.time() - t)

    # ------------------------------------------------------------------ ảnh: bộ kiểm chấm CROP CŨ
    t = time.time()
    S = SI.Scorers(A, dev, cache / "emb", a.recompute, log)
    paths = [str(all_dir / p) for p in G.image]
    with __import__("concurrent.futures").futures.ThreadPoolExecutor(8) as ex:   # read_crop: thiếu tệp -> MissingImageError
        keys = list(ex.map(lambda p: SI.read_crop(p)[0], paths, chunksize=512))
    G["img_md5"] = keys
    # chữ viết tay: STT (Kinh) + Borg LOBO-sách (27/09: B34 -> Kinh, B18 -> DungLy; common.HAND_VARIANT)
    hand_masks = {var: np.isin(G.set8.values, [s for s, v in HAND_VARIANT.items() if v == var]) for var in ("Kinh", "DungLy")}
    E = S.embed_crops(keys, paths, need_enc=True, need_vft=True, need_hand=hand_masks)
    uz = A.load("vft_universe.pt"); uni = SI.Universe(uz["U"], uz["SIM5"])
    sig = {}
    sig.update(SI.score_vft(S, G, E, uni, log))
    borg = np.isin(G.set8.values, BORG8)
    if borg.any():   # nguyên mẫu verifier_ft chứa crop Borg (cả hai sách) -> không công bố p_wood cho Borg (không dùng ở cổng)
        for k in ("p_wood_T", "p_wood_L"):
            sig[k] = np.where(borg, np.nan, sig[k])
    for k in ("lobo_pT", "lobo_pL", "lobo_nh", "lobo_cert"):
        sig[k] = np.full(len(G), np.nan)
    hand_thr = {}
    qv = POL.hand_q_of(G.set8.values, cfg)          # 28/09: mức q theo profile handwriting (vắng profile -> stt_q như cũ)
    for var, hm in hand_masks.items():
        if not hm.any():
            continue
        Gs = G[hm].reset_index(drop=True)
        qs = sorted(set(qv[hm].tolist()))
        H = SI.score_hand(S, Gs, E, uni, qs[0], log, variant=var)
        hand_thr[var] = H["thr"]
        if len(qs) > 1:                             # bộ viết tay ngoài profile (q khác) cùng biến thể: chứng nhận lại theo q của ô
            qsub = qv[hm]
            for q in qs[1:]:
                tT, tL = (hand_thr_at(A, t, var, q) for t in "TL")
                m = qsub == q
                H["lobo_cert"][m] = ((H["lobo_pT"][m] >= tT) & (H["lobo_pL"][m] >= tL)).astype(np.int8)
        for k in ("lobo_pT", "lobo_pL", "lobo_nh", "lobo_cert"):
            sig[k][hm] = H[k]
    V = SI.score_vis(S, G, E["enc"], cfg["vis_calib"], log)
    sig.update(vis_m_win_glyph=V["vis_m_win_glyph"], vis_z=V["vis_z"])
    msets = [s for s, c in cfg["gate_v2"].items() if c.get("mocr")]
    MO = SI.score_mocr(S, G, E["enc"], mrefs, cfg["mocr_t50"], msets, log)
    sig.update(m_hom=MO["m_hom"], mocr_cons=MO["mocr_cons"])
    idx4 = np.nonzero(np.isin(G.book_set.values, list(SI.VISS_DIR)))[0]
    EB = S.embed_viewB(G, idx4)
    VS = SI.score_viss(S, G, E["enc"], EB, A.load("viss_params.pt"), log)
    sig.update(VS)
    T["img_s"] = round(time.time() - t)

    # ------------------------------------------------------------------ chính sách
    BC_T, BC_L = POL.bc_flags(ady, adx, sig["vis_z"], cfg)
    S8 = G.set8.values
    bc = np.where(S8 == "L16", BC_T, BC_L)
    mocr = ST.mocr_gate(G, sig["mocr_cons"], intf, rules, old_cqf, dup, ady, adx, sig["vis_z"], cfg["gate_v2"])
    sg = dict(int_foreign=intf, rescue=rules["rescue"], similar=rules["similar"], weak_text=rules["weak_text"], mocr=mocr,
              f_blank=CF.f_blank.values, f_cut=CF.f_cut.values, f_two=CF.f_two.values, f_ink=CF.f_ink.values,
              bleed_new=CF.bleed_new.values, trunc_new=CF.trunc_new.values, tall_new=CF.tall_new.values,
              one_char_ok=CF.one_char_ok.values, dup_bbox=dup, ov_heavy=ovh, cnt=cnt, bc=bc,
              core_loss_flag=CF.core_loss_flag.values, ta=ta, lobo_nh=sig["lobo_nh"], vis_z=sig["vis_z"])
    sg["simg"] = POL.simg(S8, sig, cfg["thresholds"], cfg)
    dec, why, M = POL.decide(S8, sg, cfg)

    X = pd.DataFrame(dict(cell_uid=G.cell_uid, book_set=G.book_set, book=G.book, set8=S8, label=G.label, gold_exact=dec,
                          reason=why, evidence_level=POL.evidence(S8)))
    X["crop_chuan"] = np.where(dec == "ok", ["crops_chuan/" + uid_path(u) for u in G.cell_uid], "")
    for k in ("one_char_ok", "core_loss", "core_loss_flag", "tall_new", "f_blank", "f_cut", "f_two", "f_ink", "bleed_new",
              "trunc_new", "ink_ratio", "sq_md5", "ink_cx", "ink_cy", "status"):
        X[k] = CF[k].values
    X["crop_flags"] = CF["flags"].fillna("").values
    for k in ("p_wood_T", "p_wood_L", "viss_T", "viss_L", "viss_X", "lobo_pT", "lobo_pL", "lobo_nh", "lobo_cert", "vis_z",
              "vis_m_win_glyph", "m_hom", "mocr_cons"):
        X[k] = sig[k]
    X["ady"], X["adx"] = ady, adx
    for k, v in (("int_foreign", intf), ("rescue", rules["rescue"]), ("similar", rules["similar"]), ("weak_text", rules["weak_text"]),
                 ("mocr", mocr), ("dup_bbox", dup), ("ov_heavy", ovh), ("cnt", cnt), ("bc", bc), ("ta", ta), ("ta_sub", ta_sub),
                 ("simg", sg["simg"])):
        X[k] = v
    for k in ("A0_new", "B0_new", "H1_new", "SIMG", "TA_OK", "lai", "AINT"):
        X[k] = M[k]
    X["img_md5"] = G.img_md5.values

    # ------------------------------------------------------------------ đầu ra (crop ô ok: cache -> đích, kiểm md5)
    t = time.time()
    ok_uids = list(X.cell_uid[X.gold_exact == "ok"])
    pub_root = (Path(a.dataset_root).resolve() if a.dataset_root else all_dir.parent) if a.publish else None
    pub_books = sorted(set(G.book_set))
    if a.publish:
        # xoá ĐÚNG đầu ra cũ của bước này trong <all-dir> + bản lọc theo bộ <root>/<Bộ>/gold_exact.csv (N2) trước khi ghi
        gone = PUB.clean(all_dir, pub_root, pub_books)
        log(f"publish: dọn đầu ra cũ: {', '.join(gone) or '(không có)'}")
        dests = [all_dir]
    else:
        dests = [] if a.no_crops else [out]
        for c in PUB.CROPS_DIRS:
            if dests and (out / c).exists():
                shutil.rmtree(out / c)
    md5_sq, md5_128 = PUB.export_crops(ok_uids, dict(zip(X.cell_uid, X.sq_md5.fillna(""))), cache / "crop_chuan/sq",
                                       cache / "crop_chuan/sq128", dests)
    n_copied = len(ok_uids) if dests else 0
    P = csv_frame(X, G, md5_sq, md5_128, policy_version, cfg_sha, dev)
    # thay hằng số cũ crop_md5_eq_cache_record=True: md5 cache == md5 lúc dựng (mọi ô ok) + đọc lại từng tệp đã chép ở đích
    rec_eq = all(md5_sq.get(u, "") == m and m for u, m in zip(X.cell_uid[X.gold_exact == "ok"], X.sq_md5[X.gold_exact == "ok"]))
    crop_verify = {str(d): PUB.verify_crops(P, d) for d in dests}
    PUB._write_csv(P, out / "gold_exact.csv")
    X.to_pickle(out / "gold_exact_full.pkl")
    T["write_s"] = round(time.time() - t)

    # ------------------------------------------------------------------ đo IHR + core_loss
    from . import eval_ihr as EV
    t = time.time()
    Ht = EV.ihr_truth(G, D, cfg)
    Hm = X[["cell_uid"]].merge(Ht, on="cell_uid", how="inner")
    Xi = X.set_index("cell_uid").loc[Hm.cell_uid]
    B, seed = cfg["ihr_bootstrap"]["B"], cfg["ihr_bootstrap"]["seed"]
    ihr = dict(ok_slot_new=EV.measure((Xi.gold_exact == "ok").values, Hm, "slot_new", B, seed),
               lai_slot_new=EV.measure(Xi.lai.values, Hm, "slot_new", B, seed),
               ok_slot_old=EV.measure((Xi.gold_exact == "ok").values, Hm, "slot_old", B, seed))
    cl = {}
    covf = Path(a.ref_cov_core) if a.ref_cov_core else REPO / "measure_out/_thu_nghiem_anh_chu/TN4/v2/cov_core.pkl"
    damaged = set()
    if covf.exists():
        try:
            R = pd.read_pickle(covf)
            damaged = set(R.key[R.core_drop])
        except Exception as e:  # noqa: BLE001 — tệp tham chiếu hỏng: chỉ ảnh hưởng bảng đo core_loss (thông tin)
            log(f"không đọc được {covf} ({type(e).__name__}) — bỏ cột 'damaged' TN4")
    both_ok = Hm.y_lab.values & (Hm.slot_new.values == "1")
    for bk, bs in (("L16", "LucVanTien1916"), ("TK", "TruyenKieu1872")):
        m = Hm.book_set.values == bs
        fl = Xi.core_loss_flag.values.astype(bool) & m
        dm = np.array([u in damaged for u in Hm.cell_uid]) & m
        lai_pre = Xi.lai.values & ~Xi.AINT.values & ~Xi.mocr.values.astype(bool)
        cl[bk] = dict(n_gold=int(m.sum()), flagged=int(fl.sum()), damaged_tn4=int(dm.sum()), caught=int((fl & dm).sum()),
                      flagged_correct_both=int((fl & both_ok).sum()), flagged_wrong=int((fl & ~both_ok & (Hm.gt_char.values != "")).sum()),
                      lai_lost=int((fl & lai_pre).sum()), lai_lost_correct=int((fl & lai_pre & both_ok).sum()),
                      lai_lost_wrong=int((fl & lai_pre & ~both_ok & (Hm.gt_char.values != "")).sum()),
                      damaged_in_lai=int((dm & lai_pre).sum()), damaged_in_lai_caught=int((dm & lai_pre & fl).sum()),
                      damaged_correct_both=int((dm & both_ok).sum()), damaged_in_lai_correct_both=int((dm & lai_pre & both_ok).sum()),
                      core_loss_median_damaged=float(np.nanmedian(Xi.core_loss.values[dm])) if dm.any() else None)
    Hm.assign(damaged_tn4=[u in damaged for u in Hm.cell_uid], gold_exact=Xi.gold_exact.values,
              core_loss=Xi.core_loss.values).to_csv(out / "ihr_cells.csv", index=False)
    T["eval_s"] = round(time.time() - t)

    # ------------------------------------------------------------------ đối chiếu tham chiếu (tuỳ chọn)
    cmp = None
    if a.ref_base:
        from .compare_ref import compare
        cmp = compare(X, a.ref_base, a.ref_masks, a.ref_tn4)
        if "_slot_new_ref" in cmp:
            sref = pd.Series(cmp.pop("_slot_new_ref"), index=X.cell_uid)
            cmp["slot_new_eq_tn4"] = float((sref.loc[Hm.cell_uid].values == Hm.slot_new.values).mean())
            lai_ref = pd.Series(cmp.pop("_lai_ref_array"), index=X.cell_uid)
            ihr["lai_tn4_mask_slot_new"] = EV.measure(lai_ref.loc[Hm.cell_uid].values, Hm, "slot_new", B, seed)
    if a.ref_harness:
        CE = pd.read_csv(a.ref_harness, dtype=str, keep_default_na=False, usecols=["cell_uid", "gt_char", "slot_ok"]).set_index("cell_uid")
        CE = CE.reindex(Hm.cell_uid)
        cmp = cmp or {}
        cmp["harness_gt_char_eq"] = float((CE.gt_char.fillna("").values == Hm.gt_char.values).mean())
        cmp["harness_slot_old_eq"] = float((CE.slot_ok.fillna("").values == Hm.slot_old.values).mean())

    # ------------------------------------------------------------------ bảng + invariant
    rows = []
    CLF = CF.core_loss_flag.values.astype(bool)
    gate = prereg["gate"]
    for s in sets_in(S8):
        m = S8 == s
        r = dict(bo=s, evidence=POL.evidence([s])[0], gold=int(m.sum()))
        for st in POL.STATUSES:
            r[st] = int(((dec == st) & m).sum())
        r["lai_tinh_lai"] = int((M["lai"] & m).sum()); r["lai_TN4"] = TN4_LAI.get(s, 0)   # Borg: không có mốc TN4 (0)
        r["lai_tru_luatA"] = int((M["lai"] & M["AINT"] & m).sum())
        r["lai_tru_MOCR"] = int((M["lai"] & ~M["AINT"] & mocr & m).sum())
        # core_loss: cổng (gate) -> số ô lai mất; không cổng -> số ô ok MANG cờ (thông tin, không trừ)
        cl_n = int((M["lai"] & ~M["AINT"] & ~mocr & CLF & m).sum())
        r["lai_tru_core_loss"] = cl_n if gate else 0
        r["ok_co_core_loss"] = 0 if gate else cl_n
        rows.append(r)
    TB = pd.DataFrame(rows)
    reasons = X[X.gold_exact != "ok"].groupby(["set8", "gold_exact", "reason"]).size().rename("n").reset_index()
    lab_sha_after = sha256_file(all_dir / "labels.csv")
    inv = dict(n_gold=int(len(G)), uid_unique=bool(X.cell_uid.is_unique), labels_sha256_unchanged=lab_sha_before == lab_sha_after,
               labels_sha256=lab_sha_after, policy_version=policy_version, config_sha256=cfg_sha,
               core_loss_gate=gate, core_loss_def_eq_prereg=prereg["same_def"], md5_crop_chuan_vs_TN4=md5chk,
               ok_implies_crop=bool((X.crop_chuan[X.gold_exact == "ok"] != "").all()),
               ok_implies_one_char_ok=bool(X.one_char_ok[X.gold_exact == "ok"].all()),
               ok_subset_lai=bool((~(X.gold_exact == "ok") | X.lai).all()),
               decomposition_ok=POL.decomposition(dec == "ok", M["lai"], M["AINT"], mocr, CLF, gate)
               and bool((TB.ok == TB.lai_tinh_lai - TB.lai_tru_luatA - TB.lai_tru_MOCR - TB.lai_tru_core_loss).all()),
               crops_copied=n_copied, crop_md5_cache_eq_build=bool(rec_eq), crop_md5_dest_verify=crop_verify,
               ctx_ocr_match=int(V["ctx_ok"]),
               kim_char_ok=float(pd.to_numeric(KB.kim_char_ok, errors="coerce").mean()),
               out_not_in_dataset=not is_inside(out, REPO / "dataset"),
               assets_manifest_verified=asset_check, device=dev,
               all_gold_images_present=True, crop_no_page=no_page, crop_src_original_fallback=crop_src_fallback,
               ta_refs_no_ihr_gt="has_gt" not in z_ta and "gt_char" not in z_ta,
               ref_coverage=cov, profile_handwriting=prof)
    pub_plan = None
    if a.publish:
        root = Path(a.dataset_root).resolve() if a.dataset_root else all_dir.parent
        _r = lambda p: str(p.relative_to(REPO)) if REPO in p.parents or p == REPO else str(p)
        pub_plan = dict(all_dir=_r(all_dir), root=_r(root), root_abs=str(root),
                        books=[bs for bs in pub_books if (root / bs / "labels.csv").exists()])
    T["total_s"] = round(time.time() - t0)
    reasons.to_csv(out / "reasons.csv", index=False)
    write_md(out / "GOLD_EXACT.md", TB, reasons, ihr, cl, inv, T, cmp, prereg, a, pub_plan)
    good = (inv["labels_sha256_unchanged"] and inv["uid_unique"] and inv["ok_implies_crop"] and inv["ok_subset_lai"]
            and inv["decomposition_ok"] and inv["crop_md5_cache_eq_build"] and inv["ta_refs_no_ihr_gt"]
            and all(v["PASS"] for v in crop_verify.values()))
    pub = None
    if a.publish and not good:
        PUB.clean(all_dir, pub_root, pub_books)   # invariant hỏng -> KHÔNG giao nửa vời: gỡ crop đã chép, không ghi csv
        log(f"publish: BỎ (invariant lõi FAIL) — đã gỡ đầu ra khỏi {all_dir} + bản lọc theo bộ")
        inv["published"] = False
    elif a.publish:
        try:
            pub = PUB.publish(P, all_dir, out / "GOLD_EXACT.md", Path(pub_plan["root_abs"]), log)
        except PUB.PublishError as e:
            PUB.clean(all_dir, pub_root, pub_books)
            log(f"publish: BỎ ({e}) — đã gỡ đầu ra khỏi {all_dir} + bản lọc theo bộ")
            inv["published"] = False; inv["publish_error"] = str(e)[:500]
            good = False
        else:
            inv["published"] = True
            inv["published_png"] = pub["n_png"]
            inv["published_png_eq_ok"] = all(v == pub["ok"] for v in pub["n_png"].values())
            inv["per_book_gold_set_eq_labels"] = all(v.get("gold_set_eq_labels") for v in pub["per_book"].values())
            good = good and inv["published_png_eq_ok"] and inv["per_book_gold_set_eq_labels"]
            log(f"publish: {pub['rows']} dòng, {pub['ok']} ô ok -> {all_dir}; theo bộ: "
                + ", ".join(f"{k} {v['rows']}" for k, v in pub["per_book"].items()))
    summ = dict(version=cfg["version"], policy_version=policy_version, config_sha256=cfg_sha, api_calls=0, device=dev, profiles=prof,
                cache_dir=str(cache), timings=T, table=rows, ihr=ihr, core_loss=dict(prereg=prereg, by_book=cl),
                invariants=inv, compare_ref=cmp, thresholds=cfg["thresholds"], hand_thr=hand_thr.get("Kinh"), publish=pub)
    if borg.any():
        summ["hand_thr_by_variant"] = hand_thr
        summ["hand_variant"] = {s: HAND_VARIANT[s] for s in BORG8 if (G.set8 == s).any()}
    json.dump(summ, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    log(f"xong: {out} ({T['total_s']} s)")
    print(TB[["bo", "gold", "ok", "text_only", "uncertified", "review", "lai_tinh_lai", "lai_TN4", "ok_co_core_loss"]]
          .to_string(index=False))
    return 0 if good else 1


def _pct(x):
    return f"{100 * x:.2f}".replace(".", ",")


def write_md(path, TB, reasons, ihr, cl, inv, T, cmp, prereg, a, pub_plan=None):
    gate = prereg["gate"]
    L = ["# GOLD_EXACT — đánh giá tự động mọi ô GOLD (ảnh + chữ)", "",
         "Sinh bởi `python -m pipeline.gold_exact` (0 API). Chính sách: cấu hình **\"lai\"** của TN4 (crop chuẩn v2 + cổng hình học "
         "H1 dùng cờ một chữ của crop chuẩn + bộ kiểm ảnh chấm CROP CŨ với ngưỡng TN1) + **luật A / M-OCR** (chính sách v3). "
         + ("**core_loss là cổng** (cấu hình tái lập lượt run1). " if gate else
            "**core_loss KHÔNG còn là cổng** (đăng ký trước, kết quả âm — §3; chỉ giữ cột thông tin). ")
         + "Chỉ gắn trạng thái; không sửa nhãn; không đổi `labels.csv`.", "",
         *((f"**Profile handwriting** (TN5, đăng ký trước {(inv['profile_handwriting'].get('prereg') or {}).get('registered_at', '?')}): "
            f"bộ {', '.join(inv['profile_handwriting']['sets_present']) or '—'} — bỏ cổng {', '.join(inv['profile_handwriting']['drop']) or '—'} "
            f"(không phân biệt trên chữ viết tay có nhãn người Borg), cổng khe `{inv['profile_handwriting']['slot']}`, bộ kiểm viết tay "
            f"q = {inv['profile_handwriting']['hand_q']}, ≥ {inv['profile_handwriting']['n_hum_min']} nguyên mẫu người. Bộ khác "
            "(in/khắc) = profile printed, hành vi cũ từng ô. Chi tiết: `docs/GOLD_CHINH_XAC_2026-09-27.md` §11.", "")
           if inv.get("profile_handwriting") else ()),
         f"policy_version **{inv['policy_version']}** · `config/gold_exact.yaml` sha256 `{inv['config_sha256'][:16]}…` · "
         f"labels.csv sha256 `{inv['labels_sha256'][:16]}…` (không đổi trong lượt chạy: {inv['labels_sha256_unchanged']}).", "",
         "Trạng thái (luật đầu tiên khớp thắng): `review` (luật A, M-OCR) → `text_only` (ảnh không đúng một chữ / trượt) → "
         "`uncertified` (không qua bộ kiểm ảnh↔chữ hoặc thiếu chứng dị bản người) → `ok` (ảnh giao = crop chuẩn v2, "
         "`crops_chuan/`; cột `image` gốc vẫn giữ trong labels.csv).", "",
         "> ⚠️ Mức chắc: độ chính xác ô ok chỉ **ĐO** được ở L16/TK (nhãn người IHR, §2); KVK/L83 **ước lượng** (dị bản người); "
         "STT/Chr **suy đoán** (không có sự thật). Số ô ok phụ thuộc ngưỡng TN1 (τ = 0,995) — đổi ngưỡng thì số đổi.",
         GIOI_HAN.rstrip("\n"), "",
         f"Thiết bị nhúng: **{inv.get('device', '?')}** (cột `device`; tất định trên MPS). TA (dị bản người) KHÔNG nhìn GT người IHR "
         f"(tài sản ta_refs không có cột GT: `ta_refs_no_ihr_gt` = {inv.get('ta_refs_no_ihr_gt')}); GT IHR chỉ dùng ở §2 (đo).", "",
         "## 1. Bảng theo bộ", "",
         "| Bộ | Mức chắc | GOLD | ok | text_only | uncertified | review | lai (tính lại) | lai TN4 | − luật A | − M-OCR | "
         + ("− core_loss |" if gate else "ok mang cờ core_loss (thông tin) |"),
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    last = "lai_tru_core_loss" if gate else "ok_co_core_loss"
    for r in TB.itertuples():
        lt = f"{r.lai_TN4:,}" if r.bo in TN4_LAI else "—"      # Borg (27/09): không có mốc TN4
        L.append(f"| {r.bo} | {r.evidence} | {r.gold:,} | **{r.ok:,}** | {r.text_only:,} | {r.uncertified:,} | {r.review:,} | "
                 f"{r.lai_tinh_lai:,} | {lt} | {r.lai_tru_luatA} | {r.lai_tru_MOCR} | {getattr(r, last)} |".replace(",", "."))
    tot = TB[["gold", "ok", "text_only", "uncertified", "review", "lai_tinh_lai", "lai_TN4", "lai_tru_luatA", "lai_tru_MOCR",
              last]].sum()
    L.append(f"| **Tổng** | | {tot.gold:,} | **{tot.ok:,}** | {tot.text_only:,} | {tot.uncertified:,} | {tot.review:,} | "
             f"{tot.lai_tinh_lai:,} | {tot.lai_TN4:,} | {tot.lai_tru_luatA} | {tot.lai_tru_MOCR} | {tot[last]} |".replace(",", "."))
    L += ["", ("ok = lai (tính lại) − (lai ∩ luật A) − (lai ∩ M-OCR, ngoài luật A) − (lai ∩ core_loss, ngoài hai luật trước). "
               if gate else "ok = lai (tính lại) − (lai ∩ luật A) − (lai ∩ M-OCR, ngoài luật A) — kiểm từng ô (`decomposition_ok`). ")
          + "Chênh 'lai tính lại' ↔ 'lai TN4' = sai khác tái lập tín hiệu (nhúng lại trên MPS).", "",
          "## 2. Độ chính xác hai vế của ô ok trên nhãn người IHR (CHẮC CHẮN THEO MÁY)", "",
          "Đúng hai vế = V1+(nhãn, chữ người) ∧ khe của CROP CHUẨN = 1; strict = nhãn trùng hẳn chữ người. CI 95 % bootstrap cụm trang.", "",
          "| Tập | Bộ | ô giữ | có GT | V1+ | CI | strict | CI | lỗi (nhãn/ảnh) |", "|---|---|---:|---:|---:|---|---:|---|---|"]
    for nm, lab in (("ok_slot_new", "ok (khe crop chuẩn)"), ("lai_slot_new", "lai tính lại (khe crop chuẩn)"),
                    ("lai_tn4_mask_slot_new", "lai mặt nạ TN4 (khe crop chuẩn)"), ("ok_slot_old", "ok (khe crop cũ, tham khảo)")):
        if nm not in ihr:
            continue
        for bk, r in ihr[nm].items():
            if "both_pt" not in r:
                continue
            L.append(f"| {lab} | {bk} | {r['n_keep']} | {r['n_eval']} | **{_pct(r['both_pt'])}** | [{_pct(r['both_lo'])}–{_pct(r['both_hi'])}] | "
                     f"{_pct(r['strict_pt'])} | [{_pct(r['strict_lo'])}–{_pct(r['strict_hi'])}] | {r['both_err']} ({r['both_err_lab']}/{r['both_err_img']}) |")
    L += ["", f"TN4 cột lai (KET_QUA §5): L16 {TN4_LAI_ACC['L16']}, TK {TN4_LAI_ACC['TK']} (V1+).", "",
          "## 3. core_loss — đăng ký trước, kết quả ÂM → " + ("VẪN là cổng (cấu hình run1)" if gate else "BỎ khỏi cổng"), "",
          f"Đăng ký gốc: {prereg['registered_at']} (lượt `measure_out/_gold_exact_run1`, sha mã `{prereg['code_sha256'][:12]}`; "
          f"mã hiện tại {'TRÙNG' if prereg['same_def'] else 'KHÁC — cột core_loss đã đổi nghĩa'}). Định nghĩa: mất > "
          f"{prereg['max_loss']} mực lõi (±{prereg['band_pitch']} bước quanh tâm hộp cũ × dải cột crop chuẩn).", "",
          "Quyết định 27/09 (không dò lại ngưỡng): trên nhãn người, chốt này chỉ làm MẤT ô đúng (lượt run1: 46 ô ok ngoài luật A/M-OCR, "
          "L16 20 + TK 26, **cả 46 đúng hai vế**, 0 ô sai) và chỉ bắt 6/61 ô 'damaged' TN4 của L16 (0/8 ô damaged nằm trong lai) "
          "→ lợi ích âm; bỏ khỏi cổng, giữ `core_loss`/`core_loss_flag` làm cột thông tin. Bảng dưới đo lại mỗi lượt:", "",
          "| Bộ | GOLD | ô bị cờ | ô 'damaged' TN4 | bắt được | cờ mà đúng hai vế | cờ mà sai | " +
          ("lai bị mất | lai mất mà đúng | lai mất mà sai" if gate else "ok mang cờ | … mà đúng | … mà sai") +
          " | damaged trong lai | … bắt được | damaged đúng hai vế |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for bk, r in cl.items():
        L.append(f"| {bk} | {r['n_gold']} | {r['flagged']} | {r['damaged_tn4']} | {r['caught']} | {r['flagged_correct_both']} | "
                 f"{r['flagged_wrong']} | {r['lai_lost']} | {r['lai_lost_correct']} | {r['lai_lost_wrong']} | {r['damaged_in_lai']} | "
                 f"{r['damaged_in_lai_caught']} | {r['damaged_correct_both']} |")
    L += ["", "'damaged' TN4 = ô mất > 25 % mực LÕI của khe người khi thay crop cũ bằng crop chuẩn (TN4 s02b cov_core, L16 61 + TK 2); "
          "'đúng hai vế' đo với khe của crop chuẩn.", "", "## 4. Invariant", ""]
    for k, v in inv.items():
        L.append(f"- `{k}` = {json.dumps(v, ensure_ascii=False, default=float)}")
    if cmp:
        L += ["", "### Đối chiếu tín hiệu với số tham chiếu (base_v2 / TN4 masks)", ""]
        for k, v in cmp.items():
            L.append(f"- `{k}`: {json.dumps(v, ensure_ascii=False, default=float)}")
    L += ["", "## 5. Lý do không ok (theo bộ)", "", "| Bộ | trạng thái | lý do | n |", "|---|---|---|---:|"]
    for r in reasons.itertuples():
        L.append(f"| {r.set8} | {r.gold_exact} | {r.reason} | {r.n} |")
    L += ["", "## 6. Thời gian (giây)", "", "- " + ", ".join(f"{k} {v}" for k, v in T.items()), "",
          "## 7. Lệnh", "", "```bash",
          f".venv/bin/python -m pipeline.gold_exact --all-dir {a.all_dir} --out {a.out} --cache-dir {a.cache_dir}"
          + (" --publish" if a.publish else ""),
          "./run_pipeline.sh --book all --yes      # cả 8 bộ + gộp + bước này + nghiệm thu (--gold-exact off để bỏ)",
          ".venv/bin/python scripts/measure/gold_exact_eval.py   # invariant của bản giao -> measure_out/gold_exact/summary.json",
          "```", ""]
    if pub_plan:
        bl = ", ".join(f"`{pub_plan['root']}/{b}/gold_exact.csv`" for b in pub_plan.get("books", [])) or "(không bộ nào)"
        L += ["## 8. Bản giao", "",
              f"`{pub_plan['all_dir']}`: `gold_exact.csv` (đường dẫn tương đối thư mục này), `GOLD_EXACT.md`, `crops_chuan/`, "
              f"`crops_chuan_128/` (chỉ ô ok). Bản lọc theo bộ (`image` tương đối thư mục bộ, `crop_chuan` trỏ `../_ALL/…`) ghi ở "
              f"ĐÚNG các bộ sau (bộ có labels.csv, tập ô GOLD trùng — N3): {bl}. Bộ được dựng lại sau lượt này (`--book <Bộ>`) "
              "mất bản lọc cho tới lần `./run_pipeline.sh --merge` kế. CHECKSUMS.txt của bộ gộp được sinh lại (dòng labels.csv "
              "giữ nguyên).", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
