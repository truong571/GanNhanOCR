#!/usr/bin/env python3
"""borg_human_eval.py — nghiệm thu BẢN GIAO dataset/_BORG_NHAN_NGUOI/ (bộ crop NHÃN NGƯỜI Borg.tonch 18 + 34), 0 API, CPU, ≈ 30 s.

Đọc: dataset/_BORG_NHAN_NGUOI/{labels.csv, BUILD_INFO.json, CHECKSUMS.txt, crops/, crops_chuan/}, prepared/<Sách>/transcriptions
(chuỗi chữ người, đếm ĐỘC LẬP), dataset/_ALL/labels.csv (chỉ cột cell_uid — kiểm tách bạch). Ghi measure_out/borg_human/summary.json.

Invariant cứng (không khoá số đếm cố định — số kỳ vọng tính lại từ nguồn):
  số dòng = số chữ Lo của bản phiên (prepared) · khoá cell_uid và (book, page, idx) duy nhất, idx liền 0..N−1 mỗi trang ·
  char/syllable của từng dòng == chữ/âm người phiên tại vị trí idx (nhãn là của NGƯỜI, không phải máy) · keep_level hợp lệ ·
  luật keep TÍNH LẠI từ cột (kind, det_score, post_v1, post_v2, agree_v2, tỉ lệ bỏ chữ của trang) == tập keep ∪ keep_v5 ·
  keep_v5 == keep − (paddle_test lệch) − (chuẩn hoá '|paddle') ⇒ keep_v5 ⊆ keep ⊆ keep_high · mọi ô mức có ảnh đều có
  crops/ + crops_chuan/ tồn tại và md5 khớp cột; ô `khong` không có ảnh · θ trượt ±1 (Paddle) TÍNH LẠI từ cột paddle_test
  trên keep == BUILD_INFO và nằm trong khoảng đã biết của vòng 5 [0,94 %; 1,69 %] · không cột chia tập ·
  sha256 tệp gốc == CHECKSUMS.txt · không cell_uid nào lọt vào dataset/_ALL · tổng dung lượng ≤ 1 GB.
Mềm: tái lập ĐÚNG từng ô so với r4/r5 (BUILD_INFO.repro.exact_reproduction).

    .venv/bin/python scripts/measure/borg_human_eval.py [--dir dataset/_BORG_NHAN_NGUOI] [--out measure_out/borg_human] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BOOKS = ("SachKinhThayCaBinh", "SachDungLyHoThan")
LEVELS = ("keep_v5", "keep", "rong_995", "rong_95", "khong")   # TN10 30/09: + 2 mức mở rộng có ảnh
IMG_LEVELS = ("keep_v5", "keep", "rong_995", "rong_95")          # == params.CROP_LEVELS
THETA_KNOWN = (0.0094, 0.0169)       # r5 q06 'D|borg_keep|mid+delay' CI 95 % (θ = 1,32 %, n = 15.383)
KEEP_RULE = dict(det_min=0.2, page_skip_max=0.1, post_v1=0.999, post_v2=0.999, post_v2_high=0.99,
                 rong_995=0.995, rong_95=0.95)   # == params.KEEP (a12 + TN10 mức mở rộng)
MAX_BYTES = 1_000_000_000


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def human_chars(book: str, pn: str):
    """Đếm ĐỘC LẬP (không import pipeline.borg_human): chữ Lo của nom_clean theo câu + âm khi #chữ == #âm."""
    d = json.load(open(REPO / "prepared" / book / "transcriptions" / f"{pn}.json", encoding="utf-8"))
    ch, sy = [], []
    for s in d["sentences"]:
        cs = [c for c in s["nom_clean"] if unicodedata.category(c) == "Lo"]
        q = s.get("qn_syllables") or []
        ch += cs
        sy += (q if len(q) == len(cs) else [""] * len(cs))
    return ch, sy


def theta_est(P):
    P = np.asarray(P)
    n = len(P)
    c = {d: float((P == d).sum()) / n for d in range(-3, 4)}
    far = np.mean([c[-2], c[2], c[-3], c[3]])
    return (c[-1] + c[1] - 2 * far) / max((1 - 6 * far) - far, 1e-6)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(REPO / "dataset" / "_BORG_NHAN_NGUOI"))
    ap.add_argument("--out", default=str(REPO / "measure_out" / "borg_human"))
    ap.add_argument("--limit", type=int, default=0, help="chỉ kiểm md5 N ảnh đầu (invariant md5 = SKIP)")
    ap.add_argument("--book", default="", help="(bỏ qua — bộ luôn gồm 2 sách; giữ cho giao diện measure.py)")
    ap.add_argument("--workers", type=int, default=1, help="(không dùng)")
    a = ap.parse_args(argv)
    t0 = time.time()
    D, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    iv = []

    def add(name, expected, observed, ok, method=""):
        iv.append(dict(name=name, expected=expected, observed=observed, method=method, **{"pass": None if ok is None else bool(ok)}))

    if not (D / "labels.csv").exists():
        add("co_ban_giao", True, False, False, f"{D}/labels.csv không có — chạy .venv/bin/python -m pipeline.borg_human --stage all")
        (out / "summary.json").write_text(json.dumps(dict(invariants=iv), ensure_ascii=False, indent=1), encoding="utf-8")
        print("FAIL co_ban_giao")
        return 1
    L = pd.read_csv(D / "labels.csv", dtype=str, keep_default_na=False)
    info = json.load(open(D / "BUILD_INFO.json", encoding="utf-8"))
    L["idx_i"] = L.idx.astype(int)

    # 1. số dòng + nhãn là chữ NGƯỜI (tính lại từ prepared)
    n_src, bad_char, bad_syl, n_pages_src = 0, 0, 0, 0
    G = {k: g for k, g in L.groupby(["book", "page"])}
    missing_pages = 0
    for b in BOOKS:
        man = json.load(open(REPO / "prepared" / b / "manifest.json", encoding="utf-8"))["pages"]
        for p in man:
            ch, sy = human_chars(b, p["page_name"])
            n_src += len(ch)
            n_pages_src += 1
            g = G.get((b, p["page_name"]))
            if g is None:
                missing_pages += 1
                bad_char += len(ch)
                continue
            gi = g.set_index("idx_i")
            for i, (c, s) in enumerate(zip(ch, sy)):
                if i not in gi.index or gi.at[i, "char"] != c:
                    bad_char += 1
                elif gi.at[i, "syllable"] != s:
                    bad_syl += 1
    add("so_dong_bang_so_chu_nguoi_phien", n_src, len(L), len(L) == n_src, "đếm chữ Lo nom_clean của prepared/<Sách>/transcriptions")
    add("moi_trang_co_mat", n_pages_src, int(L.groupby(["book", "page"]).ngroups), missing_pages == 0 and L.groupby(["book", "page"]).ngroups == n_pages_src,
        "641 trang = 529 (Borg.tonch.18) + 112 (Borg.tonch.34), theo manifest")
    add("nhan_la_chu_nguoi_phien_tai_idx", 0, dict(sai_chu=bad_char, sai_am=bad_syl), bad_char == 0 and bad_syl == 0,
        "char/syllable dòng (page, idx) == chữ/âm người tại vị trí idx của trang")

    # 2. khoá
    uq = L.cell_uid.is_unique and not L.duplicated(["book", "page", "idx_i"]).any()
    contig = bool(L.groupby(["book", "page"]).idx_i.apply(lambda s: sorted(s) == list(range(len(s)))).all())
    add("khoa_duy_nhat_idx_lien", True, dict(unique=bool(uq), contiguous=contig), uq and contig, "cell_uid & (book,page,idx) duy nhất; idx = 0..N−1")
    add("nguon_nhan_nguoi_phien", "nguoi_phien", sorted(L.nguon_nhan.unique().tolist()), set(L.nguon_nhan) == {"nguoi_phien"}, "")

    # 3. luật keep tính lại + lồng nhau
    lv_ok = set(L.keep_level) <= set(LEVELS)
    add("keep_level_hop_le", list(LEVELS), sorted(L.keep_level.unique().tolist()), lv_ok, "")
    f = lambda c: pd.to_numeric(L[c], errors="coerce").fillna(-1).to_numpy()  # noqa: E731
    ps = L.groupby(["book", "page"]).kind.transform(lambda x: (x == "skip").mean()).to_numpy()
    base = (L.kind.to_numpy() == "real") & (f("det_score") >= KEEP_RULE["det_min"]) & (ps < KEEP_RULE["page_skip_max"]) & \
        (f("agree_v2") == 1) & (f("post_v1") >= KEEP_RULE["post_v1"])
    K = base & (f("post_v2") >= KEEP_RULE["post_v2"])
    KH = base & (f("post_v2") >= KEEP_RULE["post_v2_high"])
    lv = L.keep_level.to_numpy()
    in_keep = np.isin(lv, ["keep_v5", "keep"])
    base0 = (L.kind.to_numpy() == "real") & (f("det_score") >= KEEP_RULE["det_min"]) & (ps < KEEP_RULE["page_skip_max"]) & \
        (f("agree_v2") == 1)
    mnp = np.minimum(f("post_v1"), f("post_v2"))
    R995 = K | (base0 & (mnp >= KEEP_RULE["rong_995"]))
    R95 = R995 | KH | (base0 & (mnp >= KEEP_RULE["rong_95"]))
    in_r995 = np.isin(lv, ["keep_v5", "keep", "rong_995"])
    in_high = np.isin(lv, ["keep_v5", "keep", "rong_995", "rong_95"])          # = rong_95 (tên biến giữ cho các khối sau)
    add("luat_keep_tinh_lai_trung", 0, dict(keep_lech=int((K != in_keep).sum()), rong_995_lech=int((R995 != in_r995).sum()),
                                            rong_95_lech=int((R95 != in_high).sum())),
        bool((K == in_keep).all() and (R995 == in_r995).all() and (R95 == in_high).all()),
        "keep: real ∧ det≥0,2 ∧ trang bỏ chữ<10 % ∧ cùng hộp v1/v2 ∧ post_v1≥0,999 ∧ post_v2≥0,999; rong_995/rong_95: keep ∪ "
        "(cùng điều kiện hộp ∧ min(post_v1, post_v2) ≥ 0,995 / ≥ 0,95; rong_95 ⊇ keep_high cũ)")
    lech = np.array([t.startswith("lech") for t in L.paddle_test])
    normp = np.array(["|paddle" in x for x in L.chuan_hoa_nguoi_phien])
    k5 = in_keep & ~lech & ~normp
    add("keep_v5_bang_keep_tru_paddle", 0, int((k5 != (lv == "keep_v5")).sum()), bool((k5 == (lv == "keep_v5")).all()),
        "keep_v5 = keep − paddle_test lệch − chuẩn hoá '|paddle' (⇒ keep_v5 ⊆ keep ⊆ rong_995 ⊆ rong_95)")
    cnt = {x: int((lv == x).sum()) for x in LEVELS}
    add("long_nhau_keep_v5_keep_rong", True, dict(keep_v5=cnt["keep_v5"], keep=int(in_keep.sum()), rong_995=int(in_r995.sum()),
                                                   rong_95=int(in_high.sum())),
        cnt["keep_v5"] <= in_keep.sum() <= in_r995.sum() <= in_high.sum() <= len(L), "")

    # 4. ảnh
    has_img = L.image != ""
    need = np.isin(lv, IMG_LEVELS)
    add("anh_dung_muc", 0, dict(thieu_anh=int((need & ~has_img.to_numpy()).sum()), khong_ma_co_anh=int((~need & has_img.to_numpy()).sum())),
        bool((need == has_img.to_numpy()).all()), "mọi ô rong_95 trở lên có crop; ô khong không có")
    todo = L[has_img | (L.crop_chuan != "")]
    if a.limit:
        todo = todo.head(a.limit)
    miss = bad = 0
    n = 0
    for c, m in (("image", "image_md5"), ("crop_chuan", "crop_chuan_md5")):
        for rel, h in zip(todo[c], todo[m]):
            if not rel:
                continue
            n += 1
            p = D / rel
            if not p.is_file():
                miss += 1
            elif hashlib.md5(p.read_bytes()).hexdigest() != h:
                bad += 1
    add("anh_ton_tai_md5_khop", 0, dict(n=n, thieu=miss, md5_sai=bad), None if a.limit else (miss == 0 and bad == 0),
        "crops/ + crops_chuan/: tệp tồn tại, md5 == cột")
    cc_miss = int((need & (L.crop_chuan == "").to_numpy()).sum())
    add("crop_chuan_da_so", "ghi nhận", dict(o_co_anh_khong_co_crop_chuan=cc_miss), True, "crop chuẩn v2 status ≠ ok -> rỗng (chỉ báo)")
    oc = L.one_char_ok.to_numpy()
    add("one_char_ok_chi_khi_co_crop_chuan", True, bool(((oc != "") <= need).all()), bool(((oc != "") <= need).all()), "")

    # 5. θ trượt ±1 (Paddle) tính lại từ paddle_test
    kk = L[in_keep & (L.paddle_test != "").to_numpy()]
    offs = np.array([0 if t == "ok" else int(t[4:]) for t in kk.paddle_test])
    grp = (kk.book + "/" + kk.page).to_numpy()
    th = theta_est(offs)
    rng = np.random.default_rng(0)
    ks = pd.Series(np.arange(len(grp))).groupby(grp).apply(lambda s: s.to_numpy()).tolist()
    boot = [theta_est(offs[np.concatenate([ks[i] for i in rng.integers(0, len(ks), len(ks))])]) for _ in range(300)]
    ci = [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])]
    bi = info["theta"]["paddle"]["keep"]
    add("theta_paddle_tinh_lai_bang_BUILD_INFO", dict(n=bi["n"], theta=bi["theta"]), dict(n=int(len(offs)), theta=round(float(th), 4)),
        len(offs) == bi["n"] and abs(th - bi["theta"]) < 1e-4, "θ = (P(±1) − 2·far)/(1 − 7·far) trên ô keep thử được (cột paddle_test)")
    add("theta_trong_khoang_da_biet", list(THETA_KNOWN), dict(theta=round(float(th), 4), ci95=ci),
        THETA_KNOWN[0] <= th <= THETA_KNOWN[1], "θ(keep) nằm trong CI 95 % của vòng 5 (q06 D|borg_keep|mid+delay)")

    # 6. KHÔNG chia tập (A-10), checksums, tách khỏi _ALL, dung lượng
    chia = sorted({"split", "split_hint", "lobo_group"} & set(L.columns))
    add("khong_cot_chia_tap", [], chia, not chia, "quyết định A-10 (16/09): bộ giao không có cột train/val/test")
    ck = {}
    for line in (D / "CHECKSUMS.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            h, name = line.split(None, 1)
            ck[name.strip()] = h
    top = sorted(p.name for p in D.glob("*") if p.is_file() and p.name != "CHECKSUMS.txt")
    ck_bad = [n for n in top if ck.get(n) != sha256_file(D / n)]
    add("checksums_khop", [], ck_bad, not ck_bad, "sha256 tệp gốc == CHECKSUMS.txt")
    allf = REPO / "dataset" / "_ALL" / "labels.csv"
    if allf.exists():
        au = pd.read_csv(allf, dtype=str, keep_default_na=False, usecols=["cell_uid"]).cell_uid
        leak = int(au.str.startswith("BORG/").sum() + au.isin(set(L.cell_uid)).sum())
        add("tach_khoi_ALL", 0, leak, leak == 0, "không cell_uid nào của bộ này trong dataset/_ALL/labels.csv")
    size = sum(p.stat().st_size for p in D.rglob("*") if p.is_file())
    add("dung_luong_toi_da_1GB", f"≤ {MAX_BYTES}", size, size <= MAX_BYTES, "tổng byte mọi tệp của bản giao")
    rep = info.get("repro", {})
    add("tai_lap_r4_r5", True, dict(exact=rep.get("exact_reproduction"), keep=rep.get("keep"), keep_v5=rep.get("keep_v5")),
        bool(rep.get("exact_reproduction")), "MỀM: keep/keep_high/keep_paddle_ok/keep_v5 trùng từng ô r4/r5 và θ trùng q06")

    by_book = {b: {x: int(((L.book == b) & (L.keep_level == x)).sum()) for x in LEVELS} for b in BOOKS}
    summ = dict(generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"), dir=str(D.relative_to(REPO)) if D.is_relative_to(REPO) else str(D),
                limit=a.limit or None, n_cells=len(L), level_exclusive=by_book, level_total=cnt,
                cumulative=dict(keep_v5=cnt["keep_v5"], keep=int(in_keep.sum()), rong_995=int(in_r995.sum()), rong_95=int(in_high.sum())),
                with_image=int(has_img.sum()), one_char_ok=int((oc == "1").sum()),
                theta_paddle_keep=dict(n=int(len(offs)), theta=round(float(th), 4), ci95=ci),
                theta_encoder=info["theta"].get("encoder", {}), size_mb=round(size / 1e6, 1),
                build_created=info.get("created"), code_sha256=info.get("code_sha256", "")[:16],
                runtime_s=round(time.time() - t0, 1), invariants=iv)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    n_fail = sum(1 for x in iv if x["pass"] is False and x["name"] != "tai_lap_r4_r5")
    print(f"borg_human_eval: {len(L):,} ô · keep_v5 {cnt['keep_v5']:,} · keep {int(in_keep.sum()):,} · rong_995 {int(in_r995.sum()):,} · "
          f"rong_95 {int(in_high.sum()):,} · "
          f"θ(keep) {100 * th:.2f} % [{100 * ci[0]:.2f}–{100 * ci[1]:.2f}] · {size / 1e6:.0f} MB · {summ['runtime_s']} s")
    for x in iv:
        print(f"  {'SKIP' if x['pass'] is None else ('PASS' if x['pass'] else 'FAIL')} {x['name']}"
              + ("" if x["pass"] is not False else f"  kỳ vọng {x['expected']} quan sát {x['observed']}"))
    print(f"invariants PASS {sum(1 for x in iv if x['pass'])} FAIL {n_fail} -> {out / 'summary.json'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
