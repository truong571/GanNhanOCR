#!/usr/bin/env python3
"""gold_exact_eval.py — nghiệm thu BẢN GIAO của bước gold_exact (dataset/_ALL/gold_exact.csv), 0 API, CPU, ≈ 30 s.

Đọc: dataset/_ALL/{gold_exact.csv, labels.csv, CHECKSUMS.txt, crops_chuan*/}, dataset/<Bộ>/gold_exact.csv, config/gold_exact.yaml,
data/<IHR>/manifest.tsv + prepared/<IHR>/kim_raw (sự thật người qua harness đã port: pipeline/gold_exact/eval_ihr).
Ghi: measure_out/gold_exact/summary.json (≤ 8 KB, `invariants` PASS/FAIL).

Invariant (cứng, KHÔNG khoá số đếm): số dòng = số GOLD; mỗi ô đúng 1 trạng thái; ok ⊂ GOLD; không ô luật A
(self_training_rescue / s1_inter_s2_similar / văn bản yếu) nào ok — tính lại từ cột `rule` của labels.csv; ô ok có crop chuẩn
tồn tại + md5 khớp cột; labels.csv không đổi từ lúc gộp (sha256 == CHECKSUMS.txt); policy_version + config sha khớp
config/gold_exact.yaml; trạng thái tái lập được từ các cột cờ + config (policy.decide); bộ kiểm ảnh tái lập ĐÚNG từ điểm +
ngưỡng config (CSV ghi %.17g từ 28/09 -> dung sai 0; tệp cũ %.6g -> dung sai 1e-6); ok = lai − luật A − M-OCR; bản lọc theo bộ
khớp bản gộp VÀ khớp tập ô GOLD của dataset/<Bộ>/labels.csv (N3); cột `device` một giá trị.
MỀM (SKIP + cảnh báo, không FAIL — N4): bộ nguồn có labels.csv mà THIẾU dataset/<Bộ>/gold_exact.csv = bộ đó vừa được dựng lại
(`--book <Bộ>`: bước export xoá *.csv) sau lần gộp -> gợi ý `./run_pipeline.sh --merge`. Chọn cảnh báo thay vì để --book tự
gộp: gộp lại đụng dataset/_ALL của MỌI bộ + ≈ 5–20 phút, trong khi --book chỉ được phép đổi đúng một bộ.
ĐỘ CHÍNH XÁC trên nhãn người IHR (L16/TK, V1+/strict, CI bootstrap cụm trang) CHỈ BÁO — không làm cổng PASS/FAIL.

    .venv/bin/python scripts/measure/gold_exact_eval.py [--all-dir dataset/_ALL] [--out measure_out/gold_exact] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipeline.gold_exact import policy as POL  # noqa: E402
from pipeline.gold_exact.common import EVIDENCE, SETS8, load_cfg, load_gold, rd, sha256_file, to_float  # noqa: E402
from pipeline.gold_exact.per_book import gold_images  # noqa: E402
from pipeline.gold_exact.publish import checksums_labels_sha  # noqa: E402

# Luật A tính ĐỘC LẬP từ cột `rule` của labels.csv (chính sách v3): rescue, cầu tự dạng, văn bản yếu
RULE_A = {"self_training_rescue": "rescue", "s1_inter_s2_similar": "similar",
          "s1_inter_s2_direct_am_sua_dau": "weak_text", "s1_inter_s2_direct_lowp": "weak_text"}
BOOL_COLS = ["one_char_ok", "core_loss_flag", "f_blank", "f_cut", "f_two", "f_ink", "bleed_new", "trunc_new", "tall_new",
             "mocr", "simg", "H1_new", "lai", "AINT", "cnt", "bc", "dup_bbox", "ov_heavy", "int_foreign", "rescue", "similar",
             "weak_text"]
EPS_OLD = 1e-6      # CSV cũ ghi %.6g


def rule_class(rule: str) -> str:
    if rule in RULE_A:
        return RULE_A[rule]
    return "weak_text" if rule.startswith("corpus_reading") else ""


def md5f(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else ""


def checksums_line(all_dir: Path, name: str) -> str | None:
    f = all_dir / "CHECKSUMS.txt"
    if not f.exists():
        return None
    for ln in f.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) == 2 and p[1] == name:
            return p[0]
    return None


def shift_th(th: dict, d: float) -> dict:
    o = dict(th)
    for k in ("C5_T", "C5_L"):
        o[k] = [th[k][0] + d, th[k][1] + d]
    for k in ("C2_T", "C2_L"):
        o[k] = th[k] + d
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all-dir", default=str(REPO / "dataset/_ALL"))
    ap.add_argument("--root", default=None, help="thư mục các bộ nguồn (mặc định cha của --all-dir)")
    ap.add_argument("--out", default=str(REPO / "measure_out/gold_exact"))
    ap.add_argument("--config", default=str(REPO / "config/gold_exact.yaml"))
    ap.add_argument("--limit", type=int, default=0, help="chỉ kiểm md5 tệp của N ô ok đầu (chạy thử; invariant md5 = SKIP)")
    ap.add_argument("--no-ihr", action="store_true", help="bỏ phần đo trên nhãn người (chỉ báo)")
    a = ap.parse_args(argv)
    t0 = time.time()
    all_dir = Path(a.all_dir).resolve()
    root = Path(a.root).resolve() if a.root else all_dir.parent
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(a.config); cfg_sha = sha256_file(Path(a.config)); gate = POL.core_loss_gate(cfg)

    E = rd(all_dir / "gold_exact.csv")
    G = load_gold(all_dir)                                # GOLD của labels.csv (thứ tự labels.csv)
    L = rd(all_dir / "labels.csv", usecols=["cell_uid", "tier"])
    iv = []

    def add(name, expected, observed, ok, method=""):
        iv.append(dict(name=name, expected=expected, observed=observed, method=method,
                       **{"pass": None if ok is None else bool(ok)}))

    n_gold = len(G)
    add("so_dong_bang_so_GOLD", n_gold, len(E), len(E) == n_gold, "số dòng gold_exact.csv = số dòng tier GOLD của labels.csv")
    uniq = E.cell_uid.is_unique
    same = uniq and set(E.cell_uid) == set(G.cell_uid)
    add("cell_uid_duy_nhat_va_dung_tap_GOLD", True, dict(unique=bool(uniq), same_set=bool(same)), same,
        "khoá cell_uid duy nhất và trùng đúng tập ô GOLD")
    E = E.set_index("cell_uid").reindex(G.cell_uid).fillna("").reset_index()
    st = E.gold_exact.values
    okm = st == "ok"
    one = np.isin(st, POL.STATUSES)
    rs_ok = (E.reason.values[okm] == "").all() and (E.reason.values[~okm] != "").all()
    add("moi_o_dung_1_trang_thai", list(POL.STATUSES), dict(la=int((~one).sum()), reason_sai=int(not rs_ok)),
        bool(one.all()) and bool(rs_ok), "gold_exact ∈ 4 trạng thái; reason rỗng ⇔ ok")
    tier = dict(zip(L.cell_uid, L.tier))
    add("ok_tap_con_GOLD", 0, int(sum(tier.get(u) != "GOLD" for u in E.cell_uid[okm])),
        all(tier.get(u) == "GOLD" for u in E.cell_uid[okm]), "mọi ô ok là tier GOLD trong labels.csv")
    lab_eq = (E.label.values == G.label.values).all() and (E.image.values == G.image.values).all()
    add("nhan_va_image_goc_khop_labels", True, bool(lab_eq), lab_eq, "label + image (đường ảnh gốc) trùng labels.csv từng ô")

    # ---- luật A độc lập từ cột rule
    rc = np.array([rule_class(r) for r in G.rule.astype(str)], dtype=object)
    bad_ok = int((okm & (rc != "")).sum())
    add("khong_o_luat_A_nao_ok", 0, dict(ok_luat_A=bad_ok, ok_int_foreign=int((okm & (E.int_foreign.values == "1")).sum())),
        bad_ok == 0 and not (okm & (E.int_foreign.values == "1")).any(),
        "ô rule self_training_rescue / s1_inter_s2_similar / văn bản yếu (am_sua_dau, lowp, corpus_reading*) hoặc ảnh của ô khác: không ô nào ok")
    cols_eq = {k: bool(((E[k].values == "1") == (rc == k)).all()) for k in ("rescue", "similar", "weak_text")}
    add("co_luat_A_khop_cot_rule", True, cols_eq, all(cols_eq.values()), "cột rescue/similar/weak_text = phân loại lại từ rule")

    # ---- crop chuẩn của ô ok
    idx_ok = np.nonzero(okm)[0]
    chk = idx_ok[: a.limit] if a.limit else idx_ok
    miss = bad = 0
    bad_img = 0
    for i in chk:
        for c, m in (("crop_chuan", "crop_chuan_md5"), ("crop_chuan_128", "crop_chuan_128_md5")):
            f = all_dir / E[c].iat[i]
            if not E[c].iat[i] or not f.exists():
                miss += 1
            elif md5f(f) != E[m].iat[i]:
                bad += 1
        if md5f(all_dir / E.image.iat[i]) != E.image_file_md5.iat[i]:
            bad_img += 1
    part = bool(a.limit) and a.limit < len(idx_ok)
    add("o_ok_co_crop_chuan_ton_tai_md5_khop", 0, dict(kiem=int(len(chk)), thieu=miss, md5_sai=bad),
        None if part else (miss == 0 and bad == 0), "mọi ô ok: crops_chuan + crops_chuan_128 tồn tại, md5 == cột")
    add("o_ok_anh_goc_md5_khop", 0, bad_img, None if part else bad_img == 0, "md5 tệp ảnh gốc (image) == image_file_md5")
    nonok_path = int(((E.crop_chuan.values != "") & ~okm).sum() + ((E.crop_chuan_md5.values != "") & ~okm).sum())
    add("o_khong_ok_khong_co_crop", 0, nonok_path, nonok_path == 0, "ô không ok không mang đường/md5 crop chuẩn")
    n_png = {c: sum(1 for _ in (all_dir / c).rglob("*.png")) if (all_dir / c).exists() else 0
             for c in ("crops_chuan", "crops_chuan_128")}
    add("so_tep_crop_chuan_bang_so_ok", int(okm.sum()), n_png, all(v == int(okm.sum()) for v in n_png.values()),
        "không thừa/thiếu tệp crop chuẩn (bước gộp dọn bản cũ, bước publish chỉ chép ô ok)")
    oc = bool((E.one_char_ok.values[okm] == "1").all() and (E.crop_status.values[okm] == "ok").all())
    add("ok_la_mot_chu_crop_ok", True, oc, oc, "ô ok: one_char_ok = 1 và crop_status = ok")

    # ---- không đổi labels.csv; gold_exact.csv đúng bản đã publish
    sha_lab = sha256_file(all_dir / "labels.csv")
    rec = checksums_labels_sha(all_dir)
    add("labels_sha256_khong_doi_tu_luc_gop", rec, sha_lab, rec is not None and rec == sha_lab,
        "sha256 labels.csv == dòng labels.csv trong CHECKSUMS.txt (bước gộp ghi, publish giữ nguyên)")
    sha_ge = sha256_file(all_dir / "gold_exact.csv")
    add("gold_exact_csv_khop_CHECKSUMS", checksums_line(all_dir, "gold_exact.csv"), sha_ge,
        checksums_line(all_dir, "gold_exact.csv") == sha_ge, "gold_exact.csv không bị sửa sau khi publish")

    # ---- ngưỡng/config
    pv = sorted(set(E.policy_version)); cs = sorted(set(E.config_sha16))
    add("policy_version_khop_config", [str(cfg["version"])], pv, pv == [str(cfg["version"])], "policy_version mỗi dòng == config version")
    add("config_sha_khop", [cfg_sha[:16]], cs, cs == [cfg_sha[:16]], "config_sha16 mỗi dòng == sha256(config/gold_exact.yaml)[:16]")
    S8 = G.set8.values
    num = lambda c: to_float(E[c])          # đúng bit (pd.to_numeric có thể lệch 1 ulp với %.17g)
    sig = {k: (E[k].values == "1") for k in BOOL_COLS}
    sig["ta"] = E.ta.values.astype(object); sig["lobo_nh"] = num("lobo_nh")
    dec, why, M = POL.decide(S8, sig, cfg)
    n_dec = int((dec != st).sum()); n_why = int((why != E.reason.values).sum())
    add("trang_thai_tai_lap_tu_cot_va_config", 0, dict(trang_thai=n_dec, ly_do=n_why), n_dec == 0 and n_why == 0,
        "policy.decide trên các cột cờ của CSV + config hiện hành cho lại đúng gold_exact + reason từng ô")
    s = {k: num(k) for k in ("p_wood_T", "p_wood_L", "viss_T", "viss_L", "viss_X", "lobo_cert", "lobo_nh")}
    th = cfg["thresholds"]
    # CSV %.17g (từ 28/09) tái lập đúng từng bit -> dung sai 0; nhận ra tệp cũ %.6g qua độ dài chữ số của p_wood_T
    full = E.p_wood_T.str.len().max() > 12 if len(E) else True
    eps = 0.0 if full else EPS_OLD
    lo, hi = POL.simg(S8, s, shift_th(th, -eps)), POL.simg(S8, s, shift_th(th, +eps))
    simg = sig["simg"]
    n_out = int((simg & ~lo).sum() + (~simg & hi).sum())
    add("bo_kiem_anh_tai_lap_tu_diem_va_nguong_config", 0, dict(lech=n_out, dung_sai=eps), n_out == 0,
        "simg tái lập từ điểm + ngưỡng config (CSV %.17g: dung sai 0; CSV cũ %.6g: ±1e-6)")
    dv = sorted(set(E.device)) if "device" in E else ["(không có cột)"]
    add("thiet_bi_mot_gia_tri", "1 thiết bị (mps)", dv, len(dv) == 1 and dv[0] not in ("", "(không có cột)"),
        "cột device (thiết bị nhúng) có đúng một giá trị trong bản giao")
    CLF = sig["core_loss_flag"]
    dec_ok = POL.decomposition(okm, sig["lai"], sig["AINT"], sig["mocr"], CLF, gate)
    add("ok_bang_lai_tru_luatA_tru_MOCR", True, dec_ok, dec_ok,
        "ok = lai ∧ ¬luật A ∧ ¬M-OCR" + (" ∧ ¬core_loss (cổng bật)" if gate else " (core_loss KHÔNG là cổng)"))
    n_core_reason = int((E.reason.values == "core_loss_crop_chuan_cat_vao_chu").sum())
    add("core_loss_theo_config", "cổng" if gate else "0 ô lý do core_loss", n_core_reason,
        True if gate else n_core_reason == 0, "config core_loss.gate = false ⇒ không ô nào bị hạ vì core_loss")
    ev_bad = int((E.evidence_level.values != np.array([EVIDENCE.get(x, "suy_doan") for x in S8])).sum())
    add("evidence_level_khop_bo", 0, ev_bad, ev_bad == 0, "evidence_level theo bộ (L16/TK đo, KVK/L83 ước lượng, STT/Chr suy đoán)")

    # ---- bản lọc theo bộ (N3 + N4)
    pb, n3, thieu = {}, {}, []
    for bs in sorted(set(E.book_set)):
        d = root / bs
        if not (d / "labels.csv").exists():
            continue
        f = d / "gold_exact.csv"
        if not f.exists():
            thieu.append(bs); continue
        B = rd(f)
        Es = E[E.book_set == bs]
        pre = f"crops/{bs}/"
        okb = (len(B) == len(Es) and (B.cell_uid.values == Es.cell_uid.values).all()
               and (B.gold_exact.values == Es.gold_exact.values).all()
               and all(b == (e[len(pre):] if e.startswith(pre) else e) for b, e in zip(B.image, Es.image))
               and all((not e and not b) or (b.endswith("/" + e) and b.startswith("..")) for b, e in zip(B.crop_chuan, Es.crop_chuan)))
        pb[bs] = "ok" if okb else "lệch"
        gi = gold_images(d / "labels.csv")
        n3[bs] = "ok" if sorted(B.image) == gi else dict(gold_labels=len(gi), rows=len(B),
                                                        chi_labels=len(set(gi) - set(B.image)),
                                                        chi_gold_exact=len(set(B.image) - set(gi)))
    add("ban_theo_bo_khop_ban_gop", "mọi bộ có tệp: ok", pb, (all(v == "ok" for v in pb.values()) if pb else None),
        "dataset/<Bộ>/gold_exact.csv = bản gộp lọc theo book_set (image tương đối bộ, crop_chuan trỏ ../_ALL)")
    add("ban_theo_bo_trung_tap_GOLD_labels_bo", "mọi bộ có tệp: ok", n3, (all(v == "ok" for v in n3.values()) if n3 else None),
        "N3: tập ô GOLD (image) của dataset/<Bộ>/labels.csv == đúng các dòng dataset/<Bộ>/gold_exact.csv")
    add("ban_theo_bo_du_mat", [], thieu, None if thieu else True,
        "MỀM (N4): bộ nguồn thiếu gold_exact.csv (vừa --book <Bộ> sau lần gộp) -> SKIP + gợi ý ./run_pipeline.sh --merge")
    if thieu:
        print(f"CẢNH BÁO: thiếu gold_exact.csv theo bộ ở {', '.join(thieu)} (bộ đã dựng lại sau lần gộp) — "
              "chạy ./run_pipeline.sh --merge để gộp lại + sinh gold_exact")

    # ---- bảng theo bộ + độ chính xác IHR (CHỈ BÁO)
    table = []
    for s8 in SETS8:
        m = S8 == s8
        r = dict(bo=s8, evidence=EVIDENCE.get(s8), gold=int(m.sum()))
        r.update({k: int(((st == k) & m).sum()) for k in POL.STATUSES})
        table.append(r)
    totals = dict(gold=n_gold, **{k: int((st == k).sum()) for k in POL.STATUSES},
                  ok_core_loss_flag=int((okm & CLF).sum()), lai=int(sig["lai"].sum()))
    ihr = {}
    if not a.no_ihr:
        from pipeline.gold_exact import eval_ihr as EV
        from pipeline.gold_exact.common import Assets, set_lexicon
        set_lexicon(Assets())
        D = pd.DataFrame(dict(cell_uid=E.cell_uid, status=E.crop_status, ink_cx=num("ink_cx"), ink_cy=num("ink_cy")))
        H = EV.ihr_truth(G, D, cfg)
        keep = pd.Series(okm, index=E.cell_uid).loc[H.cell_uid].values
        R = EV.measure(keep, H, "slot_new", cfg["ihr_bootstrap"]["B"], cfg["ihr_bootstrap"]["seed"])
        for bk, r in R.items():
            if "both_pt" in r:
                ihr[bk] = dict(n_ok=r["n_keep"], n_eval=r["n_eval"], both_pt=r["both_pt"],
                               ci95=[round(r["both_lo"], 5), round(r["both_hi"], 5)], both_err=r["both_err"],
                               err_lab=r["both_err_lab"], err_img=r["both_err_img"], strict_pt=r["strict_pt"],
                               strict_ci95=[round(r["strict_lo"], 5), round(r["strict_hi"], 5)],
                               both_lo=r["both_lo"], both_hi=r["both_hi"])
    summ = dict(generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"), all_dir=str(all_dir), policy_version=str(cfg["version"]),
                config_sha16=cfg_sha[:16], core_loss_gate=gate, labels_sha256=sha_lab, limit=a.limit or None,
                partial_run=part, totals=totals, table=table, ihr=ihr,
                ihr_note="đúng hai vế = V1+(nhãn, chữ người) ∧ khe crop chuẩn = 1; CI bootstrap cụm trang — CHỈ BÁO, không là cổng",
                runtime_s=round(time.time() - t0, 1), invariants=iv)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    n_fail = sum(1 for x in iv if x["pass"] is False)
    print(f"gold_exact_eval: {n_gold:,} ô GOLD · ok {totals['ok']:,} · text_only {totals['text_only']:,} · "
          f"uncertified {totals['uncertified']:,} · review {totals['review']:,} · policy {cfg['version']} · {summ['runtime_s']} s")
    for bk, r in ihr.items():
        print(f"  IHR {bk}: ô ok có GT {r['n_eval']} · đúng hai vế V1+ {100*r['both_pt']:.2f} % "
              f"[{100*r['ci95'][0]:.2f}–{100*r['ci95'][1]:.2f}] · strict {100*r['strict_pt']:.2f} % (chỉ báo)")
    for x in iv:
        print(f"  {'SKIP' if x['pass'] is None else ('PASS' if x['pass'] else 'FAIL')} {x['name']}"
              + ("" if x["pass"] is not False else f"  kỳ vọng {x['expected']} quan sát {x['observed']}"))
    print(f"invariants PASS {sum(1 for x in iv if x['pass'])} FAIL {n_fail} -> {out / 'summary.json'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
