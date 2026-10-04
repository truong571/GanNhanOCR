#!/usr/bin/env python3
"""Phân tích THĂM DÒ X1 (bản 2) theo ĐÚNG giao thức vision/prereg/thamdo_X1_cua_so_vision_L16.json (OFFLINE, 0 yêu cầu).

  .venv/bin/python vision/crop_test/analyze_explore_vwindow.py
Ghi: vision/ket_qua/crop_test/X1_ket_qua.json (không có p) và X1_BAO_CAO_THAM_DO.md.
Điểm cuối chính DUY NHẤT: D1 = Δtop1(vpipe) trên ô 'bị cắt'. Mọi KTC khác chỉ MÔ TẢ (thăm dò, cùng dữ liệu đã sinh ra giả thuyết)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_crop_test as A                                      # noqa: E402  (dùng lại boot/delta/fmt: bootstrap theo trang, B=5000, hạt giống 17)

REPO = HERE.parent.parent
OUT = HERE.parent / "ket_qua" / "crop_test"
VARS_CLIPPED = ["py20", "adapt", "vw", "vw_off", "vpipe", "vpipe_flip", "vpipe_shuf", "vpipe_own", "vbox"]
ALL_V = ["base"] + VARS_CLIPPED
GUARD_VARS = ["vw", "vpipe"]
THR_OV = 0.10             # ngưỡng tràn (tỉ lệ cạnh hộp Vision) để gọi 'tràn dọc/ngang'
THR_GUARD = -0.3          # điểm %: bảo vệ 'không hại'
N_E1 = 13117              # số ô GOLD L16 có nhãn người của E1 (mẫu số quy ra %)


def strip_p(o):
    """Bỏ mọi p-value khỏi JSON (kể cả kiểm dấu chính xác, vốn bỏ qua cụm trang)."""
    if isinstance(o, dict):
        return {k: strip_p(v) for k, v in o.items() if k != "p" and not str(k).startswith("dau_exact")}
    if isinstance(o, list):
        return [strip_p(v) for v in o]
    return o


def verify_freeze():
    """Giao thức đã đóng băng và 6 tệp mã nguồn không đổi kể từ lúc đóng băng? → (ok, thông tin)."""
    import hashlib
    pj = HERE.parent / "prereg" / "thamdo_X1_cua_so_vision_L16.json"
    sf = pj.with_suffix(".sha256")
    if not (pj.exists() and sf.exists()):
        return False, "CHƯA ĐÓNG BĂNG (thiếu .sha256)"
    want = sf.read_text().split()[0]
    got = hashlib.sha256(pj.read_bytes()).hexdigest()
    proto = json.loads(pj.read_text(encoding="utf-8"))
    src = proto.get("ma_nguon_sha256", {})
    bad = [rel for rel, h in src.items() if not (REPO / rel).exists() or hashlib.sha256((REPO / rel).read_bytes()).hexdigest() != h]
    return (want == got and not bad and len(src) >= 6), f"JSON {'khớp' if want == got else 'LỆCH'}; mã nguồn {len(src) - len(bad)}/{len(src)} khớp" + (f"; lệch: {bad}" if bad else "")


def paired(d, v, mask=None):
    """Tập cặp đầy đủ (cả t_base lẫn t_v có giá trị) — để top1 gốc/biến thể in ra cùng một tập ô."""
    m = d[f"t_{v}"].notna() & d["t_base"].notna()
    if mask is not None:
        m &= mask
    return d[m]


def dl(d, v, mask=None):
    return A.delta(paired(d, v, mask), v)


def main():
    lines, res = [], {}
    sfx = "_khoi" if "--khoi" in sys.argv else ""          # thử khói: đọc/ghi tệp *_khoi (KHÔNG diễn giải)

    def log(s=""):
        print(s)
        lines.append(s)
    sf = HERE.parent / "prereg" / "thamdo_X1_cua_so_vision_L16.sha256"
    reg = sf.read_text().split()[0][:16] if sf.exists() else "CHƯA ĐÓNG BĂNG"
    d = pd.read_csv(OUT / f"X1_L16_vwindow{sfx}.csv.gz", low_memory=False)
    d["cl"] = d["page"]
    d["cl2"] = d["page"].astype(str) + "|" + d["column"].astype(str)
    clipped = d["clipped"].astype(bool)
    inv = json.loads((OUT / f"X1_invariants{sfx}.json").read_text(encoding="utf-8"))

    log(f"# THĂM DÒ X1 — crop theo hộp Vision với vùng xoá nét láng giềng khớp (L16) — giao thức sha256 {reg}…\n")
    log("> **THĂM DÒ, không phải kiểm định xác nhận.** Giả thuyết sinh ra SAU khi thấy E1/E2 trên chính các ô này (forking paths). Chỉ D1 là điểm cuối chính; hơn 60 KTC khác được in KHÔNG hiệu chỉnh đa so sánh và chỉ để MÔ TẢ. "
        "Không dùng các từ: 'đã xác nhận', 'nguyên nhân của việc E1 giảm', 'pipeline nên áp dụng', 'an toàn áp cho mọi ô', 'khái quát sang sách khác/ô Vision không đọc', 'hộp Vision đúng'. "
        "'lo > 0' nghĩa là cận dưới của KTC 95 % hai phía, tức cận một phía 97,5 %.\n")

    # ── bất biến TRƯỚC TIÊN
    checks = []
    is_full = inv["n_bi_cat"] == 601 and inv["n_bi_cat"] + inv["n_con_lai"] == 2978
    checks.append(("quần thể khớp E1: 601 ô bị cắt + 2.377 ô còn lại = 2.978 Vision khớp", is_full, f"{inv['n_bi_cat']} + {inv['n_con_lai']}"))
    checks.append(("số ô GOLD L16 có nhãn người = 13.117 (mẫu số quy %)", sfx != "" or inv.get("n_eval") == N_E1, f"{inv.get('n_eval')}"))
    if not sfx:
        ok_fr, info_fr = verify_freeze()
        checks.append(("giao thức đóng băng: mã băm JSON và các tệp mã nguồn khớp lúc đăng ký", ok_fr, info_fr))
    checks.append(("tìm thấy hộp ký hiệu Vision của chính ô (khớp hộp, 0 thất bại)", inv["own_khong_tim_thay"] == 0, f"thất bại {inv['own_khong_tim_thay']}"))
    checks.append(("trang đã xử lý và pages_denoised cùng kích thước (hệ toạ độ Vision = hệ crop)", inv["trang_lech_kich_thuoc"] == 0, f"lệch {inv['trang_lech_kich_thuoc']}/{inv['trang_kiem_tra']}"))
    checks.append(("cửa sổ THẬT của vw / vpipe / vbox đều chứa hộp V (số nguyên, kẹp biên trang)", inv["cua_so_chua_hop_V"] == 0, f"chưa chứa {inv['cua_so_chua_hop_V']}/{inv['cua_so_kiem_tra']}"))
    checks.append(("đường base tái lập crop_lib.make_crop từng điểm ảnh (mẫu ngẫu nhiên)", inv["pixel_base_khop"] == inv["pixel_kiem_tra"], f"{inv['pixel_base_khop']}/{inv['pixel_kiem_tra']}"))
    checks.append(("vpipe == make_crop trên hộp hợp, từng điểm ảnh (mẫu ≤ 150 ô bị cắt)", inv["pixel_vpipe_khop"] == inv["pixel_vpipe_kiem_tra"], f"{inv['pixel_vpipe_khop']}/{inv['pixel_vpipe_kiem_tra']}"))
    checks.append(("mực bị carve xoá ⇒ seam đã chạy (nhất quán cờ seam)", inv["seam_khong_nhat_quan"] == 0, f"{inv['seam_khong_nhat_quan']}"))
    checks.append(("E1 đọc và so được", ("E1_loi" not in inv) and ("E1_thieu_tep" not in inv) and inv.get("E1_cung_o") is True, f"{inv.get('E1_loi', inv.get('E1_cung_o'))}"))
    for v in ("base", "py20", "adapt"):
        e = inv.get(f"E1_{v}")
        ok = bool(e) and e["max_abs_ds"] is not None and e["max_abs_ds"] < 2e-3 and e["top1_lech"] == e["top1_lech_giai_thich_duoc"] and e["mau_nan_t_giong"]
        checks.append((f"{v} tái lập điểm E1 (max|Δs|<2e-3; mọi lệch top1 giải thích được bằng gần hoà |m|<2·max|Δs|; mẫu NaN giống)", ok,
                       f"n={e['so_o']} max|Δs|={e['max_abs_ds']:.1e} lệch top1={e['top1_lech']} (giải thích được {e['top1_lech_giai_thich_duoc']}) NaN giống={e['mau_nan_t_giong']}" if e else "thiếu"))
    rv = d["rival_ok"] == 1
    for v in ALL_V:
        sub = rv if v in ("base", "vw", "vpipe") else (clipped & rv)          # base/vw/vpipe được dựng cho mọi ô Vision khớp; các biến thể còn lại chỉ cho ô bị cắt
        have = float(d.loc[sub, f"have_{v}"].mean())
        checks.append((f"{v}: ≥ 99 % ô (có đối thủ) có crop", have >= 0.99, f"{have:.3f} trên {int(sub.sum())} ô"))
    allok = True
    log("## Bất biến (chạy trước mọi diễn giải)")
    for name, ok, info in checks:
        allok &= bool(ok)
        log(f"- {'PASS' if ok else 'FAIL'} {name} — {info}")
    log(f"\n**Bất biến: {'TẤT CẢ ĐẠT' if allok else 'CÓ FAIL — KHÔNG DIỄN GIẢI các kết luận D1–D4 bên dưới (chỉ in số)'}**\n")
    res["bat_bien"] = dict(tat_ca_dat=bool(allok), chi_tiet=[dict(ten=n, dat=bool(o), thong_tin=i) for n, o, i in checks])
    gate = (lambda s: s) if allok else (lambda s: "KHÔNG DIỄN GIẢI (bất biến FAIL)")

    # ── quần thể
    cl_ = d[clipped]
    vert, horiz = (cl_["ov_v"] >= THR_OV), (cl_["ov_h"] >= THR_OV)
    per_page = cl_.groupby("page").size()
    log("## Quần thể")
    log(f"- Ô L16 Vision khớp chữ nhãn: {len(d)}; 'bị cắt' (agree & hại=1): {int(clipped.sum())} trên {len(per_page)} trang (ô/trang: trung vị {per_page.median():.0f}, lớn nhất {per_page.max()}); còn lại: {int((~clipped).sum())}; mẫu số quy % = {N_E1} ô GOLD L16 có nhãn người (E1)")
    log(f"- Hướng tràn (≥ {THR_OV:.0%} cạnh hộp Vision): chỉ dọc {int((vert & ~horiz).sum())}, chỉ ngang {int((~vert & horiz).sum())}, cả hai {int((vert & horiz).sum())}, không đạt ngưỡng {int((~vert & ~horiz).sum())}; "
        f"ô bị cắt có nhãn pipeline ≠ nhãn người: {int((cl_['label_eq'] == 0).sum())}; ô còn lại: {int((d.loc[~clipped, 'label_eq'] == 0).sum())}; ô còn lại có hại=NaN (ký hiệu nghiêng): {int(d.loc[~clipped, 'harm'].isna().sum())}")
    res["quan_the"] = dict(n_agree=int(len(d)), n_bi_cat=int(clipped.sum()), n_con_lai=int((~clipped).sum()), so_trang_bi_cat=int(len(per_page)),
                           chi_doc=int((vert & ~horiz).sum()), chi_ngang=int((~vert & horiz).sum()), ca_hai=int((vert & horiz).sum()), khong_dat=int((~vert & ~horiz).sum()))

    # ── D0: tiền đề — ô 'bị cắt' có thật sự mất mực của chính ô trong crop giao nộp?
    cov = cl_["cov_base"]
    cov_o = d.loc[~clipped, "cov_base"]
    cn, con = cov.dropna(), cov_o.dropna()
    q = np.nanpercentile(cn, [10, 25, 50, 75, 90])

    def pc(x, f):
        return 100 * float(f(x).mean()) if len(x) else float("nan")
    log("\n## D0 — tiền đề (mô tả): ô 'bị cắt' có thật sự mất mực của chính ô trong crop giao nộp (cov_base)?")
    log(f"- cov_base ô bị cắt (n={len(cn)}, NaN {int(cov.isna().sum())}): phân vị 10/25/50/75/90 = {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}/{q[3]:.3f}/{q[4]:.3f}; < 0,97: {pc(cn, lambda x: x < 0.97):.1f} %; ≤ 0,90: {pc(cn, lambda x: x <= 0.90):.1f} % "
        f"| ô còn lại (n={len(con)}, NaN {int(cov_o.isna().sum())}): trung vị {np.nanmedian(con):.3f}, < 0,97: {pc(con, lambda x: x < 0.97):.1f} %, ≤ 0,90: {pc(con, lambda x: x <= 0.90):.1f} %")
    rg_c, rg_o = d.loc[clipped, "ring"].dropna(), d.loc[~clipped, "ring"].dropna()
    log(f"- vòng mực quanh V (V ±3 px; mực ngoài/mực trong V; mô tả — V chặt vào mực hay còn mực lân cận) phân vị 25/50/75: ô bị cắt {rg_c.quantile(.25):.3f}/{rg_c.quantile(.5):.3f}/{rg_c.quantile(.75):.3f}; ô còn lại {rg_o.quantile(.25):.3f}/{rg_o.quantile(.5):.3f}/{rg_o.quantile(.75):.3f}")
    ero = cl_["er_own_base"].dropna()
    log(f"- mực lạ tuyệt đối của crop giao nộp (foA_base, chia cho mực chính ô): ô bị cắt trung vị {np.nanmedian(cl_['foA_base']):.3f}; carve xoá mực CHÍNH Ô ở base: {pc(ero, lambda x: x > 0):.1f} % ô bị cắt "
        f"(trung vị phần bị xoá trong số đó {np.nanmedian(ero[ero > 0]) if (ero > 0).any() else float('nan'):.3f}); seam chạy ở base: trên {100 * float(cl_['seam_top_base'].mean()):.0f} %, dưới {100 * float(cl_['seam_bot_base'].mean()):.0f} % ô bị cắt")
    res["D0"] = dict(cov_pv=[float(x) for x in q], cov_lt097=pc(cn, lambda x: x < 0.97) / 100, cov_le090=pc(cn, lambda x: x <= 0.90) / 100, cov_con_lai_lt097=pc(con, lambda x: x < 0.97) / 100,
                     carve_xoa_chu_chinh_base=pc(ero, lambda x: x > 0) / 100, n_nan_cov_bi_cat=int(cov.isna().sum()), n_nan_cov_con_lai=int(cov_o.isna().sum()))

    # ── D1: điểm cuối chính
    log(f"\n## Ô 'bị cắt' (n={int(clipped.sum())}): Δ so với crop giao nộp, tập cặp đầy đủ theo từng biến thể\n")
    main_t = {}
    for v in VARS_CLIPPED:
        main_t[v] = dl(d, v, clipped)
        log(f"- {v:11s} {A.fmt(main_t[v])} | loại {int(clipped.sum()) - main_t[v]['n']}/{int(clipped.sum())} ô (thiếu điểm)")
    res["bi_cat"] = main_t
    prim = main_t["vpipe"]["top1_pp"]
    ok1 = prim["lo"] > 0
    flips = {}
    for v in VARS_CLIPPED:
        pp = paired(d, v, clipped)
        b = int(((pp[f"t_{v}"] == 1) & (pp["t_base"] == 0)).sum())
        c = int(((pp[f"t_{v}"] == 0) & (pp["t_base"] == 1)).sum())
        flips[v] = dict(dung_hon=b, sai_hon=c, rong=b - c, rong_pct_E1=100.0 * (b - c) / N_E1,
                        dau_exact_hai_phia=float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else None)
    log("\n### Số ô lật so với crop giao nộp (ô bị cắt) và quy mô trên 13.117 ô GOLD L16 có nhãn người")
    for v in VARS_CLIPPED:
        f = flips[v]
        log(f"- {v:11s} đúng hơn {f['dung_hon']:3d}, sai hơn {f['sai_hon']:3d}, ròng {f['rong']:+4d} ({f['rong_pct_E1']:+.2f} % ô GOLD L16 có nhãn người)")
    res["lat"] = strip_p(flips)
    hw = (prim["hi"] - prim["lo"]) / 2
    if ok1:
        txt = "ĐÁNG CHÚ Ý (cận dưới > 0) — vẫn chỉ là thăm dò"
    elif prim["hi"] < 0:
        txt = "TỆ HƠN crop giao nộp (cả KTC < 0)"
    elif prim["hi"] < 2.0:
        txt = "KTC chứa 0 và loại trừ hiệu ứng ≥ +2 điểm"
    else:
        txt = "KTC chứa 0 và còn chứa hiệu ứng dương ≥ +2 điểm (null chưa đủ độ nhạy)"
    log(f"\n**D1 — điểm cuối chính** Δtop1(vpipe) trên ô bị cắt: {prim['mean']:+.2f} điểm [{prim['lo']:+.2f}; {prim['hi']:+.2f}] (nửa rộng ≈ {hw:.2f}; dự kiến 2,2–2,4) → {gate(txt)}")
    sl = main_t["vpipe"]["s_label"]
    mh = main_t["vpipe"]["m_hom"]
    log(f"- đồng điểm cuối mô tả: Δs_label {sl['mean']:+.4f} [{sl['lo']:+.4f}; {sl['hi']:+.4f}]; Δm_hom {mh['mean']:+.4f} [{mh['lo']:+.4f}; {mh['hi']:+.4f}]; ô đổi crop {100 * main_t['vpipe']['chg']:.1f} %\n")
    res["D1"] = dict(ok=bool(ok1) if allok else None, top1_pp=prim, nua_rong=hw)

    # ── tầng mô tả của D1 (không ngưỡng)
    log("### Tầng mô tả của Δtop1(vpipe) trên ô bị cắt (n hiển thị; n < 30 chỉ để xem, không diễn giải)")
    strata = {}
    cb = d["cov_base"]
    defs = [("cov_base ≤ 0,90 (mất ≥ 10 % mực chính ô)", clipped & (cb <= 0.90)), ("cov_base (0,90; 0,97)", clipped & (cb > 0.90) & (cb < 0.97)), ("cov_base ≥ 0,97 (gần như không mất)", clipped & (cb >= 0.97)),
            ("tràn chỉ dọc", clipped & (d["ov_v"] >= THR_OV) & (d["ov_h"] < THR_OV)), ("tràn chỉ ngang", clipped & (d["ov_v"] < THR_OV) & (d["ov_h"] >= THR_OV)),
            ("tràn cả hai", clipped & (d["ov_v"] >= THR_OV) & (d["ov_h"] >= THR_OV)), ("tràn dọc < 10 % (gồm cả tràn ngang)", clipped & (d["ov_v"] < THR_OV)),
            ("|dy_pitch| < 0,5 (tâm V gần tâm ô)", clipped & (d["dy_pitch"].abs() < 0.5)), ("|dy_pitch| ≥ 0,5 (V lệch xa: có thể là ký hiệu láng giềng)", clipped & (d["dy_pitch"].abs() >= 0.5)),
            ("nhãn pipeline = nhãn người", clipped & (d["label_eq"] == 1)), ("crop vpipe khác crop giao nộp", clipped & (d["chg_vpipe"] == 1)),
            ("chồng lấn hộp ký hiệu khác < 0,1 (Δ của vpipe)", clipped & (d["own_overlap"] < 0.1))]
    ovv = d.loc[clipped & (d["ov_v"] >= THR_OV), "ov_v"]
    if len(ovv) >= 9:
        t1, t2 = ovv.quantile([1 / 3, 2 / 3])
        defs += [(f"liều: tràn dọc tam phân vị thấp (≥ 10 % … {t1:.2f})", clipped & (d["ov_v"] >= THR_OV) & (d["ov_v"] <= t1)),
                 (f"liều: tràn dọc tam phân vị giữa ({t1:.2f} … {t2:.2f}]", clipped & (d["ov_v"] > t1) & (d["ov_v"] <= t2)), (f"liều: tràn dọc tam phân vị cao (> {t2:.2f})", clipped & (d["ov_v"] > t2))]
    for name, mk in defs:
        pp = paired(d, "vpipe", mk)
        if len(pp) < 2:
            log(f"- {name}: n={len(pp)}")
            continue
        r = A.delta(pp, "vpipe")
        t = r["top1_pp"]
        strata[name] = dict(n=int(len(pp)), mean=t["mean"], lo=t["lo"], hi=t["hi"])
        log(f"- {name}: n={len(pp):4d}  top1 {r['top1_base']:5.1f}→{r['top1_var']:5.1f}  Δ {t['mean']:+6.2f} [{t['lo']:+6.2f}; {t['hi']:+6.2f}]")
    res["tang"] = strata
    log()

    # ── độ nhạy theo cụm và dấu hiệu cặp (mô tả)
    pp = paired(d, "vpipe", clipped)
    dd = (pp["t_vpipe"] - pp["t_base"]) * 100.0
    m2, lo2, hi2, n2, _ = A.boot(dd.values, pp["cl2"].values)
    log(f"- độ nhạy: KTC theo trang×cột cho D1 = {m2:+.2f} [{lo2:+.2f}; {hi2:+.2f}] (số cụm {pp['cl2'].nunique()} so với {pp['cl'].nunique()} trang); "
        f"cặp bất đồng b={flips['vpipe']['dung_hon']} (vpipe đúng, giao nộp sai), c={flips['vpipe']['sai_hon']}; kiểm dấu chính xác (mô tả; BỎ QUA cụm trang nên có thể mạnh hơn KTC theo trang) "
        + (f"p={flips['vpipe']['dau_exact_hai_phia']:.3f}" if flips['vpipe']['dau_exact_hai_phia'] is not None else "p=—") + "\n")

    # ── đối chứng (mô tả)
    cc = clipped & d["t_base"].notna() & d["t_vpipe"].notna() & d["t_vpipe_flip"].notna() & d["t_vpipe_shuf"].notna()
    ccs = d[cc]
    log(f"## Đối chứng (MÔ TẢ; tập cặp đầy đủ cả 4 biến thể: n={int(cc.sum())}) — kỳ vọng vpipe > đối chứng theo cấu tạo (cửa sổ cùng độ tràn nhưng SAI thông tin); KHÔNG phải bằng chứng lợi so với crop giao nộp, chỉ D1 nói điều đó")
    ctrl = {}
    for v in ("vpipe", "vpipe_flip", "vpipe_shuf"):
        r = A.delta(ccs, v)
        ctrl[v] = r
        log(f"- {v:11s} {A.fmt(r)}")
    for a_, b_ in (("vpipe", "vpipe_shuf"), ("vpipe", "vpipe_flip")):
        x = (ccs[f"t_{a_}"] - ccs[f"t_{b_}"]) * 100.0
        m_, l_, h_, n_, _ = A.boot(x.values, ccs["cl"].values)
        ctrl[f"{a_}_tru_{b_}"] = dict(mean=m_, lo=l_, hi=h_, n=n_)
        log(f"  · {a_} − {b_}: {m_:+.2f} điểm [{l_:+.2f}; {h_:+.2f}] (n={n_})")
    sub = cc & (d["ov_v"] >= THR_OV)
    xs = d[sub]
    x = (xs["t_vpipe"] - xs["t_vpipe_flip"]) * 100.0
    m_, l_, h_, n_, _ = A.boot(x.values, xs["cl"].values)
    log(f"  · riêng ô tràn dọc ≥ 10 % (flip chỉ có nghĩa ở đó): vpipe − vpipe_flip = {m_:+.2f} [{l_:+.2f}; {h_:+.2f}] (n={n_})\n")
    ctrl["vpipe_tru_flip_tran_doc"] = dict(mean=m_, lo=l_, hi=h_, n=n_)
    res["doi_chung"] = strip_p(ctrl)

    # ── D3: bảo vệ trên ô Vision khớp còn lại
    rest = ~clipped
    log(f"## D3 — nếu áp luật cho MỌI ô Vision khớp: ô còn lại (n={int(rest.sum())}; cơ sở top1 {100 * d.loc[rest, 't_base'].mean():.2f} %)")
    guard = {}
    for v in GUARD_VARS:
        r = dl(d, v, rest)
        guard[v] = r
        ch = paired(d, v, rest & (d[f"chg_{v}"] == 1))
        rc = A.delta(ch, v) if len(ch) > 1 else None
        nan_part = dl(d, v, rest & d["harm"].isna())
        log(f"- {v:6s} {A.fmt(r)}")
        log(f"    trong đó chỉ ô crop thật sự đổi (n={len(ch)}, {100 * len(ch) / max(int(rest.sum()), 1):.1f} % ô còn lại): " + (A.fmt(rc) if rc else "—"))
        log(f"    riêng ký hiệu nghiêng (hại=NaN): " + (A.fmt(nan_part) if nan_part["n"] > 1 else f"n={nan_part['n']}"))
        guard[v]["chi_o_doi_crop"] = rc
        guard[v]["ky_hieu_nghieng"] = nan_part
    g = guard["vpipe"]["top1_pp"]
    ok3 = g["lo"] >= THR_GUARD
    allv = paired(d, "vpipe", None)
    net_all = int(((allv["t_vpipe"] == 1) & (allv["t_base"] == 0)).sum() - ((allv["t_vpipe"] == 0) & (allv["t_base"] == 1)).sum())
    log(f"\n**D3** Δtop1(vpipe) ô còn lại: cận dưới {g['lo']:+.2f} ≥ {THR_GUARD} → {gate('đạt (chỉ nghĩa là cận dưới ≥ −0,3; KHÔNG chứng minh vô hại)' if ok3 else 'KHÔNG đạt (cận dưới < −0,3: không loại trừ được hại ≥ 0,3 điểm; có thể do độ rộng KTC)')}; "
        f"ròng khi áp cho MỌI ô Vision khớp (bị cắt + còn lại): {net_all:+d} ô = {100.0 * net_all / N_E1:+.2f} % ô GOLD L16 có nhãn người\n")
    res["D3"] = dict(ok=bool(ok3) if allok else None, top1_pp=g, rong_moi_o=net_all, con_lai=strip_p(guard))

    # ── cơ chế (liên hệ, không phải nhân quả)
    log("## Cơ chế (LIÊN HỆ, không phải nhân quả; ô bị cắt; trước tighten, trên trang đã xử lý): trung bình, và Δ so với base (KTC theo trang)")
    log("| biến thể | cov | foA (mực ngoài V / mực chính ô) | nbA (mực hộp ký hiệu khác / mực chính ô) | er_own (mực chính ô bị carve xoá) | foO (điểm tối ngoài V trên điểm ảnh gốc, Otsu theo cửa sổ) | mm (lề nhỏ nhất, px) | seam trên/dưới chạy | khung giao nộp cao×rộng (trung vị) |")
    log("|---|---|---|---|---|---|---|---|---|")
    mech = {}
    for v in ALL_V:
        row, mech[v] = [], {}
        sub_v = clipped & d[f"have_{v}"].eq(1)
        for k in ("cov", "foA", "nbA", "er_own", "foO", "mm"):
            x = d.loc[sub_v, f"{k}_{v}"]
            fm = ".2f" if k == "mm" else ".3f"
            if v == "base":
                row.append(f"{np.nanmean(x):{fm}}")
                mech[v][k] = dict(mean=float(np.nanmean(x)))
            else:
                bsub = sub_v & d[f"{k}_base"].notna() & d[f"{k}_{v}"].notna()
                m_, l_, h_, n_, _ = A.boot((d.loc[bsub, f"{k}_{v}"] - d.loc[bsub, f"{k}_base"]).values, d.loc[bsub, "cl"].values)
                row.append(f"{np.nanmean(x):{fm}} (Δ {m_:+{fm}} [{l_:+{fm}}; {h_:+{fm}}])")
                mech[v][k] = dict(mean=float(np.nanmean(x)), d_mean=m_, d_lo=l_, d_hi=h_)
        st, sb = 100 * float(d.loc[sub_v, f"seam_top_{v}"].mean()), 100 * float(d.loc[sub_v, f"seam_bot_{v}"].mean())
        fh, fw = float(np.nanmedian(d.loc[sub_v, f"fh_{v}"])), float(np.nanmedian(d.loc[sub_v, f"fw_{v}"]))
        row.append(f"{st:.0f} % / {sb:.0f} %")
        row.append(f"{fh:.0f}×{fw:.0f}")
        mech[v]["seam"] = dict(tren=st, duoi=sb)
        log(f"| {v} | " + " | ".join(row) + " |")
    log()
    pw = paired(d, "vpipe_own", clipped)
    pw = pw[pw["t_vpipe"].notna()]
    x = (pw["t_vpipe_own"] - pw["t_vpipe"]) * 100.0
    m_, l_, h_, n_, _ = A.boot(x.values, pw["cl"].values)
    log(f"- vpipe_own − vpipe (có loại mực hộp ký hiệu khác thêm giá trị không?): {m_:+.2f} điểm [{l_:+.2f}; {h_:+.2f}] (n={n_}); nbA của vpipe_own ≈ 0 theo cấu tạo (không diễn giải); "
        f"vpipe_own chỉ ở ô chồng lấn hộp khác < 0,1: " + (A.fmt(A.delta(paired(d, 'vpipe_own', clipped & (d['own_overlap'] < 0.1)), 'vpipe_own')) if (clipped & (d['own_overlap'] < 0.1)).any() else '—'))
    log("- Điều đọc được (mô tả): so sánh foA/nbA/er_own của py20, adapt, vw với base và vpipe ở bảng trên. Chỉ nói 'mực lạ tuyệt đối tăng/không tăng khi nới', không nói đó là nguyên nhân.\n")
    res["co_che"] = mech
    res["vpipe_own_tru_vpipe"] = dict(mean=m_, lo=l_, hi=h_, n=n_)
    res["quy_tac"] = dict(D1="cận dưới KTC95 % của Δtop1(vpipe) > 0 trên ô bị cắt", dieu_kien_so_sanh="mọi so sánh trên tập cặp đầy đủ; threshold so không làm tròn")

    (OUT / f"X1_ket_qua{sfx}.json").write_text(json.dumps(strip_p(res), ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (OUT / f"X1_BAO_CAO_THAM_DO{sfx}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT / ('X1_BAO_CAO_THAM_DO' + sfx + '.md')}")


if __name__ == "__main__":
    main()
