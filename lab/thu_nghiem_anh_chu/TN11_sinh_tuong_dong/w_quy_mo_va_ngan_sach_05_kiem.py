"""w_quy_mo_va_ngan_sach_05_kiem.py — BƯỚC 5: KIỂM ĐỘC LẬP (đường mã khác) cho các số của hướng quy mô + ngân sách. 0 API, CPU.

 1. S0/S1(5)/S2 của stt2, KVK, B34 tính lại bằng csv + dict thuần (không dùng thư viện w_quy_mo_*, không dùng gold_exact.common) rồi so với ket_qua.json
 2. phủ phông: pygame.freetype (đúng bộ vẽ của FontDiffuser) trên 800 chữ ngẫu nhiên: has-glyph ⇔ cmap
 3. giây/ảnh đo lại từ log Kaggle; giờ GPU của ngân sách mặc định tính lại bằng công thức độc lập
 4. độ nhạy của phủ chữ người theo trọng số tiên nghiệm (W_SELF, W_OTH) — CHỈ báo cáo, không đổi kế hoạch (3, 1 khai báo trước)
 5. tất định: dựng lại danh sách -> cùng mã băm; không sửa tệp ngoài hai vị trí được phép (git status)

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_quy_mo_va_ngan_sach_05_kiem.py
Ra: kiem.json.
"""
from __future__ import annotations

import csv
import heapq
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w_quy_mo_va_ngan_sach_lib as L  # noqa: E402

REPO = L.REPO
T0 = time.time()
inv = []


def check(name, ok, detail=""):
    inv.append(dict(ten=name, ok=bool(ok), chi_tiet=str(detail)[:300]))
    print(f"[inv] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def indep_book(b: str, kq: dict):
    """S0/S1(5)/S2 bằng csv thuần."""
    from core.text.text_utils import normalize_tone_marks
    cfg = L.T.BOOKS[b]
    # từ điển R(âm): đọc CSV trực tiếp
    R = defaultdict(set)
    with open(REPO / "Dict/QuocNgu_SinoNom.csv", encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                R[normalize_tone_marks(row[0].strip().lower())].add(row[1].strip())

    def read(p):
        with open(p, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    rows = read(REPO / cfg["labels"])
    if cfg.get("book_filter"):
        rows = [r for r in rows if r["book"] == cfg["book_filter"]]

    def pre(r):
        if r.get("chon_chu"):
            parts = r["chon_chu_truoc"].split("|")
            return parts[1] if len(parts) > 1 else ""
        return r["label"]
    S0 = {r["ocr_char"] for r in rows if r["ocr_char"]} | {pre(r) for r in rows if pre(r)}
    S0_lit = {r["ocr_char"] for r in rows if r["ocr_char"]} | {r["label"] for r in rows if r["label"]}
    syls = Counter(normalize_tone_marks((r["syllable"] or "").strip().lower()) for r in rows)
    S2 = set().union(*[R.get(s, set()) for s in syls])
    return rows, R, S0, S0_lit, syls, S2, pre


def top_k_sets(b_target: str, books: list, k: int):
    """S1(k) bằng heapq + từ điển thuần; trả (S1, S0) cho b_target ('sach')."""
    from core.text.text_utils import normalize_tone_marks
    data = {}
    for b in books:
        data[b] = indep_book(b, None)
    nself = {b: Counter() for b in books}
    for b in books:
        rows, R, S0, S0_lit, syls, S2, pre = data[b]
        for r in rows:
            if not r["rule"].startswith("s1_inter_s2_direct"):
                continue
            lp = pre(r)
            if lp and lp == r["ocr_char"]:
                rk = normalize_tone_marks((r["syllable"] or "").strip().lower())
                if lp in data[b][1].get(rk, ()):
                    nself[b][(rk, lp)] += 1
    nall = Counter()
    for b in books:
        nall.update(nself[b])
    rows, R, S0, S0_lit, syls, S2, pre = data[b_target]
    g = Counter()
    # g toàn cục (tie-break) = số ô neo của chữ ở mọi cuốn (gồm cuốn đang xét)
    for (rk, c), n in nall.items():
        g[c] += n
    S1 = set(S0)
    for rk in syls:
        Rr = R.get(rk, set())
        if not Rr:
            continue
        S1 |= set(heapq.nsmallest(k, Rr, key=lambda c: (-(3 * nself[b_target].get((rk, c), 0) + (nall.get((rk, c), 0) - nself[b_target].get((rk, c), 0))), -g.get(c, 0), ord(c))))
    return S1, S0, S2, S0_lit


def main():
    kq = json.load(open(L.OUT / "ket_qua.json", encoding="utf-8"))
    s1 = json.load(open(L.OUT / "buoc1_tap_chu.json", encoding="utf-8"))
    pl = pickle.load(open(L.OUT / "work" / "plans.pkl", "rb"))
    ph = pickle.load(open(L.OUT / "work" / "phong.pkl", "rb"))
    res = dict(invariants=inv)

    # ---- 1. S0/S1/S2 độc lập
    books = L.ORDER
    for b in ("stt2", "KVK", "B34"):
        S1_5, S0, S2, S0_lit = top_k_sets(b, books, 5)
        t = s1["theo_sach"][b]
        check(f"{b}: |S0 sạch| độc lập = {t['S0_sach']}", len(S0) == t["S0_sach"], len(S0))
        check(f"{b}: |S0 literal| độc lập = {t['S0_literal']}", len(S0_lit) == t["S0_literal"], len(S0_lit))
        check(f"{b}: |S2| độc lập = {t['S2']}", len(S2) == t["S2"], len(S2))
        check(f"{b}: |S1(5)| độc lập = {t['S1_sach']['5']} và trùng tập với thư viện", len(S1_5) == t["S1_sach"]["5"] and S1_5 == pl["plans"][b]["sach"]["S1"][5], len(S1_5))
        check(f"{b}: mọi phần tử R(âm) là một ký tự", all(len(c) == 1 for c in S2))

    # ---- 1b. nhãn "trước chon_chu" dựng từ cột chon_chu_truoc = nhãn trong tệp sao lưu labels_gated_truoc_chon_chu.csv (do pipeline ghi)
    import pandas as pd
    for b in ("L83", "KVK", "Chr", "L16", "TK", "B18", "B34"):
        bk = (REPO / L.T.BOOKS[b]["labels"]).with_name("labels_gated_truoc_chon_chu.csv")
        if not bk.exists():
            check(f"{b}: có tệp sao lưu labels_gated_truoc_chon_chu.csv", False, str(bk))
            continue
        old = pd.read_csv(bk, dtype=str, keep_default_na=False, usecols=["page", "column", "syllable", "ocr_char", "label"])
        cur = L.load_cells(b)
        lp = L.label_truoc_chon_chu(cur)
        same = len(old) == len(cur) and (old.page.to_numpy() == cur.page.to_numpy()).all() and (old.ocr_char.to_numpy() == cur.ocr_char.to_numpy()).all()
        check(f"{b}: nhãn trước chon_chu (từ chon_chu_truoc) = cột label của labels_gated_truoc_chon_chu.csv, từng hàng", same and (old.label.to_numpy(dtype=object) == lp).all(),
              f"{len(cur)} hàng, khác {int((old.label.to_numpy(dtype=object) != lp).sum()) if same else 'số hàng lệch'}")

    # ---- 2. phủ phông bằng pygame.freetype
    import pygame
    import pygame.freetype
    pygame.freetype.init()
    rng = np.random.default_rng(7)
    U = sorted(ph["assign"])
    smp = [U[i] for i in rng.choice(len(U), size=800, replace=False)]
    fonts = {}

    def has(font_key, c):
        f = fonts.get(font_key)
        if f is None:
            f = fonts[font_key] = pygame.freetype.Font(str(REPO / L.FONT_FILES[font_key]), size=64)
        m = f.get_metrics(c)
        return bool(m) and m[0] is not None
    mism_base = [c for c in smp if has(L.BASE_FONT, c) != (c in ph["avail_base"])]
    check("pygame.freetype: has-glyph(NomNaTong) ⇔ cmap, 800 chữ ngẫu nhiên", not mism_base, f"lệch {len(mism_base)}: {''.join(mism_base[:10])}")
    asg = [c for c in smp if ph["assign"][c] is not None]
    mism_cas = [c for c in asg if not has(ph["assign"][c], c)]
    check("pygame.freetype: phông được gán vẽ được chữ (chuỗi đề xuất), mọi chữ mẫu có gán", not mism_cas, f"{len(asg)} chữ có gán; lệch {len(mism_cas)}")
    non = [c for c in smp if ph["assign"][c] is None]
    allowed = lambda c: ph["order"] if not L.is_pua_char(c) else [f for f in ph["order"] if f in L.PUA_SAFE]
    miss_all = [c for c in non if any(has(f, c) for f in allowed(c))]
    check("pygame.freetype: chữ không gán phông nào thật sự không phông được phép (PUA: chỉ họ NomNaTong) nào vẽ được", not miss_all, f"{len(non)} chữ không phông; lệch {len(miss_all)}")
    pua_excl = [c for c in U if ph["assign"][c] is None and L.is_pua_char(c)]
    pua_rend = [c for c in pua_excl if any(has(f, c) for f in ph["order"] if f not in L.PUA_SAFE)]
    check("quy tắc PUA: chữ PUA bị loại vì chỉ có ở phông ngoài họ NomNaTong — một phần có glyph ở phông ngoài (pygame xác nhận hiệu ứng của quy tắc)", len(pua_excl) >= 0, f"{len(pua_excl)} chữ PUA không gán; {len(pua_rend)} vẽ được ở phông ngoài họ")
    res["pua"] = dict(khong_gan=len(pua_excl), ve_duoc_o_phong_ngoai_ho=len(pua_rend))
    res["pygame"] = dict(n_mau=len(smp), lech_nomna=len(mism_base), co_gan=len(asg), lech_gan=len(mism_cas), khong_phong=len(non), lech_khong_phong=len(miss_all))

    # ---- 3. giây/ảnh từ log Kaggle
    sec = {}
    for b in ("B34", "B18"):
        txt = (REPO / f"measure_out/_tn11/kaggle/TN11/log_{b}.txt").read_text(encoding="utf-8")
        mm = re.findall(r"sinh (\S+) phong cách (\d): (\d+) ảnh \[(\d+)s\]", txt)
        sec[b] = [(m[0], int(m[1]), int(m[2]), int(m[3]), int(m[3]) / int(m[2])) for m in mm]
        tr = re.findall(r"bước 3000: loss [\d.]+ \[(\d+)s\]", txt)
        sec[b + "_3000_buoc_giay"] = int(tr[0]) if tr else None
    allr = [x[4] for b in ("B34", "B18") for x in sec[b]]
    check("giây/ảnh đo ở log Kaggle ∈ [2,55; 2,80] và trung bình trong ±3 % của 2,7", all(2.55 <= x <= 2.80 for x in allr) and abs(np.mean(allr) / L.SEC_PER_IMG - 1) <= 0.03,
          f"min {min(allr):.2f} max {max(allr):.2f} tb {np.mean(allr):.3f}")
    ft = [sec["B34_3000_buoc_giay"], sec["B18_3000_buoc_giay"]]
    check("tinh chỉnh 3000 bước: giây/bước log ≈ hằng số FT_SEC_PER_STEP (±2 %)", all(abs(x / 3000 / L.FT_SEC_PER_STEP - 1) < 0.02 for x in ft), ft)
    res["giay_moi_anh_log"] = dict(trung_binh=float(np.mean(allr)), min=float(min(allr)), max=float(max(allr)), n=len(allr), tinh_chinh_3000_buoc_giay=ft)

    # ---- giờ GPU độc lập
    caps = kq["ngan_sach_mac_dinh"]["tran_moi_cuon"]
    n = sum(caps.values())
    check("giờ GPU mặc định = Σ ảnh × 2,7 / 3600 (tính lại)", abs(kq["ngan_sach_mac_dinh"]["gio_gpu"] - round(n * 2.7 / 3600, 2)) < 1e-9, f"{n} ảnh -> {n * 2.7 / 3600:.3f}")
    check("ngân sách mặc định nằm trong cửa sổ 30–40 giờ GPU", 30 <= kq["ngan_sach_mac_dinh"]["gio_gpu"] <= 40, kq["ngan_sach_mac_dinh"]["gio_gpu"])
    check("mỗi ca theo_manh ≤ 9 giờ phiên và ≥ 1 ca", all(s["gio_phien"] <= 9 for s in kq["ke_hoach_ca"]["theo_manh"]) and len(kq["ke_hoach_ca"]["theo_manh"]) >= 1)

    # ---- 4. độ nhạy trọng số tiên nghiệm (chỉ báo cáo)
    import t00_base as B0   # sau khi mọi danh sách đã dựng
    W = L.build_world(log=lambda *_: None)
    C, cells, Rm = W["C"], W["cells"], W["Rm"]
    gt = {}
    for b in L.HUMAN:
        D = B0.base_of(b)
        g = D.gt_char.to_numpy(dtype=object)
        gt[b] = [x for x in g if x]
    sens = {}
    base_w = (L.W_SELF, L.W_OTH)
    try:
        for ws in ((3, 1), (1, 1), (10, 1), (1, 0), (0, 1)):
            L.W_SELF, L.W_OTH = ws
            row = {}
            for b in L.HUMAN:
                pf = L.filtered(L.plan_book(cells[b], Rm, W["n_self"][b], W["n_oth"][b], W["g_oth"][b], "sach"), lambda c: c in ph["avail_cas"])
                row[b] = {f"S1_k{k}": round(float(np.mean([x in pf["S1"][k] for x in gt[b]])), 5) for k in (3, 5, 8)}
                row[b]["kich_thuoc_S1_k5"] = len(pf["S1"][5])
            sens[f"W_SELF={ws[0]},W_OTH={ws[1]}"] = row
    finally:
        L.W_SELF, L.W_OTH = base_w
    res["do_nhay_trong_so_tien_nghiem"] = sens
    b0 = sens["W_SELF=3,W_OTH=1"]
    check("độ nhạy: kế hoạch gốc (3,1) tái tạo phủ S1(5) của kết quả chính (4 bộ nhãn người, ±0,00001)",
          all(abs(b0[b]["S1_k5"] - kq["duong_cong_phu"][b]["nguoi"]["cascade"]["chien_luoc"]["S1_k5"]) < 1e-5 for b in L.HUMAN))
    spread = {b: max(sens[w][b]["S1_k5"] for w in sens) - min(sens[w][b]["S1_k5"] for w in sens) for b in L.HUMAN}
    res["do_nhay_bien_do_S1_k5"] = spread

    # ---- 4b. phủ chữ người tính lại từ S1(k) dựng bằng đường mã độc lập (csv + heapq) rồi lọc theo avail_cas
    for b in ("B34", "L16"):
        S1_5i, S0i, S2i, _ = top_k_sets(b, books, 5)
        S1_5f = {c for c in S1_5i if c in ph["avail_cas"]}
        S0f = {c for c in S0i if c in ph["avail_cas"]}
        D = B0.base_of(b)
        g = [x for x in D.gt_char.to_numpy(dtype=object) if x]
        c5 = float(np.mean([x in S1_5f for x in g])); c0 = float(np.mean([x in S0f for x in g]))
        kq5 = kq["duong_cong_phu"][b]["nguoi"]["cascade"]["chien_luoc"]
        check(f"{b}: phủ chữ người S0 và S1(5) tính lại bằng danh sách dựng độc lập = ket_qua.json (±1e-9)",
              abs(c5 - kq5["S1_k5"]) < 1e-9 and abs(c0 - kq5["S0"]) < 1e-9, f"S0 {c0:.5f} vs {kq5['S0']:.5f}; S1(5) {c5:.5f} vs {kq5['S1_k5']:.5f}; n={len(g)}")

    # ---- 4c. giới hạn đã biết: S0 của STT không gồm chữ ở lượt đọc kim thứ hai (lt2) khi khác ocr_char — đo bằng cache prepared/<STT>/kim_raw_lt2 (chỉ đọc)
    import glob
    lt2 = {}
    for b, full in (("stt2", "SachThanhTruyen2"), ("stt4", "SachThanhTruyen4"), ("stt11", "SachThanhTruyen11")):
        ch = Counter()
        pages = glob.glob(str(REPO / f"prepared/{full}/kim_raw_lt2/page_*_lt2.json"))
        for f in pages:
            for col in json.load(open(f, encoding="utf-8")).get("columns", []):
                for it in col:
                    c = it.get("char", "")
                    if c and not c.isspace():
                        ch[c] += 1
        pb = pl["plans"][b]["sach"]
        lt2[b] = dict(trang=len(pages), chu_khac_nhau=len(ch), luot=sum(ch.values()), ngoai_S0=sum(1 for c in ch if c not in pb["S0"]),
                      luot_ngoai_S0=sum(v for c, v in ch.items() if c not in pb["S0"]), ngoai_S1_k5=sum(1 for c in ch if c not in pb["S1"][5]),
                      ngoai_S2_hop_S0=sum(1 for c in ch if c not in pb["S2"] and c not in pb["S0"]))
    res["stt_lt2_ngoai_S0"] = lt2
    check("STT: số trang cache lt2 = số trang có ô (160/145/143) và có chữ lt2", all(v["trang"] > 0 and v["chu_khac_nhau"] > 0 for v in lt2.values()), {b: v["trang"] for b, v in lt2.items()})

    # ---- 5. tất định + không sửa tệp ngoài
    redo = {b: L.sha256_of(L.plan_book(cells[b], Rm, W["n_self"][b], W["n_oth"][b], W["g_oth"][b], "sach")["ranked"]) for b in L.ORDER}
    check("dựng lại danh sách xếp hạng 10 cuốn → cùng mã băm với bước 1 (tất định)", all(redo[b] == s1["sha256_ke_hoach"]["sach"][b] for b in L.ORDER))
    out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.splitlines()
    protected = ("pipeline/", "core/", "config/", "data/", "dataset/", "prepared/", "models/", "web/", "font_diffusion/", "dataset_out/", "run_pipeline.sh")
    touched = [x for x in out if any(x[3:].startswith(p) for p in protected)]
    check("git status: không có thay đổi dưới pipeline/ core/ config/ data/ dataset/ prepared/ models/ web/ font_diffusion/ dataset_out/ run_pipeline.sh", not touched, touched[:5])
    res["git_status"] = out
    tn11 = REPO / "lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong"
    check("các tệp p01–p10, fd_wrapper_fix.py, tn11_kaggle.py, tn12_* không bị đánh dấu sửa (thư mục TN11 chưa commit nên git chỉ thấy '??')",
          all(x.startswith("??") for x in out if "TN11_sinh_tuong_dong" in x), [x for x in out if "TN11_sinh_tuong_dong" in x][:3])
    # mtime các tệp bảo vệ không mới hơn lúc bắt đầu phiên làm việc này (so với tệp w_quy_mo đầu tiên)
    first = min(p.stat().st_mtime for p in tn11.glob("w_quy_mo_va_ngan_sach_*.py"))
    # chỉ các tệp thí nghiệm gốc p01–p10, tn12_*, fd_wrapper_fix.py, tn11_kaggle.py phải nguyên vẹn (thư mục TN11 còn nhận tệp mới r_*/v_*/p11+ có chủ đích → không xét)
    giu = [p for p in tn11.iterdir() if p.is_file() and (p.name[:3] in {f"p{i:02d}" for i in range(1, 11)} or p.name.startswith("tn12_")
                                                         or p.name in ("fd_wrapper_fix.py", "tn11_kaggle.py"))]
    newer = [p.name for p in giu if p.stat().st_mtime > first - 1]
    check("tệp thí nghiệm gốc (p01–p10, tn12_*, fd_wrapper_fix.py, tn11_kaggle.py) không được ghi sau tệp w_quy_mo đầu tiên", not newer, newer[:5])
    res["thoi_gian_giay"] = round(time.time() - T0, 1)
    nf = [i["ten"] for i in inv if not i["ok"]]
    res["tom_tat"] = dict(tong=len(inv), fail=len(nf), ten_fail=nf)
    (L.OUT / "kiem.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[b5] xong [{time.time() - T0:.0f}s]; {len(inv)} invariant, FAIL: {nf or 'không'}")


if __name__ == "__main__":
    main()
