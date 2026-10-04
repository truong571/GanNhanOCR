"""r_goi_du_lieu_01_ro_nhan_nguoi.py — HƯỚNG 1: RÒ NHÃN NGƯỜI trong danh sách chữ của gói Kaggle (TN11 full).

(a) KIỂM KÊ ĐẦU VÀO bằng THỰC THI: chạy lại ĐÚNG đường mã của khâu quy mô (w_quy_mo_va_ngan_sach_lib.build_world + plan_book) dưới móc kiểm toán
    sys.addaudithook ghi mọi tệp được MỞ -> liệt kê tệp dữ liệu (ngoài mã) đã đọc; so với mẫu tên tệp nhãn người; so ranked với plans.pkl.
(b) DỰNG LẠI ĐỘC LẬP: mã mới (không nạp w_quy_mo_*/tn8lib/t00_base...), đọc CHỈ 8 cột (book page column syllable ocr_char label tier rule) của bảng TRƯỚC chon_chu
    (labels_gated_truoc_chon_chu.csv; STT: labels_final.csv), tự gán phông (cmap + kiểm vẽ PIL) -> so từng chữ với chars_<cuốn>.txt, so mã băm.
(c) NHẠY: nếu dựng từ bảng SAU chon_chu (cột label cuối) thì danh sách B18/B34 khác bao nhiêu.
(d) GREP gói (tên tệp trong zip, JSON, mã, siêu dữ liệu PNG/safetensors) tìm dấu hiệu nhãn người.

    cd measure_out/_tn11/full/review/goi_du_lieu && PYTHONDONTWRITEBYTECODE=1 ../../../../../.venv/bin/python -B ../../../../../lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_goi_du_lieu_01_ro_nhan_nguoi.py
Ra: 01_ro_nhan_nguoi.json, rebuild.pkl (danh sách dựng lại, cho script 03).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
_OPENED: dict = {}


def _hook(event, args):
    if event == "open":
        p = args[0]
        if isinstance(p, (str, bytes, os.PathLike)):
            try:
                _OPENED[os.fsdecode(p)] = _OPENED.get(os.fsdecode(p), 0) + 1
            except Exception:  # noqa: BLE001
                pass


import os  # noqa: E402

sys.addaudithook(_hook)

import collections  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_goi_du_lieu_00_lib as Lb  # noqa: E402

HUMAN_PAT = re.compile(r"(\.xlsx$|manifest\.tsv|nom_transcriptions|_BORG_NHAN_NGUOI|nhan_nguoi|TAP_DANH_GIA|SinoNom_Char|gt_char|NomDB|ihr_|eval_ihr|"
                       r"borg_human|borg_endtoend|t00_base|tn6lib|models/chon_chu|hum_B|/data/(Sach|Luc|Truyen))", re.I)
CODE_EXT = (".py", ".pyc", ".so", ".dylib", ".pth", ".pyi", ".typed")


def data_files_opened() -> list:
    """Tệp dữ liệu (không phải mã/thư viện) trong REPO (trừ .venv) đã mở, rút gọn thư mục crop."""
    out = {}
    root = str(Lb.REPO) + os.sep
    for p, n in _OPENED.items():
        if not p.startswith(root) or "/.venv/" in p or p.endswith(CODE_EXT) or "/__pycache__/" in p:
            continue
        rel = p[len(root):]
        out[rel] = n
    return sorted(out.items())


# ============================================================================================ (b) dựng lại độc lập
def load_table(b: str, which: str) -> pd.DataFrame:
    """which = 'pre' (TRƯỚC chon_chu) | 'post' (cột label cuối). Chỉ 8 cột; STT lọc cột book."""
    rel = Lb.PREF[b] if which == "pre" else Lb.POST[b]
    d = Lb.rd(Lb.REPO / rel, usecols=lambda c: c in Lb.PLAN_COLS)
    if b in Lb.STT:
        d = d[d.book == Lb.BOOK_VAL[b]]
    return d.reset_index(drop=True)


def plan_from_tables(tabs: dict, R_key, R_of, k: int = Lb.K) -> dict:
    """Danh sách chữ ưu tiên mỗi cuốn — cài đặt LẠI từ đặc tả (w_quy_mo_va_ngan_sach_lib docstring) không gọi mã đó.
    tabs[b] = DataFrame cột syllable, ocr_char, label, rule. Trả {b: dict(S0, S1, ranked, best, e, g, demand, n_cells)}."""
    cells, Rm = {}, {}
    for b, d in tabs.items():
        rk = [R_key(s) for s in d.syllable]
        cells[b] = (rk, d.label.to_numpy(dtype=object), d.ocr_char.to_numpy(dtype=object), d.rule.to_numpy(dtype=object))
        for x in set(rk):
            if x not in Rm:
                Rm[x] = frozenset(R_of(x))
    n_self = {}
    for b, (rk, lab, kim, rule) in cells.items():
        cnt: Counter = Counter()
        for x, l, kk, r in zip(rk, lab, kim, rule):
            if l and l == kk and r.startswith(Lb.ANCHOR_PREFIX) and l in Rm[x]:
                cnt[(x, l)] += 1
        n_self[b] = cnt
    n_all: Counter = Counter()
    for c in n_self.values():
        n_all.update(c)
    res = {}
    for b, (rk, lab, kim, rule) in cells.items():
        n_oth = n_all - n_self[b]
        g: Counter = Counter()
        for (x, c), n in n_oth.items():
            g[c] += n
        for (x, c), n in n_self[b].items():
            g[c] += n
        e: Counter = Counter()
        for a, kk in zip(lab, kim):
            if a and kk and a != kk:
                e[a] += 1; e[kk] += 1
            elif a or kk:
                e[a or kk] += 1
        S0 = set(e)
        rk_counts = Counter(rk)
        best, demand = {}, Counter()
        for x, ncell in rk_counts.items():
            R = Rm[x]
            if not R:
                continue
            sc = sorted(R, key=lambda c: (-(3 * n_self[b].get((x, c), 0) + 1 * n_oth.get((x, c), 0)), -g.get(c, 0), ord(c)))
            for i, c in enumerate(sc, 1):
                b0 = best.get(c)
                if b0 is None or i < b0:
                    best[c] = i; demand[c] = ncell
                elif i == b0:
                    demand[c] += ncell
        S2 = set(best)
        r0 = sorted(S0, key=lambda c: (-e[c], -g.get(c, 0), ord(c)))
        r1 = sorted((c for c in S2 if c not in S0), key=lambda c: (best[c], -demand[c], -g.get(c, 0), ord(c)))
        res[b] = dict(S0=S0, S2=S2, S1k=S0 | {c for c, r in best.items() if r <= k}, ranked=r0 + r1, n0=len(r0), best=best, e=e, g=g, demand=demand,
                      n_cells=len(rk), n_anchor=int(sum(n_self[b].values())))
    return res


# ---- gán phông độc lập: chuỗi cố định (do khâu quy mô chọn theo họ phông; ở đây coi là THAM SỐ và kiểm riêng ở script 03)
CHAIN = [("NomNaTong[fd]", "font_diffusion/fonts/NomNaTong-Regular.ttf"), ("HanaMinB", "font_diffusion/fonts/HanaMinB.ttf"),
         ("HanNomA", "font_diffusion/fonts/HAN NOM A.ttf"), ("NomNaTongLight2", "font_diffusion/fonts/NomNaTongLight2.ttf"),
         ("HanaMinA", "font_diffusion/fonts/HanaMinA.ttf"), ("NomNaTong[fonts]", "fonts/NomNaTong-Regular.ttf"),
         ("PlangothicP1", "fonts/PlangothicP1-Regular.ttf"), ("PlangothicP2", "fonts/PlangothicP2-Regular.ttf")]
PUA_SAFE = {"NomNaTong[fd]", "NomNaTong[fonts]", "NomNaTongLight2"}


def pil_bad(font_rel: str, chars, size: int = 64) -> set:
    """Chữ vẽ rỗng hoặc trùng ô .notdef của chính phông (PIL/FreeType), cùng cách kiểm của khâu quy mô nhưng mã viết lại."""
    from PIL import Image, ImageDraw, ImageFont
    import hashlib
    cm = Lb.cmap_union(Lb.REPO / font_rel)
    ft = ImageFont.truetype(str(Lb.REPO / font_rel), size)
    W = size * 3

    def draw(ch):
        im = Image.new("L", (W, W), 0)
        ImageDraw.Draw(im).text((size // 2, size // 2), ch, font=ft, fill=255)
        return im
    sigs = set()
    for cp in (0xFFFF, 0x2FFFE, 0x10FFFE):
        if cp not in cm:
            im = draw(chr(cp))
            if im.getbbox() is not None:
                sigs.add(hashlib.md5(im.tobytes()).hexdigest())
    bad = set()
    for c in chars:
        im = draw(c)
        if im.getbbox() is None or hashlib.md5(im.tobytes()).hexdigest() in sigs:
            bad.add(c)
    return bad


def assign_chain(chars, log=print) -> dict:
    """chữ -> tên phông đầu tiên trong CHAIN có glyph (cmap hợp) và vẽ được; PUA chỉ phông họ NomNaTong; None nếu không có."""
    cms = {n: Lb.cmap_union(Lb.REPO / p) for n, p in CHAIN}
    bad: set = set()
    for it in range(5):
        a = {}
        for c in chars:
            cp = ord(c)
            cand = [n for n, _ in CHAIN if (not Lb.is_pua(c) or n in PUA_SAFE)]
            a[c] = next((n for n in cand if cp in cms[n] and (n, c) not in bad), None)
        by = collections.defaultdict(list)
        for c, n in a.items():
            if n:
                by[n].append(c)
        new = set()
        for n, cs in by.items():
            pr = dict(CHAIN)[n]
            for c in pil_bad(pr, cs):
                if (n, c) not in bad:
                    new.add((n, c))
        log(f"  [assign] vong {it}: {sum(len(v) for v in by.values())} chữ có phông, {len(new)} mới vẽ rỗng/.notdef")
        if not new:
            return a
        bad |= new
    return a


# ============================================================================================ main
def main():
    t0 = time.time()
    inv = Lb.Inv()
    res: dict = dict(tao_luc=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0], pandas=pd.__version__)
    # ---------------------------------------------------------------- (0) kiểm kê đầu vào quyết định (số dòng do mã tìm, không gõ tay)
    T = Lb.HERE

    def ln(fname, snippet, base=T):
        for i, line in enumerate((base / fname).read_text(encoding="utf-8").split("\n"), 1):
            if snippet in line:
                return f"{(base / fname).relative_to(Lb.REPO).as_posix()}:{i}"
        return f"{fname}:KHÔNG_THẤY({snippet[:30]})"
    res["0_dau_vao_quyet_dinh"] = dict(
        danh_sach_chu=[
            dict(nguon="bảng nhãn 10 cuốn: STT=dataset_out/labels_final.csv(lọc book); 7 sách còn lại=prepared/<S>/dataset_out/labels_gated.csv (SAU chon_chu)", cot="book page column syllable ocr_char label tier rule chon_chu chon_chu_truoc",
                 dong=[ln("w_quy_mo_va_ngan_sach_lib.py", "PLAN_COLS = ["), ln("w_quy_mo_va_ngan_sach_lib.py", "L = pd.read_csv(labels_path(b)")]),
            dict(nguon="NHÃN TRƯỚC chon_chu suy từ cột chon_chu_truoc ('<tầng>|<nhãn>')", dong=[ln("w_quy_mo_va_ngan_sach_lib.py", "def label_truoc_chon_chu"), ln("w_quy_mo_va_ngan_sach_lib.py", "L[\"lab_pre\"] = label_truoc_chon_chu(L)")]),
            dict(nguon="từ điển R(âm) = Dict/QuocNgu_SinoNom.csv qua pipeline.gold_exact.common.R_of (không có nhãn người; tệp sinh 2026-08-19, trước dữ liệu người vào repo 20/09 và 27/09)",
                 dong=[ln("w_quy_mo_va_ngan_sach_lib.py", "C = T.lex()")]),
            dict(nguon="ô neo = rule bắt đầu s1_inter_s2_direct ∧ nhãn trước chon_chu == chữ kim ∧ ∈ R(âm) (tiên nghiệm xếp hạng R(âm))", dong=[ln("w_quy_mo_va_ngan_sach_lib.py", "def anchors_counter"), ln("w_quy_mo_va_ngan_sach_lib.py", "def plan_book")]),
            dict(nguon="gán phông: cmap fontTools + kiểm vẽ PIL trên font_diffusion/fonts/*, fonts/*; PUA chỉ họ NomNaTong", dong=[ln("w_quy_mo_va_ngan_sach_lib.py", "def robust_assign")]),
            dict(nguon="p11: chars = ranked lọc phông [:|S1(k=5) lọc phông|]; font_of_<cuốn>.tsv", dong=[ln("p11_dung_goi_toan_bo.py", "chars = F[\"ranked\"][:cap]")])],
        anh_phong_cach=[
            dict(nguon="bảng nhãn SAU chon_chu (load_all) -> neo/<ma>.csv", dong=[ln("w_hien_trang_du_lieu_lib.py", "def load_all"), ln("w_hien_trang_du_lieu_01_kiem_ke.py", "D = L.load_all(ma)")]),
            dict(nguon="dataset/<Bộ>/gold_exact.csv: cờ hình học f_blank f_cut f_two bleed_new trunc_new tall_new dup_bbox ov_heavy int_foreign one_char_ok crop_status (one_char_ok gồm ¬f_ink: trung vị mực theo (bộ,nhãn CUỐI))",
                 dong=[ln("w_hien_trang_du_lieu_05_tong_hop.py", "GEO = ["), ln("w_hien_trang_du_lieu_05_tong_hop.py", "def clean_flags"), ln("pipeline/gold_exact/signals_geom.py", "X[\"one_char_ok\"]", Lb.REPO)]),
            dict(nguon="bộ lọc ứng viên: tier==GOLD ∧ trong_pub ∧ cờ sạch ∧ cạnh ngắn ≥ 24; ±25 % kích thước trung vị; mực ∈ [p25,p75]; chữ ≥ 5 ô neo; xếp theo khoảng cách chuẩn hoá; KHÔNG lọc rule/chon_chu/gate_reason",
                 dong=[ln("w_hien_trang_du_lieu_05_tong_hop.py", "Ng = N[(N.tier == \"GOLD\")"), ln("w_hien_trang_du_lieu_05_tong_hop.py", "S = Rk[(Rk.h.between")]),
            dict(nguon="p11: ảnh phong cách = hạng 1, 2 của style_ung_vien/<cuốn>.csv -> crop_chuan_cache_128 -> Otsu nhị phân, ô vuông lề 10 %, 128x128", dong=[ln("p11_dung_goi_toan_bo.py", "def style_image"), ln("p11_dung_goi_toan_bo.py", "src = REPO / str(r.crop_chuan_cache_128)")])],
        dau_vao_la_nhan_nguoi_hay_suy_ra_tu_nhan_nguoi=dict(
            nhan_Nom_nguoi=("KHÔNG — không đường nào mở xlsx/manifest.tsv/nom_transcriptions/_BORG_NHAN_NGUOI/gt_char (kiểm bằng móc open() ở (a), (b))"),
            am_QN_la_phien_am_NGUOI_cua_4_bo=("CÓ, là ĐẦU VÀO hợp lệ của pipeline (không phải nhãn Nôm): B18/B34 cột ChuQN_txt (pipeline/tools/ingest_borg_book.py:27, XLSX_READ dòng 77); "
                                              "L16/TK trường translation + ô cột VoTT do người vẽ (pipeline/tools/ingest_ihr_book.py:11-14); TAP_DANH_GIA.md mục 3. Danh sách S1(k=5) lấy R(âm) của các âm này"),
            chon_chu_B18_B34=("mô hình + nguyên mẫu người học trên nhãn Nôm NGƯỜI của sách KIA (LOBO): config/chon_chu.yaml SachKinhThayCaBinh model hand_B34 human [B34]; SachDungLyHoThan model hand_B18 human [B18] "
                              "— nên bảng SAU chon_chu của B18/B34 bị ảnh hưởng gián tiếp bởi nhãn người của sách kia")))
    # ---------------------------------------------------------------- (a) chạy lại đường mã gốc dưới móc kiểm toán
    sys.path.insert(0, str(Lb.REPO / "lab" / "thu_nghiem_anh_chu" / "TN11_sinh_tuong_dong"))
    import w_quy_mo_va_ngan_sach_lib as QL  # noqa: E402
    mark = len(_OPENED)
    _OPENED.clear()
    W = QL.build_world(log=lambda *a, **k: None)
    plans_re = {b: QL.plan_book(W["cells"][b], W["Rm"], W["n_self"][b], W["n_oth"][b], W["g_oth"][b], "sach") for b in Lb.ORDER}
    opened_a = data_files_opened()
    res["a_duong_ma_goc_chay_lai"] = dict(
        tep_du_lieu_da_mo=[dict(tep=f, lan=n) for f, n in opened_a],
        tep_khop_mau_nhan_nguoi=[f for f, _ in opened_a if HUMAN_PAT.search(f)],
        ghi_chu="các tệp ngoài REPO/.venv và mã .py/.pyc/.so bị lọc; mở bằng open() (pandas/torch/json) đều được móc bắt")
    inv.check("a_duong_ma_goc_khong_mo_tep_khop_mau_nhan_nguoi", not res["a_duong_ma_goc_chay_lai"]["tep_khop_mau_nhan_nguoi"],
              res["a_duong_ma_goc_chay_lai"]["tep_khop_mau_nhan_nguoi"])
    pk = pickle.load(open(Lb.FULL / "quy_mo_va_ngan_sach" / "work" / "plans.pkl", "rb"))["plans"]
    same_pkl = {b: (plans_re[b]["ranked"] == pk[b]["sach"]["ranked"]) for b in Lb.ORDER}
    res["a_chay_lai_khop_plans_pkl"] = same_pkl
    inv.check("a_chay_lai_duong_goc_ranked_trung_plans.pkl_10_cuon", all(same_pkl.values()), [b for b, v in same_pkl.items() if not v])
    # dấu vết nhãn người trong cột đọc: cột đọc là PLAN_COLS; liệt kê tên cột của 4 bảng evaluation_only và đánh dấu cột lạ
    res["a_cot_doc"] = dict(PLAN_COLS_goc=QL.PLAN_COLS, ghi_chu="khâu quy mô đọc cột label/ocr_char/syllable/rule/tier/chon_chu/chon_chu_truoc; KHÔNG đọc gt_char/ref/f_hum/f_vW/slot/pos_*")
    # ---------------------------------------------------------------- (b) dựng lại độc lập
    _OPENED.clear()
    from pipeline.gold_exact import common as C  # chỉ R_key/R_of (Dict/QuocNgu_SinoNom.csv + core.text.*)
    pre = {b: load_table(b, "pre") for b in Lb.ORDER}
    P = plan_from_tables(pre, C.R_key, C.R_of)
    S15 = set().union(*[P[b]["S1k"] for b in Lb.ORDER])
    print(f"[b] ứng viên S1(5) hợp 10 cuốn: {len(S15)} chữ — gán phông...", flush=True)
    assign = assign_chain(sorted(S15))
    avail = {c for c, n in assign.items() if n}
    mine = {}
    for b in Lb.ORDER:
        p = P[b]
        rf = [c for c in p["ranked"] if c in avail]
        cap = len({c for c in p["S1k"] if c in avail})
        chars = rf[:cap]
        assert set(chars) == {c for c in p["S1k"] if c in avail}, f"{b}: tiền tố ≠ S1(5) lọc phông"
        mine[b] = chars
    opened_b = data_files_opened()
    res["b_duong_doc_lap_tep_da_mo"] = [dict(tep=f, lan=n) for f, n in opened_b]
    res["b_tep_khop_mau_nhan_nguoi"] = [f for f, _ in opened_b if HUMAN_PAT.search(f)]
    inv.check("b_duong_doc_lap_khong_mo_tep_khop_mau_nhan_nguoi", not res["b_tep_khop_mau_nhan_nguoi"], res["b_tep_khop_mau_nhan_nguoi"])
    labs_opened = sorted({f for f, _ in opened_b if "labels" in f})
    inv.check("b_duong_doc_lap_chi_doc_bang_truoc_chon_chu_va_labels_final(STT)", set(labs_opened) <= set(Lb.PREF.values()) and len(labs_opened) == len(set(Lb.PREF.values())),
              labs_opened)
    # so từng cuốn với chars_<b>.txt
    cmp = {}
    plan = json.loads((Lb.PACK / "plan.json").read_text(encoding="utf-8"))
    for b in Lb.ORDER:
        txt = [c for c in (Lb.SPEC / f"chars_{b}.txt").read_text(encoding="utf-8").split("\n") if c]
        pkc = list(plan["books"][b]["chars"]) + list(plan["books"][b]["uncovered"])
        # kế hoạch gói giữ thứ tự covered trước, uncovered sau: so tập + thứ tự covered
        eq_txt = (mine[b] == txt)
        diff_set = sorted(set(mine[b]) ^ set(txt))
        cmp[b] = dict(n_mine=len(mine[b]), n_chars_txt=len(txt), bang_nhau_tung_chu_va_thu_tu=eq_txt, so_chu_khac_tap=len(diff_set),
                      chu_khac=[f"U+{ord(c):05X}" for c in diff_set[:12]],
                      sha_mine=Lb.sha256_json(mine[b]), sha_chars_txt=Lb.sha256_json(txt),
                      sha_ranked_truoc_loc_phong_mine=Lb.sha256_json(P[b]["ranked"]),
                      n0_S0=len(P[b]["S0"]), n_S1_5=len(P[b]["S1k"]), n_anchor_truoc=P[b]["n_anchor"])
        if not eq_txt:
            # vị trí lệch đầu tiên
            i = next((i for i, (x, y) in enumerate(zip(mine[b], txt)) if x != y), min(len(mine[b]), len(txt)))
            cmp[b]["lech_dau_tien_o_vi_tri"] = i
    # mã băm kế hoạch trước lọc phông của khâu quy mô (buoc1_tap_chu.json) so với của ta
    b1 = json.load(open(Lb.FULL / "quy_mo_va_ngan_sach" / "buoc1_tap_chu.json", encoding="utf-8"))
    for b in Lb.ORDER:
        cmp[b]["sha_ranked_truoc_loc_phong_khau_quy_mo"] = b1["sha256_ke_hoach"]["sach"][b]
        cmp[b]["ranked_truoc_loc_trung_khau_quy_mo"] = (cmp[b]["sha_ranked_truoc_loc_phong_mine"] == b1["sha256_ke_hoach"]["sach"][b])
    res["b_so_voi_chars_txt"] = cmp
    res["b_so_cuon_trung_tuyet_doi_voi_chars_txt"] = sorted(b for b, v in cmp.items() if v["bang_nhau_tung_chu_va_thu_tu"])
    inv.check("b_dung_lai_doc_lap_trung_chars_txt_9_cuon_tru_KVK (từng chữ, đúng thứ tự)", all(v["bang_nhau_tung_chu_va_thu_tu"] for b, v in cmp.items() if b != "KVK"),
              {b: v["so_chu_khac_tap"] for b, v in cmp.items() if not v["bang_nhau_tung_chu_va_thu_tu"]})
    inv.check("b_ranked_truoc_loc_phong_trung_ma_bam_khau_quy_mo_9_cuon_tru_KVK", all(v["ranked_truoc_loc_trung_khau_quy_mo"] for b, v in cmp.items() if b != "KVK"),
              [b for b, v in cmp.items() if not v["ranked_truoc_loc_trung_khau_quy_mo"]])
    inv.check("b_B18_va_B34_trung_chars_txt_tuyet_doi (hai bộ có nhãn người cần tách chon_chu)", cmp["B18"]["bang_nhau_tung_chu_va_thu_tu"] and cmp["B34"]["bang_nhau_tung_chu_va_thu_tu"])
    # ---------------------------------------------------------------- (b3) KVK: giải thích 5 chữ lệch (np_geo xoá âm trong bảng SAU; bảng TRƯỚC còn âm rác)
    key = ["book", "page", "column", "nom_idx", "syl_idx"]
    kp = Lb.rd(Lb.REPO / Lb.PREF["KVK"], usecols=lambda c: c in key + Lb.PLAN_COLS)
    ko = Lb.rd(Lb.REPO / Lb.POST["KVK"], usecols=lambda c: c in key + ["syllable"])
    kp["_k"] = kp[key].agg("|".join, axis=1); ko["_k"] = ko[key].agg("|".join, axis=1)
    hy = kp.merge(ko[["_k", "syllable"]].rename(columns={"syllable": "syl_post"}), on="_k", how="left")
    syl_diff = hy[hy.syllable != hy.syl_post]
    hy["syllable"] = hy.syl_post
    P2 = plan_from_tables({**pre, "KVK": hy[Lb.PLAN_COLS]}, C.R_key, C.R_of)
    S2k = {c for c in P2["KVK"]["S1k"] if c in avail}
    kvk_lai = [c for c in P2["KVK"]["ranked"] if c in avail][:len(S2k)]
    txt_kvk = [c for c in (Lb.SPEC / "chars_KVK.txt").read_text(encoding="utf-8").split("\n") if c]
    extra = sorted(set(mine["KVK"]) - set(txt_kvk), key=ord)
    why = {}
    for c in extra:
        ss = Counter(s for s in syl_diff.syllable if c in C.R_of(s))
        why[f"U+{ord(c):05X}"] = dict(ss.most_common(3))
    res["b3_KVK_giai_thich"] = dict(o_am_khac_giua_bang_truoc_va_sau=int(len(syl_diff)), vi_du_am_truoc=dict(Counter(syl_diff.syllable).most_common(6)),
                                    am_sau_la=dict(Counter(syl_diff.syl_post).most_common(3)), chu_chi_co_o_duong_doc_lap=[f"U+{ord(c):05X}" for c in extra],
                                    chu_do_am_truoc_nao_sinh_ra=why, lai_bang_truoc_nhung_am_cua_bang_sau_bang_chars_txt=(kvk_lai == txt_kvk))
    inv.check("b3_KVK_lech_chi_do_am_np_geo: bảng TRƯỚC + cột âm của bảng SAU ⇒ trùng chars_KVK.txt từng chữ", kvk_lai == txt_kvk,
              f"{len(extra)} chữ thừa ở đường độc lập; {len(syl_diff)} ô khác âm")
    # ---------------------------------------------------------------- (b2) bảng trước chon_chu khớp cột chon_chu_truoc của bảng sau (cách khâu quy mô suy ra nhãn trước)
    chk = {}
    for b in Lb.ORDER:
        if b in Lb.STT:
            continue
        key = ["book", "page", "column", "nom_idx", "syl_idx"]
        A = Lb.rd(Lb.REPO / Lb.POST[b], usecols=lambda c: c in key + ["label", "tier", "rule", "chon_chu", "chon_chu_truoc", "syllable", "ocr_char"])
        B = Lb.rd(Lb.REPO / Lb.PREF[b], usecols=lambda c: c in key + ["label", "tier", "rule", "syllable", "ocr_char"])
        A["_k"] = A[key].agg("|".join, axis=1); B["_k"] = B[key].agg("|".join, axis=1)
        M = A.merge(B[["_k", "label", "tier", "rule", "syllable", "ocr_char"]], on="_k", suffixes=("", "_pre"), how="outer", indicator=True)
        both = M[M._merge == "both"]
        m = (both.chon_chu != "").to_numpy()
        parts = both.chon_chu_truoc[m].str.split("|", n=2, expand=True)
        lp = both.label.to_numpy(dtype=object).copy()
        lp[m] = parts[1].fillna("").to_numpy(dtype=object) if parts.shape[1] > 1 else ""
        chk[b] = dict(n_o=int(len(A)), chi_o_bang_sau=int((M._merge == "left_only").sum()), chi_o_bang_truoc=int((M._merge == "right_only").sum()),
                      o_chon_chu_khac_rong=int(m.sum()), nhan_doi_boi_chon_chu=int((both.label != both.label_pre).sum()),
                      ti_le_nhan_doi=round(float((both.label != both.label_pre).mean()), 4), tier_doi=int((both.tier != both.tier_pre).sum()),
                      rule_doi=int((both.rule != both.rule_pre).sum()), kim_doi=int((both.ocr_char != both.ocr_char_pre).sum()), am_doi=int((both.syllable != both.syllable_pre).sum()),
                      nhan_truoc_suy_tu_cot_chon_chu_truoc_khop_bang_truoc=int((lp != both.label_pre.to_numpy(dtype=object)).sum()) == 0)
    res["b2_chon_chu_doi_bang_nao"] = chk
    inv.check("b2_nhan_truoc_chon_chu_suy_tu_cot_khop_tep_truoc_7_cuon", all(v["nhan_truoc_suy_tu_cot_chon_chu_truoc_khop_bang_truoc"] for v in chk.values()))
    # ---------------------------------------------------------------- (c) độ nhạy: dựng từ bảng SAU chon_chu
    post = {b: load_table(b, "post") for b in Lb.ORDER}
    Q = plan_from_tables(post, C.R_key, C.R_of)
    S15q = set().union(*[Q[b]["S1k"] for b in Lb.ORDER])
    assign_q = assign_chain(sorted(S15q - S15), log=lambda *a: None)
    assign.update(assign_q)
    avail_q = {c for c, n in assign.items() if n}
    sens = {}
    for b in Lb.ORDER:
        s_pre = {c for c in P[b]["S1k"] if c in avail}
        s_post = {c for c in Q[b]["S1k"] if c in avail_q}
        r_pre, r_post = [c for c in P[b]["ranked"] if c in avail][:len(s_pre)], [c for c in Q[b]["ranked"] if c in avail_q][:len(s_post)]
        sens[b] = dict(n_pre=len(s_pre), n_post=len(s_post), chi_o_pre=len(s_pre - s_post), chi_o_post=len(s_post - s_pre),
                       so_chu_khac_tap=len(s_pre ^ s_post), ti_le_khac_tap_pct=round(100 * len(s_pre ^ s_post) / max(len(s_pre | s_post), 1), 2),
                       thu_tu_giong_nhau=(r_pre == r_post), vi_tri_chu_trong_top500_khac=int(len(set(r_pre[:500]) ^ set(r_post[:500]))),
                       S0_pre=len(P[b]["S0"]), S0_post=len(Q[b]["S0"]), S0_khac=len(P[b]["S0"] ^ Q[b]["S0"]))
    res["c_nhay_bang_sau_chon_chu"] = sens
    inv.check("c_danh_sach_B18_B34_THAT_SU_khac_neu_dung_bang_sau_chon_chu (chứng tỏ lựa chọn 'trước' có tác dụng)",
              sens["B18"]["so_chu_khac_tap"] > 0 and sens["B34"]["so_chu_khac_tap"] > 0, {b: sens[b]["so_chu_khac_tap"] for b in ("B18", "B34")})
    # ---------------------------------------------------------------- lưu dựng lại cho script 03
    rb = dict(mine=mine, ranked={b: P[b]["ranked"] for b in Lb.ORDER}, n0={b: P[b]["n0"] for b in Lb.ORDER}, best={b: P[b]["best"] for b in Lb.ORDER},
              e={b: dict(P[b]["e"]) for b in Lb.ORDER}, g={b: dict(P[b]["g"]) for b in Lb.ORDER}, demand={b: dict(P[b]["demand"]) for b in Lb.ORDER},
              S0={b: P[b]["S0"] for b in Lb.ORDER}, S1k={b: P[b]["S1k"] for b in Lb.ORDER}, assign=assign)
    with open(Lb.OUT / "rebuild.pkl", "wb") as f:
        pickle.dump(rb, f, protocol=4)
    # ---------------------------------------------------------------- (d) grep gói
    res["d_grep_goi"] = grep_pack()
    inv.check("d_ten_tep_trong_zip_khong_khop_mau_nhan_nguoi", not res["d_grep_goi"]["zip_ten_khop"], res["d_grep_goi"]["zip_ten_khop"])
    inv.check("d_json_goi_khong_chua_khoa_hay_chuoi_nhan_nguoi", not res["d_grep_goi"]["json_khop"], res["d_grep_goi"]["json_khop"])
    inv.check("d_siêu_dữ_liệu_PNG_và_safetensors_không_chứa_đường_dẫn_nhãn", not res["d_grep_goi"]["png_safetensors_khop"], res["d_grep_goi"]["png_safetensors_khop"])
    res["bat_bien"] = inv.summary()
    res["bat_bien_chi_tiet"] = inv.rows
    res["giay"] = round(time.time() - t0, 1)
    Lb.jdump(res, "01_ro_nhan_nguoi.json")
    print(f"[01] xong {res['giay']}s; bất biến {res['bat_bien']['dat']}/{res['bat_bien']['tong']} rớt={res['bat_bien']['rot']}")


def grep_pack() -> dict:
    pat = re.compile(r"(gt_char|nhan_nguoi|nhãn người|SinoNom_Char|\.xlsx|manifest\.tsv|nom_transcriptions|BORG_NHAN|evaluation_only|TAP_DANH_GIA|IHR|borg|ground[_ ]truth|human)", re.I)
    out = dict(mau=pat.pattern)
    # 1) tên tệp trong zip
    zn = []
    with zipfile.ZipFile(Lb.ZIP) as z:
        names = z.namelist()
    out["zip_so_muc"] = len(names)
    out["zip_ten_khop"] = [n for n in names if re.search(r"(nhan|human|gt_|truth|ground|xlsx|manifest\.tsv|labels|borg|ihr|\.csv$|\.tsv$|\.pkl$)", n, re.I)]
    out["zip_duoi_tep"] = dict(Counter(Path(n).suffix.lower() for n in names))
    # 2) JSON của gói: khoá + chuỗi (trừ danh sách chữ trong plan.json)
    jh = []
    for fn in ("plan.json", "manifest.json", "fonts.json"):
        t = (Lb.PACK / fn).read_text(encoding="utf-8")
        d = json.loads(t)
        keys = set()

        def walk(o, top=True):
            if isinstance(o, dict):
                for k, v in o.items():
                    keys.add(str(k))
                    walk(v, False)
            elif isinstance(o, list):
                for v in o[:5]:
                    walk(v, False)
        walk(d)
        for k in keys:
            if pat.search(k):
                jh.append(f"{fn}:khoa:{k}")
        # chuỗi ngắn (không phải chuỗi chữ dài) khớp mẫu
        for m in re.finditer(r'"([^"\\]{1,120})"', t):
            s = m.group(1)
            if pat.search(s) and len(s) < 100:
                jh.append(f"{fn}:chuoi:{s[:60]}")
    out["json_khop"] = sorted(set(jh))
    out["json_khoa_plan"] = sorted(json.loads((Lb.PACK / "plan.json").read_text(encoding="utf-8")).keys())
    # 3) mã trong gói: tệp .py ngoài mã FontDiffuser gốc (core/, fd_wrapper_fix.py, w_runner_khung_chay.py) và toàn bộ code/font_diffusion (mẫu hẹp)
    own = [Lb.PACK / "w_runner_khung_chay.py", Lb.PACK / "code" / "fd_wrapper_fix.py", Lb.PACK / "code" / "core" / "ranking" / "fontdiffusion_gen.py"]
    out["ma_rieng"] = {}
    for p in own:
        hits = []
        for i, ln in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
            if pat.search(ln):
                hits.append(f"{i}: {ln.strip()[:130]}")
        out["ma_rieng"][p.name] = hits
    strict = re.compile(r"(gt_char|SinoNom_Char|\.xlsx|manifest\.tsv|nom_transcriptions|BORG_NHAN|evaluation_only|TAP_DANH_GIA|IHR-NomDB|nhan_nguoi)", re.I)
    up = []
    for p in sorted((Lb.PACK / "code" / "font_diffusion").rglob("*.py")):
        for i, ln in enumerate(p.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
            if strict.search(ln):
                up.append(f"{p.relative_to(Lb.PACK).as_posix()}:{i}: {ln.strip()[:100]}")
    out["ma_fontdiffuser_goc_khop_mau_hep"] = up
    # 4) siêu dữ liệu PNG (khối tEXt/iTXt) và đầu safetensors
    from PIL import Image
    hits = []
    for p in sorted((Lb.PACK / "data").rglob("*.png")):
        with Image.open(p) as im:
            info = {k: str(v)[:80] for k, v in im.info.items()}
        if info:
            hits.append(f"{p.relative_to(Lb.PACK).as_posix()}: {info}")
    out["png_co_siêu_dữ_liệu"] = hits
    st = []
    for p in sorted((Lb.PACK / "ckpt").glob("*.safetensors")):
        with open(p, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            head = json.loads(f.read(n).decode("utf-8"))
        meta = head.get("__metadata__", {})
        if meta:
            st.append(f"{p.name}: {json.dumps(meta, ensure_ascii=False)[:200]}")
    out["safetensors_metadata"] = st
    out["png_safetensors_khop"] = [h for h in hits + st if pat.search(h)]
    return out


if __name__ == "__main__":
    main()
