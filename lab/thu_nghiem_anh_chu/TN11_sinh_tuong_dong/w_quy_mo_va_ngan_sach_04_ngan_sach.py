"""w_quy_mo_va_ngan_sach_04_ngan_sach.py — BƯỚC 4: giờ GPU theo chiến lược, ngân sách mặc định, kế hoạch chia ca, danh sách mẫu, ket_qua.json. 0 API, CPU.

Mô hình chi phí (đề bài + log Kaggle): 2,7 giây/ảnh/GPU (measure_out/_tn11/kaggle/TN11/log_B34.txt: 2,63–2,74; log_B18.txt: 2,67–2,75), hai GPU chạy song song
(mỗi GPU một tiến trình) ⇒ "giờ GPU" = Σ ảnh × 2,7/3600 (cộng mọi GPU), "giờ phiên" (đồng hồ tường) = giờ GPU / 2. Ca ≤ 9 giờ phiên gồm 0,25 giờ chi phí cố định
(nạp FontDiffuser 20,9 s đo ở log_B34.txt:103; phần còn lại — pip, nén kết quả — là GIẢ ĐỊNH). Hạn mức Kaggle (phiên ≤ 12 giờ, ≈ 30 giờ GPU/tuần) lấy từ đề bài, chưa xác minh.
Số ảnh = số (cuốn, chữ) × số phong cách; chữ đã lọc theo chuỗi phông đề xuất (chữ không phông nào có không sinh được).

Ngân sách mặc định = QUY TẮC "S1(k = 5), 1 phong cách/cuốn" (tái lập được từ nhãn mới nhất, không cần nhãn người); so với phân bổ tối ưu (quy hoạch động nhóm-ba-lô) trên đường cong phủ.
Kế hoạch ca: (a) "theo_manh" = ca s lấy phần thứ s (theo hạng ưu tiên) của MỌI cuốn, các mảnh chia hai làn GPU bằng LPT; (b) "theo_cuon" = nguyên cuốn mỗi làn.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_quy_mo_va_ngan_sach_04_ngan_sach.py
Ra: ket_qua.json, chu_uu_tien_B34.tsv, chu_uu_tien_B34_chars.txt.
"""
from __future__ import annotations

import itertools
import json
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w_quy_mo_va_ngan_sach_lib as L  # noqa: E402

K_DEFAULT = 5
STYLES_DEFAULT = 1
SAMPLE_BOOK = "B34"
LANE_CAP = int((L.SHIFT_H - L.SHIFT_OVERHEAD_H) * 3600.0 / L.SEC_PER_IMG)     # ảnh tối đa mỗi làn trong một ca ≤ 9 giờ
GRID = 100


def hours(n):
    return dict(anh=int(n), gio_gpu=round(L.gpu_hours(n), 2), gio_phien_2gpu=round(L.wall_hours(n), 2),
                so_ca_toi_thieu=int(math.ceil(n / (L.N_GPU * LANE_CAP))) if n else 0)


def lpt(items: list, n_bins: int):
    """items = [(khóa, kích thước)] -> (bins, loads) theo LPT (lớn trước vào thùng ít nhất)."""
    bins = [[] for _ in range(n_bins)]; loads = [0] * n_bins
    for key, n in sorted(items, key=lambda kv: -kv[1]):
        k = int(np.argmin(loads)); bins[k].append((key, n)); loads[k] += n
    return bins, loads


def shift_layered(caps: dict, styles: int, n_shifts: int) -> list:
    """Ca s = phần thứ s (theo hạng) của mọi cuốn; mảnh chia hai làn bằng LPT."""
    out = []
    for s in range(n_shifts):
        shards = []
        for b, cap in caps.items():
            a, e = round(cap * s / n_shifts), round(cap * (s + 1) / n_shifts)
            if e > a:
                shards.append(((b, a, e), (e - a) * styles))
        bins, loads = lpt(shards, L.N_GPU)
        out.append(dict(ca=s + 1, lan=[[dict(cuon=k[0], hang_tu=k[1] + 1, hang_den=k[2], anh=n) for k, n in bn] for bn in bins],
                        anh_moi_lan=loads, gio_phien=round(max(loads) * L.SEC_PER_IMG / 3600 + L.SHIFT_OVERHEAD_H, 2),
                        gio_gpu=round(sum(loads) * L.SEC_PER_IMG / 3600, 2)))
    return out


def shift_by_book(sizes: dict, styles: int) -> list:
    items = [(b, n * styles) for b, n in sizes.items()]
    total = sum(n for _, n in items)
    ns = max(1, math.ceil(total / (L.N_GPU * LANE_CAP)))
    while True:
        bins, loads = lpt(items, ns * L.N_GPU)
        if max(loads) <= LANE_CAP:
            break
        ns += 1
    order = sorted(range(len(bins)), key=lambda i: -loads[i])
    out = []
    for s in range(ns):
        pair = [order[s], order[len(order) - 1 - s]]
        out.append(dict(ca=s + 1, lan=[[dict(cuon=k, anh=n) for k, n in bins[i]] for i in pair], anh_moi_lan=[loads[i] for i in pair],
                        gio_phien=round(max(loads[i] for i in pair) * L.SEC_PER_IMG / 3600 + L.SHIFT_OVERHEAD_H, 2),
                        gio_gpu=round(sum(loads[i] for i in pair) * L.SEC_PER_IMG / 3600, 2)))
    return out


def main():
    t0 = time.time()
    pl = pickle.load(open(L.OUT / "work" / "plans.pkl", "rb"))
    ph = pickle.load(open(L.OUT / "work" / "phong.pkl", "rb"))
    s1 = json.load(open(L.OUT / "buoc1_tap_chu.json", encoding="utf-8"))
    sf = json.load(open(L.OUT / "phong.json", encoding="utf-8"))
    su = json.load(open(L.OUT / "phu.json", encoding="utf-8"))
    P = pl["plans"]
    inv = []

    def check(name, ok, detail=""):
        inv.append(dict(ten=name, ok=bool(ok), chi_tiet=str(detail)[:300]))
        print(f"[inv] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)

    Fc = {b: L.filtered(P[b]["sach"], lambda c: c in ph["avail_cas"]) for b in L.ORDER}
    Fn = {b: L.filtered(P[b]["sach"], lambda c: c in ph["avail_base"]) for b in L.ORDER}
    SK = ["S0"] + [f"S1_k{k}" for k in L.KS_ALL] + ["S2_hop_S0"]

    def size(F, b, name):
        p = F[b]
        return len(p["S0"]) if name == "S0" else len(p["S2"] | p["S0"]) if name == "S2_hop_S0" else len(p["S1"][int(name.split("k")[1])])

    # ---------------------------------------------------------------- 1. chi phí theo chiến lược
    chi_phi = {}
    for name in SK:
        row = dict(theo_cuon={b: size(Fc, b, name) for b in L.ORDER}, theo_cuon_NomNaTong_don={b: size(Fn, b, name) for b in L.ORDER})
        tot = sum(row["theo_cuon"].values()); totn = sum(row["theo_cuon_NomNaTong_don"].values())
        row["cap_cuon_chu_cascade"] = tot; row["cap_cuon_chu_NomNaTong_don"] = totn
        row["1_phong_cach"] = hours(tot); row["2_phong_cach"] = hours(2 * tot); row["3_phong_cach"] = hours(3 * tot)
        row["1_phong_cach_NomNaTong_don"] = hours(totn)
        row["chua_loc_phong"] = sum(len(P[b]["sach"]["S0"]) if name == "S0" else len(P[b]["sach"]["S2"] | P[b]["sach"]["S0"]) if name == "S2_hop_S0"
                                    else len(P[b]["sach"]["S1"][int(name.split("k")[1])]) for b in L.ORDER)
        chi_phi[name] = row

    # ---------------------------------------------------------------- 2. phủ theo chiến lược (thước tốt nhất sẵn có mỗi cuốn)
    def truth_of(b):
        R = su["sach"][b]
        if b in L.HUMAN:
            return "nguoi[ĐO]", R["duong_cong"]["nguoi"]["cascade"]
        if b in ("L83", "KVK"):
            return "diban[ƯL]", R["duong_cong"]["diban"]["cascade"]
        return "gold-giu-ra[xấp xỉ]", R["giu_ra"]["gold"]

    def cov_at(b, name):
        return truth_of(b)[1]["chien_luoc"][name]

    phu_chien_luoc = {}
    for name in SK:
        per = {b: round(cov_at(b, name), 5) for b in L.ORDER}
        phu_chien_luoc[name] = dict(theo_cuon=per, tb_4_bo_nhan_nguoi=round(float(np.mean([per[b] for b in L.HUMAN])), 5),
                                    tb_10_bo_hon_hop=round(float(np.mean(list(per.values()))), 5),
                                    min_10_bo=round(min(per.values()), 5))
    thuoc = {b: truth_of(b)[0] for b in L.ORDER}

    # ---------------------------------------------------------------- 3. cửa sổ 30–40 giờ GPU
    cua_so = {}
    for m in (1, 2, 3):
        rows = []
        for k in L.KS_ALL:
            n = m * chi_phi[f"S1_k{k}"]["cap_cuon_chu_cascade"]
            rows.append(dict(k=k, **hours(n), trong_cua_so_30_40=bool(30 <= L.gpu_hours(n) <= 40), duoi_40=bool(L.gpu_hours(n) <= 40),
                             phu_tb_10_bo=phu_chien_luoc[f"S1_k{k}"]["tb_10_bo_hon_hop"], phu_tb_4_bo_nguoi=phu_chien_luoc[f"S1_k{k}"]["tb_4_bo_nhan_nguoi"]))
        n0 = m * chi_phi["S0"]["cap_cuon_chu_cascade"]
        rows.insert(0, dict(k="S0", **hours(n0), trong_cua_so_30_40=bool(30 <= L.gpu_hours(n0) <= 40), duoi_40=bool(L.gpu_hours(n0) <= 40),
                            phu_tb_10_bo=phu_chien_luoc["S0"]["tb_10_bo_hon_hop"], phu_tb_4_bo_nguoi=phu_chien_luoc["S0"]["tb_4_bo_nhan_nguoi"]))
        cua_so[f"{m}_phong_cach"] = rows

    # ---------------------------------------------------------------- 4. phân bổ tối ưu (quy hoạch động nhóm-ba-lô) so với quy tắc đồng đều S1(k=5)
    fine_len = len(su["FINE_NS"])
    styles1 = {b: 1 for b in L.ORDER}
    jmin = {b: math.ceil(len(Fc[b]["S0"]) / GRID) for b in L.ORDER}      # ràng buộc: mọi chữ kim/nhãn (S0 lọc phông) đều có ảnh sinh

    def with_floor(src):
        out = {}
        for b in L.ORDER:
            c = np.array(src[b], dtype=float); c[: jmin[b]] = -1e6; out[b] = c
        return out
    best_fine = {b: truth_of(b)[1]["fine"] for b in L.ORDER}                 # thước tốt nhất sẵn có: nhãn người (4 bộ), dị bản (2), GOLD giữ-ra (4) — dùng để CHẤM
    gold_fine = {b: su["sach"][b]["giu_ra"]["gold"]["fine"] for b in L.ORDER}  # thước KHÔNG nhãn người: phủ giữ-ra theo khối trang của nhãn GOLD (10 cuốn)

    def at(src, b, N):
        return float(src[b][min(int(math.ceil(N / GRID)), fine_len - 1)])

    def score(caps):
        per = {b: round(at(best_fine, b, caps[b]), 5) for b in L.ORDER}
        return per, round(float(np.mean(list(per.values()))), 5), round(float(np.mean([per[b] for b in L.HUMAN])), 5)

    # kiểm DP bằng vét cạn trên ví dụ nhỏ
    rng = np.random.default_rng(0)
    okbf = True
    for _ in range(30):
        cs = {f"b{i}": np.sort(rng.random(5)) for i in range(3)}
        st = {f"b{i}": int(rng.integers(1, 3)) for i in range(3)}
        U_ = int(rng.integers(3, 12))
        caps_, obj_ = L.knapsack_caps(cs, 1, U_, st)
        best = max(sum(cs[f"b{i}"][j[i]] for i in range(3)) for j in itertools.product(range(5), repeat=3)
                   if sum(j[i] * st[f"b{i}"] for i in range(3)) <= U_)
        okbf &= abs(obj_ - best) < 1e-9 and sum(caps_[f"b{i}"] * st[f"b{i}"] for i in range(3)) <= U_
    check("quy hoạch động nhóm-ba-lô = vét cạn trên 30 ví dụ nhỏ ngẫu nhiên (mục tiêu và ràng buộc ngân sách)", okbf)

    uni = {b: size(Fc, b, f"S1_k{K_DEFAULT}") for b in L.ORDER}
    M_uni_grid = sum(math.ceil(v / GRID) * GRID for v in uni.values())
    per_rule, tb10_rule, tb4_rule = score(uni)
    budgets = (("30_gio_gpu", int(30 * 3600 / L.SEC_PER_IMG)), ("35_gio_gpu", int(35 * 3600 / L.SEC_PER_IMG)),
               ("40_gio_gpu", int(40 * 3600 / L.SEC_PER_IMG)), (f"bang_chi_phi_S1_k{K_DEFAULT}", M_uni_grid))
    pb_free, pb_best = {}, {}
    for label, M in budgets:
        un = (M // len(L.ORDER)) // GRID
        per_un = {b: round(at(best_fine, b, max(un, jmin[b]) * GRID), 5) for b in L.ORDER}
        for name, src, dst in (("khong_nhan_nguoi", gold_fine, pb_free), ("theo_thuoc_tot_nhat", best_fine, pb_best)):
            caps, _ = L.knapsack_caps(with_floor(src), GRID, M // GRID, styles1)
            per, tb10, tb4 = score(caps)
            dst[label] = dict(ngan_sach_anh=int(M), gio_gpu=round(L.gpu_hours(M), 2), tran_moi_cuon=caps, tong_anh=int(sum(caps.values())),
                              phu_theo_cuon=per, phu_tb_10_bo=tb10, phu_tb_4_bo_nguoi=tb4,
                              dong_deu_N_moi_cuon=int(un * GRID), dong_deu_phu_tb_10_bo=round(float(np.mean(list(per_un.values()))), 5),
                              dong_deu_phu_tb_4_bo_nguoi=round(float(np.mean([per_un[b] for b in L.HUMAN])), 5))
    for dst in (pb_free, pb_best):
        dst[f"bang_chi_phi_S1_k{K_DEFAULT}"].update(quy_tac_S1_k5_phu_theo_cuon=per_rule, quy_tac_S1_k5_phu_tb_10_bo=tb10_rule, quy_tac_S1_k5_phu_tb_4_bo_nguoi=tb4_rule)
    kr = f"bang_chi_phi_S1_k{K_DEFAULT}"
    check("DP theo thước tốt nhất ≥ quy tắc S1(k=5) và ≥ DP không nhãn người, cùng ngân sách, chấm cùng thước (lưới 100 chữ)",
          pb_best[kr]["phu_tb_10_bo"] >= tb10_rule - 1e-9 and pb_best[kr]["phu_tb_10_bo"] >= pb_free[kr]["phu_tb_10_bo"] - 1e-9,
          f"{pb_best[kr]['phu_tb_10_bo']} >= {tb10_rule}, {pb_free[kr]['phu_tb_10_bo']}")
    check("DP: mọi trần ≥ |S0| lọc phông và tổng ảnh ≤ ngân sách (cả hai bản)",
          all(all(r["tran_moi_cuon"][b] >= len(Fc[b]["S0"]) for b in L.ORDER) and r["tong_anh"] <= r["ngan_sach_anh"] for dst in (pb_free, pb_best) for r in dst.values()))
    check("DP không nhãn người: đầu vào chỉ là đường cong giữ-ra của nhãn GOLD (khoá 'giu_ra'/'gold' của phu.json), không khoá 'nguoi'/'diban'",
          all(len(gold_fine[b]) == fine_len for b in L.ORDER))

    # ---------------------------------------------------------------- 5. ngân sách mặc định + kế hoạch ca
    caps_def = dict(uni)
    n_def = sum(caps_def.values()) * STYLES_DEFAULT
    n_shifts = max(1, math.ceil(n_def / (L.N_GPU * LANE_CAP)))
    ca_manh = shift_layered(caps_def, STYLES_DEFAULT, n_shifts)
    ca_cuon = shift_by_book(caps_def, STYLES_DEFAULT)
    # kiểm: mỗi (cuốn, hạng) đúng một lần; mỗi ca ≤ 9 giờ phiên; tổng ảnh khớp
    cover = {b: np.zeros(caps_def[b], dtype=np.int16) for b in L.ORDER}
    for s in ca_manh:
        for lane in s["lan"]:
            for it in lane:
                cover[it["cuon"]][it["hang_tu"] - 1: it["hang_den"]] += 1
    check("kế hoạch theo_manh: mỗi (cuốn, hạng) ≤ trần được gán đúng một lần", all(int(c.min()) == 1 and int(c.max()) == 1 for c in cover.values()))
    check("kế hoạch theo_manh: tổng ảnh các ca = tổng ảnh mặc định", sum(sum(s["anh_moi_lan"]) for s in ca_manh) == n_def)
    check("kế hoạch theo_manh: mọi ca ≤ 9 giờ phiên", all(s["gio_phien"] <= L.SHIFT_H + 1e-9 for s in ca_manh), [s["gio_phien"] for s in ca_manh])
    got = Counter()
    for s in ca_cuon:
        for lane in s["lan"]:
            for it in lane:
                got[it["cuon"]] += it["anh"]
    check("kế hoạch theo_cuon: mỗi cuốn đúng một lần, đủ số ảnh", all(got[b] == caps_def[b] * STYLES_DEFAULT for b in L.ORDER) and len(got) == len(L.ORDER))
    check("kế hoạch theo_cuon: mọi ca ≤ 9 giờ phiên", all(s["gio_phien"] <= L.SHIFT_H + 1e-9 for s in ca_cuon), [s["gio_phien"] for s in ca_cuon])

    # ---------------------------------------------------------------- 6. hợp nhất theo ngân sách mặc định
    lists_def = {b: Fc[b]["ranked"][:caps_def[b]] for b in L.ORDER}
    check("danh sách mặc định mỗi cuốn = đúng S1(k=5) lọc phông (tính chất tiền tố)", all(set(lists_def[b]) == Fc[b]["S1"][K_DEFAULT] for b in L.ORDER))
    cnt = Counter(c for v in lists_def.values() for c in v)
    hop_def = dict(so_cap=sum(len(v) for v in lists_def.values()), hop=len(cnt), chu_o_ge2_cuon=sum(1 for v in cnt.values() if v >= 2),
                   chu_o_10_cuon=sum(1 for v in cnt.values() if v == len(L.ORDER)))
    assign = ph["assign"]
    inc = Counter(assign[c] for c in cnt)
    hop_def["tang_theo_phong"] = {f: int(inc.get(f, 0)) for f in ph["order"]}
    hop_def["khong_phong"] = int(inc.get(None, 0))
    per_font_cap = {b: dict(Counter(assign[c] for c in lists_def[b])) for b in L.ORDER}
    ky_uoc = {f"U+{ord(c):05X}": c for c in cnt if assign[c] is None}

    stt_b = ("stt2", "stt4", "stt11")
    stt_union = set().union(*[set(lists_def[b]) for b in stt_b])
    chung_stt = dict(tong_cap=sum(len(lists_def[b]) for b in stt_b), hop=len(stt_union),
                     tiet_kiem_anh=sum(len(lists_def[b]) for b in stt_b) - len(stt_union),
                     tiet_kiem_gio_gpu=round(L.gpu_hours(sum(len(lists_def[b]) for b in stt_b) - len(stt_union)), 2),
                     ghi_chu="TUỲ CHỌN, chưa xác minh ba quyển cùng một tay chép; bỏ yêu cầu 'phong cách riêng từng cuốn' cho STT")
    cn = {}
    for name in ("S0", "S1_k3", "S1_k5", "S1_k8", "S2_hop_S0"):
        cc, nn = chi_phi[name]["cap_cuon_chu_cascade"], chi_phi[name]["cap_cuon_chu_NomNaTong_don"]
        row = dict(anh_cascade=cc, anh_NomNaTong_don=nn, them_do_cascade=cc - nn, ty_le_them_pct=round(100 * (cc - nn) / cc, 2),
                   gio_gpu_cascade=round(L.gpu_hours(cc), 2), gio_gpu_NomNaTong_don=round(L.gpu_hours(nn), 2))
        if name != "S2_hop_S0":
            cov_c = [su["sach"][b]["duong_cong"]["nguoi"]["cascade"]["chien_luoc"][name] for b in L.HUMAN]
            cov_n = [su["sach"][b]["duong_cong"]["nguoi"]["nomna"]["chien_luoc"][name] for b in L.HUMAN]
            row["phu_chu_nguoi_tb4_cascade"] = round(float(np.mean(cov_c)), 5); row["phu_chu_nguoi_tb4_NomNaTong_don"] = round(float(np.mean(cov_n)), 5)
            row["phu_ung_vien_tb10_cascade"] = round(float(np.mean([su["sach"][b]["phu_ung_vien_cascade"]["chien_luoc"][name] for b in L.ORDER])), 5)
            row["phu_ung_vien_tb10_NomNaTong_don"] = round(float(np.mean([su["sach"][b]["phu_ung_vien_nomna"]["chien_luoc"][name] for b in L.ORDER])), 5)
        cn[name] = row

    # ---------------------------------------------------------------- 7. danh sách mẫu của một cuốn
    sb = SAMPLE_BOOK
    pb = P[sb]["sach"]
    rk = Fc[sb]["ranked"]
    capb = caps_def[sb]
    with open(L.OUT / "chu_uu_tien_B34.tsv", "w", encoding="utf-8", newline="") as f:
        f.write("hang\tchu\tma_unicode\tkhoi\ttrong_S0\tso_o_bang_chung\thang_tien_nghiem_tot_nhat\ttrong_S1_k3\ttrong_S1_k5\ttrong_S1_k8\tphong\ttrong_ngan_sach_mac_dinh\n")
        for i, c in enumerate(rk, 1):
            f.write(f"{i}\t{c}\tU+{ord(c):05X}\t{L.block_of(c)}\t{int(c in pb['S0'])}\t{pb['e'].get(c, 0)}\t{pb['best'].get(c, '')}\t"
                    f"{int(c in Fc[sb]['S1'][3])}\t{int(c in Fc[sb]['S1'][5])}\t{int(c in Fc[sb]['S1'][8])}\t{assign[c]}\t{int(i <= capb)}\n")
    (L.OUT / "chu_uu_tien_B34_chars.txt").write_text("\n".join(rk[:capb]) + "\n", encoding="utf-8")
    check("danh sách mẫu B34: tệp chars.txt = đúng S1(k=5) lọc phông, không trùng", set(rk[:capb]) == Fc[sb]["S1"][K_DEFAULT] and len(set(rk[:capb])) == capb)

    # ---------------------------------------------------------------- 8. phương án ngân sách (so sánh)
    n_ft_steps = 3000
    ft_gpu_h_per_book = n_ft_steps * L.FT_SEC_PER_STEP / 3600
    def pa(label, n_img, ghi_chu, extra_gpu_h=0.0, cov=None):
        g = L.gpu_hours(n_img) + extra_gpu_h
        return dict(ten=label, anh=int(n_img), gio_gpu=round(g, 2), gio_phien_2gpu=round(g / L.N_GPU, 2),
                    so_ca=int(math.ceil((g / L.N_GPU) / (L.SHIFT_H - L.SHIFT_OVERHEAD_H))), trong_30_40=bool(30 <= g <= 40), duoi_40=bool(g <= 40),
                    phu_tb_10_bo=cov, ghi_chu=ghi_chu)
    cs = lambda k: chi_phi[k]["cap_cuon_chu_cascade"]
    PA = [
        pa("S0 x 1 phong cach", cs("S0"), "chỉ chữ pipeline đã đọc/gán", cov=phu_chien_luoc["S0"]["tb_10_bo_hon_hop"]),
        pa("S0 x 2 phong cach", 2 * cs("S0"), "hai ảnh phong cách/cuốn, không mở rộng ứng viên", cov=phu_chien_luoc["S0"]["tb_10_bo_hon_hop"]),
        pa("S1(3) x 1 phong cach", cs("S1_k3"), "", cov=phu_chien_luoc["S1_k3"]["tb_10_bo_hon_hop"]),
        pa("S1(5) x 1 phong cach [MAC DINH]", cs("S1_k5"), "quy tắc mặc định", cov=phu_chien_luoc["S1_k5"]["tb_10_bo_hon_hop"]),
        pa("S1(8) x 1 phong cach", cs("S1_k8"), "vượt 40 giờ GPU", cov=phu_chien_luoc["S1_k8"]["tb_10_bo_hon_hop"]),
        pa("S2 x 1 phong cach", cs("S2_hop_S0"), "cận trên; không khả thi", cov=phu_chien_luoc["S2_hop_S0"]["tb_10_bo_hon_hop"]),
        pa("S1(5) x 1 + tinh chinh 3000 buoc x 10 cuon", cs("S1_k5"), f"tinh chỉnh {n_ft_steps} bước x {L.FT_SEC_PER_STEP} s/bước = {ft_gpu_h_per_book:.2f} giờ GPU/cuốn "
           "(log_B34.txt:139: 10.327 s); kết quả đo trước: tinh chỉnh nhẹ làm xấu đi", extra_gpu_h=10 * ft_gpu_h_per_book, cov=phu_chien_luoc["S1_k5"]["tb_10_bo_hon_hop"]),
        pa("S1(3) x 1 + 2 phong cach cho S0 cua 4 bo nhan nguoi", cs("S1_k3") + sum(len(Fc[b]["S0"]) for b in L.HUMAN),
           "phong cách thứ hai chỉ cho chữ S0 của B18/B34/L16/TK để đo độ nhạy kiểu nét", cov=phu_chien_luoc["S1_k3"]["tb_10_bo_hon_hop"]),
    ]
    ft_only = dict(gio_gpu_moi_cuon=round(ft_gpu_h_per_book, 2), gio_gpu_10_cuon=round(10 * ft_gpu_h_per_book, 2),
                   con_lai_cho_sinh_anh_trong_40h=round(40 - 10 * ft_gpu_h_per_book, 2),
                   anh_toi_da_cho_sinh_trong_40h=int((40 - 10 * ft_gpu_h_per_book) * 3600 / L.SEC_PER_IMG))

    # bản literal (đúng định nghĩa đề bài: cột label cuối ∪ kim): phủ phông để đối chiếu; mọi chữ phải nằm trong tập U đã xét phông ở bước 2
    lit_font = {}
    ok_sub_U = True
    for b in L.ORDER:
        pl_ = P[b]["literal"]
        strat_l = {"S0": pl_["S0"], "S1_k3": pl_["S1"][3], "S1_k5": pl_["S1"][5], "S1_k8": pl_["S1"][8], "S2_hop_S0": pl_["S2"] | pl_["S0"]}
        lit_font[b] = {}
        for name, st in strat_l.items():
            ok_sub_U &= all(c in assign for c in st)
            n_ = len(st)
            lit_font[b][name] = dict(n=n_, nomna_pct=round(100 * sum(1 for c in st if c in ph["avail_base"]) / n_, 2),
                                     cascade_pct=round(100 * sum(1 for c in st if c in ph["avail_cas"]) / n_, 2))
    check("mọi chữ của S0/S1/S2 bản literal nằm trong tập U đã xét phông (bước 2)", ok_sub_U)
    # phương án hỗn hợp theo bộ có thước đo: B18/B34 k=8, L16/TK k=3, còn lại k=5 (CHỌN k THEO ĐƯỜNG CONG NHÃN NGƯỜI => chỉ tham khảo)
    mix_k = {b: K_DEFAULT for b in L.ORDER}
    mix_k.update({"B18": 8, "B34": 8, "L16": 3, "TK": 3})
    mix_sizes = {b: size(Fc, b, f"S1_k{mix_k[b]}") for b in L.ORDER}
    n_mix = sum(mix_sizes.values())
    mix_cov = {b: round(cov_at(b, f"S1_k{mix_k[b]}"), 5) for b in L.ORDER}
    rule_cov = {b: round(cov_at(b, f"S1_k{K_DEFAULT}"), 5) for b in L.ORDER}
    hon_hop = dict(k_theo_cuon=mix_k, tran_moi_cuon=mix_sizes, **hours(n_mix), phu_theo_cuon=mix_cov, phu_quy_tac_S1_k5_theo_cuon=rule_cov,
                   phu_tb_10_bo=round(float(np.mean(list(mix_cov.values()))), 5), phu_tb_4_bo_nguoi=round(float(np.mean([mix_cov[b] for b in L.HUMAN])), 5),
                   phu_tb_10_bo_quy_tac=round(float(np.mean(list(rule_cov.values()))), 5), phu_tb_4_bo_nguoi_quy_tac=round(float(np.mean([rule_cov[b] for b in L.HUMAN])), 5),
                   ghi_chu="k chọn theo đường cong chữ người của 4 bộ (B18/B34 còn thiếu phủ; L16/TK đã bão hoà) ⇒ trần chọn bằng nhãn người, chỉ tham khảo")
    check("phương án hỗn hợp k: ≤ 40 giờ GPU", L.gpu_hours(n_mix) <= 40, round(L.gpu_hours(n_mix), 2))

    # ---------------------------------------------------------------- 9. tổng hợp JSON
    theo_sach = {}
    for b in L.ORDER:
        t1 = s1["theo_sach"][b]; tf = sf["theo_sach"][b]
        theo_sach[b] = dict(n_o=t1["n_o"], n_am=t1["n_am"], n_am_co_R=t1["n_am_co_R"],
                            S0_literal=t1["S0_literal"], S0_sach=t1["S0_sach"],
                            S1_literal={k: t1["S1_literal"][k] for k in ("3", "5", "8")}, S1_sach={k: t1["S1_sach"][k] for k in ("3", "5", "8")},
                            S2=t1["S2"], S2_hop_S0=t1["S2_hop_S0_sach"],
                            loc_phong_cascade={k: size(Fc, b, k) for k in ("S0", "S1_k3", "S1_k5", "S1_k8", "S2_hop_S0")},
                            phu_phong=tf, phu_phong_literal=lit_font[b], thuoc_do=thuoc[b])
    duong = {}
    for b in L.ORDER:
        R = su["sach"][b]; duong[b] = {}
        for t, row in R["duong_cong"].items():
            duong[b][t] = dict(nhan=row["nhan"], n_o=row["n_o"], ty_le_chu_dung_bang_kim=row["ty_le_chu_dung_bang_kim"],
                               cascade=dict(NS=row["cascade"]["NS"], chien_luoc={k: row["cascade"]["chien_luoc"][k] for k in ("S0", "S1_k3", "S1_k5", "S1_k8", "S2_hop_S0")},
                                            tran=row["cascade"]["tran"], N_dat=row["cascade"]["N_dat"]),
                               NomNaTong_don=dict(NS=row["nomna"]["NS"], chien_luoc={k: row["nomna"]["chien_luoc"][k] for k in ("S0", "S1_k3", "S1_k5", "S1_k8", "S2_hop_S0")},
                                                  tran=row["nomna"]["tran"]),
                               ci95_cum_trang_cascade=row.get("ci95_cum_trang_cascade"),
                               bien_the_V1plus=(dict(NS=row["bien_the_V1plus_cascade"]["NS"], tran=row["bien_the_V1plus_cascade"]["tran"],
                                                     chien_luoc={k: row["bien_the_V1plus_cascade"]["chien_luoc"][k] for k in ("S0", "S1_k3", "S1_k5", "S1_k8", "S2_hop_S0")})
                                                if "bien_the_V1plus_cascade" in row else None))
        duong[b]["_giu_ra_theo_khoi_trang"] = {t: dict(n_o=r["n_o"], NS=r["NS"], chien_luoc={k: r["chien_luoc"][k] for k in ("S0", "S1_k3", "S1_k5", "S1_k8", "S2_hop_S0")}, tran=r["tran"])
                                               for t, r in R["giu_ra"].items()}
        duong[b]["_phu_ung_vien_cascade"] = dict(NS=R["phu_ung_vien_cascade"]["NS"], ca_tap_NS=R["phu_ung_vien_cascade"]["ca_tap_NS"],
                                                 chien_luoc={k: R["phu_ung_vien_cascade"]["chien_luoc"][k] for k in ("S0", "S1_k3", "S1_k5", "S1_k8", "S2_hop_S0")},
                                                 ca_tap_chien_luoc={k: R["phu_ung_vien_cascade"]["ca_tap_chien_luoc"][k] for k in ("S0", "S1_k3", "S1_k5", "S1_k8", "S2_hop_S0")})
    all_inv = []
    for part, name in ((s1, "b1"), (sf, "b2"), (su, "b3")):
        all_inv += [dict(buoc=name, **i) for i in part["invariants"]]
    all_inv += [dict(buoc="b4", **i) for i in inv]
    res = dict(
        meta=dict(tao="TN11 / hướng quy_mo_va_ngan_sach", ngay="2026-10-04", script=sorted(p.name for p in Path(__file__).parent.glob("w_quy_mo_va_ngan_sach_*.py")),
                  nguon_nhan=s1["nguon"], ghi_chu_mode="mọi số 'sach' = nhãn trước chon_chu ∪ kim (không ảnh hưởng gián tiếp của nhãn người); 'literal' = cột label cuối (đối chiếu)",
                  chuoi_phong=ph["order"], thuoc_phu=thuoc),
        gia_dinh=dict(giay_moi_anh=L.SEC_PER_IMG, so_gpu=L.N_GPU, tran_ca_gio=L.SHIFT_H, chi_phi_co_dinh_ca_gio=L.SHIFT_OVERHEAD_H, phien_toi_da_gio=L.SESSION_MAX_H,
                      han_muc_tuan_gio=L.WEEK_QUOTA_H, anh_toi_da_moi_lan_moi_ca=LANE_CAP, giay_moi_buoc_tinh_chinh=L.FT_SEC_PER_STEP,
                      nguon="đề bài + measure_out/_tn11/kaggle/TN11/log_B34.txt, log_B18.txt; hạn mức Kaggle chưa xác minh ở hướng này"),
        theo_sach=theo_sach, hop_nhat_buoc1=s1["hop_nhat"], phong=dict(chuoi_de_xuat=sf["chuoi_de_xuat"], so_sanh_chuoi=sf["so_sanh_chuoi"],
                                                                     tang_theo_phong_U=sf["tang_theo_phong_U"], U=sf["U"], hop_nhat=sf["hop_nhat"],
                                                                     chu_khong_phong=sf["chu_khong_phong"], do_dam=sf["do_dam_net_tuong_doi"],
                                                                     font_file=sf["font_file"], theo_khoi_unicode=sf["theo_khoi_unicode"]),
        chi_phi_theo_chien_luoc=chi_phi, phu_theo_chien_luoc=phu_chien_luoc, cua_so_30_40=cua_so, duong_cong_phu=duong, NS=su["NS"],
        phan_bo_khong_nhan_nguoi=pb_free, phan_bo_theo_thuoc_tot_nhat_CHI_THAM_KHAO=pb_best, cascade_so_voi_NomNaTong_don=cn, tuy_chon_chung_phong_cach_STT=chung_stt,
        ngan_sach_mac_dinh=dict(quy_tac=f"S1(k={K_DEFAULT}) lọc phông, {STYLES_DEFAULT} phong cách/cuốn", tran_moi_cuon=caps_def, phong_cach_moi_cuon=STYLES_DEFAULT,
                                tong_anh=n_def, **hours(n_def), hop_nhat=hop_def, theo_phong_moi_cuon=per_font_cap, chu_khong_phong_trong_ds=ky_uoc,
                                phu_theo_cuon={b: round(cov_at(b, f"S1_k{K_DEFAULT}"), 5) for b in L.ORDER},
                                phu_tb_10_bo=phu_chien_luoc[f"S1_k{K_DEFAULT}"]["tb_10_bo_hon_hop"], phu_tb_4_bo_nguoi=phu_chien_luoc[f"S1_k{K_DEFAULT}"]["tb_4_bo_nhan_nguoi"]),
        phuong_an=PA, phuong_an_hon_hop_k=hon_hop, tinh_chinh=ft_only, ke_hoach_ca=dict(theo_manh=ca_manh, theo_cuon=ca_cuon, so_ca=n_shifts),
        danh_sach_mau=dict(cuon=sb, tsv="chu_uu_tien_B34.tsv", chars_txt="chu_uu_tien_B34_chars.txt", n_chu_mac_dinh=capb, n_chu_toan_danh_sach=len(rk)),
        invariants=all_inv, thoi_gian_giay=round(time.time() - t0, 1))
    nf = [i for i in all_inv if not i["ok"]]
    res["tom_tat_invariants"] = dict(tong=len(all_inv), fail=len(nf), ten_fail=[i["ten"] for i in nf][:20])
    (L.OUT / "ket_qua.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[b4] xong [{time.time() - t0:.0f}s]; {len(all_inv)} invariant, FAIL: {[i['ten'] for i in nf] or 'không'}")


if __name__ == "__main__":
    main()
