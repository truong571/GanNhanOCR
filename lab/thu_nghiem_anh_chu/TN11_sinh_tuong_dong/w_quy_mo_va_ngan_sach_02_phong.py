"""w_quy_mo_va_ngan_sach_02_phong.py — BƯỚC 2: phủ phông (cmap fontTools) của NomNaTong-Regular.ttf và của chuỗi phông dự phòng. 0 API, CPU.

Hỏi: bao nhiêu chữ của S0 / S1(k) / S2 mỗi cuốn có glyph trong NomNaTong-Regular.ttf (phông nội dung mặc định của wrapper FontDiffusionGenerator:
core/ranking/fontdiffusion_gen.py `get_available_chars_for_font` BỎ IM LẶNG chữ thiếu), và chuỗi phông nào thêm được bao nhiêu chữ.
Chuỗi đề xuất theo QUY TẮC (không cảm tính): họ Song-Minh trước (xếp tham lam theo số chữ thêm được trên U = ∪ (S0 ∪ S2) của 10 cuốn, NomNaTong
[font_diffusion] cố định đầu), rồi họ Khải, họ Gothic (Plangothic) cuối cùng — để ảnh nội dung của chữ hiếm khác kiểu chữ NomNaTong ít nhất có thể.
Họ kiểu chữ phân theo TÊN phông/giấy phép, chưa kiểm hình bằng mắt (quy ước CLAUDE.md: không dùng LLM nhìn ảnh để đo); độ đậm nét tương đối so với
NomNaTong đo bằng mã (tỉ lệ mực trung bình trên chữ chung) là chỉ báo phụ.
Mã PUA mang nghĩa theo từng phông ⇒ chữ PUA chỉ vẽ bằng phông họ NomNaTong (PUA_SAFE trong lib); chữ PUA chỉ có ở phông khác bị loại, ghi ở chu_khong_phong.ly_do.
Kiểm vẽ thật (PIL/FreeType): chữ được gán mà vẽ rỗng / trùng ô .notdef ⇒ chuyển phông kế tiếp (lặp đến khi ổn định).

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_quy_mo_va_ngan_sach_02_phong.py
Ra: phong.json, work/phong.pkl.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w_quy_mo_va_ngan_sach_lib as L  # noqa: E402

TH_GAIN_MIN = 1       # phông chỉ vào chuỗi nếu thêm ≥ chừng này chữ trên U


def greedy_family(base_set: set, fonts: list, cmaps: dict, U: set, order: list):
    """Thêm tham lam các phông trong `fonts` (cùng họ) cho đến khi không phông nào thêm ≥ TH_GAIN_MIN chữ chưa phủ."""
    covered = set(base_set)
    pool = list(fonts)
    while pool:
        rem = U - covered
        gains = {f: sum(1 for c in rem if ord(c) in cmaps[f]) for f in pool}
        f = max(pool, key=lambda x: (gains[x], -list(L.FONT_FILES).index(x)))      # hoà: phông khai báo trước trong FONT_FILES
        if gains[f] < TH_GAIN_MIN:
            break
        order.append(f)
        covered |= {c for c in rem if ord(c) in cmaps[f]}
        pool.remove(f)
    return covered


def ink_ratio(font_key: str, base_key: str, chars: list, size: int = 64) -> tuple:
    """Tỉ lệ mực trung bình (phông / NomNaTong) trên các chữ cả hai cùng có. Chỉ báo độ đậm nét, KHÔNG phải độ giống hình dạng."""
    from PIL import Image, ImageDraw, ImageFont
    fa = ImageFont.truetype(str(L.REPO / L.FONT_FILES[font_key]), size)
    fb = ImageFont.truetype(str(L.REPO / L.FONT_FILES[base_key]), size)
    W = size * 3
    ra = []
    for c in chars:
        v = []
        for f in (fa, fb):
            im = Image.new("L", (W, W), 0)
            ImageDraw.Draw(im).text((size // 2, size // 2), c, font=f, fill=255)
            v.append(float(np.asarray(im, dtype=np.float32).mean()) / 255.0)
        if v[1] > 0 and v[0] > 0:
            ra.append(v[0] / v[1])
    return (float(np.median(ra)) if ra else float("nan"), len(ra))


def main():
    t0 = time.time()
    d = pickle.load(open(L.OUT / "work" / "plans.pkl", "rb"))
    P = d["plans"]
    cm = L.all_cmaps()
    U = set().union(*[P[b]["sach"]["S0"] | P[b]["sach"]["S2"] for b in L.ORDER])
    S0u = set().union(*[P[b]["sach"]["S0"] for b in L.ORDER])
    res = dict(font_file={k: dict(duong_dan=p, mb=round((L.REPO / p).stat().st_size / 1e6, 1), n_cmap=len(cm[k]), ho=L.FONT_FAMILY[k])
                          for k, p in L.FONT_FILES.items()}, invariants=[])
    inv = res["invariants"]

    def check(name, ok, detail=""):
        inv.append(dict(ten=name, ok=bool(ok), chi_tiet=str(detail)[:300]))
        print(f"[inv] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)

    # ---- phủ từng phông trên U, S0 toàn kho
    for k in L.FONT_FILES:
        res["font_file"][k]["phu_U"] = sum(1 for c in U if ord(c) in cm[k])
        res["font_file"][k]["phu_S0_hop"] = sum(1 for c in S0u if ord(c) in cm[k])
    # ---- chọn chuỗi theo quy tắc họ
    order = [L.BASE_FONT]
    covered = {c for c in U if ord(c) in cm[L.BASE_FONT]}
    for fam in ("Song-Minh", "Khải", "Gothic"):
        fonts = [k for k in L.FONT_FILES if L.FONT_FAMILY[k] == fam and k not in order]
        covered = greedy_family(covered, fonts, cm, U, order)
    res["chuoi_de_xuat"] = order
    res["chuoi_de_xuat_ho"] = [L.FONT_FAMILY[k] for k in order]
    # so sánh chuỗi tham lam thuần (bỏ qua họ) để biết cái giá của việc ưu tiên họ
    gorder = [L.BASE_FONT]
    gcov = {c for c in U if ord(c) in cm[L.BASE_FONT]}
    gcov = greedy_family(gcov, [k for k in L.FONT_FILES if k != L.BASE_FONT], cm, U, gorder)
    res["chuoi_tham_lam_thuan"] = gorder
    # so sánh với chuỗi sẵn có của repo (pipeline/ground_truth/audit_grid._FONT_PREFERENCE) và NomNaTong một mình, trên U / hợp S0 / hợp S1(5)
    S15u = set().union(*[P[b]["sach"]["S1"][5] for b in L.ORDER])
    repo_chain = [L.BASE_FONT, "Han-nom Minh 1.42.otf", "HanaMinA.ttf", "HanaMinB.ttf", "PlangothicP1-Regular.ttf", "PlangothicP2-Regular.ttf"]
    so_sanh = {}
    for cname, ch, pua in (("chi_NomNaTong", [L.BASE_FONT], False), ("chuoi_audit_grid_repo", repo_chain, False), ("chuoi_de_xuat", order, False),
                           ("chuoi_de_xuat_PUA_an_toan", order, True)):
        row = {}
        for uname, uni in (("U", U), ("hop_S0", S0u), ("hop_S1_k5", S15u)):
            a = L.assign_fonts(sorted(uni), ch, cm, None, set(L.PUA_SAFE) if pua else None)
            cnt = Counter(a.values())
            row[uname] = dict(n=len(uni), khong_phong=int(cnt.get(None, 0)), gan_gothic=int(sum(v for k, v in cnt.items() if k and L.FONT_FAMILY[k] == "Gothic")),
                              so_phong_dung=int(sum(1 for k in cnt if k)))
        so_sanh[cname] = row
    res["so_sanh_chuoi"] = so_sanh
    res["so_sanh_chuoi_ghi_chu"] = "chỉ theo cmap (chưa kiểm vẽ); 'chuoi_de_xuat_PUA_an_toan' áp thêm quy tắc PUA chỉ phông họ NomNaTong — đây là chuỗi được dùng"

    # ---- gán phông có kiểm vẽ trên U (chuỗi đề xuất) và trên U chỉ NomNaTong
    assign, bad, hist = L.robust_assign(sorted(U), order, cm, pua_safe=set(L.PUA_SAFE))
    res["kiem_ve"] = dict(vong=hist, so_chu_bi_chuyen_phong=len(bad))
    inc = Counter(v for v in assign.values())
    res["tang_theo_phong_U"] = [dict(phong=f, ho=L.FONT_FAMILY[f], them=int(inc.get(f, 0))) for f in order]
    res["khong_phong_nao_U"] = int(inc.get(None, 0))
    unc = sorted((c for c, f in assign.items() if f is None), key=ord)
    pua_loai = [c for c in unc if L.is_pua_char(c) and any(ord(c) in cm[f] for f in order)]       # có glyph ở phông ngoài họ NomNaTong nhưng mã PUA chưa xác minh
    res["chu_khong_phong"] = [dict(c=c, cp=f"U+{ord(c):05X}", khoi=L.block_of(c),
                                   ly_do="PUA chỉ có ở phông ngoài họ NomNaTong (mã PUA theo phông, chưa xác minh cùng bảng)" if c in set(pua_loai) else "không phông nào trong chuỗi có glyph")
                              for c in unc]
    res["chu_khong_phong_theo_khoi"] = dict(Counter(L.block_of(c) for c in unc))
    res["chu_khong_phong_theo_ly_do"] = dict(khong_phong_nao=len(unc) - len(pua_loai), pua_chi_phong_ngoai_NomNaTong=len(pua_loai))
    res["quy_tac_PUA"] = dict(phong_an_toan=list(L.PUA_SAFE), ghi_chu="chữ PUA chỉ vẽ bằng phông họ NomNaTong; HAN NOM A/B, HanaMin, Plangothic bị loại cho PUA")
    base_cov = sum(1 for c in U if ord(c) in cm[L.BASE_FONT])
    res["U"] = dict(n=len(U), nomna=base_cov, nomna_pct=round(100 * base_cov / len(U), 2),
                    cascade=len(U) - len(unc), cascade_pct=round(100 * (len(U) - len(unc)) / len(U), 2))
    # độ phủ NomNaTong theo khối Unicode (U và S0)
    blk_u, blk_b, blk_s0, blk_s0b, blk_s0c = Counter(), Counter(), Counter(), Counter(), Counter()
    for c in U:
        blk_u[L.block_of(c)] += 1
        blk_b[L.block_of(c)] += int(ord(c) in cm[L.BASE_FONT])
    for c in S0u:
        blk_s0[L.block_of(c)] += 1
        blk_s0b[L.block_of(c)] += int(ord(c) in cm[L.BASE_FONT])
        blk_s0c[L.block_of(c)] += int(assign.get(c) is not None)
    res["theo_khoi_unicode"] = {k: dict(U=blk_u[k], U_nomna=blk_b[k], S0=blk_s0.get(k, 0), S0_nomna=blk_s0b.get(k, 0), S0_cascade=blk_s0c.get(k, 0))
                                for k in sorted(set(blk_u) | set(blk_s0))}

    # ---- avail
    avail_base = {c for c in U if ord(c) in cm[L.BASE_FONT]}
    avail_cas = {c for c, f in assign.items() if f is not None}
    # ---- phủ mỗi cuốn × chiến lược: NomNaTong một mình, tầng phông; tăng theo phông
    per = {}
    for b in L.ORDER:
        p = P[b]["sach"]
        strat = {"S0": p["S0"], "S1_k3": p["S1"][3], "S1_k5": p["S1"][5], "S1_k8": p["S1"][8], "S2_hop_S0": p["S2"] | p["S0"]}
        row = {}
        for name, s in strat.items():
            n = len(s)
            nb = sum(1 for c in s if c in avail_base)
            nc = sum(1 for c in s if c in avail_cas)
            incb = Counter(assign.get(c) for c in s)
            row[name] = dict(n=n, nomna=nb, nomna_pct=round(100 * nb / n, 2), cascade=nc, cascade_pct=round(100 * nc / n, 2),
                             tang={f: int(incb.get(f, 0)) for f in order if incb.get(f, 0)}, khong_phong=int(incb.get(None, 0)))
        per[b] = row
    res["theo_sach"] = per
    # hợp toàn kho theo chiến lược
    hop = {}
    for name, g in [("S0", lambda p: p["S0"]), ("S1_k3", lambda p: p["S1"][3]), ("S1_k5", lambda p: p["S1"][5]), ("S1_k8", lambda p: p["S1"][8]),
                    ("S2_hop_S0", lambda p: p["S2"] | p["S0"])]:
        u = set().union(*[g(P[b]["sach"]) for b in L.ORDER])
        incu = Counter(assign.get(c) for c in u)
        hop[name] = dict(n=len(u), nomna=sum(1 for c in u if c in avail_base), cascade=sum(1 for c in u if c in avail_cas),
                         tang={f: int(incu.get(f, 0)) for f in order if incu.get(f, 0)}, khong_phong=int(incu.get(None, 0)))
        hop[name]["nomna_pct"] = round(100 * hop[name]["nomna"] / hop[name]["n"], 2)
        hop[name]["cascade_pct"] = round(100 * hop[name]["cascade"] / hop[name]["n"], 2)
    res["hop_nhat"] = hop
    # ảnh hưởng lên ô: bao nhiêu ô có chữ kim không phông nào vẽ được
    # (tính ở bước 3 cùng nhãn); ở đây chỉ số chữ kim PUA/thiếu phông trong S0 mỗi cuốn
    # ---- độ đậm nét tương đối (chỉ báo phụ)
    rng = np.random.default_rng(0)
    ink = {}
    for f in order[1:]:
        common = sorted((c for c in U if ord(c) in cm[f] and ord(c) in cm[L.BASE_FONT]), key=ord)
        if common:
            common = [common[i] for i in sorted(rng.choice(len(common), size=min(600, len(common)), replace=False))]
        ink[f] = ink_ratio(f, L.BASE_FONT, common) if common else (float("nan"), 0)
    res["do_dam_net_tuong_doi"] = {f: dict(trung_vi=round(v[0], 3), n_chu=v[1]) for f, v in ink.items()}
    res["ghi_chu_do_dam"] = "trung vị của (tỉ lệ mực phông / tỉ lệ mực NomNaTong) trên chữ chung, 64 px; chỉ phản ánh độ đậm nét"
    # ---- invariants
    ok_sub = all(assign[c] is not None and ord(c) in cm[assign[c]] for c in assign if assign[c] is not None)
    check("mọi chữ được gán đều có trong cmap của phông gán", ok_sub)
    check("cascade ≥ NomNaTong một mình trên U, mọi cuốn × chiến lược",
          all(per[b][s]["cascade"] >= per[b][s]["nomna"] for b in per for s in per[b]) and res["U"]["cascade"] >= res["U"]["nomna"])
    blank_f, notdef_f = L.render_check(assign)
    check("sau gán cuối: 0 chữ vẽ rỗng/.notdef ở phông gán", len(blank_f) + len(notdef_f) == 0, f"rỗng {len(blank_f)}, notdef {len(notdef_f)}")
    # kiểm độc lập phủ NomNaTong bằng đường mã của FontDiffuser (is_char_in_font)
    sys.path.insert(0, str(L.REPO / "font_diffusion"))
    try:
        from src.tools.utils import is_char_in_font
        rs = np.random.default_rng(1)
        smp = [sorted(U)[i] for i in rs.choice(len(U), size=400, replace=False)]
        mism = sum(is_char_in_font(str(L.REPO / L.FONT_FILES[L.BASE_FONT]), c) != (c in avail_base) for c in smp)
        check("is_char_in_font của FontDiffuser trùng cmap của ta (400 chữ ngẫu nhiên)", mism == 0, f"lệch {mism}")
    except Exception as e:  # noqa: BLE001
        check("is_char_in_font của FontDiffuser trùng cmap của ta", False, f"không nạp được: {type(e).__name__}: {e}")
    res["thoi_gian_giay"] = round(time.time() - t0, 1)
    (L.OUT / "phong.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(L.OUT / "work" / "phong.pkl", "wb") as f:
        pickle.dump(dict(order=order, assign=assign, avail_base=avail_base, avail_cas=avail_cas, bad=bad), f, protocol=4)
    print(json.dumps(dict(chuoi=order, tang=res["tang_theo_phong_U"], U=res["U"], khong_phong=res["khong_phong_nao_U"],
                          hop={k: (v["n"], v["nomna_pct"], v["cascade_pct"]) for k, v in hop.items()}), ensure_ascii=False, indent=1))
    print(f"[b2] xong [{time.time() - t0:.0f}s]; FAIL: {[i['ten'] for i in inv if not i['ok']] or 'không'}")


if __name__ == "__main__":
    main()
