"""T7 — CẦU NỐI TỰ DẠNG HAI CHIỀU: đo trước, quyết sau.

CÂU HỎI
-------
`consensus.py:100` bắc cầu bằng `similar_dict.get(ocr_char)` — MỘT CHIỀU. Nhưng
`Dict/SinoNom_Similar.csv` là bảng "20 chữ giống nhất" của mỗi chữ, tức một danh sách
k-láng-giềng, nên nó BẤT ĐỐI XỨNG: X có thể nằm trong top-20 của Y mà Y không nằm trong
top-20 của X. Mọi cầu nối chỉ tồn tại theo chiều ngược hiện đang bị bỏ qua.

Đo được trên bộ vừa dựng: 600 ô đang bị giữ lại có ĐÚNG MỘT cầu theo chiều ngược và
KHÔNG có cầu nào theo chiều xuôi. Ví dụ `lấy/柩→穊`, `biển/渡→𣷷`, `cho/床→咮`.

VÌ SAO KHÔNG THỂ TRẢ LỜI "ĐÚNG BAO NHIÊU" MỘT CÁCH TRỰC TIẾP
------------------------------------------------------------
Dự án KHÔNG có ground truth người (docs/KE_HOACH_TONG_THE_2026-08-22.md §0). Nên không
có phép đo nào cho ra "đúng 87%". Thứ đo được là: **600 ô này có hành xử giống hệt cái
tầng cầu nối đã được chấp nhận hay không**, dưới con mắt của một kênh KHÔNG tham gia
gán nhãn.

Kênh đó là `sem_score` (nghĩa Hán trong Unihan `kDefinition`). Tính chất đã đo
(docs/NGHIEN_CUU_UNIHAN_KDEFINITION_2026-08-22.md):

    điểm > 0,05  ->  độ chính xác 100,0%  (nhưng chỉ bắt được 303/2002 cặp đúng)
    điểm = 0     ->  KHÔNG NÓI LÊN GÌ — 47,7% cặp ĐÚNG cũng cho 0

Hệ quả bắt buộc phải nhớ khi đọc kết quả bên dưới:

    TỶ LỆ XÁC NHẬN CAO   = bằng chứng THUẬN, dùng được
    TỶ LỆ XÁC NHẬN THẤP  = KHÔNG phải bằng chứng NGHỊCH

Nên thí nghiệm này KHÔNG BAO GIỜ có thể kết luận "cầu ngược sai". Nó chỉ có thể kết luận
"cầu ngược được xác nhận NGANG tầng đã chấp nhận" hoặc "không đủ bằng chứng để nới".
Đó là bất đối xứng thật của công cụ, không phải thận trọng khách sáo.

CỔNG HỢP LỆ — kiểm TRƯỚC khi đọc kết quả
-----------------------------------------
G1  cỡ mẫu >= N_MIN. Dưới ngưỡng thì mọi khác biệt đều là nhiễu.
G2  KÊNH CHẤM PHẢI CÓ LỰC TRÊN CHÍNH DỮ LIỆU NÀY. Kiểm bằng cách chấm tầng cầu XUÔI đã
    được chấp nhận và một ĐỐI CHỨNG XÁO TRỘN (thay chữ cầu bằng một chữ khác lấy ngẫu
    nhiên trong đúng tập ứng viên của âm đó). Nếu cầu xuôi KHÔNG vượt đối chứng thì kênh
    chấm mù trên dữ liệu này -> báo "KHÔNG ĐO ĐƯỢC", không được suy diễn tiếp.

    🔴 G2 NHƯ TRÊN ĐÃ CHẠY VÀ KHÔNG QUA (2026-08-25): cầu xuôi 15,1% vs xáo trộn 10,8%
       = 1,40x < 2,0x. Theo đúng đăng ký, phán quyết của lượt đó là KHÔNG ĐO ĐƯỢC, và
       phán quyết ấy GIỮ NGUYÊN trong hồ sơ. Không xoá, không viết lại.

G2' SỬA THIẾT KẾ CỔNG (đăng ký lại 2026-08-25, SAU khi đã thấy kết quả — đọc kèm cảnh
    báo hậu nghiệm ở dưới). Lý do sửa KHÔNG phụ thuộc vào kết quả: một cổng hợp lệ phải
    kiểm kênh chấm trên quần thể ta TIN (đối chứng dương), chứ không phải trên quần thể
    dùng làm MỐC SO SÁNH. Hai câu hỏi khác nhau bị G2 gộp làm một:
        · "phép đo có hợp lệ không?"  -> đối chứng dương = GOLD TRỰC TIẾP
        · "cầu ngược có bằng cầu xuôi không?" -> đó là LUẬT QUYẾT ĐỊNH, không phải cổng
    Cầu xuôi yếu (1,40x) là một PHÁT HIỆN VỀ CHÍNH TẦNG CẦU XUÔI — luật cầu nối chọn chữ
    theo HÌNH DẠNG còn kênh này chấm theo NGHĨA — chứ không phải bằng chứng rằng phép đo
    hỏng. Cổng sửa lại:
        G2'a  GOLD TRỰC TIẾP phải vượt đối chứng >= CONTROL_MIN_RATIO   (đối chứng dương)
        G2'b  CHÍNH quần thể đang xét phải tách khỏi đối chứng của nó ở p < P_MAX

    ⚠️ CẢNH BÁO HẬU NGHIỆM, phải đọc cùng mọi con số bên dưới: sai sót thiết kế này được
       nhận ra SAU khi đã nhìn thấy kết quả thuận. Dù lý do sửa là chính đáng và độc lập
       với kết quả, việc sửa cổng rồi quyết trên CÙNG một lượt nhìn là yếu hơn hẳn một
       phép thử đăng ký sạch. Kết quả dưới đây là GỢI Ý MẠNH, không phải bằng chứng chốt.
G3  bảng tự dạng có được SẮP theo độ giống không. Nếu không thì thứ hạng KHÔNG phải đại
    lượng độ giống và tuyệt đối không được dùng làm bằng chứng (cùng ràng buộc G3.1 của
    T5). Kiểm bằng tương quan thứ hạng của các cặp có mặt ở CẢ HAI chiều.

LUẬT QUYẾT ĐỊNH — chốt TRƯỚC khi chạy
--------------------------------------
NỚI sang hai chiều CHỈ KHI cả ba đồng thời:
    (a) tỷ lệ xác nhận của cầu NGƯỢC không thấp hơn cầu XUÔI quá CONFIRM_MAX_DROP
    (b) cầu NGƯỢC vượt đối chứng xáo trộn ít nhất CONTROL_MIN_RATIO lần
    (c) sản lượng thêm >= YIELD_MIN ô — dưới mức đó không đáng nới một chốt chặn
Không đủ -> GIỮ MỘT CHIỀU. "Không đủ bằng chứng để nới" là một kết luận hợp lệ.

RÀNG BUỘC ĐẦU RA — y như T4/T5
------------------------------
Lab này KHÔNG đổi bộ nhãn đã công bố. `BRIDGE_TWO_WAY` mặc định TẮT. Đầu ra tối đa là
một khuyến nghị kèm số. Đổi tier là đổi bộ giao nộp, phải do người quyết.

    python -m pipeline.lab.t7_bridge_2way            # chạy phép đo
    python -m pipeline.lab.t7_bridge_2way --csv ...  # kèm xuất 600 ô ra tệp để xem tay
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import random
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "dataset_out" / "labels_final.csv"
DICT_QN = REPO / "Dict" / "QuocNgu_SinoNom.csv"
DICT_SIM = REPO / "Dict" / "SinoNom_Similar.csv"

# ---------------------------------------------------------------- CỜ TẮT-MỞ
# Mặc định TẮT: bộ đã công bố dựng bằng cầu MỘT CHIỀU, và lab không được đụng vào nó.
# Bật lên chỉ để ĐO xem hai chiều cho thêm gì. Muốn đưa vào sản xuất thì phải sửa
# consensus.py và dựng lại pipeline — có chủ ý, không phải hệ quả phụ của một lần chạy lab.
BRIDGE_TWO_WAY = False

# ------------------------------------------------------- NGƯỠNG CHỐT TRƯỚC
N_MIN = 300                 # G1
CONTROL_MIN_RATIO = 2.0     # (b)
CONFIRM_MAX_DROP = 0.02     # (a) — điểm tuyệt đối
YIELD_MIN = 300             # (c)
P_MAX = 0.01                # G2'b — quần thể đang xét phải tách khỏi đối chứng của nó
TAU_CONFIRM = 0.05          # cùng ngưỡng "chính xác 100%" của sem_score
SEED = 20260825

GIU_LAI = ("REVIEW", "SILVER_uncalibrated")


def nrm(s: str | None) -> str:
    return unicodedata.normalize("NFC", (s or "").strip().lower())


def doc_tu_dien() -> dict[str, list[str]]:
    """âm Quốc ngữ -> các chữ mà từ điển cho là đọc âm ấy (giữ thứ tự xuất hiện)."""
    out: dict[str, list[str]] = {}
    with open(DICT_QN, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            a, c = nrm(r["QuocNgu"]), (r["SinoNom"] or "").strip()
            if a and c:
                out.setdefault(a, [])
                if c not in out[a]:
                    out[a].append(c)
    return out


def doc_tu_dang() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(xuôi, ngược). xuôi[X] = danh sách top-K của X. ngược[Y] = các X có Y trong top-K."""
    xuoi: dict[str, list[str]] = {}
    nguoc: dict[str, list[str]] = {}
    with open(DICT_SIM, encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        c0, c1 = rd.fieldnames[0], rd.fieldnames[1]
        for r in rd:
            a = (r[c0] or "").strip()
            try:
                lst = [str(x).strip() for x in ast.literal_eval(r[c1] or "[]")]
            except (ValueError, SyntaxError):
                continue
            if not a:
                continue
            xuoi[a] = lst
            for b in lst:
                if b:
                    nguoc.setdefault(b, []).append(a)
    return xuoi, nguoc


def cau_noi(ocr_char: str, ung_vien: list[str],
            xuoi: dict[str, list[str]], nguoc: dict[str, list[str]],
            hai_chieu: bool) -> list[str]:
    """Trả các chữ vừa GIỐNG `ocr_char` vừa là cách đọc hợp lệ của âm.

    Chiều xuôi là đúng cái consensus.py đang làm. Chiều ngược chỉ được cộng vào khi
    `hai_chieu` bật — và luôn xếp SAU, để `cau_noi(...)[0]` giữ nguyên nghĩa cũ.
    """
    R = set(ung_vien)
    ra = [s for s in dict.fromkeys(xuoi.get(ocr_char, [])) if s in R]
    if hai_chieu:
        ra += [s for s in dict.fromkeys(nguoc.get(ocr_char, [])) if s in R and s not in ra]
    return ra


# ============================================================== CỔNG HỢP LỆ
def g3_bang_co_sap_khong(xuoi: dict[str, list[str]], n_mau: int = 4000) -> dict:
    """Bảng tự dạng có được sắp theo ĐỘ GIỐNG không?

    Không có cột điểm nên phải suy ra: với các cặp (X,Y) có mặt ở CẢ HAI chiều, nếu danh
    sách được sắp theo độ giống thì thứ hạng của Y trong X phải tương quan THUẬN với thứ
    hạng của X trong Y (cùng một đại lượng, đo hai lần). Sắp ngẫu nhiên -> tương quan ~0.

    Cổng này KHÔNG chặn thí nghiệm; nó quyết định được phép dùng THỨ HẠNG làm bằng chứng
    hay không. Ràng buộc y hệt G3.1 của T5.
    """
    rng = random.Random(SEED)
    keys = sorted(xuoi)
    rng.shuffle(keys)
    xs: list[int] = []
    ys: list[int] = []
    for x in keys:
        for i, y in enumerate(xuoi[x]):
            j = xuoi.get(y, [])
            if x in j:
                xs.append(i)
                ys.append(j.index(x))
        if len(xs) >= n_mau:
            break
    if len(xs) < 100:
        return {"n": len(xs), "rho": None, "co_sap": None}
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = sum((a - mx) ** 2 for a in xs) ** 0.5
    sy = sum((b - my) ** 2 for b in ys) ** 0.5
    rho = cov / (sx * sy) if sx and sy else 0.0
    return {"n": n, "rho": round(rho, 4), "co_sap": rho > 0.30}


# ================================================================ PHÉP ĐO
def gom_theo_anh_xa(pop: list[dict], khoa_chu: str, khoa_am: str, scorer) -> dict:
    """Tỷ lệ xác nhận tính theo ÁNH XẠ (chữ-OCR -> chữ nhãn), KHÔNG theo ô.

    VÌ SAO BẮT BUỘC: các ô KHÔNG độc lập. Lỗi OCR có tính hệ thống, nên cùng một ánh xạ
    lặp lại hàng chục lần. Đo được ở lượt 2026-08-25: 600 ô cầu ngược chỉ là 183 ánh xạ,
    8 ánh xạ đầu chiếm 46% số ô, và 84 ô được xác nhận thật ra chỉ thuộc 18 ánh xạ — riêng
    3 ánh xạ đã chiếm 59/84. Tính ở mức Ô là đếm cùng một bằng chứng nhiều chục lần và
    thổi p-value lên giả tạo. n hiệu dụng là số ÁNH XẠ.
    """
    g: dict[tuple[str, str], list[dict]] = {}
    for r in pop:
        g.setdefault(((r.get("ocr_char") or "").strip(), r[khoa_chu]), []).append(r)
    n_cham = n_xn = 0
    for (_oc, chu), rs in g.items():
        diem = [scorer.score(chu, r.get(khoa_am) or nrm(r["syllable"])) for r in rs]
        diem = [v for v in diem if v is not None]
        if not diem:
            continue
        n_cham += 1
        if max(diem) > TAU_CONFIRM:
            n_xn += 1
    return {"n_anh_xa": len(g), "n_cham_duoc": n_cham, "n_xac_nhan": n_xn,
            "ty_le": (n_xn / n_cham) if n_cham else None}


def _p_hai_ty_le(m: dict, c: dict) -> float | None:
    """p hai phía cho chênh lệch hai tỷ lệ (xấp xỉ chuẩn). Không có scipy thì trả None."""
    try:
        from scipy import stats
    except ImportError:
        return None
    k1, n1 = m["n_xac_nhan"], m["n_cham_duoc"]
    k2, n2 = c["n_xac_nhan"], c["n_cham_duoc"]
    if not n1 or not n2:
        return None
    p = (k1 + k2) / (n1 + n2)
    se = (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5
    if se == 0:
        return None
    z = (k1 / n1 - k2 / n2) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def ty_le_xac_nhan(cap: list[tuple[str, str]], scorer) -> dict:
    """(chữ, âm) -> tỷ lệ được kênh nghĩa XÁC NHẬN (điểm > TAU_CONFIRM).

    Mẫu số là số cặp CHẤM ĐƯỢC, không phải tổng — cặp mà Unihan không có nghĩa thì kênh
    này câm, gộp vào mẫu số là tự pha loãng tín hiệu bằng chỗ mình không biết.
    """
    n_cham = n_xac_nhan = 0
    for ch, am in cap:
        v = scorer.score(ch, am)
        if v is None:
            continue
        n_cham += 1
        if v > TAU_CONFIRM:
            n_xac_nhan += 1
    return {"n_cap": len(cap), "n_cham_duoc": n_cham, "n_xac_nhan": n_xac_nhan,
            "ty_le": (n_xac_nhan / n_cham) if n_cham else None}


def thu_thap(hai_chieu: bool) -> dict:
    """Gom bốn quần thể cần so, từ chính bộ nhãn trên đĩa."""
    qn = doc_tu_dien()
    xuoi, nguoc = doc_tu_dang()
    rows = list(csv.DictReader(open(LABELS, encoding="utf-8")))

    cau_xuoi_gold: list[dict] = []      # tầng ĐÃ CHẤP NHẬN — mốc so sánh
    truc_tiep_gold: list[dict] = []     # tầng mạnh nhất — chỉ để tham chiếu
    nguoc_moi: list[dict] = []          # quần thể đang xét
    for r in rows:
        if r["tier"] == "GOLD" and "s1_inter_s2_similar" in r["rule"]:
            cau_xuoi_gold.append(r)
        elif r["tier"] == "GOLD" and "s1_inter_s2_direct" in r["rule"]:
            truc_tiep_gold.append(r)
        elif r["tier"] in GIU_LAI:
            am, oc = nrm(r["syllable"]), (r["ocr_char"] or "").strip()
            R = qn.get(am)
            if not R or not oc:
                continue
            fx = cau_noi(oc, R, xuoi, nguoc, hai_chieu=False)
            if fx:
                continue                      # chiều xuôi đã có -> rớt vì lý do KHÁC
            # PHẢI dùng tham số, không được ghim cứng True: `--mot-chieu` là ĐỐI CHỨNG
            # của chính lab này (quần thể phải rỗng). Ghim cứng là tự vô hiệu đối chứng.
            fn = cau_noi(oc, R, xuoi, nguoc, hai_chieu=hai_chieu)
            if len(fn) == 1:
                nguoc_moi.append({**r, "_cau": fn[0], "_am": am})
    return {"qn": qn, "xuoi": xuoi, "nguoc": nguoc, "rows": rows,
            "cau_xuoi_gold": cau_xuoi_gold, "truc_tiep_gold": truc_tiep_gold,
            "nguoc_moi": nguoc_moi}


def doi_chung_xao_tron(pop: list[dict], qn: dict[str, list[str]],
                       khoa_chu: str, khoa_am: str) -> list[tuple[str, str]]:
    """Thay chữ thật bằng một chữ khác lấy trong ĐÚNG tập ứng viên của âm đó.

    Âm giữ nguyên nên hồ sơ nghĩa giữ nguyên; chỉ chữ đổi. Đây là đối chứng đúng: nếu
    kênh nghĩa chấm quần thể thật ngang với quần thể xáo trộn thì nó không đo gì cả.
    """
    rng = random.Random(SEED)
    ra: list[tuple[str, str]] = []
    for r in pop:
        am = r.get(khoa_am) or nrm(r["syllable"])
        khac = [c for c in qn.get(am, []) if c != r[khoa_chu]]
        if khac:
            ra.append((rng.choice(khac), am))
    return ra


def chay(hai_chieu: bool = True, xuat_csv: Path | None = None) -> dict:
    from pipeline.tools.sem_score import SemScorer, load_meanings

    d = thu_thap(hai_chieu)
    qn, nguoc_moi = d["qn"], d["nguoc_moi"]
    scorer = SemScorer(load_meanings(), qn)

    print("=" * 68)
    print(" T7 — CẦU NỐI TỰ DẠNG HAI CHIỀU")
    print("=" * 68)
    print(f"  cờ BRIDGE_TWO_WAY (mặc định)   : {BRIDGE_TWO_WAY}")
    print(f"  chạy phép đo với hai_chieu     : {hai_chieu}")
    print(f"  bộ nhãn                        : {LABELS.relative_to(REPO)}")

    # ---- CỔNG ----
    print("\n--- CỔNG HỢP LỆ (kiểm TRƯỚC khi đọc kết quả) ---")
    g1 = len(nguoc_moi) >= N_MIN
    print(f"  G1 cỡ mẫu {len(nguoc_moi):,} >= {N_MIN}                       : "
          f"{'QUA' if g1 else 'KHÔNG QUA'}")

    g3 = g3_bang_co_sap_khong(d["xuoi"])
    print(f"  G3 bảng tự dạng có sắp theo độ giống?          : "
          f"rho={g3['rho']} trên {g3['n']:,} cặp hai chiều -> "
          f"{'CÓ' if g3['co_sap'] else 'KHÔNG'}")
    if not g3["co_sap"]:
        print("     -> THỨ HẠNG KHÔNG được dùng làm bằng chứng (ràng buộc G3.1 của T5).")

    # Tính TẤT CẢ quần thể trước rồi mới gác cổng — nếu bỏ chạy ngay ở G2 thì mất luôn
    # thông tin quan trọng nhất: kênh chấm mù HẲN, hay chỉ mù với riêng nhãn cầu nối.
    xn_truc = ty_le_xac_nhan(
        [(r["label"], nrm(r["syllable"])) for r in d["truc_tiep_gold"][:8000]], scorer)
    dc_truc = ty_le_xac_nhan(
        doi_chung_xao_tron(d["truc_tiep_gold"][:8000], qn, "label", ""), scorer)
    xn_xuoi = ty_le_xac_nhan([(r["label"], nrm(r["syllable"])) for r in d["cau_xuoi_gold"]],
                             scorer)
    dc_xuoi = ty_le_xac_nhan(doi_chung_xao_tron(d["cau_xuoi_gold"], qn, "label", ""), scorer)
    xn_nguoc = ty_le_xac_nhan([(r["_cau"], r["_am"]) for r in nguoc_moi], scorer)
    dc_nguoc = ty_le_xac_nhan(doi_chung_xao_tron(nguoc_moi, qn, "_cau", "_am"), scorer)

    def _bang(tieu_de, cap):
        print(f"\n--- {tieu_de} ---")
        print(f"  {'quần thể':40} {'n cặp':>7} {'chấm được':>10} {'xác nhận':>9}  tỷ lệ")
        for ten, m in cap:
            t = "  n/a" if m["ty_le"] is None else f"{100*m['ty_le']:5.1f}%"
            phu = f"  ({100*m['n_cham_duoc']/max(1,m['n_cap']):.0f}% chấm được)"
            print(f"  {ten:40} {m['n_cap']:>7,} {m['n_cham_duoc']:>10,} "
                  f"{m['n_xac_nhan']:>9,}  {t}{phu}")

    _bang("TỶ LỆ ĐƯỢC KÊNH NGHĨA XÁC NHẬN (điểm > 0,05)",
          [("GOLD trực tiếp", xn_truc), ("  đối chứng xáo trộn", dc_truc),
           ("GOLD cầu XUÔI (mốc so sánh)", xn_xuoi), ("  đối chứng xáo trộn", dc_xuoi),
           ("cầu NGƯỢC (quần thể đang xét)", xn_nguoc), ("  đối chứng xáo trộn", dc_nguoc)])

    def _ty(m, c):
        return (m["ty_le"] / c["ty_le"]) if (m["ty_le"] and c["ty_le"]) else None

    ty_truc, ty_xuoi = _ty(xn_truc, dc_truc), _ty(xn_xuoi, dc_xuoi)
    ty_nguoc = _ty(xn_nguoc, dc_nguoc)
    p_nguoc = _p_hai_ty_le(xn_nguoc, dc_nguoc)

    print("\n  G2  (ĐĂNG KÝ BAN ĐẦU) kênh phải có lực trên tầng CẦU XUÔI:")
    print(f"      cầu xuôi {100*(xn_xuoi['ty_le'] or 0):.1f}% vs xáo trộn "
          f"{100*(dc_xuoi['ty_le'] or 0):.1f}% = {ty_xuoi:.2f}x -> "
          f"{'QUA' if (ty_xuoi or 0) >= CONTROL_MIN_RATIO else 'KHÔNG QUA'}")
    print("      phán quyết theo đăng ký ban đầu: KHÔNG ĐO ĐƯỢC — giữ nguyên trong hồ sơ.")

    g2a = (ty_truc or 0) >= CONTROL_MIN_RATIO
    g2b = p_nguoc is not None and p_nguoc < P_MAX   # ở mức ô; xem lại ở mức ánh xạ bên dưới
    print("\n  G2' (SỬA THIẾT KẾ — đọc kèm cảnh báo hậu nghiệm trong docstring):")
    print(f"      a) đối chứng DƯƠNG: GOLD trực tiếp {100*(xn_truc['ty_le'] or 0):.1f}% vs "
          f"{100*(dc_truc['ty_le'] or 0):.1f}% = {ty_truc:.2f}x -> {'QUA' if g2a else 'KHÔNG QUA'}")
    print(f"      b) chính quần thể xét tách khỏi đối chứng: "
          f"{100*(xn_nguoc['ty_le'] or 0):.1f}% vs {100*(dc_nguoc['ty_le'] or 0):.1f}% "
          f"= {ty_nguoc:.2f}x, p={p_nguoc:.2g} < {P_MAX} -> {'QUA' if g2b else 'KHÔNG QUA'}")
    g2 = g2a and g2b
    if not g2:
        print("\n  🔴 KHÔNG ĐO ĐƯỢC. Dừng, không suy diễn tiếp.")
        return {"ket_luan": "KHONG_DO_DUOC", "n_nguoc": len(nguoc_moi),
                "g1": g1, "g2_dang_ky_ban_dau": False, "g2_sua": g2, "g3": g3}

    # ---- ĐO ----
    # tái lặp: cùng một ánh xạ (chữ-OCR -> chữ cầu) xuất hiện trên bao nhiêu TRANG khác nhau
    anh_xa = collections.defaultdict(set)
    for r in nguoc_moi:
        anh_xa[(r["ocr_char"], r["_cau"])].add((r["book"], r["page"]))
    lap = sorted(anh_xa.items(), key=lambda kv: -len(kv[1]))
    n_lap = sum(1 for v in anh_xa.values() if len(v) >= 3)
    print(f"\n--- TÁI LẶP (một ánh xạ lặp trên nhiều trang độc lập thì khó là nhiễu) ---")
    print(f"  {len(anh_xa):,} ánh xạ phân biệt · {n_lap:,} ánh xạ xuất hiện trên >= 3 trang")
    print("  hay gặp nhất: " + ", ".join(f"{a}→{b} ({len(v)} trang)" for (a, b), v in lap[:6]))

    # ---- ĐƠN VỊ PHÂN TÍCH ĐÚNG: ÁNH XẠ ----
    ax_xuoi = gom_theo_anh_xa(d["cau_xuoi_gold"], "label", "", scorer)
    ax_nguoc = gom_theo_anh_xa(nguoc_moi, "_cau", "_am", scorer)
    ax_dc_x = gom_theo_anh_xa([{**r, "label": c} for r, (c, _a) in
                               zip(d["cau_xuoi_gold"],
                                   doi_chung_xao_tron(d["cau_xuoi_gold"], qn, "label", ""))],
                              "label", "", scorer)
    ax_dc_n = gom_theo_anh_xa([{**r, "_cau": c} for r, (c, _a) in
                               zip(nguoc_moi,
                                   doi_chung_xao_tron(nguoc_moi, qn, "_cau", "_am"))],
                              "_cau", "_am", scorer)
    print("\n--- TÍNH LẠI THEO ÁNH XẠ (ô KHÔNG độc lập — xem docstring gom_theo_anh_xa) ---")
    print(f"  {'quần thể':40} {'ánh xạ':>7} {'chấm được':>10} {'xác nhận':>9}  tỷ lệ")
    for ten, m in (("GOLD cầu XUÔI (mốc so sánh)", ax_xuoi),
                   ("  đối chứng xáo trộn", ax_dc_x),
                   ("cầu NGƯỢC (quần thể đang xét)", ax_nguoc),
                   ("  đối chứng xáo trộn", ax_dc_n)):
        t = "  n/a" if m["ty_le"] is None else f"{100*m['ty_le']:5.1f}%"
        print(f"  {ten:40} {m['n_anh_xa']:>7,} {m['n_cham_duoc']:>10,} "
              f"{m['n_xac_nhan']:>9,}  {t}")
    p_ax = _p_hai_ty_le(ax_nguoc, ax_dc_n)
    print(f"  cầu NGƯỢC vs đối chứng, ở mức ánh xạ: p = "
          f"{'n/a' if p_ax is None else f'{p_ax:.3g}'}  (n hiệu dụng = "
          f"{ax_nguoc['n_cham_duoc']}, KHÔNG phải {xn_nguoc['n_cham_duoc']})")

    # ---- LUẬT QUYẾT ĐỊNH ----
    # Chấm luật trên ĐƠN VỊ ÁNH XẠ — đơn vị ô đã bị bác ở trên.
    a_ok = (ax_nguoc["ty_le"] is not None and ax_xuoi["ty_le"] is not None
            and ax_nguoc["ty_le"] >= (ax_xuoi["ty_le"] - CONFIRM_MAX_DROP))
    b_ok = (ax_dc_n["ty_le"] not in (None, 0)
            and ax_nguoc["ty_le"] >= CONTROL_MIN_RATIO * ax_dc_n["ty_le"])
    c_ok = len(nguoc_moi) >= YIELD_MIN
    print("\n--- LUẬT QUYẾT ĐỊNH (chốt TRƯỚC khi chạy) ---")
    print(f"  (a) không thấp hơn cầu xuôi quá {CONFIRM_MAX_DROP:.0%}   : "
          f"{'ĐẠT' if a_ok else 'KHÔNG ĐẠT'}")
    print(f"  (b) vượt đối chứng >= {CONTROL_MIN_RATIO}x               : "
          f"{'ĐẠT' if b_ok else 'KHÔNG ĐẠT'}")
    print(f"  (c) sản lượng thêm >= {YIELD_MIN} ô                : "
          f"{'ĐẠT' if c_ok else 'KHÔNG ĐẠT'} ({len(nguoc_moi):,})")
    thong = a_ok and b_ok and c_ok and g1
    print(f"\n  => {'NỚI sang hai chiều' if thong else 'GIỮ MỘT CHIỀU'}")
    print("     Nhắc lại giới hạn công cụ: điểm 0 KHÔNG nói lên gì (47,7% cặp ĐÚNG cũng")
    print("     cho 0). Kết quả này KHÔNG chứng minh cầu ngược sai, chỉ nói có/không đủ")
    print("     bằng chứng THUẬN để nới một chốt chặn đặt có chủ ý.")

    if xuat_csv:
        with open(xuat_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "book", "page", "column", "syllable", "ocr_char",
                        "chu_cau_de_xuat", "tier_hien_tai", "rule_hien_tai",
                        "diem_nghia", "so_trang_lap"])
            for r in nguoc_moi:
                v = scorer.score(r["_cau"], r["_am"])
                w.writerow([r["image"], r["book"], r["page"], r["column"], r["syllable"],
                            r["ocr_char"], r["_cau"], r["tier"], r["rule"],
                            "" if v is None else f"{v:.4f}",
                            len(anh_xa[(r["ocr_char"], r["_cau"])])])
        print(f"\n  đã xuất {len(nguoc_moi):,} ô -> {xuat_csv}")

    return {"ket_luan": "NOI" if thong else "GIU_MOT_CHIEU",
            "n_nguoc": len(nguoc_moi), "g1": g1, "g2": g2, "g3": g3,
            "xac_nhan": {"truc_tiep": xn_truc, "cau_xuoi": xn_xuoi, "cau_nguoc": xn_nguoc,
                         "dc_xuoi": dc_xuoi, "dc_nguoc": dc_nguoc},
            "quyet_dinh": {"a": a_ok, "b": b_ok, "c": c_ok}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mot-chieu", action="store_true",
                    help="chạy với cầu MỘT CHIỀU (đối chứng: quần thể phải rỗng)")
    ap.add_argument("--csv", type=Path, default=None, help="xuất các ô cầu ngược ra tệp")
    ap.add_argument("--json", type=Path, default=None, help="ghi kết quả dạng JSON")
    a = ap.parse_args(argv)
    kq = chay(hai_chieu=not a.mot_chieu, xuat_csv=a.csv)
    if a.json:
        a.json.write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
