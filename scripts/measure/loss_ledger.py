#!/usr/bin/env python
"""loss_ledger.py — SỔ KẾ TOÁN MẤT MÁT cho 3 sách mới (2026-09-23; 0 token LLM).

Trả lời "ô lý thuyết → ô sinh ra → GOLD ảnh, mất ở đâu và vì khâu nào", cộng 3 phép đo phụ trợ:

  ledger  (0 API)  labels_gated.csv + columns.csv + từ điển R(âm) → mỗi ô non-GOLD gán ĐÚNG MỘT nguyên nhân gốc
                   (QN / kim / từ điển / hộp-crop / căn chỉnh) theo thứ tự ưu tiên; ra <Book>.csv + <Book>_cells.csv
  imgq    (0 API)  chất lượng ảnh/chữ trên ảnh prepared (≥20 trang/sách) + đối chứng 3 cuốn STT:
                   mật độ mực, px/chữ, tỉ lệ chữ DÍNH (chiếu mực tại khe giữa 2 hộp kim), bước cột/hàng + CV,
                   cờ crop (bleed/blank/truncated) lấy từ labels
  qn      (0 API)  chất lượng OCR quốc ngữ: âm không hợp lệ, âm R(âm)=rỗng, cột thiếu/thừa âm, sai dấu,
                   parity 6/8 từ measure_out/<book>/qn_ocr/summary.json
  kim     (0 API)  chất lượng kim từ cache: kim ∈ R(âm) theo tier; kim vs dị bản (measure_out/auto_precision*/cross)
  kimapi  (API)    dò 2 giả thuyết cải tiến: (i) kim trên CROP 1 CỘT vs kim cả trang, (ii) kim trên ảnh otsu vs prepared.
                   Chỉ số so sánh = % vị trí kim ∈ R(âm) (không cần GT). Cache theo md5 → chạy lại 0 lần gọi.

Chạy:
  .venv/bin/python scripts/measure/loss_ledger.py --book all
  .venv/bin/python scripts/measure/loss_ledger.py --book LucVanTien1883 --steps ledger,qn
  .venv/bin/python scripts/measure/loss_ledger.py --book all --steps kimapi --kim-budget 48 --kim-cols 10 --kim-pages 6

Ra measure_out/loss_ledger/. CHỈ ĐỌC repo (trừ thư mục --out). Mã thoát 1 nếu có invariant FAIL.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from core.text.dictionary import load_qn_to_nom, load_similarity_dict  # noqa: E402

DICT_QN = REPO / "dict" / "QuocNgu_SinoNom.csv"
DICT_SIM = REPO / "dict" / "SinoNom_Similar.csv"

# ---------------------------------------------------------------------------- #
# Hồ sơ sách. `chars_theo` = TRẦN LÝ THUYẾT số ô (số câu × số chữ/câu), nguồn ghi ở `theo_src`.
# ---------------------------------------------------------------------------- #
BOOKS: dict[str, dict] = {
    "LucVanTien1883": dict(
        out="dataset_out_LucVanTien1883", ds="dataset_LucVanTien1883",
        pages="prepared/LucVanTien1883", trans="prepared/LucVanTien1883/transcriptions",
        n_pages_src=105, verses=2088, chars_theo=2088 // 2 * 14,
        theo_src="2.088 câu lục-bát = 1.044 cặp × 14 chữ (measure_out/LucVanTien1883/qn_ocr: verses.expected=2088)",
        cross="measure_out/auto_precision/cross/LucVanTien1883/cells.csv",
        qn_ocr="measure_out/LucVanTien1883/qn_ocr/summary.json"),
    "KimVanKieu1884": dict(
        out="dataset_out_KimVanKieu1884_b1", ds="dataset_KimVanKieu1884",
        pages="prepared/KimVanKieu1884", trans="prepared_b1/KimVanKieu1884/transcriptions",
        n_pages_src=163, verses=3256, chars_theo=3256 // 2 * 14,
        theo_src="3.256 câu lục-bát = 1.628 cặp × 14 chữ (docs/BAO_CAO_TONG_HOP §1; QN chỉ có 3.251 dòng)",
        cross="measure_out/auto_precision_b1/cross/KimVanKieu1884/cells.csv",
        qn_ocr="measure_out/KimVanKieu1884/qn_ocr/summary.json"),
    "Chrestomathie1872": dict(
        out="dataset_out_Chrestomathie1872", ds="dataset_Chrestomathie1872",
        pages="prepared/Chrestomathie1872", trans="prepared/Chrestomathie1872/transcriptions",
        n_pages_src=65, verses=None, chars_theo=8692,
        theo_src="nom.total_chars_est=8.692 (measure_out/Chrestomathie1872/chresto_map/summary.json; văn xuôi, 419 cột)",
        cross=None,
        qn_ocr="measure_out/Chrestomathie1872/chresto_map/summary.json"),
}
STT_BOOKS = ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]

# Nhãn nguyên nhân gốc: mã → (khâu, mô tả)
CAUSES = {
    "E_canh_khongkhop": ("can_chinh", "âm QN = 'khongkhop' (verse-map không gióng được câu) → ô không có ngữ cảnh"),
    "E_quarantine": ("can_chinh", "quarantine_dup (F1: 1 ảnh gán 2 cột) — lỗi gióng cột/hộp"),
    "D_crop_bad": ("hop_crop", "cổng (c): crop blank/truncated → REVIEW"),
    "D_box_low_conf": ("hop_crop", "cổng (a'): hộp ink_cut|detector_low → GOLD_text_only (văn bản đúng, ảnh không dùng được)"),
    "B_kim_cross_similar": ("kim", "cổng (d): chữ kim khác dị bản VÀ gần hình → nghi kim đọc nhầm"),
    "A_qn_khong_hop_le": ("qn", "rule not_plausible: âm QN không phải âm tiếng Việt hợp lệ"),
    "A_qn_sai_dau": ("qn", "s1_inter_s2_direct_am_sua_dau: kim khớp SAU khi sửa dấu → QN OCR sai dấu"),
    "A_qn_am_la": ("qn", "âm hợp lệ nhưng R(âm) = rỗng → âm sai chính tả/OCR lọt lưới (KHÔNG phải thiếu từ điển, xem §c)"),
    "B_kim_cau_gan_hinh": ("kim", "s1_inter_s2_similar: kim đọc chữ GẦN HÌNH của một chữ trong R(âm)"),
    "B_kim_ngoai_R": ("kim", "kim là chữ Nôm có trong từ điển nhưng ∉ R(âm) → kim đọc sai chữ"),
    "B_kim_chu_la": ("kim", "chữ kim KHÔNG có trong toàn bộ từ điển Nôm → kim bịa/chữ hiếm"),
    "X_khac": ("khac", "còn lại (low_posterior, confusion_fix…)"),
}


def _int(x, d=0):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return d


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dicts() -> tuple[dict[str, set], set]:
    q = load_qn_to_nom(str(DICT_QN))
    R = {k: set(v) for k, v in q.items()}
    allnom = set().union(*R.values()) if R else set()
    return R, allnom


# ---------------------------------------------------------------------------- #
# 1. SỔ KẾ TOÁN
# ---------------------------------------------------------------------------- #
def classify(row: dict, R: dict[str, set], allnom: set) -> str:
    """Gán ĐÚNG MỘT nguyên nhân gốc cho ô non-GOLD (thứ tự ưu tiên = thứ tự if)."""
    base = row["rule"].split("|")[0]
    gate = row.get("gate_reason", "").split(":")[0]
    syl = (row.get("syllable") or "").lower()
    ch = row.get("ocr_char") or ""
    ds = _int(row.get("dict_support"))
    if "quarantine" in row["rule"]:
        return "E_quarantine"
    if syl == "khongkhop":
        return "E_canh_khongkhop"
    if gate == "crop_bad":
        return "D_crop_bad"
    if gate == "box_low_conf":
        return "D_box_low_conf"
    if gate == "cross_similar":
        return "B_kim_cross_similar"
    if base == "not_plausible":
        return "A_qn_khong_hop_le"
    if base.endswith("_am_sua_dau"):
        return "A_qn_sai_dau"
    if ds == 0:
        return "A_qn_am_la"
    if base in ("s1_inter_s2_similar", "s1_inter_s2_direct_lowp"):
        return "B_kim_cau_gan_hinh"
    if base.startswith("syl_ctx") or base in ("no_context", "low_posterior"):
        return "B_kim_chu_la" if ch and ch not in allnom else "B_kim_ngoai_R"
    return "X_khac"


def step_ledger(book: str, cfg: dict, out: Path, R, allnom) -> dict:
    ds_dir = REPO / "dataset" / book
    if not ds_dir.exists():
        ds_dir = REPO / cfg["ds"]
    o = REPO / "prepared" / book / "dataset_out"
    if not o.exists():
        o = REPO / "prepared_b1" / book / "dataset_out_b1"
    if not o.exists():
        o = REPO / cfg["out"]
    lab_p = o / "labels_gated.csv"
    lab_p = lab_p if lab_p.exists() else o / "labels_final.csv"
    lab = read_csv(lab_p)
    cols = {(c["book"], c["page"], c["column"]): c for c in read_csv(ds_dir / "columns.csv")}
    summary = json.loads((o / "summary.json").read_text(encoding="utf-8"))

    built = len(lab)
    theo = cfg["chars_theo"]
    pages_built = len({r["page"] for r in lab})

    rows_cells = []
    cnt, cnt_meq = Counter(), Counter()
    gold = 0
    for r in lab:
        col = cols.get((r["book"], r["page"], r["column"]))
        meq = bool(col) and col["n_ocr"] == col["n_qn"]
        if r["tier"] == "GOLD":
            gold += 1
            cause = "-"
        else:
            cause = classify(r, R, allnom)
            cnt[cause] += 1
            cnt_meq[(cause, meq)] += 1
        rows_cells.append(dict(image=r.get("image", ""), page=r["page"], column=r["column"],
                               nom_idx=r.get("nom_idx", ""), syl_idx=r.get("syl_idx", ""),
                               tier=r["tier"], rule=r["rule"], gate_reason=r.get("gate_reason", ""),
                               syllable=r.get("syllable", ""), ocr_char=r.get("ocr_char", ""),
                               dict_support=r.get("dict_support", ""), box_source=r.get("box_source", ""),
                               count_source=r.get("count_source", ""), n_ocr=(col or {}).get("n_ocr", ""),
                               n_qn=(col or {}).get("n_qn", ""), n_det=(col or {}).get("n_det", ""),
                               m_eq_n=int(meq), cause=cause,
                               khau=("-" if cause == "-" else CAUSES[cause][0])))

    with open(out / f"{book}_cells.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_cells[0]))
        w.writeheader()
        w.writerows(rows_cells)

    # bảng sổ kế toán
    led = [dict(muc="0_o_ly_thuyet", n=theo, pct_theo=100.0, khau="-", mo_ta=cfg["theo_src"]),
           dict(muc="1_o_sinh_ra", n=built, pct_theo=round(built / theo * 100, 1), khau="-",
                mo_ta=f"labels_gated.csv của {cfg['out']} ({pages_built} trang)"),
           dict(muc="1b_khong_sinh", n=theo - built, pct_theo=round((theo - built) / theo * 100, 1), khau="qn/can_chinh",
                mo_ta="ô lý thuyết không thành ô: trang/cột không dựng được + cột QN thiếu âm (N = số âm QN của cột)"),
           dict(muc="2_GOLD_anh", n=gold, pct_theo=round(gold / theo * 100, 1), khau="-",
                mo_ta=f"tier GOLD trong {lab_p.name} ({round(gold / built * 100, 1)} % ô sinh ra)")]
    for c, n in sorted(cnt.items(), key=lambda kv: -kv[1]):
        led.append(dict(muc="3_" + c, n=n, pct_theo=round(n / theo * 100, 1), khau=CAUSES[c][0],
                        mo_ta=f"{CAUSES[c][1]} | M==N {cnt_meq[(c, True)]} / M!=N {cnt_meq[(c, False)]}"))
    with open(out / f"{book}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["muc", "n", "pct_theo", "khau", "mo_ta"])
        w.writeheader()
        w.writerows(led)

    by_khau = Counter()
    for c, n in cnt.items():
        by_khau[CAUSES[c][0]] += n
    return dict(labels=str(lab_p.relative_to(REPO)), o_ly_thuyet=theo, o_sinh_ra=built,
                khong_sinh=theo - built, trang=pages_built, trang_nguon=cfg["n_pages_src"],
                GOLD_anh=gold, GOLD_pct_o_sinh=round(gold / built * 100, 1),
                GOLD_pct_ly_thuyet=round(gold / theo * 100, 1),
                non_gold=built - gold, theo_nguyen_nhan=dict(cnt.most_common()),
                theo_khau={k: v for k, v in by_khau.most_common()},
                tier=dict(Counter(r["tier"] for r in lab).most_common()),
                summary_json_gold=summary.get("tier_counts", {}) or None)


# ---------------------------------------------------------------------------- #
# 2. CHẤT LƯỢNG ẢNH / CHỮ
# ---------------------------------------------------------------------------- #
def page_img_metrics(cache: Path, touch_frac: float = 0.20):
    """Chỉ số một trang từ cache kim (hộp chữ) + ảnh prepared. Trả None nếu cache hỏng."""
    import numpy as np
    from PIL import Image
    d = json.loads(cache.read_text(encoding="utf-8"))
    img = REPO / d["image"]
    if not img.exists():
        return None
    g = np.asarray(Image.open(img).convert("L"))
    ink = g < 128
    hs, ws, rp, cx, nb, nt, dark = [], [], [], [], 0, 0, []
    for col in d.get("columns") or []:
        if len(col) < 2:
            continue
        x1 = min(c["bbox"][0] for c in col)
        x2 = max(c["bbox"][2] for c in col)
        cx.append((x1 + x2) / 2)
        prof = ink[:, max(0, x1):x2].sum(1).astype(float)
        rows = []
        for c in col:
            b = c["bbox"]
            hs.append(b[3] - b[1])
            ws.append(b[2] - b[0])
            rows.extend(range(max(0, b[1]), min(len(prof), b[3])))
        if not rows:
            continue
        m = float(np.median(prof[rows]))
        dark.append(float(ink[min(rows):max(rows) + 1, max(0, x1):x2].mean()))
        if m <= 0:
            continue
        for a, b in zip(col, col[1:]):
            if b["bbox"][1] - a["bbox"][3] > 0.6 * (a["bbox"][3] - a["bbox"][1]):
                continue                                   # khe giữa 2 TẦNG, không phải giữa 2 chữ
            gy = (a["bbox"][3] + b["bbox"][1]) / 2
            lo, hi = int(gy - 4), int(gy + 5)
            v = float(prof[max(0, lo):hi].min()) if hi > max(0, lo) else 0.0
            nb += 1
            nt += int(v > touch_frac * m)
            rp.append(b["bbox"][1] - a["bbox"][1])
    if not hs:
        return None
    return dict(ink_page=float(ink.mean()), ink_col=float(np.mean(dark)) if dark else 0.0,
                h=float(np.median(hs)), w=float(np.median(ws)), n_touch=nt, n_bound=nb,
                rpitch=float(np.median(rp)) if rp else 0.0,
                rcv=float(np.std(rp) / np.mean(rp)) if rp else 0.0,
                cpitch=float(np.median(np.diff(sorted(cx)))) if len(cx) > 1 else 0.0,
                levels=int(len(np.unique(g))))


def orig_contrast(book: str, n: int = 20):
    """Tương phản ảnh GỐC data/<book>/pages (chưa nhị phân): Michelson (p95−p5)/(p95+p5) + tách lớp Otsu."""
    import numpy as np
    from PIL import Image
    fs = sorted(glob.glob(str(REPO / "data" / book / "pages" / "*")))
    if not fs:
        return None
    fs = [fs[i] for i in range(0, len(fs), max(1, len(fs) // n))][:n]
    mich, sep = [], []
    for f in fs:
        g = np.asarray(Image.open(f).convert("L")).astype(float)
        p5, p95 = np.percentile(g, [5, 95])
        mich.append((p95 - p5) / max(p95 + p5, 1e-6))
        h = np.histogram(g, bins=256, range=(0, 256))[0].astype(float)
        p = h / h.sum()
        w = np.cumsum(p)
        m = np.cumsum(p * np.arange(256))
        den = w * (1 - w)
        den[den == 0] = 1e-9
        sep.append(float(np.max((m[-1] * w - m) ** 2 / den) / max(g.var(), 1e-6)))
    return dict(n_pages=len(fs), michelson=round(float(np.mean(mich)), 3), otsu_sep=round(float(np.mean(sep)), 3))


def step_imgq(out: Path, n_pages: int) -> dict:
    import numpy as np
    res = {}
    for book in list(BOOKS) + STT_BOOKS:
        pdir = REPO / (BOOKS[book]["pages"] if book in BOOKS else f"prepared/{book}")
        fs = sorted(glob.glob(str(pdir / "detected" / "*_ocr_cache.json")))
        if not fs:
            continue
        fs = [fs[i] for i in range(0, len(fs), max(1, len(fs) // n_pages))][:n_pages]
        ms = [m for m in (page_img_metrics(Path(f)) for f in fs) if m]
        nb = sum(m["n_bound"] for m in ms)
        r = dict(n_pages=len(ms), n_bound=nb,
                 muc_trang_pct=round(float(np.mean([m["ink_page"] for m in ms])) * 100, 2),
                 muc_trong_cot_pct=round(float(np.mean([m["ink_col"] for m in ms])) * 100, 2),
                 px_chu_cao=round(float(np.median([m["h"] for m in ms])), 1),
                 px_chu_rong=round(float(np.median([m["w"] for m in ms])), 1),
                 chu_dinh_pct=round(sum(m["n_touch"] for m in ms) / max(nb, 1) * 100, 1),
                 buoc_hang=round(float(np.median([m["rpitch"] for m in ms])), 1),
                 buoc_hang_cv=round(float(np.mean([m["rcv"] for m in ms])), 3),
                 buoc_cot=round(float(np.median([m["cpitch"] for m in ms])), 1),
                 muc_xam_so_muc=int(np.median([m["levels"] for m in ms])))
        oc = orig_contrast(book)
        if oc:
            r["anh_goc"] = oc
        res[book] = r
    # cờ crop thật lấy từ labels của từng build
    for book, cfg in BOOKS.items():
        p = REPO / cfg["out"] / "labels_gated.csv"
        p = p if p.exists() else REPO / cfg["out"] / "labels_final.csv"
        lab = read_csv(p)
        c = Counter(r.get("crop_quality_flag", "") for r in lab)
        n = sum(v for k, v in c.items() if k)
        res[book]["crop_flag"] = {k or "(REVIEW-khong-crop)": v for k, v in c.most_common()}
        res[book]["crop_bleed_pct"] = round(c.get("bleed", 0) / max(n, 1) * 100, 1)
    lab = read_csv(REPO / "dataset_out" / "labels.csv")
    c = Counter(r.get("crop_quality_flag", "") for r in lab)
    n = sum(v for k, v in c.items() if k)
    for b in STT_BOOKS:
        res.setdefault(b, {})
    res["STT(gộp 3 cuốn)"] = dict(crop_flag={k or "(REVIEW-khong-crop)": v for k, v in c.most_common()},
                                  crop_bleed_pct=round(c.get("bleed", 0) / max(n, 1) * 100, 1),
                                  nguon="dataset_out/labels.csv")
    (out / "imgq.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------- #
# 3. CHẤT LƯỢNG QN
# ---------------------------------------------------------------------------- #
_TONE_RE = None


def tone_to_dict_style(s: str) -> str:
    """Dời dấu thanh từ nguyên âm SAU về nguyên âm TRƯỚC trong oa/oe/uy (kiểu chính tả của từ điển:
    hoà→hòa, hoạ→họa, luỵ→lụy). QN in/OCR dùng kiểu mới nên R(âm) rỗng oan."""
    global _TONE_RE
    if _TONE_RE is None:
        import re
        _TONE_RE = re.compile(r"([ou])([aeoy])([̣̀́̃̉])")
    t = unicodedata.normalize("NFD", s)
    m = _TONE_RE.search(t)
    return unicodedata.normalize("NFC", t[:m.start()] + m.group(1) + m.group(3) + m.group(2) + t[m.end():]) if m else s


def step_qn(book: str, cfg: dict, R) -> dict:
    ds_dir = REPO / "dataset" / book
    if not ds_dir.exists():
        ds_dir = REPO / cfg["ds"]
    cols = read_csv(ds_dir / "columns.csv")
    out_dir = REPO / "prepared" / book / "dataset_out"
    if not out_dir.exists():
        out_dir = REPO / "prepared_b1" / book / "dataset_out_b1"
    if not out_dir.exists():
        out_dir = REPO / cfg["out"]
    p = out_dir / "labels_gated.csv"
    lab = read_csv(p if p.exists() else out_dir / "labels_final.csv")
    n_syl = len(lab)
    imp = sum(1 for r in lab if r["rule"].split("|")[0] == "not_plausible")
    kk = sum(1 for r in lab if (r.get("syllable") or "").lower() == "khongkhop")
    ds0 = sum(1 for r in lab if _int(r.get("dict_support")) == 0)
    tone = sum(1 for r in lab if r["rule"].split("|")[0].endswith("_am_sua_dau"))
    oov = sum(1 for r in lab if (r.get("syllable") or "").lower() not in R)
    dif = Counter(_int(c["n_ocr"]) - _int(c["n_qn"]) for c in cols)
    res = dict(o=n_syl, cot=len(cols),
               am_khong_hop_le=imp, am_khong_hop_le_pct=round(imp / n_syl * 100, 2),
               am_khongkhop=kk, am_ngoai_tu_dien=oov, am_ngoai_tu_dien_pct=round(oov / n_syl * 100, 2),
               am_R_rong=ds0, am_R_rong_pct=round(ds0 / n_syl * 100, 2),
               am_sai_dau_kim_cuu_duoc=tone,
               cot_M_eq_N=dif.get(0, 0), cot_M_eq_N_pct=round(dif.get(0, 0) / len(cols) * 100, 1),
               cot_kim_thieu=sum(v for k, v in dif.items() if k < 0),
               cot_kim_thua=sum(v for k, v in dif.items() if k > 0),
               hist_n_ocr_tru_n_qn=dict(sorted(dif.items())))
    qs = REPO / cfg["qn_ocr"]
    if qs.exists():
        s = json.loads(qs.read_text(encoding="utf-8"))
        if "verses" in s:
            v = s["verses"]
            res["cau_QN_dung_so_am_6_8_pct"] = round(v.get("parity_rate", 0) * 100, 1)
            res["cau_QN_n"] = v.get("n")
            res["conf_tesseract_tb"] = v.get("conf_mean")
        if "qn" in s:                                     # Chrestomathie (chresto_map)
            res["tesseract_oov_rate_pct"] = round(s["qn"].get("oov_rate", 0) * 100, 2)
            res["qn_n_syll"] = s["qn"].get("n_syll")
            res["nom_total_chars_est"] = s.get("nom", {}).get("total_chars_est")
    # nguồn QN (B1'): tier theo qn_source
    tdir = REPO / cfg["trans"]
    src = {}
    for tf in sorted(glob.glob(str(tdir / "page_*.json"))):
        t = json.loads(Path(tf).read_text(encoding="utf-8"))
        for c in t["columns"]:
            for key in ("verse_odd", "verse_even"):
                v = c.get(key) or {}
                if v:
                    src[v.get("qn_source") or "ocr"] = src.get(v.get("qn_source") or "ocr", 0) + 1
    res["cau_theo_qn_source"] = src
    # lỗi vị trí dấu thanh (hoà/hòa): ô R(âm) rỗng chỉ vì chính tả dấu — chữa 0 API
    n_t = k_t = 0
    for r in lab:
        if _int(r.get("dict_support")) != 0:
            continue
        s = (r.get("syllable") or "").lower()
        s2 = tone_to_dict_style(s)
        if s2 != s and s2 in R:
            n_t += 1
            k_t += int((r.get("ocr_char") or "") in R[s2])
    res["loi_vi_tri_dau_thanh"] = dict(o=n_t, o_kim_khop_R_am_chuan=k_t,
                                       chu_thich="chuẩn hoá hoà→hòa, hoạ→họa, luỵ→lụy trước khi tra R(âm)")
    return res


# ---------------------------------------------------------------------------- #
# 4. CHẤT LƯỢNG KIM (0 API)
# ---------------------------------------------------------------------------- #
def step_kim(book: str, cfg: dict, R) -> dict:
    p = REPO / cfg["out"] / "labels_gated.csv"
    lab = read_csv(p if p.exists() else REPO / cfg["out"] / "labels_final.csv")
    byt = defaultdict(lambda: [0, 0])
    for r in lab:
        s = (r.get("syllable") or "").lower()
        if s not in R:
            continue
        t = byt[r["tier"]]
        t[0] += 1
        t[1] += int((r.get("ocr_char") or "") in R[s])
    res = dict(kim_trong_R_theo_tier={k: dict(n=v[0], pct=round(v[1] / max(v[0], 1) * 100, 1))
                                      for k, v in sorted(byt.items())},
               kim_trong_R_tong_pct=round(sum(v[1] for v in byt.values()) / max(sum(v[0] for v in byt.values()), 1) * 100, 1))
    if cfg["cross"] and (REPO / cfg["cross"]).exists():
        cells = read_csv(REPO / cfg["cross"])
        per = defaultdict(Counter)
        for x in cells:
            if x.get("ref_pua") == "1" or not x.get("ref_nom"):
                continue
            k = (x["ref"], x["tier"])
            per[k]["n"] += 1
            eq = x.get("ocr_char") == x["ref_nom"]
            per[k]["kim_eq_ref"] += int(eq)
            per[k]["ref_trong_R"] += int(x["ref_nom"] in R.get((x.get("syllable") or "").lower(), ()))
            if not eq:
                per[k]["khac_di_the_cung_am"] += int(x.get("di_the_cung_am") == "1")
                per[k]["khac_gan_hinh"] += int(x.get("similar") == "1")
        res["so_di_ban"] = {f"{a}|{b}": dict(n=v["n"],
                                             kim_eq_ref_pct=round(v["kim_eq_ref"] / v["n"] * 100, 1),
                                             ref_trong_R_am_pct=round(v["ref_trong_R"] / v["n"] * 100, 1),
                                             khac_gan_hinh=v["khac_gan_hinh"],
                                             khac_di_the_cung_am=v["khac_di_the_cung_am"])
                            for (a, b), v in sorted(per.items())}
        # phân loại LỖI KIM theo tier: so CHỮ KIM (không phải nhãn) với chữ dị bản
        S = load_similarity_dict(str(DICT_SIM))
        tax = defaultdict(Counter)
        for x in cells:
            if x.get("ref_pua") == "1" or not x.get("ref_nom"):
                continue
            k, r0, s = x.get("ocr_char") or "", x["ref_nom"], (x.get("syllable") or "").lower()
            c = tax[x["tier"]]
            c["n"] += 1
            if k == r0:
                c["kim_dung"] += 1
            elif k in R.get(s, ()):
                c["di_the_cung_am"] += 1
            elif r0 in S.get(k, []) or k in S.get(r0, []):
                c["gan_hinh_voi_ref"] += 1
            else:
                c["chu_khac_han"] += 1
        res["loi_kim_theo_tier"] = {t: {k: (v if k == "n" else f"{v} ({round(v / c['n'] * 100, 1)} %)")
                                        for k, v in c.items()} for t, c in sorted(tax.items())}
    return res


# ---------------------------------------------------------------------------- #
# 5. DÒ CẢI TIẾN KIM (API) — crop 1 cột vs cả trang; otsu vs prepared
# ---------------------------------------------------------------------------- #
def step_kimapi(book: str, cfg: dict, out: Path, R, budget, n_cols: int, n_pages: int) -> dict:
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(REPO / "scripts" / "measure"))
    import auto_precision as ap                                  # kim_ocr_patch + kim_chars

    cache = out / "kim_probe"
    cache.mkdir(parents=True, exist_ok=True)
    trans = {Path(f).stem: json.loads(Path(f).read_text(encoding="utf-8"))
             for f in sorted(glob.glob(str(REPO / cfg["trans"] / "page_*.json")))}
    caches = sorted(glob.glob(str(REPO / cfg["pages"] / "detected" / "*_ocr_cache.json")))

    def syls_of(page, column):
        t = trans.get(page)
        if not t:
            return []
        for c in t["columns"]:
            if int(c["column"]) == column:
                return [s.lower() for s in (c.get("syllables") or [])]
        return []

    def in_R(chars, syls):
        n = min(len(chars), len(syls))
        return sum(int(chars[i] in R.get(syls[i], ())) for i in range(n)), len(syls)

    # (i) crop 1 cột vs cả trang, trên các cột có N âm == số chữ kim của trang
    picks = []
    for f in caches:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        page = Path(f).stem.replace("_ocr_cache", "")
        for ci, col in enumerate(d.get("columns") or [], 1):
            sy = syls_of(page, ci)
            if len(col) >= 6 and len(sy) == len(col):
                picks.append((f, page, ci, col, sy))
    step = max(1, len(picks) // n_cols)
    picks = [picks[i] for i in range(0, len(picks), step)][:n_cols]
    col_rows = []
    for f, page, ci, col, sy in picks:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        x1 = min(c["bbox"][0] for c in col) - 8
        x2 = max(c["bbox"][2] for c in col) + 8
        y1 = min(c["bbox"][1] for c in col) - 8
        y2 = max(c["bbox"][3] for c in col) + 8
        im = Image.open(REPO / d["image"]).convert("RGB")
        p = cache / f"{book}_{page}_c{ci:02d}.png"
        im.crop((max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2))).save(p)
        r = ap.kim_ocr_patch(p, 1, cache, budget, f"{book} col {page} c{ci}")
        kc = ap.kim_chars(r.get("boxes"))
        pg = [c["char"] for c in col]
        a, n = in_R(pg, sy)
        b, _ = in_R(kc, sy)
        col_rows.append(dict(page=page, col=ci, n_am=n, n_kim_trang=len(pg), n_kim_cot=len(kc),
                             inR_trang=a, inR_cot=b, status=r.get("status"),
                             kim_trang="".join(pg), kim_cot="".join(kc)))

    # (ii) ổn định theo tiền xử lý: ảnh GỐC data/ (chưa stretch/otsu) và otsu(gốc) vs ảnh prepared (đã cache)
    man = json.loads((REPO / cfg["pages"] / "manifest.json").read_text(encoding="utf-8"))
    srcf = {p["page_name"]: p.get("source_file", "") for p in man.get("pages", [])}
    step = max(1, len(caches) // n_pages)
    pg_rows = []
    for f in [caches[i] for i in range(0, len(caches), step)][:n_pages]:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        page = Path(f).stem.replace("_ocr_cache", "")
        orig = REPO / "data" / book / "pages" / srcf.get(page, "")
        if not srcf.get(page) or not orig.exists():
            continue
        pgc = [c["char"] for cl in (d.get("columns") or []) for c in cl]
        sy = [s for ci in range(1, len(d.get("columns") or []) + 1) for s in syls_of(page, ci)]
        Rall = set().union(*[R.get(s, set()) for s in sy]) if sy else set()

        def stat(chars):
            return dict(n=len(chars),
                        trong_R_trang_pct=round(sum(c in Rall for c in chars) / max(len(chars), 1) * 100, 1),
                        jaccard_vs_prepared=round(len(set(pgc) & set(chars)) / max(len(set(pgc) | set(chars)), 1), 3))

        g = np.asarray(Image.open(orig).convert("L"))
        h = np.histogram(g, bins=256, range=(0, 256))[0].astype(float)
        pr = h / h.sum()
        w = np.cumsum(pr)
        m = np.cumsum(pr * np.arange(256))
        den = w * (1 - w)
        den[den == 0] = 1e-9
        th = int(np.argmax((m[-1] * w - m) ** 2 / den))
        po = cache / f"{book}_{page}_goc.png"
        Image.fromarray(g).save(po)
        pt = cache / f"{book}_{page}_otsu.png"
        Image.fromarray(((g > th) * 255).astype("uint8")).save(pt)
        r1 = ap.kim_ocr_patch(po, 1, cache, budget, f"{book} goc {page}")
        r2 = ap.kim_ocr_patch(pt, 1, cache, budget, f"{book} otsu {page}")
        c1, c2 = ap.kim_chars(r1.get("boxes")), ap.kim_chars(r2.get("boxes"))
        pg_rows.append(dict(page=page, src=srcf[page], otsu_thr=th, n_am_QN=len(sy),
                            prepared=stat(pgc), goc=stat(c1), otsu=stat(c2),
                            status=f"{r1.get('status')}/{r2.get('status')}"))

    with open(out / f"{book}_kimapi_cols.csv", "w", encoding="utf-8", newline="") as fh:
        if col_rows:
            w2 = csv.DictWriter(fh, fieldnames=list(col_rows[0]))
            w2.writeheader()
            w2.writerows(col_rows)
    (out / f"{book}_kimapi_pages.json").write_text(json.dumps(pg_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = [r for r in col_rows if r["status"] == "ok"]
    tot = sum(r["n_am"] for r in ok) or 1

    def avg(key, sub, nd=1):
        v = [r[key][sub] for r in pg_rows if r[key]["n"]]
        return round(float(np.mean(v)), nd) if v else None

    return dict(crop_1_cot=dict(n_cot=len(ok), n_am=tot,
                                inR_kim_ca_trang_pct=round(sum(r["inR_trang"] for r in ok) / tot * 100, 1),
                                inR_kim_crop_cot_pct=round(sum(r["inR_cot"] for r in ok) / tot * 100, 1),
                                cot_dung_so_chu_trang=sum(int(r["n_kim_trang"] == r["n_am"]) for r in ok),
                                cot_dung_so_chu_crop=sum(int(r["n_kim_cot"] == r["n_am"]) for r in ok)),
                tien_xu_ly=dict(n_trang=len(pg_rows),
                                n_chu_tb=dict(prepared=avg("prepared", "n"), goc=avg("goc", "n"), otsu=avg("otsu", "n")),
                                trong_R_trang_pct=dict(prepared=avg("prepared", "trong_R_trang_pct"),
                                                       goc=avg("goc", "trong_R_trang_pct"),
                                                       otsu=avg("otsu", "trong_R_trang_pct")),
                                jaccard_vs_prepared=dict(goc=avg("goc", "jaccard_vs_prepared", 3),
                                                         otsu=avg("otsu", "jaccard_vs_prepared", 3)),
                                chi_tiet=f"{book}_kimapi_pages.json"))


# ---------------------------------------------------------------------------- #
def ceilings(ledger: dict) -> dict:
    """Trần lợi ích: chữa HOÀN HẢO một khâu thì GOLD ảnh tăng tối đa bao nhiêu ô.

    Các nhánh nguyên nhân LOẠI TRỪ NHAU (mỗi ô đúng 1 nguyên nhân) nên cộng được; ô 'khong_sinh'
    chỉ vào trần của khâu QN/căn chỉnh (cận TRÊN: giả định mọi ô mới đều thành GOLD)."""
    out = {}
    for book, r in ledger.items():
        k = r["theo_khau"]
        g, theo = r["GOLD_anh"], r["o_ly_thuyet"]
        rows = [("kim", k.get("kim", 0), 0), ("qn", k.get("qn", 0), r["khong_sinh"]),
                ("hop_crop", k.get("hop_crop", 0), 0), ("can_chinh", k.get("can_chinh", 0), 0),
                ("khac", k.get("khac", 0), 0)]
        out[book] = {kh: dict(o_them_toi_da=n + extra, GOLD_neu_chua=g + n + extra,
                              GOLD_pct_ly_thuyet=round((g + n + extra) / theo * 100, 1),
                              diem_tang=round((n + extra) / theo * 100, 1))
                     for kh, n, extra in rows if n + extra}
        out[book]["_hien_tai"] = dict(GOLD=g, pct_ly_thuyet=round(g / theo * 100, 1), o_ly_thuyet=theo)
    return out


def invariants(res: dict) -> list[dict]:
    inv = []
    for book, r in res.get("ledger", {}).items():
        s = r["GOLD_anh"] + r["non_gold"]
        inv.append(dict(name=f"{book}:tong_o_khop", expected=r["o_sinh_ra"], observed=s,
                        status="PASS" if s == r["o_sinh_ra"] else "FAIL"))
        s2 = sum(r["theo_nguyen_nhan"].values())
        inv.append(dict(name=f"{book}:nguyen_nhan_phu_het_non_gold", expected=r["non_gold"], observed=s2,
                        status="PASS" if s2 == r["non_gold"] else "FAIL"))
        inv.append(dict(name=f"{book}:o_sinh_ra_le_ly_thuyet", expected=True,
                        observed=r["o_sinh_ra"] <= r["o_ly_thuyet"],
                        status="PASS" if r["o_sinh_ra"] <= r["o_ly_thuyet"] else "FAIL"))
    for book, r in res.get("kim", {}).items():
        for k, v in (r.get("so_di_ban") or {}).items():
            inv.append(dict(name=f"{book}:{k}:ref_trong_R_am>=99", expected=">=99",
                            observed=v["ref_trong_R_am_pct"],
                            status="PASS" if v["ref_trong_R_am_pct"] >= 99 else "FAIL"))
    q = res.get("imgq", {})
    if q:
        for b in BOOKS:
            if b in q and "chu_dinh_pct" in q[b]:
                inv.append(dict(name=f"imgq:{b}:co_do_chu_dinh", expected="0..100",
                                observed=q[b]["chu_dinh_pct"],
                                status="PASS" if 0 <= q[b]["chu_dinh_pct"] <= 100 else "FAIL"))
    return inv


def main(argv=None) -> int:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--book", default="all", help="all | " + " | ".join(BOOKS))
    ap_.add_argument("--steps", default="ledger,imgq,qn,kim")
    ap_.add_argument("--out", default="measure_out/loss_ledger")
    ap_.add_argument("--pages", type=int, default=20, help="số trang mẫu cho imgq")
    ap_.add_argument("--kim-budget", type=int, default=0, help="trần số lần gọi API kim (step kimapi)")
    ap_.add_argument("--kim-cols", type=int, default=10)
    ap_.add_argument("--kim-pages", type=int, default=6)
    a = ap_.parse_args(argv)

    books = list(BOOKS) if a.book == "all" else [a.book]
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R, allnom = dicts()
    res: dict = dict(measure="loss_ledger", date=time.strftime("%Y-%m-%d"),
                     cli="scripts/measure/loss_ledger.py " + " ".join(sys.argv[1:]),
                     dict_qn=dict(n_am=len(R), n_chu_nom=len(allnom), file="dict/QuocNgu_SinoNom.csv"))
    if "ledger" in steps:
        res["ledger"] = {b: step_ledger(b, BOOKS[b], out, R, allnom) for b in books}
        res["tran_loi_ich"] = ceilings(res["ledger"])
    if "qn" in steps:
        res["qn"] = {b: step_qn(b, BOOKS[b], R) for b in books}
    if "kim" in steps:
        res["kim"] = {b: step_kim(b, BOOKS[b], R) for b in books}
    if "imgq" in steps:
        res["imgq"] = step_imgq(out, a.pages)
    if "kimapi" in steps:
        sys.path.insert(0, str(REPO / "scripts" / "measure"))
        import auto_precision as apx
        bud = apx.KimBudget(out / "kim_calls.json", a.kim_budget)
        res["kimapi"] = {b: step_kimapi(b, BOOKS[b], out, R, bud, a.kim_cols, a.kim_pages) for b in books}
        res["kimapi"]["_calls"] = bud.state["calls"]
    res["runtime_s"] = round(time.time() - t0, 1)
    res["invariants"] = invariants(res)
    res["inv_pass"] = sum(1 for i in res["invariants"] if i["status"] == "PASS")
    res["inv_fail"] = sum(1 for i in res["invariants"] if i["status"] == "FAIL")
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out}/summary.json  ({res['runtime_s']} s, invariants {res['inv_pass']} PASS / {res['inv_fail']} FAIL)")
    return 1 if res["inv_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
