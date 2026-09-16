"""B-4 · H2-lite — đọc bảng quyết định LỚP người đã điền -> ĐỀ XUẤT mục mới cho config/decisions.yaml.

VÌ SAO CÓ TỆP NÀY
-----------------
KhoiB/h2/lop_review.html (xay_hang_doi_lop.py) cho người xem 12 crop/mã và chọn cho từng lớp
(sách, âm, mã đa số, mã thiểu số). Người xuất `lop_decisions_filled.csv` (cột như template:
id_lop, book, syllable, ma_da_so, ma_thieu_so, n_da_so, n_thieu_so, e3_verdict, quyet, nguoi_ky,
ngay, xuat_xu). Tệp này dịch `quyet` sang mục yaml theo schema pipeline/decisions.py:

  quyet                 -> mục yaml                                   -> hiệu lực khi build (PASS 1c)
  dung                  (không sinh mục; chỉ ghi report)               giữ cả hai mã
  di_the:<chuẩn>        di_the {quan_sat:[M,L], chuan, book, quy_tac}  label_canonical = chuẩn (label KHÔNG đổi)
  nham:<mã đúng>        lop_nham {syllable, ocr:<mã sai>, den:<mã đúng>, book}   ghi đè theo sách  [MỞ RỘNG *]
  nham:<mã sai>-><mã đúng>   dạng tường minh của nham (khi mã đúng là mã thứ ba)
  mo                    lop_nham {syllable, ocr:<mã thiểu số>, den: REVIEW, book}  ô -> REVIEW    [MỞ RỘNG *]
  (trống)               bỏ qua, đếm 'chua_quyet'

[*] Schema hiện có (pipeline/decisions.py:47 DEN = ("REVIEW",); :235 lop_nham không có `book`) và
    build (align_engine/build_dataset.py:719 chỉ áp lop_nham cho âm ∈ AM_DA_QUYET = {'người'}) CHƯA
    thi hành được `den:<mã>` và `book` — công cụ vẫn ghi đúng ý người ký và BÁO số mục cần mở rộng;
    `--nham-nhu-di-the` ghi nham thành di_the (chỉ đổi label_canonical) để áp được ngay bằng schema cũ.

TRẠNG THÁI: `da_ky` CHỈ khi `xuat_xu` không rỗng VÀ tệp tồn tại trong repo VÀ `nguoi_ky` không rỗng;
ngược lại `cho_ky`. Công cụ KHÔNG ghi vào config/ — chỉ in KhoiB/h2/decisions_lop_proposed.yaml
(= decisions.yaml hiện có + mục mới; mục cùng cặp/sách đã có trong config được GIỮ hoặc cập nhật
nếu đang cho_ky) và apply_lop_report.json (số ô ảnh hưởng đo trên dataset_out_v3/labels.csv,
ngoài khoá QĐ-01).

    .venv/bin/python -m pipeline.tools.apply_lop_decisions --filled KhoiB/h2/lop_decisions_filled.csv
    .venv/bin/python -m pipeline.tools.apply_lop_decisions --gia-dinh-e3      # ước lượng nếu ký toàn bộ theo E3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipeline import decisions as dcs                                    # noqa: E402
from pipeline.decisions import BOOKS, MUC, nfc, uni                      # noqa: E402

NA = dict(dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
QUYET_RE = re.compile(r"^(dung|mo|di_the:(?P<dt>\S)|nham:(?:(?P<sai>\S)->)?(?P<dung>\S))$")
H2_HTML = "KhoiB/h2/lop_review.html"


def phan_tich_quyet(q: str, M: str, L: str) -> dict:
    """-> {loai: dung|mo|di_the|nham, chuan?, sai?, dung?} hoặc raise ValueError."""
    q = nfc(q)
    if not q:
        return {"loai": ""}
    m = QUYET_RE.match(q)
    if not m:
        raise ValueError(f"quyet {q!r} không hợp lệ (dung | mo | di_the:<mã> | nham:<mã đúng> | nham:<mã sai>-><mã đúng>)")
    if q == "dung":
        return {"loai": "dung"}
    if q == "mo":
        return {"loai": "mo", "sai": L}
    if m.group("dt"):
        c = m.group("dt")
        if c not in (M, L):
            raise ValueError(f"di_the:{c} — chuẩn phải ∈ {{{M}, {L}}}")
        return {"loai": "di_the", "chuan": c}
    dung, sai = m.group("dung"), m.group("sai")
    if sai is None:
        sai = L if dung == M else (M if dung == L else L)      # mã thứ ba -> ghi đè mã THIỂU SỐ (chủ thể của lớp)
    if sai == dung:
        raise ValueError(f"nham: mã sai {sai} trùng mã đúng {dung}")
    return {"loai": "nham", "sai": sai, "dung": dung, "ma_thu_ba": dung not in (M, L)}


def trang_thai(row) -> tuple[str, str]:
    """da_ky chỉ khi xuat_xu tồn tại trên đĩa VÀ có nguoi_ky; trả (trang_thai, lý do)."""
    xx, nk = nfc(row.get("xuat_xu", "")), nfc(row.get("nguoi_ky", ""))
    if not xx:
        return "cho_ky", "thieu_xuat_xu"
    p = Path(xx) if Path(xx).is_absolute() else REPO / xx.split("#", 1)[0]
    if not p.is_file():
        return "cho_ky", f"xuat_xu_khong_ton_tai:{xx}"
    if not nk:
        return "cho_ky", "thieu_nguoi_ky"
    return "da_ky", ""


def dem_o(labels: pd.DataFrame):
    """Bộ đếm ô có label, NGOÀI khoá QĐ-01, theo (book, syllable, label) + theo tier."""
    lab = labels[(labels["label"] != "") & (labels["qd01_locked"] != "1")]
    c_all = Counter(zip(lab["book"], lab["syllable"], lab["label"]))
    c_gold = Counter(zip(lab[lab["tier"] == "GOLD"]["book"], lab[lab["tier"] == "GOLD"]["syllable"],
                         lab[lab["tier"] == "GOLD"]["label"]))
    c_lab_book = Counter(zip(lab["book"], lab["label"]))       # so_o của di_the (như sinh_decisions: mọi tier có label)
    am_theo_ma: dict[tuple, Counter] = defaultdict(Counter)    # (book, label) -> Counter(syllable): di_the lan sang âm khác?
    for b, l, s in zip(lab["book"], lab["label"], lab["syllable"]):
        am_theo_ma[(b, l)][s] += 1
    return c_all, c_gold, c_lab_book, am_theo_ma


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--filled", default="", help="CSV người đã điền (mặc định: template nếu --gia-dinh-e3)")
    ap.add_argument("--template", default=str(REPO / "KhoiB/h2/lop_decisions_template.csv"))
    ap.add_argument("--config", default=str(REPO / "config/decisions.yaml"))
    ap.add_argument("--labels", default=str(REPO / "dataset_out_v3/labels.csv"))
    ap.add_argument("--out", default=str(REPO / "KhoiB/h2/decisions_lop_proposed.yaml"))
    ap.add_argument("--report", default=str(REPO / "KhoiB/h2/apply_lop_report.json"))
    ap.add_argument("--gia-dinh-e3", action="store_true",
                    help="điền quyet giả định (MOT/không E3 -> nham:<mã đa số>, mo -> mo) để ƯỚC LƯỢNG; mọi mục cho_ky")
    ap.add_argument("--nham-nhu-di-the", action="store_true",
                    help="ghi nham:<mã> thành mục di_the (chỉ đổi label_canonical) để áp được bằng schema hiện có")
    a = ap.parse_args()

    out_p = Path(a.out)
    if out_p.resolve().is_relative_to((REPO / "config").resolve()):
        raise SystemExit("[apply_lop] 🔴 không ghi vào config/ — chỉ đề xuất; người ký chép tay sau khi xem")

    src = a.filled or a.template
    df = pd.read_csv(src, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    need = {"id_lop", "book", "syllable", "ma_da_so", "ma_thieu_so", "quyet"}
    if not need <= set(df.columns):
        raise SystemExit(f"[apply_lop] 🔴 {src} thiếu cột {sorted(need - set(df.columns))}")
    for c in ("nguoi_ky", "ngay", "xuat_xu", "e3_verdict", "n_da_so", "n_thieu_so"):
        if c not in df.columns:
            df[c] = ""
    if df["id_lop"].duplicated().any():
        raise SystemExit(f"[apply_lop] 🔴 id_lop trùng: {df[df['id_lop'].duplicated()]['id_lop'].tolist()[:5]}")
    if a.gia_dinh_e3:
        df["quyet"] = ["mo" if v == "mo" else f"nham:{M}" for v, M in zip(df["e3_verdict"], df["ma_da_so"])]
        df["nguoi_ky"], df["xuat_xu"] = "", ""

    labels = pd.read_csv(a.labels, **NA)
    c_all, c_gold, c_lab_book, am_theo_ma = dem_o(labels)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    doc = {m: (cfg.get(m) if cfg.get(m) is not None else []) for m in MUC[:3]}
    doc["khoa_o"] = cfg.get("khoa_o") or {"cells": "config/qd01_cells.csv", "decisions": "config/qd01a_decisions.csv"}
    dt_by_pair = {(frozenset(x["quan_sat"]), x.get("book")): x for x in doc["di_the"]}
    ids = {x["id"] for m in MUC[:3] for x in doc[m]}

    # ---- dịch từng dòng ----
    rows_out, loi, tk = [], [], Counter()
    for r in df.to_dict("records"):
        b, s, M, L = r["book"], nfc(r["syllable"]).lower(), nfc(r["ma_da_so"]), nfc(r["ma_thieu_so"])
        if b not in BOOKS[1:]:
            loi.append(f"{r['id_lop']}: book {b!r} ∉ {BOOKS[1:]}")
            continue
        try:
            q = phan_tich_quyet(r["quyet"], M, L)
        except ValueError as ex:
            loi.append(f"{r['id_lop']}: {ex}")
            continue
        if not q["loai"]:
            tk["chua_quyet"] += 1
            continue
        tt, ly_do = trang_thai(r)
        rows_out.append(dict(r, _q=q, _tt=tt, _ly_do=ly_do, _b=b, _s=s, _M=M, _L=L))
        tk[q["loai"]] += 1
        tk[f"tt_{tt}"] += 1
    if loi:
        print("[apply_lop] 🔴 dòng lỗi (bỏ qua):\n  " + "\n  ".join(loi[:30]), file=sys.stderr)

    # ---- gộp theo (cặp/mục, quyết) qua 3 sách: cùng quyết ở cả 3 sách -> một mục book: null ----
    nhom: dict[tuple, list] = defaultdict(list)
    for r in rows_out:
        q = r["_q"]
        if q["loai"] == "dung":
            continue
        if q["loai"] == "di_the" or (q["loai"] == "nham" and a.nham_nhu_di_the):
            chuan = q.get("chuan") or q["dung"]
            k = ("di_the", frozenset([r["_M"], r["_L"]]) | {chuan}, chuan, r["_s"] if q["loai"] == "nham" else "")
        elif q["loai"] == "nham":
            k = ("nham", r["_s"], q["sai"], q["dung"])
        else:
            k = ("mo", r["_s"], q["sai"], "REVIEW")
        nhom[k].append(r)

    new_dt, new_ln, cap_nhat, anh_huong, canh_bao = [], [], [], Counter(), []
    o_theo_muc = {}
    can_mo_rong = Counter()
    for k, rs in nhom.items():
        books = sorted({r["_b"] for r in rs})
        tts = {r["_tt"] for r in rs}
        tt = "da_ky" if tts == {"da_ky"} else "cho_ky"       # một sách chưa ký -> cả mục cho_ky (không ký thay)
        gop = len(books) == 3
        scopes = [None] if gop else books
        for bk in scopes:
            rr = rs if gop else [r for r in rs if r["_b"] == bk]
            bks = list(BOOKS[1:]) if gop else [bk]
            r0 = rr[0]
            nk = "; ".join(sorted({nfc(r["nguoi_ky"]) for r in rr if nfc(r["nguoi_ky"])})) or None
            ng = "; ".join(sorted({nfc(r["ngay"]) for r in rr if nfc(r["ngay"])})) or None
            xx = sorted({nfc(r["xuat_xu"]) for r in rr if nfc(r["xuat_xu"])})
            xx = xx[0].split("#", 1)[0] if xx else ""
            ids_lop = ", ".join(r["id_lop"] for r in rr)
            if k[0] == "di_the":
                codes = sorted(k[1]); chuan = k[2]
                so_o = {c: int(sum(c_lab_book.get((x, c), 0) for x in bks)) for c in codes}
                n_doi = int(sum(c_all.get((x, r0["_s"], c), 0) for x in bks for c in codes if c != chuan))
                # đơn vị (sách, âm): ô đổi label_canonical đếm trên MỌI âm mang mã ≠ chuẩn trong sách (như apply_di_the)
                n_doi_moi_am = int(sum(c_lab_book.get((x, c), 0) for x in bks for c in codes if c != chuan))
                idd = f"dt_{'_'.join(uni(c)[2:].lower() for c in codes)}" + (f"_{bk}" if bk else "")
                # apply_di_the khoá theo (label, book) — KHÔNG theo âm: mã ≠ chuẩn đang đọc âm khác cũng bị đổi
                am_khac = Counter()
                for x in bks:
                    for c in codes:
                        if c != chuan:
                            am_khac.update({k: v for k, v in am_theo_ma[(x, c)].items() if k != r0["_s"]})
                lan = ", ".join(f"{k} {v}" for k, v in am_khac.most_common(6))
                if am_khac:
                    canh_bao.append(f"{idd}: di_the {'/'.join(codes)}->{chuan} sách {bk or 'mọi'} "
                                    f"lan sang {sum(am_khac.values())} ô âm khác ({lan})")
                quy_tac = "da_so_ngu_lieu" if chuan == max(codes, key=lambda c: so_o[c]) else "unicode"
                la_nham = k[3] != ""
                ex = dt_by_pair.get((frozenset(codes), bk))
                item = {
                    "id": idd, "quan_sat": codes, "unicode": {c: uni(c) for c in codes}, "chuan": chuan,
                    "book": bk, "quy_tac": quy_tac, "so_o": so_o,
                    "kiem_unicode": ("H2-lite: người ký theo crop — OCR nhầm khác chữ, KHÔNG phải dị thể Unicode (ghi thành di_the vì --nham-nhu-di-the)"
                                     if la_nham else "H2-lite: người ký theo crop (không tự kiểm Unihan)"),
                    "ghi_chu": (f"H2-lite lớp {ids_lop}: âm '{r0['_s']}', {' / '.join(f'{c} {so_o[c]}' for c in codes)}; "
                                f"E3 {r0['e3_verdict'] or '—'}; ô đổi label_canonical (âm này, ngoài khoá) {n_doi}, mọi âm {n_doi_moi_am}"
                                + (f" — LAN sang âm khác: {lan}" if am_khac else "")),
                    "xuat_xu": xx, "trang_thai": tt,
                }
                if nk: item["nguoi_ky"] = nk
                if ng: item["ngay"] = ng
                if ex is not None and ex.get("trang_thai") != "da_ky":
                    ex.update({kk: item[kk] for kk in ("chuan", "quy_tac", "so_o", "ghi_chu", "xuat_xu", "trang_thai")})
                    for kk in ("nguoi_ky", "ngay"):
                        if kk in item: ex[kk] = item[kk]
                    cap_nhat.append(ex["id"]); idd = ex["id"]
                elif ex is not None:
                    loi.append(f"{ids_lop}: cặp {codes} sách {bk} đã da_ky trong config ({ex['id']}) — không ghi đè")
                    continue
                else:
                    if idd in ids:
                        idd = idd + "_h2"
                    item["id"] = idd; ids.add(idd); new_dt.append(item)
                o_theo_muc[idd] = {"loai": "di_the" if not la_nham else "nham_nhu_di_the", "book": bk, "o_am_nay": n_doi,
                                   "o_moi_am": n_doi_moi_am, "trang_thai": tt}
                anh_huong["di_the_label_canonical_am_nay"] += n_doi
                anh_huong["di_the_label_canonical_moi_am"] += n_doi_moi_am
            else:
                syl, sai, den = k[1], k[2], k[3]
                n = int(sum(c_all.get((x, syl, sai), 0) for x in bks))
                n_g = int(sum(c_gold.get((x, syl, sai), 0) for x in bks))
                idd = f"ln_{syl}_{uni(sai)[2:].lower()}_{'review' if den == 'REVIEW' else uni(den)[2:].lower()}" + (f"_{bk}" if bk else "")
                if idd in ids:
                    idd += "_h2"
                ids.add(idd)
                item = {"id": idd, "syllable": syl, "ocr": sai, "unicode": uni(sai), "den": den, "pham_vi": "ngoai_khoa",
                        "xuat_xu": xx, "trang_thai": tt,
                        "ghi_chu": (f"H2-lite lớp {ids_lop}: " + (f"ghi đè {sai}->{den} theo sách" if den != "REVIEW" else f"mờ -> {sai} về REVIEW")
                                    + f"; ô ngoài khoá {n} (GOLD {n_g}); E3 {r0['e3_verdict'] or '—'}")}
                if bk:
                    item["book"] = bk                                  # MỞ RỘNG: schema lop_nham chưa có book
                    can_mo_rong["lop_nham_book"] += 1
                if den != "REVIEW":
                    can_mo_rong["lop_nham_den_ma"] += 1               # MỞ RỘNG: DEN chỉ có REVIEW
                if any(r["_q"].get("ma_thu_ba") for r in rr):
                    item["ghi_chu"] += " · mã đúng là mã THỨ BA: cần xác nhận cả mã đa số"
                    can_mo_rong["nham_ma_thu_ba_can_xac_nhan"] += 1
                if nk: item["nguoi_ky"] = nk
                if ng: item["ngay"] = ng
                new_ln.append(item)
                o_theo_muc[idd] = {"loai": k[0], "book": bk, "o": n, "o_gold": n_g, "trang_thai": tt}
                anh_huong["nham_ghi_de" if den != "REVIEW" else "mo_review"] += n
    if loi:
        print("[apply_lop] 🔴 mâu thuẫn với config (bỏ qua):\n  " + "\n  ".join(loi[-10:]), file=sys.stderr)

    doc["di_the"] = doc["di_the"] + new_dt
    doc["lop_nham"] = doc["lop_nham"] + new_ln
    out_p.parent.mkdir(parents=True, exist_ok=True)
    header = Path(a.config).read_text(encoding="utf-8").split("\ncorpus_readings:", 1)[0]
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=110, default_flow_style=False)
    out_p.write_text(header + "\n# ---- H2-lite (pipeline/tools/apply_lop_decisions.py) ĐỀ XUẤT — chưa chép vào config/ ----\n"
                     + body, encoding="utf-8")

    # ---- kiểm bằng loader hiện có ----
    kiem = {"loader": "ok", "loi": ""}
    try:
        d = dcs.load(out_p)
        kiem["report"] = {m: {kk: d.report()[m][kk] for kk in ("n", "da_ky", "cho_ky")} for m in MUC[:3]}
    except SystemExit as ex:
        kiem = {"loader": "tu_choi", "loi": str(ex)}

    n_dung = sum(1 for r in rows_out if r["_q"]["loai"] == "dung")
    rep = {
        "nguon": src, "gia_dinh_e3": a.gia_dinh_e3, "nham_nhu_di_the": a.nham_nhu_di_the,
        "dong": {"tong": int(len(df)), **{k: int(v) for k, v in tk.items()}},
        "muc_moi": {"di_the": len(new_dt), "lop_nham": len(new_ln), "cap_nhat_cho_ky": cap_nhat, "dung_khong_sinh_muc": n_dung},
        "o_anh_huong_ngoai_khoa": dict(anh_huong),
        "o_theo_muc": o_theo_muc,
        "can_mo_rong_schema": dict(can_mo_rong),
        "canh_bao_di_the_lan_am_khac": canh_bao,
        "ghi_chu_thi_hanh": ("decisions.py:47 DEN=('REVIEW',) và :235 lop_nham không có `book`; "
                             "build_dataset.py:719 chỉ áp lop_nham cho âm ∈ AM_DA_QUYET={'người'} — "
                             "mục lop_nham H2 cần mở rộng cả hai chỗ trước khi có hiệu lực"),
        "kiem_loader": kiem, "loi": loi, "out": str(out_p),
    }
    Path(a.report).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[apply_lop] {src}: {len(df)} dòng · quyết {len(rows_out)} (dung {n_dung}, di_the {tk['di_the']}, nham {tk['nham']}, "
          f"mo {tk['mo']}) · chưa quyết {tk['chua_quyet']} · da_ky {tk['tt_da_ky']} / cho_ky {tk['tt_cho_ky']}")
    print(f"[apply_lop] mục mới: di_the {len(new_dt)} (+cập nhật cho_ky {len(cap_nhat)}), lop_nham {len(new_ln)} "
          f"· ô ảnh hưởng ngoài khoá: {dict(anh_huong)} · cần mở rộng schema: {dict(can_mo_rong)}")
    for w in canh_bao:
        print(f"[apply_lop] ⚠ {w}")
    print(f"[apply_lop] loader hiện có: {kiem['loader']} {kiem.get('loi', '')[:160]}")
    print(f"[apply_lop] -> {out_p} · {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
