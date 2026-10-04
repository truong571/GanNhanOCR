#!/usr/bin/env python3
"""Kiểm thử OFFLINE vision_crosscheck.py bằng dữ liệu tổng hợp có đáp án biết trước (0 yêu cầu).
  .venv/bin/python vision/tests/test_vision_crosscheck.py"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


def load(env):
    for k, v in env.items():
        os.environ[k] = str(v)
    name = f"vc_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(name, CODE / "vision_crosscheck.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    tmp = Path(tempfile.mkdtemp(prefix="vc_test_"))
    (tmp / "dict").mkdir()
    (tmp / "dict" / "QuocNgu_SinoNom.csv").write_text("quoc_ngu,sino_nom\nba,三\nba,巴\nbon,四\ncuong,強\n", encoding="utf-8")
    (tmp / "dict" / "SinoNom_Similar.csv").write_text("Input Character,Top 20 Similar Characters\n強,\"['强', '弶']\"\n", encoding="utf-8")
    vc = load(dict(VISION_REPO_ROOT=tmp, VISION_CACHE_ROOT=tmp / "cache", VISION_AUDIT_OUT=tmp / "out"))
    R = vc.load_dict()
    sim = vc.load_similar()
    # một cột 10 ô 40×40 (x 100–140), bước 45
    rows = []
    for k in range(10):
        rows.append(dict(book_key="T", book="t", page="page_0001", column=1, tier="GOLD", syllable="ba", label="三", ocr_char="三", x0=100, y0=50 + 45 * k, x1=140, y1=90 + 45 * k))
    pdf = pd.DataFrame(rows)

    def sym(ch, k, conf=0.95, dy=0.0, dx=0.0, size=32, ang=0.0):
        cx, cy = 120 + dx * 40, 70 + 45 * k + dy * 40
        return (ch, conf, cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, ang)
    syms = [
        sym("三", 0),                                   # k0 khớp, căn giữa
        sym("三", 1, dy=0.30),                          # k1 khớp, hộp pipeline cao hơn glyph 0,3 ô
        sym("巴", 2, conf=0.9),                         # k2 Vision nói chữ khác (cùng URO, không phải dị thể)
        # k3: không có ký hiệu
        # k4: không có ký hiệu của chính ô 4 (xem C7)
        ("a", 0.99, 100, 50 + 45 * 6, 140, 90 + 45 * 6, 0.0),  # k6: chỉ có chữ Latin → bị lọc → no_vision
        sym("三", 7, conf=0.3),                         # k7: khớp nhưng tin cậy thấp
        sym("三", 8, conf=0.9),                         # k8: khớp (đủ 3 ô agree conf≥0,5 cho độ lệch theo cột)
    ]
    D = pd.DataFrame(vc.analyse_page(pdf, [list(s) for s in syms], R, sim=sim))
    st = D["status"].tolist()
    check("C1.khop_can_giua", st[0] == "agree" and abs(D.loc[0, "dy"]) < 0.02 and abs(D.loc[0, "dx"]) < 0.02, f"{st[0]} dy={D.loc[0, 'dy']:.3f}")
    check("C2.khop_lech_hop_0_3_theo_buoc_cot", st[1] == "agree" and abs(D.loc[1, "dy"] - 0.30) < 0.02 and abs(D.loc[1, "dy_pitch"] - 0.30 * 40 / 45) < 0.02, f"dy={D.loc[1, 'dy']:.3f} dy_pitch={D.loc[1, 'dy_pitch']:.3f}")
    check("C3.khac_chu_thuoc_URO_la_disagree", st[2] == "disagree" and D.loc[2, "v"] == "巴" and D.loc[2, "v_class"] == "URO", f"{st[2]}")
    check("C4.khong_ky_hieu", st[3] == "no_vision", st[3])
    check("C5.chu_Latin_bi_loc", st[6] == "no_vision", st[6])
    check("C6.tin_cay_thap_van_ghi", st[7] == "agree" and D.loc[7, "vconf"] < 0.5, f"conf={D.loc[7, 'vconf']}")

    # trượt: ô 5 nhãn khác → ký hiệu '三' tại ô 5 không phải của ô 5; ô 4 nhãn '三' → 'shifted' lệch 45/40 ô
    pdf2 = pdf.copy()
    pdf2.loc[5, "label"] = "巴"; pdf2.loc[5, "ocr_char"] = "巴"
    D2 = pd.DataFrame(vc.analyse_page(pdf2, [list(sym("三", 5))], R, sim=sim))
    check("C7.phat_hien_lech_anh_chu_1_o", D2.loc[4, "status"] == "shifted" and abs(D2.loc[4, "shift_dy"] - 45 / 40) < 0.05 and D2.loc[5, "status"] == "disagree",
          f"ô4={D2.loc[4, 'status']} dy={D2.loc[4, 'shift_dy']} ô5={D2.loc[5, 'status']}")
    # ứng viên KHÔNG duy nhất → không báo trượt
    pdf2b = pdf.copy()
    pdf2b.loc[3, "label"] = "巴"; pdf2b.loc[3, "ocr_char"] = "巴"; pdf2b.loc[5, "label"] = "巴"; pdf2b.loc[5, "ocr_char"] = "巴"   # hai ký hiệu '三' lân cận đều do ô 'disagree' giữ
    D2b = pd.DataFrame(vc.analyse_page(pdf2b, [list(sym("三", 5)), list(sym("三", 3))], R, sim=sim))
    check("C7b.ung_vien_khong_duy_nhat_khong_bao_truot", D2b.loc[4, "status"] != "shifted", D2b.loc[4, "status"])
    # ghép 1–1
    D3 = pd.DataFrame(vc.analyse_page(pdf.iloc[:2].copy(), [list(sym("三", 0, dy=0.1)), list(sym("三", 0, dy=0.45))], R, sim=sim))
    check("C8.ghep_1_1", D3.loc[0, "status"] == "agree" and abs(D3.loc[0, "dy"] - 0.1) < 0.02, f"dy={D3.loc[0, 'dy']:.3f}")

    # dị thể giản/phồn (SinoNom_Similar) không tính bất đồng; lớp mã nhãn
    pdf4 = pdf.iloc[:2].copy()
    pdf4["label"] = ["強", "\U0002029A"]; pdf4["ocr_char"] = pdf4["label"]; pdf4["syllable"] = ["cuong", "nom"]
    D4 = pd.DataFrame(vc.analyse_page(pdf4, [list(sym("强", 0)), list(sym("羅", 1))], R, sim=sim))
    check("C11.di_the_gian_phon_la_variant_khong_phai_disagree", D4.loc[0, "status"] == "variant" and D4.loc[1, "status"] == "disagree" and D4.loc[1, "label_class"] == "ExtB+",
          f"{D4['status'].tolist()} lớp={D4['label_class'].tolist()}")

    # chỉ số hại hình học: glyph lệch hẳn ra ngoài cửa sổ pad → harm=1; căn giữa → 0; ký hiệu nghiêng >3° → không đo (NaN)
    D5 = pd.DataFrame(vc.analyse_page(pdf.iloc[:3].copy(), [list(sym("三", 0)), list(sym("三", 1, dy=0.55)), list(sym("三", 2, ang=8.0))], R, sim=sim))
    check("C12.chi_so_hai_hinh_hoc", D5.loc[0, "harm"] == 0.0 and D5.loc[1, "harm"] == 1.0 and np.isnan(D5.loc[2, "harm"]), f"{D5['harm'].tolist()}")

    # bỏ-một-ô: cột lệch HẰNG 0,2 bước → mô hình hằng đúng (~100 %), 'không dịch' sai (~0 %)
    rows6 = [dict(book="T", page="p", column=1, y0=50 + 45 * k, y1=90 + 45 * k, pitch=45.0, dy_pitch=0.2 + (0.01 if k % 2 else -0.01)) for k in range(8)]
    cm, errs = vc.column_models(pd.DataFrame(rows6))
    check("C13.bo_mot_o_hang_so", len(cm) == 1 and vc._frac_le(errs["const"]) == 1.0 and vc._frac_le(errs["zero"]) == 0.0 and len(errs["zero"]) == 8,
          f"hằng={vc._frac_le(errs['const'])} không dịch={vc._frac_le(errs['zero'])} n={len(errs['zero'])}")

    # đầu-cuối: run_cells trên kho tổng hợp (pages/*.json + dataset/*/labels.csv)
    (tmp / "dataset" / "KimVanKieu1884").mkdir(parents=True)
    lab = pdf.copy()
    lab["bbox"] = lab.apply(lambda r: json.dumps([int(r.x0), int(r.y0), int(r.x1), int(r.y1)]), axis=1)
    lab["image"] = "x.png"; lab["page"] = "page_0001"; lab["column"] = 1; lab["tier"] = "GOLD"; lab["syllable"] = "ba"
    lab[["image", "page", "column", "ocr_char", "syllable", "label", "tier", "bbox"]].assign(book="kimvankieu1884", rule="").to_csv(tmp / "dataset" / "KimVanKieu1884" / "labels.csv", index=False)
    cid = "test-cfg"
    (tmp / "cache" / cid / "pages" / "KVK").mkdir(parents=True)
    (tmp / "cache" / cid / "pages" / "KVK" / "page_0001.json").write_text(json.dumps(dict(book="KVK", page="page_0001", cols=["ch", "conf", "x0", "y0", "x1", "y1", "ang"], sym=[list(s) for s in syms])), encoding="utf-8")
    vc2 = load(dict(VISION_REPO_ROOT=tmp, VISION_CACHE_ROOT=tmp / "cache", VISION_AUDIT_OUT=tmp / "out"))
    vc2.run_cells(cid, None, log=lambda *a: None)
    summ = json.loads((tmp / "out" / cid / "summary.json").read_text())
    dc = pd.read_csv(tmp / "out" / cid / "disagree_candidates.csv")
    off = pd.read_csv(tmp / "out" / cid / "offsets.csv")
    check("C9.dau_cuoi_ung_vien_bat_dong_URO_va_khong_con_recrop_theo_o", len(dc) == 1 and dc.iloc[0]["v"] == "巴" and not (tmp / "out" / cid / "recrop.csv").exists()
          and not (tmp / "out" / cid / "rescue.csv").exists(), f"ứng viên={len(dc)}")
    check("C10.do_lech_theo_cot_va_summary", len(off) == 1 and off.iloc[0]["n"] == 3 and "loo_tong" in summ and "theo_lop_nhan" in summ and (tmp / "out" / cid / "column_model.csv").exists(),
          f"{off.to_dict('records')}")
    nf = [n for n, ok in RES if not ok]
    print(f"TỔNG: pass={len(RES) - len(nf)} fail={len(nf)} n={len(RES)}")
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
