"""Selftest B4' cổng cơ chế (pipeline/remediation/mechanism_gates.py) + export tầng GOLD_text_only.

    .venv/bin/python -m pipeline.remediation.mechanism_gates_selftest      # exit 0 = pass

Dữ liệu GIẢ (không đọc dataset_out*), kiểm: bật/tắt theo config, từng cổng (a)(b)(c)(d), thứ tự
ưu tiên, không mutate, idempotent, bản sao byte khi tắt, "nan" là âm 難 không bị nuốt, export
ghi GOLD_text_only vào labels.csv nhưng KHÔNG copy ảnh, export không có tầng mới = như cũ,
make_dataset_docs chỉ thêm khối khi có tầng.
"""
from __future__ import annotations

import csv
import hashlib
import io
import shutil
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from pipeline.remediation import mechanism_gates as mg

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}  {detail}")


def _row(i: int, **kw) -> dict:
    base = dict(image=f"gold/b_page_0001_c01_{i:03d}.png", book="b", page="page_0001", column="1",
                ocr_char="城", syllable="thành", label="城", unicode="U+57CE", label_level="char",
                tier="GOLD", rule="s1_inter_s2_direct", s3_cosine="", ink_pct="0.10", crop_w="90",
                crop_h="120", image_md5=f"md5{i:03d}", seg_flag="ok", bbox="[0, 0, 10, 10]",
                nom_idx=str(i), syl_idx=str(i), n_ocr="14", n_qn="14", n_det="14",
                count_source="equal_qn", box_source="detector", label_canonical="城",
                tier_v3="CHAR_A", crop_quality_flag="ok")
    base.update(kw)
    return base


def _synthetic() -> pd.DataFrame:
    rows = [
        _row(1),                                                                   # sạch
        _row(2, n_det="15"),                                                       # (a) n_det ≠ N
        _row(3, box_source="midpoint"),                                            # (a) midpoint
        _row(4, box_source="split", n_det="13"),                                   # (a) cả hai
        _row(5, rule="s1_inter_s2_similar", tier_v3="CHAR_B", label="㝵", unicode="U+3775",
             label_canonical="㝵"),                                                # (b) cầu
        _row(6, rule="s1_inter_s2_direct_lowp", tier_v3="CHAR_B"),                 # (b) CHAR_B không rule
        _row(7, rule="s1_inter_s2_direct_am_sua_dau"),                             # (b) sửa dấu
        _row(8, crop_quality_flag="blank", box_source="midpoint"),                 # (c) thắng (a)
        _row(9, tier="SYLLABLE", label="", unicode="", label_level="syllable",
             image="syllable/b_page_0001_c01_009.png", rule="syl_ctx:tone",
             crop_quality_flag="truncated"),                                       # (c) SYLLABLE
        _row(10, tier="REVIEW", label="", unicode="", label_level="", rule="no_context",
             image="", crop_quality_flag="blank"),                                 # không đụng
        _row(11, tier="QUARANTINE", rule="s1_inter_s2_direct|quarantine_dup", n_det="15"),  # không đụng
        _row(12),                                                                  # (d) similar
        _row(13),                                                                  # (d) khác, không gần hình
        _row(14),                                                                  # (d) 2 ref: 1 khớp
        _row(15),                                                                  # (d) ref PUA → bỏ
        _row(16, rule="s1_inter_s2_similar", tier_v3="CHAR_B"),                    # (d) thắng (b)
        _row(17, syllable="nan", ocr_char="難", label="難", unicode="U+96E3", label_canonical="難"),
        _row(18, n_det="", n_qn="14"),                                             # n_det rỗng ≠ "14" → (a)
    ]
    return pd.DataFrame(rows, dtype=str)


def _cross() -> dict:
    img = lambda i: f"gold/b_page_0001_c01_{i:03d}.png"
    rows = [
        dict(image=img(12), eq="0", similar="1", ref_pua="0", ref="R1"),
        dict(image=img(13), eq="0", similar="0", ref_pua="0", ref="R1"),
        dict(image=img(14), eq="0", similar="1", ref_pua="0", ref="R1"),
        dict(image=img(14), eq="1", similar="0", ref_pua="0", ref="R2"),
        dict(image=img(15), eq="0", similar="1", ref_pua="1", ref="R1"),
        dict(image=img(16), eq="0", similar="1", ref_pua="0", ref="R1"),
        dict(image=img(1), eq="1", similar="0", ref_pua="0", ref="R1"),
    ]
    tmp = Path(tempfile.mkdtemp()) / "cells.csv"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    return mg.load_cross(tmp)


def test_enabled() -> None:
    print("[1] bật/tắt theo config")
    check("None → tắt", mg.gates_enabled(None) is False)
    check("{} → tắt", mg.gates_enabled({}) is False)
    check("layout stt → tắt", mg.gates_enabled({"name": "x", "layout": "stt"}) is False)
    check("layout lithograph → bật", mg.gates_enabled({"name": "x", "layout": "lithograph"}) is True)
    check("lithograph + mechanism_gates false → tắt",
          mg.gates_enabled({"name": "x", "layout": "lithograph", "mechanism_gates": False}) is False)
    check("mechanism_gates true (không layout) → bật", mg.gates_enabled({"name": "x", "mechanism_gates": True}) is True)
    try:
        mg.gates_enabled({"name": "x", "mechanism_gates": "yes"})
        check("mechanism_gates sai kiểu → ValueError", False)
    except ValueError:
        check("mechanism_gates sai kiểu → ValueError", True)
    check("--enable on ghi đè", mg.gates_enabled({"layout": "stt"}, "on") is True)
    check("--enable off ghi đè", mg.gates_enabled({"layout": "lithograph"}, "off") is False)
    d = Path(tempfile.mkdtemp())
    (d / "c.yaml").write_text("books:\n  - name: KimVanKieu1884\n    layout: lithograph\n"
                              "  - name: SachThanhTruyen2\n    pdf: x.pdf\n", encoding="utf-8")
    check("load_book_cfg không phân biệt hoa/thường",
          (mg.load_book_cfg(d / "c.yaml", "kimvankieu1884") or {}).get("layout") == "lithograph")
    check("load_book_cfg sách STT → tắt", mg.gates_enabled(mg.load_book_cfg(d / "c.yaml", "SachThanhTruyen2")) is False)
    check("load_book_cfg sách lạ → None", mg.load_book_cfg(d / "c.yaml", "Khac") is None)


def test_apply() -> None:
    print("[2] áp cổng trên dữ liệu giả")
    df = _synthetic()
    snap = df.copy()
    out, rep = mg.apply_gates(df, _cross())
    check("không mutate đầu vào", df.equals(snap))
    check("số dòng giữ nguyên", len(out) == len(df))
    T = dict(zip(out["nom_idx"], out["tier"]))
    R = dict(zip(out["nom_idx"], out["gate_reason"]))
    check("ô sạch giữ GOLD, gate_reason rỗng", T["1"] == "GOLD" and R["1"] == "")
    check("(a) n_det ≠ N → GOLD_text_only", T["2"] == mg.TIER_TEXT_ONLY and R["2"] == mg.G_NDET)
    check("(a) midpoint → GOLD_text_only", T["3"] == mg.TIER_TEXT_ONLY and R["3"] == "box_not_detector:midpoint")
    check("(a) cả hai → lý do nối '+'", R["4"] == "n_det_ne_n_qn+box_not_detector:split", R["4"])
    a2 = out[out["nom_idx"] == "2"].iloc[0]
    check("(a) giữ label/unicode/syllable/rule", a2["label"] == "城" and a2["unicode"] == "U+57CE"
          and a2["syllable"] == "thành" and a2["rule"] == "s1_inter_s2_direct" and a2["label_level"] == "char")
    b5 = out[out["nom_idx"] == "5"].iloc[0]
    check("(b) cầu → SYLLABLE, label/unicode rỗng, label_level syllable",
          b5["tier"] == "SYLLABLE" and b5["label"] == "" and b5["unicode"] == "" and b5["label_level"] == "syllable")
    check("(b) chữ gốc còn ở label_canonical + rule hậu tố",
          b5["label_canonical"] == "㝵" and b5["rule"] == "s1_inter_s2_similar|gate:bridge_similar")
    check("(b) tier_v3 CHAR_B không rule similar vẫn hạ", T["6"] == "SYLLABLE" and R["6"] == mg.G_BRIDGE)
    check("(b) am_sua_dau → SYLLABLE", T["7"] == "SYLLABLE" and R["7"] == mg.G_TONE)
    c8 = out[out["nom_idx"] == "8"].iloc[0]
    check("(c) blank thắng (a): REVIEW, giữ label, label_level rỗng",
          c8["tier"] == "REVIEW" and c8["label"] == "城" and c8["label_level"] == "" and R["8"] == "crop_bad:blank")
    check("(c) SYLLABLE truncated → REVIEW", T["9"] == "REVIEW" and R["9"] == "crop_bad:truncated")
    check("REVIEW blank không đụng", T["10"] == "REVIEW" and R["10"] == "")
    check("QUARANTINE không đụng", T["11"] == "QUARANTINE" and R["11"] == "")
    check("(d) bất đồng gần hình → REVIEW", T["12"] == "REVIEW" and R["12"] == mg.G_CROSS)
    d13 = out[out["nom_idx"] == "13"].iloc[0]
    check("(d) bất đồng không gần hình → giữ GOLD + di_ban_khac=1", d13["tier"] == "GOLD" and d13["di_ban_khac"] == "1")
    d14 = out[out["nom_idx"] == "14"].iloc[0]
    check("(d) một tham chiếu khớp → không hạ, không cờ", d14["tier"] == "GOLD" and d14["di_ban_khac"] == "" and R["14"] == "")
    check("(d) tham chiếu PUA bỏ qua", T["15"] == "GOLD" and R["15"] == "")
    check("(d) thắng (b)", T["16"] == "REVIEW" and R["16"] == mg.G_CROSS)
    check("'nan' là âm 難, còn nguyên", out[out["nom_idx"] == "17"].iloc[0]["syllable"] == "nan")
    check("n_det rỗng ≠ N → (a)", T["18"] == mg.TIER_TEXT_ONLY)
    raw = rep["gates_raw_hits"]
    check("báo cáo trúng thô: ndet 3, box 3, bridge 3, tone 1, crop GOLD 1/SYL 1, cross 2, khác 1",
          raw[mg.G_NDET] == 3 and raw[mg.G_BOX] == 3 and raw[mg.G_BRIDGE] == 3 and raw[mg.G_TONE] == 1
          and raw[mg.G_CROP]["GOLD"] == 1 and raw[mg.G_CROP]["SYLLABLE"] == 1 and raw[mg.G_CROSS] == 2
          and raw["cross_di_ban_khac"] == 1, str(raw))
    # GOLD 15 → còn 1,13,14,15,17 = 5; text_only 2,3,4,18 = 4; SYLLABLE 5,6,7 = 3; REVIEW 10 + 8,9,12,16 = 5
    check("tier_before/after đúng", rep["tier_before"]["GOLD"] == 15 and rep["tier_after"]["GOLD"] == 5
          and rep["tier_after"][mg.TIER_TEXT_ONLY] == 4 and rep["tier_after"]["SYLLABLE"] == 3
          and rep["tier_after"]["REVIEW"] == 5, str(rep["tier_after"]))
    check("coverage ảnh GOLD = 5/15", rep["gold_image"]["coverage_pct"] == round(100 * 5 / 15, 1))
    check("images_to_export = GOLD+SYLLABLE sau = 8", rep["images_to_export"] == 8)
    out2, rep2 = mg.apply_gates(out, _cross())
    check("idempotent: áp lần 2 không đổi gì", out2.equals(out) and rep2["decided_total"] == rep["decided_total"])
    out3, rep3 = mg.apply_gates(df, None)
    check("không --cross: (d) không chạy, ô 12 giữ GOLD, không khoá proxy",
          dict(zip(out3["nom_idx"], out3["tier"]))["12"] == "GOLD" and "proxy_agree_any_ref" not in rep3)
    old = df.drop(columns=["tier_v3", "box_source", "crop_quality_flag"])
    out4, _ = mg.apply_gates(old, None)
    check("thiếu cột thế hệ cũ: không lỗi, chỉ cổng có cột chạy",
          dict(zip(out4["nom_idx"], out4["tier"]))["3"] == "GOLD"
          and dict(zip(out4["nom_idx"], out4["tier"]))["2"] == mg.TIER_TEXT_ONLY)


def test_run_cli(tmp: Path) -> None:
    print("[3] CLI run: tắt = sao byte, bật = thêm cột")
    src = tmp / "labels_final.csv"
    _synthetic().to_csv(src, index=False)
    cfg = tmp / "c.yaml"
    cfg.write_text("books:\n  - name: SachThanhTruyen2\n    pdf: x.pdf\n  - name: LVT\n    layout: lithograph\n",
                   encoding="utf-8")
    with redirect_stdout(io.StringIO()):
        rep = mg.run(src, tmp / "off.csv", cfg, "SachThanhTruyen2", None, "auto", tmp / "off.json")
    check("STT: tắt", rep["enabled"] is False)
    check("STT: out byte-identical với in",
          hashlib.md5((tmp / "off.csv").read_bytes()).hexdigest() == hashlib.md5(src.read_bytes()).hexdigest())
    check("STT: báo cáo JSON ghi", (tmp / "off.json").exists())
    with redirect_stdout(io.StringIO()):
        rep = mg.run(src, tmp / "on.csv", cfg, "lvt", None, "auto", None)
    on = pd.read_csv(tmp / "on.csv", dtype=str, keep_default_na=False)
    check("LVT: bật (khớp tên không phân biệt hoa/thường)", rep["enabled"] is True)
    check("LVT: có cột gate_reason + di_ban_khac", "gate_reason" in on.columns and "di_ban_khac" in on.columns)
    check("LVT: báo cáo mặc định cạnh out", (tmp / "mechanism_gates_report.json").exists())
    check("LVT: 'nan' giữ nguyên qua CSV", (on["syllable"] == "nan").sum() == 1)
    rc = mg.main(["--in", str(src), "--out", str(tmp / "x.csv")])
    check("--enable auto thiếu --config/--book → exit 2", rc == 2)


def test_export(tmp: Path) -> None:
    print("[4] export tầng GOLD_text_only")
    from pipeline.export_final_dataset import export_dataset, TRACE, GIAO_NOP
    check("TRACE có gate_reason/di_ban_khac", "gate_reason" in TRACE and "di_ban_khac" in TRACE)
    check("GIAO_NOP vẫn 12 cột", len(GIAO_NOP) == 12)
    df = _synthetic()
    out, _ = mg.apply_gates(df, _cross())
    src_root = tmp / "src"
    for rel in out["image"]:
        if rel:
            (src_root / rel).parent.mkdir(parents=True, exist_ok=True)
            (src_root / rel).write_bytes(b"png")
    lab = src_root / "labels_gated.csv"
    out.to_csv(lab, index=False)
    dst = tmp / "dataset"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = export_dataset(lab, src_root, dst)
    rows = list(csv.DictReader(open(dst / "labels.csv", encoding="utf-8")))
    tiers = {r["tier"] for r in rows}
    check("export rc 0", rc == 0)
    check("labels.csv có GOLD_text_only, không REVIEW/QUARANTINE",
          mg.TIER_TEXT_ONLY in tiers and not ({"REVIEW", "QUARANTINE"} & tiers), str(tiers))
    to = [r for r in rows if r["tier"] == mg.TIER_TEXT_ONLY]
    check("GOLD_text_only 4 dòng, giữ label", len(to) == 4 and all(r["label"] for r in to))
    check("ảnh GOLD_text_only KHÔNG copy", not any((dst / r["image"]).exists() for r in to))
    check("ảnh GOLD/SYLLABLE có copy", all((dst / r["image"]).exists() for r in rows if r["tier"] in ("GOLD", "SYLLABLE")))
    n_img = sum(1 for p in dst.rglob("*.png"))
    check("số ảnh export = GOLD + SYLLABLE = 8", n_img == 8, str(n_img))
    tr = list(csv.DictReader(open(dst / "labels_trace.csv", encoding="utf-8")))
    check("labels_trace có gate_reason, cùng số dòng", "gate_reason" in tr[0] and len(tr) == len(rows))
    check("log nói rõ tầng text_only", "GOLD_text_only" in buf.getvalue())
    # KHÔNG có tầng mới → hành vi cũ: mọi dòng usable đều copy ảnh, log không nhắc tầng mới
    lab0 = src_root / "labels_final.csv"
    df.to_csv(lab0, index=False)
    dst0 = tmp / "dataset0"
    buf0 = io.StringIO()
    with redirect_stdout(buf0):
        export_dataset(lab0, src_root, dst0)
    rows0 = list(csv.DictReader(open(dst0 / "labels.csv", encoding="utf-8")))
    check("không tầng mới: copy = số dòng usable (16), log không nhắc GOLD_text_only",
          sum(1 for _ in dst0.rglob("*.png")) == len(rows0) == 16 and "GOLD_text_only" not in buf0.getvalue(),
          f"{sum(1 for _ in dst0.rglob('*.png'))} vs {len(rows0)}")
    # make_dataset_docs: khối tầng chỉ khi có
    from pipeline.tools import make_dataset_docs as md
    s1 = md.stats(dst / "labels.csv")
    s0 = md.stats(dst0 / "labels.csv")
    check("docs: README có khối GOLD_text_only khi có tầng", "GOLD_text_only" in md.readme(s1) and "GOLD_text_only" in md.datasheet(s1))
    check("docs: README KHÔNG có khối khi không tầng", "GOLD_text_only" not in md.readme(s0) and "GOLD_text_only" not in md.datasheet(s0))


def main() -> int:
    print("=" * 64)
    print("MECHANISM GATES SELFTEST")
    print("=" * 64)
    tmp = Path(tempfile.mkdtemp(prefix="mg_selftest_"))
    try:
        test_enabled()
        test_apply()
        test_run_cli(tmp)
        test_export(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
