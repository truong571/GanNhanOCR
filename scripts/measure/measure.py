#!/usr/bin/env python3
"""measure.py — runner duy nhất cho bộ đo scripts/measure/ (2026-09-21).

Gọi lần lượt các mô-đun đo (mỗi mô-đun là 1 script độc lập, chỉ phụ thuộc .venv + tesseract),
gom mọi summary.json thành measure_out/SUMMARY.json và sinh measure_out/REPORT.md.

    .venv/bin/python scripts/measure/measure.py --all                 # toàn bộ sách, toàn bộ trang
    .venv/bin/python scripts/measure/measure.py --book LucVanTien1883 # chỉ các phép đo của 1 sách
    .venv/bin/python scripts/measure/measure.py --all --limit 8       # chạy thử (invariants toàn sách = SKIP)
    .venv/bin/python scripts/measure/measure.py --all --steps layout,code_facts

Bố cục đầu ra (quy ước: mỗi phép đo một thư mục con, không đè summary của phép đo khác):
    measure_out/<book>/layout/            layout_lithograph.py   (LVT1883, KVK1884)
    measure_out/<book>/qn_ocr/            qn_print_ocr.py        (LVT1883, KVK1884)
    measure_out/Chrestomathie1872/chresto_map/   chresto_map.py
    measure_out/detector_transfer/        detector_transfer.py   (liên sách: LVT+KVK+STT đối chứng)
    measure_out/box_ref/                  box_ref_eval.py        (hộp legacy vs pitch_decode so ô tham chiếu, LVT+KVK)
    measure_out/code_facts/               code_facts.py → docs/PIPELINE_FACTS.json
    measure_out/SUMMARY.json, REPORT.md, logs/<step>.log

Mã thoát: 0 = mọi invariant cứng PASS; 1 = có invariant cứng FAIL; 2 = có bước chạy lỗi / thiếu summary.
Invariant "mềm" (SOFT_INVARIANTS) chỉ cảnh báo, không đổi mã thoát — mỗi mục có lý do ghi rõ.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
HERE = Path(__file__).resolve().parent

LITHO_BOOKS = ["LucVanTien1883", "KimVanKieu1884"]
# 2026-09-23: 3 bộ còn lại trong data/. IHR_BOOKS = mộc bản IHR-NomDB CÓ NHÃN NGƯỜI (tập ĐÁNH GIÁ);
# PTCL_BOOK = bản CHÉP TAY R.987, chỉ có dị bản 1871/1872 làm tham chiếu.
IHR_BOOKS = ["LucVanTien1916", "TruyenKieu1872"]
# 2026-09-23 (vòng 7): PTCL (bản chép tay R.987) ĐÃ LOẠI KHỎI PHẠM VI — thư mục data/ của nó
# không còn trên đĩa và phép đo bố cục (docs/CHAY_3_BO_CON_LAI_2026-09-23.md §3) cho thấy adapter
# hiện có không dùng được: mỗi cột 2 tầng mà tầng trên là LỜI BÌNH CHỮ HÁN, không có số câu neo,
# QN chỉ có dị bản 1871/1872. Vì vậy PTCL KHÔNG còn trong ALL_BOOKS/ALL_STEPS; `ptcl_layout.py`
# giữ lại làm hồ sơ lịch sử và vẫn chạy được bằng tay nếu có ai đặt lại thư mục data/.
PTCL_BOOK = "TruyenKieuPhongTinhCoLuc"
ALL_BOOKS = LITHO_BOOKS + ["Chrestomathie1872"] + IHR_BOOKS
ALL_STEPS = ["code_facts", "layout", "qn_ocr", "chresto_map", "detector_transfer", "box_ref",
             "ihr_layout", "ihr_endtoend"]

# Invariant FAIL được xếp "mềm" (không đổi mã thoát) — chỉ khi có lý do đo đạc rõ ràng.
SOFT_INVARIANTS: dict[str, dict[str, str]] = {
    "ihr_layout": {
        "chieu_muc_doc_lap_khop_bbox_+-1":
            "Đối chứng chiếu mực là phép ĐỘC LẬP yếu trên bản quét 37-40 px/chữ: tự tương quan hay "
            "khoá vào hài nên đếm lệch (LVT1916 0/20 trang khớp ±1, TK1872 17/20). Bố cục CHÍNH THỨC "
            "lấy từ ô cột do người vẽ (VoTT) và đã được kiểm chéo bằng SỐ CÂU (ceil(n_verses/2)) — "
            "invariant cot_bbox_khop_so_cau PASS ở cả hai sách. Ghi lại để không tin nhầm phép đo này.",
        "chieu_muc_buoc_cot_khop_bbox":
            "Cùng lý do; bước cột đo được đúng ở TK1872 (20/20) nhưng lệch ở LVT1916 (9/20).",
    },
    "detector_transfer": {
        "stt_control_pct_cols_eq_pipeline_cfg":
            "Đối chứng STT dùng MỌI cột (kể cả cột chữ dày dy≈63 px M>N); ngưỡng 90 % là của cột OCR=QN "
            "trong pipeline. Kết quả 70–89 % trên 27–81 cột là số đo thật, không phải lỗi mô-đun.",
        "stt_control_pct_cols_eq_N_verified":
            "Cùng lý do; n cột có N kim == chiếu mực rất nhỏ (16–50) nên 1 cột lệch đã đổi vài pp.",
    },
}

# Chỉ số then chốt in trong REPORT.md: (nhãn, đường dẫn chấm trong summary.json)
KEY_METRICS: dict[str, list[tuple[str, str]]] = {
    "layout": [
        ("trang văn bản", "category_counts.text"),
        ("cột/tầng (hist t0)", "cols_per_tier.tier0"),
        ("cặp lục bát", "pairs.total"),
        ("cặp kỳ vọng", "pairs.expected_full_book"),
        ("câu từ bố cục", "pairs.verses_from_layout"),
        ("6/8 khớp", "chars_per_col.rate"),
        ("số câu đọc đúng (final)", "numbers.final_ok"),
        ("số câu kỳ vọng", "numbers.expected"),
        ("tess↔kNN đồng ý", "numbers.agreement"),
        ("kNN CV acc", "numbers.digit_knn.knn3_acc"),
        ("A==B số cột", "cols_method_agreement"),
        ("Viterbi lệch", "verse_chain.n_mismatch"),
        ("pitch cột px", "col_pitch_px.median"),
    ],
    "qn_ocr": [
        ("trang QN", "pages.qn"),
        ("dòng thơ", "verses.n"),
        ("dòng kỳ vọng", "verses.expected"),
        ("chuỗi tổng", "verses.chain_total"),
        ("parity 6/8", "verses.parity_rate"),
        ("neo n", "anchors.n"),
        ("neo đúng giá trị", "anchors.value_ok_rate"),
        ("neo đúng vị trí", "anchors.position_ok_rate"),
        ("2 phương pháp đồng ý", "margin_digits.agreement.rate"),
        ("trang trôi", "flags.drift_pages"),
        ("trang cần xem", "flags.review_list"),
        ("CER dòng đọc mắt", "visual_truth.cer"),
        ("số dòng đọc mắt", "visual_truth.lines"),
    ],
    "chresto_map": [
        ("truyện QN", "map.n_stories"),
        ("dòng QN thân", "qn.n_body_lines"),
        ("âm tiết QN", "qn.n_syll"),
        ("OOV", "qn.oov_rate"),
        ("trang Nôm", "nom.n_pages"),
        ("cột Nôm", "nom.n_cols"),
        ("chữ/cột đầy", "nom.chars_per_full_col_med_pitch"),
        ("pitch↔runs đồng ý", "nom.agreement_pitch_vs_runs_le2"),
        ("ước chữ Nôm", "nom.total_chars_est"),
        ("ranh giới tự động TP", "map.boundary_auto.tp"),
        ("NomNaOCR", "nomna.status"),
    ],
    "detector_transfer": [
        ("trang đo", "pages.n"),
        ("cấu hình tốt nhất thạch bản", "best.litho.config"),
        ("% cột đúng (thạch bản)", "best.litho.pct_cols_M_eq_N"),
        ("LVT raw/stretch/otsu @0.2", "per_book.LucVanTien1883.table"),
        ("KVK raw/stretch/otsu @0.2", "per_book.KimVanKieu1884.table"),
        ("STT2 pipeline_cfg", "per_book.SachThanhTruyen2.pipeline_cfg.eq_excl"),
        ("STT4 pipeline_cfg", "per_book.SachThanhTruyen4.pipeline_cfg.eq_excl"),
        ("STT11 pipeline_cfg", "per_book.SachThanhTruyen11.pipeline_cfg.eq_excl"),
        ("chiếu vs N (LVT/KVK)", "agreement.nchars_proj_vs_N"),
        ("thời gian s", "runtime_s"),
    ],
    "box_ref": [
        ("trang đo", "pages.n"),
        ("LVT I5 n_det==N @0.15 (27 trang)", "per_book.LucVanTien1883.columns.prepared@0.15.I5_n_det_eq_N_pct"),
        ("LVT ô ok IoU≥0,5: legacy@0.15", "per_book.LucVanTien1883.cells_verified.legacy@0.15_prepared.ok_iou50_pct"),
        ("LVT ô ok IoU≥0,5: pitch", "per_book.LucVanTien1883.cells_verified.pitch_prepared.ok_iou50_pct"),
        ("LVT cắt vào thân chữ: legacy / pitch", "per_book.LucVanTien1883.cells_verified.legacy@0.15_prepared.cut_glyph_pct"),
        ("KVK I5 n_det==N @0.15 (27 trang)", "per_book.KimVanKieu1884.columns.prepared@0.15.I5_n_det_eq_N_pct"),
        ("KVK ô ok IoU≥0,5: legacy@0.15", "per_book.KimVanKieu1884.cells_verified.legacy@0.15_prepared.ok_iou50_pct"),
        ("KVK ô ok IoU≥0,5: pitch", "per_book.KimVanKieu1884.cells_verified.pitch_prepared.ok_iou50_pct"),
        ("thời gian s", "runtime_s"),
    ],
    "ihr_layout": [
        ("trang", "pages.n"),
        ("trang có ô cột", "pages.with_bbox"),
        ("cột/trang", "columns.hist_n_cols_bbox"),
        ("cột = 1 cặp lục bát", "columns.rate_col_is_couplet"),
        ("câu đúng 6/8", "columns.rate_verse_len_rule"),
        ("cột khớp số câu", "columns.rate_bbox_eq_expected"),
        ("px/chữ", "geometry.px_per_char_med"),
        ("bước cột px", "geometry.col_pitch_med"),
        ("kim ×1 đúng GT", "kim_scale.per_scale.1.rate_eq_gt"),
        ("kim ×3 đúng GT", "kim_scale.per_scale.3.rate_eq_gt"),
        ("kim ×1 cột đủ 14", "kim_scale.per_scale.1.rate_cols_full14"),
        ("lượt kim mới", "kim_scale.api_calls"),
    ],
    "ihr_endtoend": [
        ("ô sinh", "cells.produced"),
        ("ô GT", "gt.n_chars"),
        ("coverage ô", "cells.coverage_cells"),
        ("GOLD ảnh n", "precision.gold_anh.with_gt"),
        ("GOLD ảnh ĐÚNG", "precision.gold_anh.precision"),
        ("GOLD ảnh CI95", "precision.gold_anh.ci95"),
        ("GOLD (kể text_only) ĐÚNG", "precision.gold_tat_ca.precision"),
        ("GOLD coverage", "coverage.gold_tat_ca"),
        ("kim thô đúng GT", "kim_raw.precision"),
        ("nhãn == kim ở ô GOLD", "kim_raw.nhan_bang_kim_tren_GOLD"),
        ("REVIEW ĐÚNG", "precision.review.precision"),
        ("mốc patch 22/09", "baseline_patch_2026_09_22.precision"),
        ("chênh so mốc", "delta_vs_baseline.precision"),
    ],
    "ptcl_layout": [
        ("tờ phân tích", "pages.analyzed"),
        ("cột/tờ", "columns.hist_n_cols"),
        ("bước cột px", "columns.col_pitch_med"),
        ("ranh giới tầng y", "columns.tier_y_med"),
        ("bước chữ tầng dưới px", "columns.char_pitch_bottom_med"),
        ("chữ/cột tầng dưới (chiếu mực)", "columns.chars_bottom_med"),
        ("kim: chữ tầng dưới trung vị", "kim.bottom_per_col_med"),
        ("kim: cột đúng 14", "kim.rate_bottom_14"),
        ("nền dị bản 1871↔1872", "reference.rate"),
        ("lượt kim mới", "kim.api_calls"),
    ],
    "code_facts": [
        ("git HEAD", "git_head"),
        ("tệp FACTS", "facts_file"),
        ("kích thước FACTS B", "facts_bytes"),
        ("ghim '9' (tất cả / prod)", "counts.column_pins"),
        ("ghim '9' prod", "counts.column_pins_prod"),
        ("ghim đã biết tìm thấy", "counts.known_pins_found"),
        ("bước CLI / tham số", "counts.cli_steps"),
        ("khoá JSON cache / không đọc", "counts.json_keys"),
        ("sách trong config", "config_books"),
        ("checkpoint thiếu", "checkpoints_missing"),
        ("mã pipeline/core chưa commit", "git.dirty_pipeline_core"),
        ("dataset_out bị xoá (tracked)", "git.dataset_out.n_deleted_in_worktree"),
    ],
}


# ----------------------------------------------------------------------------- tiện ích
def dig(d, path: str):
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1000 else f"{v:.0f}"
    if isinstance(v, (list, tuple)):
        if len(v) > 4 and any(isinstance(x, dict) or (isinstance(x, str) and len(x) > 40) for x in v):
            return f"[{len(v)} mục]"
        s = ", ".join(fmt(x) for x in v[:4])
        return f"[{s}{', …' if len(v) > 4 else ''}]" if not isinstance(v, str) else v
    if isinstance(v, dict):
        items = list(v.items())[:6]
        return "{" + ", ".join(f"{k}: {fmt(x)}" for k, x in items) + ("…" if len(v) > 6 else "") + "}"
    s = str(v).replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= 90 else s[:87] + "…"


def rel(p: Path | str) -> str:
    s = str(p)
    pre = str(REPO) + os.sep
    return s[len(pre):] if s.startswith(pre) else s


# ----------------------------------------------------------------------------- các bước
def plan_steps(books: list[str], steps: list[str], a) -> list[dict]:
    """Trả về danh sách bước: {step, book, cli(list), out_dir, summary}."""
    root = Path(a.out)
    plan = []
    w = ["--workers", str(a.workers)]
    lim = ["--limit", str(a.limit)] if a.limit else []
    if "code_facts" in steps:
        plan.append(dict(step="code_facts", book="(repo)", out=root / "code_facts",
                         cli=[PY, str(HERE / "code_facts.py"), "--summary", str(root / "code_facts" / "summary.json")],
                         summary=root / "code_facts" / "summary.json"))
    for b in books:
        if b in LITHO_BOOKS:
            if "layout" in steps:
                out = root / b / "layout"
                cli = [PY, str(HERE / "layout_lithograph.py"), "--book", b, "--out", str(out), *w, *lim]
                if a.layout_detector:
                    cli.append("--detector")
                plan.append(dict(step="layout", book=b, out=out, cli=cli, summary=out / "summary.json"))
            if "qn_ocr" in steps:
                out = root / b / "qn_ocr"
                cli = [PY, str(HERE / "qn_print_ocr.py"), "--book", b, "--out", str(out), *w, *lim,
                       "--engine", a.qn_engine]
                if a.limit and b == "KimVanKieu1884":
                    cli += ["--start", "26"]  # canvas 1..26 là bìa/tựa Pháp: chạy thử từ trang có QN
                plan.append(dict(step="qn_ocr", book=b, out=out, cli=cli, summary=out / "summary.json"))
        elif b == "Chrestomathie1872" and "chresto_map" in steps:
            out = root / b / "chresto_map"
            cli = [PY, str(HERE / "chresto_map.py"), "--book", b, "--out", str(out), *w, *lim,
                   "--nomna-pages", str(a.nomna_pages)]
            plan.append(dict(step="chresto_map", book=b, out=out, cli=cli, summary=out / "summary.json"))
        elif b in IHR_BOOKS:
            if "ihr_layout" in steps:
                out = root / b / "ihr_layout"
                cli = [PY, str(HERE / "ihr_layout.py"), "--book", b, "--out", str(out), *lim,
                       "--kim-pages", str(a.ihr_kim_pages)]
                plan.append(dict(step="ihr_layout", book=b, out=out, cli=cli, summary=out / "summary.json"))
            if "ihr_endtoend" in steps and (REPO / "prepared_ihr" / b / "dataset_out" / "labels_gated.csv").exists():
                out = root / b / "ihr_endtoend"
                cli = [PY, str(HERE / "ihr_endtoend_eval.py"), "--book", b, "--out", str(out)]
                plan.append(dict(step="ihr_endtoend", book=b, out=out, cli=cli, summary=out / "summary.json"))
        elif b == PTCL_BOOK and "ptcl_layout" in steps and (REPO / "data" / PTCL_BOOK).is_dir():
            out = root / b / "ptcl_layout"
            cli = [PY, str(HERE / "ptcl_layout.py"), "--out", str(out), *w, *lim,
                   "--kim-pages", str(a.ptcl_kim_pages)]
            plan.append(dict(step="ptcl_layout", book=b, out=out, cli=cli, summary=out / "summary.json"))
    if "detector_transfer" in steps and any(b in LITHO_BOOKS for b in books):
        out = root / "detector_transfer"
        cli = [PY, str(HERE / "detector_transfer.py"), "--book", "all", "--out", str(out), *w, *lim,
               "--pages", str(a.det_pages), "--stt-pages", str(a.stt_pages)]
        plan.append(dict(step="detector_transfer", book="LVT+KVK+STT", out=out, cli=cli,
                         summary=out / "summary.json"))
    if "box_ref" in steps and any(b in LITHO_BOOKS for b in books):
        # (2026-09-22) hộp detector legacy vs pitch_decode so với ô tham chiếu tự động (kim + chiếu mực)
        out = root / "box_ref"
        cli = [PY, str(HERE / "box_ref_eval.py"), "--book", "all", "--out", str(out), *w, *lim,
               "--pages", str(a.det_pages)]
        plan.append(dict(step="box_ref", book="LVT+KVK", out=out, cli=cli, summary=out / "summary.json"))
    return plan


def run_step(st: dict, log_dir: Path, dry: bool) -> dict:
    log = log_dir / f"{st['step']}_{st['book'].replace('+', '_').strip('()')}.log"
    st["log"] = log
    if dry:
        st.update(rc=None, seconds=0.0)
        return st
    Path(st["out"]).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", TF_ENABLE_ONEDNN_OPTS="0", PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(st["cli"]) + "\n\n")
        fh.flush()
        rc = subprocess.call(st["cli"], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, env=env)
    st.update(rc=rc, seconds=round(time.time() - t0, 1))
    return st


def collect(st: dict) -> dict:
    """Đọc summary.json của bước, phân loại invariants (cứng/mềm/skip)."""
    rec = dict(step=st["step"], book=st["book"], cli=" ".join(rel(c) if c.startswith(str(REPO)) else c for c in st["cli"]),
               rc=st["rc"], seconds=st["seconds"], out_dir=rel(st["out"]), log=rel(st["log"]),
               summary=rel(st["summary"]), summary_found=Path(st["summary"]).exists())
    inv_rows, key = [], {}
    if rec["summary_found"]:
        try:
            s = json.loads(Path(st["summary"]).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            s = {}
            rec["summary_error"] = str(e)
        soft = SOFT_INVARIANTS.get(st["step"], {})
        partial = bool(s.get("limit")) or bool(s.get("partial_run")) or bool(st.get("limit"))
        for iv in s.get("invariants", []) or []:
            if not isinstance(iv, dict):
                continue
            p = iv.get("pass")
            status = "SKIP" if p is None else ("PASS" if p else ("SOFT_FAIL" if iv.get("name") in soft else "FAIL"))
            reason = soft.get(iv.get("name"), "")
            if status == "FAIL" and partial:
                status, reason = "SOFT_FAIL", "chạy thử --limit: mẫu nhỏ, invariant chỉ có nghĩa khi chạy đủ sách"
            inv_rows.append(dict(name=iv.get("name"), expected=iv.get("expected"), observed=iv.get("observed"),
                                 status=status, method=iv.get("method"),
                                 **({"soft_reason": reason} if status == "SOFT_FAIL" else {})))
        for label, path in KEY_METRICS.get(st["step"], []):
            key[label] = dig(s, path)
        rec["partial"] = partial
    rec["invariants"] = inv_rows
    rec["key_metrics"] = key
    rec["n_pass"] = sum(r["status"] == "PASS" for r in inv_rows)
    rec["n_fail"] = sum(r["status"] == "FAIL" for r in inv_rows)
    rec["n_soft_fail"] = sum(r["status"] == "SOFT_FAIL" for r in inv_rows)
    rec["n_skip"] = sum(r["status"] == "SKIP" for r in inv_rows)
    # CSV/đầu ra chi tiết (không đệ quy vào cache/debug)
    outs = []
    od = Path(st["out"])
    if od.exists():
        for p in sorted(od.iterdir()):
            if p.is_file() and p.suffix.lower() in (".csv", ".tsv", ".json") and p.name != "summary.json":
                outs.append(rel(p))
    rec["outputs"] = outs
    rec["ok"] = rec["rc"] == 0 and rec["summary_found"] and rec["n_fail"] == 0
    return rec


# ----------------------------------------------------------------------------- báo cáo
def write_report(recs: list[dict], root: Path, args_text: str, total_s: float, exit_code: int) -> Path:
    L = []
    L.append(f"# Báo cáo đo đạc tự động — measure.py\n")
    L.append(f"Sinh lúc {datetime.now():%Y-%m-%d %H:%M:%S} · lệnh `{args_text}` · tổng {total_s:.0f} s · mã thoát {exit_code}\n")
    L.append("> Quy ước: số liệu trong đây đến từ script tái lập được. Agent/phiên sau **không chạy lại phép đo bằng LLM**; "
             "chỉ đọc SUMMARY.json/REPORT.md, nếu nghi ngờ thì đổi tham số script hoặc thêm invariant.\n")
    L.append("## 1. Các bước\n")
    L.append("| # | phép đo | sách | rc | thời gian | invariants PASS / FAIL / mềm / SKIP | summary |")
    L.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(recs, 1):
        st = "✅" if r["ok"] else ("⚠️" if r["rc"] == 0 and r["summary_found"] else "❌")
        L.append(f"| {i} | {st} {r['step']} | {r['book']} | {fmt(r['rc'])} | {r['seconds']:.0f} s | "
                 f"{r['n_pass']} / {r['n_fail']} / {r['n_soft_fail']} / {r['n_skip']} | `{r['summary']}` |")
    L.append("")
    fails = [(r, iv) for r in recs for iv in r["invariants"] if iv["status"] in ("FAIL", "SOFT_FAIL")]
    crashed = [r for r in recs if r["rc"] != 0 or not r["summary_found"]]
    L.append("## 2. Invariants FAIL\n")
    if not fails and not crashed:
        L.append("Không có invariant nào FAIL.\n")
    for r in crashed:
        L.append(f"- ❌ **{r['step']} / {r['book']}**: rc={r['rc']}, summary {'có' if r['summary_found'] else 'THIẾU'} — xem `{r['log']}`")
    if fails:
        L.append("| phép đo | sách | invariant | kỳ vọng | quan sát | loại | lý do (mềm) |")
        L.append("|---|---|---|---|---|---|---|")
        for r, iv in fails:
            L.append(f"| {r['step']} | {r['book']} | {iv['name']} | {fmt(iv['expected'])} | {fmt(iv['observed'])} | "
                     f"{'FAIL' if iv['status'] == 'FAIL' else 'mềm'} | {fmt(iv.get('soft_reason', ''))} |")
    L.append("")
    L.append("## 3. Chỉ số then chốt theo phép đo\n")
    for r in recs:
        L.append(f"### {r['step']} — {r['book']}\n")
        L.append(f"`{r['cli']}`\n")
        if r["key_metrics"]:
            L.append("| chỉ số | giá trị |")
            L.append("|---|---|")
            for k, v in r["key_metrics"].items():
                L.append(f"| {k} | {fmt(v)} |")
            L.append("")
        L.append("<details><summary>Invariants (" + f"{len(r['invariants'])}" + ")</summary>\n")
        L.append("| trạng thái | tên | kỳ vọng | quan sát |")
        L.append("|---|---|---|---|")
        for iv in r["invariants"]:
            L.append(f"| {iv['status']} | {iv['name']} | {fmt(iv['expected'])} | {fmt(iv['observed'])} |")
        L.append("\n</details>\n")
        if r["outputs"]:
            L.append("Tệp chi tiết: " + ", ".join(f"`{p}`" for p in r["outputs"]) + f" · log `{r['log']}`\n")
    L.append("## 4. Bố cục đầu ra\n")
    L.append("```\nmeasure_out/<book>/layout/ · measure_out/<book>/qn_ocr/ · measure_out/Chrestomathie1872/chresto_map/\n"
             "measure_out/detector_transfer/ · measure_out/code_facts/ (+ docs/PIPELINE_FACTS.json)\n"
             "measure_out/SUMMARY.json · measure_out/REPORT.md · measure_out/logs/*.log · measure_out/_cache/ (OCR cache theo md5 ảnh)\n```\n")
    p = root / "REPORT.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", action="append", default=[], help="sách (lặp lại được, hoặc 'a,b'); mặc định --all nếu không có")
    ap.add_argument("--all", action="store_true", help="mọi sách + detector_transfer + code_facts")
    ap.add_argument("--steps", default=",".join(ALL_STEPS), help=f"tập con của {ALL_STEPS}")
    ap.add_argument("--out", default=str(REPO / "measure_out"), help="thư mục gốc đầu ra")
    ap.add_argument("--limit", type=int, default=0, help="chạy thử N trang mỗi phép đo (invariants toàn sách = SKIP)")
    ap.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    ap.add_argument("--qn-engine", default="tesseract", choices=["tesseract", "vietocr"])
    ap.add_argument("--nomna-pages", type=int, default=3, help="chresto_map: số trang Nôm chạy NomNaOCR (0 = tắt)")
    ap.add_argument("--det-pages", type=int, default=27, help="detector_transfer: trang thạch bản mỗi sách (27 → CI ≈ ±3,5 điểm; 9 trang ≈ ±6)")
    ap.add_argument("--stt-pages", type=int, default=3, help="detector_transfer: trang đối chứng mỗi sách STT")
    ap.add_argument("--ihr-kim-pages", type=int, default=10,
                    help="ihr_layout: số trang gọi kim mỗi hệ số phóng (cache theo md5 -> chạy lại 0 lượt)")
    ap.add_argument("--ptcl-kim-pages", type=int, default=6,
                    help="ptcl_layout: số tờ gọi kim (cache theo md5 -> chạy lại 0 lượt)")
    ap.add_argument("--layout-detector", action="store_true", help="layout: bật CenterNet làm phương pháp 2 (chậm)")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in kế hoạch")
    ap.add_argument("--report-only", action="store_true", help="không chạy, chỉ gom summary hiện có → SUMMARY/REPORT")
    a = ap.parse_args(argv)

    books = [b for grp in a.book for b in grp.split(",") if b]
    if a.all or not books:
        books = ALL_BOOKS
    bad = [b for b in books if b not in ALL_BOOKS]
    if bad:
        print(f"sách không biết: {bad}; chọn trong {ALL_BOOKS}")
        return 2
    steps = [s for s in a.steps.split(",") if s]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        print(f"bước không biết: {bad}; chọn trong {ALL_STEPS}")
        return 2
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    plan = plan_steps(books, steps, a)
    for st in plan:
        st["limit"] = a.limit
    print(f"measure.py: {len(plan)} bước, sách {books}, out {rel(root)}" + (f" · CHẠY THỬ --limit {a.limit}" if a.limit else ""))
    prev = {}
    if a.report_only and (root / "SUMMARY.json").exists():
        try:  # giữ lại rc/thời gian của lần chạy trước cho các bước không chạy lại
            prev = {(r["step"], r["book"]): r for r in json.loads((root / "SUMMARY.json").read_text(encoding="utf-8")).get("results", [])}
        except Exception:  # noqa: BLE001
            prev = {}
    t_all = time.time()
    recs = []
    for i, st in enumerate(plan, 1):
        if a.report_only:
            pr = prev.get((st["step"], st["book"]), {})
            st.update(rc=pr.get("rc", 0), seconds=float(pr.get("seconds", 0.0)),
                      log=log_dir / f"{st['step']}_{st['book'].replace('+', '_').strip('()')}.log")
        else:
            run_step(st, log_dir, a.dry_run)
        if a.dry_run:
            print(f"  [{i}/{len(plan)}] {st['step']:18s} {st['book']:14s} $ {' '.join(rel(c) if c.startswith(str(REPO)) else c for c in st['cli'])}")
            continue
        r = collect(st)
        recs.append(r)
        mark = "OK  " if r["ok"] else ("WARN" if r["rc"] == 0 and r["summary_found"] else "ERR ")
        print(f"  [{i}/{len(plan)}] {mark} {r['step']:18s} {r['book']:14s} {r['seconds']:6.0f}s  "
              f"inv PASS {r['n_pass']} FAIL {r['n_fail']} mềm {r['n_soft_fail']} SKIP {r['n_skip']}")
    if a.dry_run:
        return 0
    total_s = time.time() - t_all
    hard_fail = sum(r["n_fail"] for r in recs)
    crashed = [r for r in recs if r["rc"] != 0 or not r["summary_found"]]
    exit_code = 2 if crashed else (1 if hard_fail else 0)
    args_text = "measure.py " + " ".join(sys.argv[1:])
    summary = dict(
        generated_at=datetime.now().isoformat(timespec="seconds"), cli=args_text, python=PY,
        total_seconds=round(total_s, 1), exit_code=exit_code, books=books, steps=steps, limit=a.limit or None,
        totals=dict(steps=len(recs), crashed=len(crashed), inv_pass=sum(r["n_pass"] for r in recs),
                    inv_fail=hard_fail, inv_soft_fail=sum(r["n_soft_fail"] for r in recs),
                    inv_skip=sum(r["n_skip"] for r in recs)),
        soft_invariants=SOFT_INVARIANTS, results=recs,
    )
    (root / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    rp = write_report(recs, root, args_text, total_s, exit_code)
    print(f"tổng {total_s:.0f} s · invariants PASS {summary['totals']['inv_pass']} FAIL {hard_fail} "
          f"mềm {summary['totals']['inv_soft_fail']} SKIP {summary['totals']['inv_skip']} · lỗi bước {len(crashed)}")
    for r in recs:
        for iv in r["invariants"]:
            if iv["status"] == "FAIL":
                print(f"  FAIL {r['step']}/{r['book']}: {iv['name']} expected={fmt(iv['expected'])} observed={fmt(iv['observed'])}")
    for r in crashed:
        print(f"  ERR  {r['step']}/{r['book']}: rc={r['rc']} → {r['log']}")
    print(f"SUMMARY: {rel(root / 'SUMMARY.json')} · REPORT: {rel(rp)} · mã thoát {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
