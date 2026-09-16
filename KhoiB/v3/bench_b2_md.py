"""Sinh KhoiB/v3/BENCH_B2.md từ bench_b2_results.json (valtest) + bench_b2_results_all.json (all) — K5."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
R = {"val+test": json.load(open(D / "bench_b2_results.json")), "all": json.load(open(D / "bench_b2_results_all.json"))}
CFG = ["văn bản thuần (CALIB)", "+corpus (neo LOO cap 2,0)", "+corpus+ảnh λ=0.25", "+corpus+ảnh λ=0.5", "+corpus+ảnh λ=1.0", "+ảnh λ=0.25 không corpus"]
KIND_VI = {"none": "none (m=n, không nhiễu)", "drop_char": "drop_char (OCR rụng 1 chữ)", "drop_syl": "drop_syl (QN rụng 1 âm)", "drop2_char": "drop2_char (OCR rụng 2 chữ liền)"}
L = ["# BENCHMARK GÁC B-2 — phát xạ ảnh trong DP, mô hình OOF thật (K5, 16/09/2026)", "",
     "Script `KhoiB/v3/bench_b2.py` (đường mã sản xuất: `pipeline/align_engine/visual_emission.py` VisualEmitter + "
     "`build_dataset.anchored_cost_fn` + `anchor_align.realign_column(cost_ij=…)`), seed 2026, `thuc_nghiem.perturb` như "
     "`lab/tham_dinh_2026-09-16/dp_vis5.py`; mô hình 5 fold Kaggle `KhoiB/v3/p_visual_oof_v3_results/models/fold{k}.pt`, fold của TRANG "
     "(md5(\"book|page\")%5) → mọi cột đều out-of-fold. Kết quả: `bench_b2_results.json` (val+test), `bench_b2_results_all.json` (all); log `bench_b2_*.log`.", "",
     "Đáp án = đường chéo của cột m = n toàn match (cột đã ghép đúng số). Ghép sai = cặp (i, j) khác đáp án / tổng cặp; "
     "khe đúng = cột có đúng khe ins/del ở đúng chỗ (none: không khe). Thời gian = DP của cả tập cho 1 cấu hình (MPS; phát xạ ảnh tính 1 lần cho mọi cấu hình).", ""]
for name, r in R.items():
    L += [f"## Tập `{r['set']}` — {r['n_cols']:,} cột m=n≥{10} · hộp OCR thô top-1 OOF {r['top1_ocr_raw_box']['hit']:,}/{r['top1_ocr_raw_box']['n']:,} = "
          f"{r['top1_ocr_raw_box']['hit']/r['top1_ocr_raw_box']['n']:.3f} · cột theo fold {r['fold_cols']} · phát xạ {r.get('emission_sec', '?')}s", "",
          "| Kịch bản | Cấu hình | ghép sai | % | khe đúng | % | DP (s) |", "|---|---|---|---|---|---|---|"]
    for k in ["none", "drop_char", "drop_syl", "drop2_char"]:
        for c in CFG:
            x = r["by_kind"][k][c]
            b = "**" if c == "+corpus+ảnh λ=0.25" else ""
            L.append(f"| {KIND_VI[k] if c == CFG[0] else ''} | {c} | {b}{x['wrong']}/{x['n_pairs']:,}{b} | {b}{x['wrong_rate']:.2%}{b} | {b}{x['gap_ok']}/{x['n_cols']}{b} | {b}{x['gap_ok_rate']:.1%}{b} | {x.get('sec', '')} |")
    a = r["acceptance"]
    L += ["", f"Điều kiện chấp nhận (`{a['gate_config']}` so `{a['baseline']}`): none không tệ hơn = **{a['none_not_worse']}** "
          f"(sai {a['none']['base'][0]}→{a['none']['gate'][0]}, khe {a['none']['base'][1]}→{a['none']['gate'][1]}/{r['n_cols']}); drop_char tốt hơn = **{a['drop_char_better']}** "
          f"(sai {a['drop_char']['base'][0]}→{a['drop_char']['gate'][0]}, khe {a['drop_char']['base'][1]}→{a['drop_char']['gate'][1]}); "
          f"drop2_char tốt hơn = {a['drop2_char']['better']} (sai {a['drop2_char']['base'][0]}→{a['drop2_char']['gate'][0]}, khe {a['drop2_char']['base'][1]}→{a['drop2_char']['gate'][1]}) → **{'PASS' if a['PASS'] else 'FAIL'}**.", ""]
    dv = r["none_deviations"].get(a["gate_config"], [])
    if dv:
        L += [f"Cột lệch ở `none` với λ=0,25 ({len(dv)}):", ""]
        for d in dv:
            L.append(f"- `{'/'.join(map(str, d['col']))}` (m={d['m']}): del {d['dels']}, ins {d['inss']}, cặp lệch {d['pairs_off']}. "
                     "Dòng QN có token dính `mìnhép` (2 âm 1 token) nên cột thật có 21 âm/20 hộp — đáp án đường chéo SAI ở 4 cặp "
                     "(押↔thịt, 烟↔một, 弟↔ngày, 将↔một). Ảnh đọc từng hộp: " +
                     ", ".join(f"{d['chars'][i]}→{d['argmax'][i]} {d['p_argmax'][i]}" for i in range(2, 9)) +
                     ". Đường ghép +ảnh đúng 3 cặp mà đáp án sai (烟↔thịt, 弟↔một, 将↔ngày) và đặt khe ins[7] đúng chỗ thiếu hộp; sai 命↔nhặm và del 琰. "
                     "Trong bộ giao nộp 7 ô này (nom_idx 1–7) đều REVIEW/no_context (`dataset_out/labels_final.csv` dòng tệp 9113–9119 (chỉ số pandas 9111–9117)) — không ảnh hưởng ô usable.")
        L.append("")
# đường cong λ
L += ["## Đường cong λ (drop_char, ghép sai %) — λ ≥ 1 hại", "", "| tập | +corpus (λ=0) | λ=0,25 | λ=0,5 | λ=1,0 |", "|---|---|---|---|---|"]
for name, r in R.items():
    d = r["by_kind"]["drop_char"]
    L.append(f"| {name} | {d[CFG[1]]['wrong_rate']:.2%} / khe {d[CFG[1]]['gap_ok_rate']:.1%} | **{d[CFG[2]]['wrong_rate']:.2%} / {d[CFG[2]]['gap_ok_rate']:.1%}** | {d[CFG[3]]['wrong_rate']:.2%} / {d[CFG[3]]['gap_ok_rate']:.1%} | {d[CFG[4]]['wrong_rate']:.2%} / {d[CFG[4]]['gap_ok_rate']:.1%} |")
    n = r["by_kind"]["none"]
    L.append(f"| {name} (none) | {n[CFG[1]]['wrong']} | {n[CFG[2]]['wrong']} | {n[CFG[3]]['wrong']} | {n[CFG[4]]['wrong']} |")
L += ["", "## Kết luận", "",
      "1. λ=0,25 (config `step2.visual_emission.lambda`) là điểm tốt nhất trên cả 4 kịch bản × 2 tập; λ=0,5 vẫn tốt hơn văn bản, λ=1,0 hại "
      "(none sai 0,50–0,73%, drop_char 1,6–2,2% > +corpus). Đúng như `dp_vis5.py` trên mô hình cũ (khe 86,6→95,4%).",
      "2. Trên val+test (525 cột — ít hơn mốc 600 của nhiệm vụ; tập `all` 2.547 cột đủ): **PASS** trọn vẹn. Trên `all`: FAIL đúng 1 cột vì đáp án đường chéo sai "
      "(token QN dính) — kênh ảnh đúng hơn đáp án; drop_char/drop_syl/drop2_char đều tốt hơn rõ (ghép sai ÷3, khe đúng +7–10 điểm).",
      "3. Ảnh KHÔNG thay được ngữ liệu: `+ảnh λ=0,25 không corpus` ≈ `+corpus` (0,66–0,67% drop_char); hai kênh bổ sung nhau (ảnh+corpus 0,22–0,26%).",
      "4. Chi phí: phát xạ ảnh 24 s / 2.547 cột (53.658 hộp, MPS) ≈ 0,45 ms/hộp; DP thêm ≈ 0,1 s / 2.547 cột. Trên build 448 trang: xem `docs/BAO_CAO_KHOI_B_2026-09-16.md` §B-2.",
      "5. Khuyến nghị cho lần thăng cấp kế tiếp: **chưa bật cờ `--visual-emission` mặc định**. Benchmark gác qua, nhưng build thật đổi 235 cột / 403 cặp "
      "(DANH_MUC ước ≤≈50), 90% trong REVIEW, và 1 ô QĐ-01 rơi pending (2.011 locked + 1 pending thay vì 2.012) — cần Khối C chấm mù các cặp đổi (≈100 ô) "
      "trước khi bật; bật thì phải kèm sửa khoá QĐ-01 (ô trôi → QĐ-01a) và ghi 4 cột sidecar vào labels_trace.", ""]
(D / "BENCH_B2.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L)[:6000])
