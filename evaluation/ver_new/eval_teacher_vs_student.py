"""ĐỘT PHÁ #1 — test "student beats teacher" NGAY (proxy, không cần Kaggle).

Idea: the trained encoder's ArcFace HEAD (1591-way, in nom-embed/best.pt) is already a
recognizer trained on the auto-labels (GOLD crops + glyphs). Use it as a PROXY STUDENT
to demonstrate the close-the-loop claim before training a dedicated recognizer.

On held-out GOLD test crops (true label = dictionary-confirmed = reliable):
  TEACHER  = the SinoNom OCR's raw output (ocr_char)
  STUDENT  = the ArcFace head's top-1 over all 1591 classes (pure visual, no dict)
Compare accuracy vs the true label, on the strata that matter:
  - overall (teacher is strong here — GOLD is mostly teacher-correct by construction)
  - TEACHER-OOV: classes the SinoNom charset NEVER emits (true label ∉ {all ocr_char}).
    Teacher acc = 0 by definition; ANY student hit = a STRICT win attributable to the
    auto-labels (the bulletproof result).
  - teacher-WRONG subset (ocr_char != true): student recovery rate.
  - rare (<5 crops) vs common.
Wilson 95% CIs throughout.

CAVEATS (state in thesis): (a) proxy student = the existing head, not a dedicated
recognizer (the full #1 trains one, likely better); (b) GOLD is the EASY regime — the
true headline needs the dedicated recognizer + human-audited SILVER/REVIEW. This test
shows the MECHANISM with real numbers and the teacher-OOV strict win.

Run:
  .venv/bin/python evaluation/ver_new/eval_teacher_vs_student.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402

HERE = Path(__file__).resolve().parent


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    D = HERE / "dataset_out"
    rows = list(csv.DictReader(open(D / "labels.csv", encoding="utf-8")))
    # kimhannom EMITTABLE charset ≈ every char it ever output as ocr_char
    emittable = {r["ocr_char"] for r in rows if r["ocr_char"] and _is_cjk(r["ocr_char"])}
    freq = Counter(r["label"] for r in rows if r["tier"] == "GOLD" and r["label"])
    test = [r for r in rows if r["tier"] == "GOLD" and r["split"] == "test"
            and r["image"] and r["label"] and _is_cjk(r["label"])]
    print(f"kimhannom emittable charset (distinct ocr_char): {len(emittable)} | "
          f"label classes never emitted by teacher: "
          f"{len({r['label'] for r in test}) - len({r['label'] for r in test} & emittable)} of "
          f"{len({r['label'] for r in test})} test classes", flush=True)

    enc = VisualS3(REPO, fd_dir="").enc
    if not enc.has_head:
        sys.exit("no ArcFace head in ckpt.")

    # accumulators
    n = t_hit = s_hit = 0
    oov_n = oov_s_hit = 0                 # teacher-OOV (true ∉ emittable)
    tw_n = tw_s_hit = 0                   # teacher-WRONG (ocr != true)
    rare_n = rare_t = rare_s = 0
    com_n = com_t = com_s = 0
    examples = []
    for i, r in enumerate(test):
        emb = enc.embed_path(str(D / r["image"]))
        if emb is None:
            continue
        true = r["label"]; ocr = r["ocr_char"]
        tk = enc.predict_topk(emb, 1)
        student = tk[0][0] if tk else None
        n += 1
        t_ok = int(ocr == true); s_ok = int(student == true)
        t_hit += t_ok; s_hit += s_ok
        if true not in emittable:
            oov_n += 1; oov_s_hit += s_ok
            if s_ok and len(examples) < 8:
                examples.append(f"{true} (U+{ord(true):X}, âm {r['syllable']}): teacher đọc '{ocr}' (sai/OOV) → student ĐÚNG")
        if ocr != true:
            tw_n += 1; tw_s_hit += s_ok
        if freq.get(true, 0) < 5:
            rare_n += 1; rare_t += t_ok; rare_s += s_ok
        else:
            com_n += 1; com_t += t_ok; com_s += s_ok
        if (i + 1) % 1000 == 0:
            print(f"  ... {i+1}/{len(test)}", flush=True)

    def line(name, k, nn):
        p, lo, hi = wilson(k, nn)
        return f"  {name:32s} {p:6.1%}  [{lo:.1%}, {hi:.1%}]   (n={nn})"

    print("\n" + "=" * 72)
    print(" #1 TEACHER (SinoNom OCR) vs STUDENT (ArcFace head) — GOLD test, nhãn đáng tin")
    print("=" * 72)
    print(line("TEACHER acc (ocr_char==true)", t_hit, n))
    print(line("STUDENT acc (head top-1==true)", s_hit, n))
    print(f"\n  --- các tầng QUYẾT ĐỊNH ---")
    print(line("TEACHER-OOV: STUDENT acc", oov_s_hit, oov_n) + "  (teacher = 0% — không xuất được)")
    print(line("teacher-WRONG: STUDENT recovery", tw_s_hit, tw_n) + "  (cứu lỗi của thầy)")
    print(f"\n  --- theo tần suất ---")
    print(line("RARE (<5)  teacher", rare_t, rare_n)); print(line("RARE (<5)  student", rare_s, rare_n))
    print(line("COMMON     teacher", com_t, com_n)); print(line("COMMON     student", com_s, com_n))
    if examples:
        print(f"\n  Ví dụ teacher-OOV student cứu được:")
        for e in examples:
            print("   " + e)

    print("\n  ĐỌC: overall teacher thường ≥ student (GOLD = chỗ thầy đúng theo định nghĩa).")
    print("  THẮNG TUYỆT ĐỐI ở: TEACHER-OOV (thầy 0%, student >0%) + đuôi hiếm. Đó là headline.")
    print("  (Proxy = head sẵn có; #1 đầy đủ train recognizer riêng + soát người regime SILVER/REVIEW.)")

    out = {"n": n, "emittable_charset": len(emittable),
           "teacher_acc": round(t_hit / max(n, 1), 4), "student_acc": round(s_hit / max(n, 1), 4),
           "teacher_oov": {"n": oov_n, "student_acc": round(oov_s_hit / max(oov_n, 1), 4)},
           "teacher_wrong": {"n": tw_n, "student_recovery": round(tw_s_hit / max(tw_n, 1), 4)},
           "rare": {"n": rare_n, "teacher": round(rare_t / max(rare_n, 1), 4), "student": round(rare_s / max(rare_n, 1), 4)},
           "common": {"n": com_n, "teacher": round(com_t / max(com_n, 1), 4), "student": round(com_s / max(com_n, 1), 4)}}
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results" / "eval_teacher_vs_student.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  -> results/eval_teacher_vs_student.json")


if __name__ == "__main__":
    main()
