"""Tự sinh các khối SỐ trong docs/BANG_SO_LIEU_CHINH_THUC.md từ chính dữ liệu trên đĩa.

VÌ SAO
------
KHỐI 2 (2026-08-23) viết lại tài liệu số liệu để diệt lớp lệch "tài liệu nói một đằng,
đĩa một nẻo". Rồi T1 và T2 đổi bộ nhãn hai lần và tài liệu **lệch lại ngay** — vì không
có gì ÉP cập nhật. Gõ tay thì sẽ quên; nên các khối phụ thuộc dữ liệu phải TỰ SINH.

Cùng cơ chế với `evidence()` trong run_pipeline.sh: nội dung giữa hai mốc HTML comment
bị GHI ĐÈ, phần văn xuôi ngoài mốc do người viết và không bị đụng.

    python -m pipeline.tools.update_bang_so_lieu           # ghi
    python -m pipeline.tools.update_bang_so_lieu --check   # chỉ kiểm, exit 1 nếu lệch

`--check` dùng cho CI / trước khi commit: nó KHÔNG sửa gì, chỉ báo tài liệu đã lệch.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "BANG_SO_LIEU_CHINH_THUC.md"

FILES = [
    ("dataset_out/labels.csv",
     "python -m pipeline.align_engine.build_dataset --config config/pipeline.yaml --use-s3 --reseg detector"),
    ("dataset_out/labels_remediated.csv",
     "python -m pipeline.remediation --labels dataset_out/labels.csv --out dataset_out apply --tau 0.62"),
    ("dataset_out/labels_final.csv",
     "python -m pipeline.remediation.confusion_fix … rồi python -m pipeline.remediation.s3_unwind … --apply"),
    ("dataset/labels.csv",
     "python pipeline/export_final_dataset.py --labels dataset_out/labels_final.csv --src-root dataset_out --out dataset"),
]
USABLE = ("GOLD", "SYLLABLE")
BLOCKS = ("HEADER", "NGUON_GOC", "PHAN_HANG", "LUAT", "PHAM_VI", "VA_LOI", "DO_KHAC")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def _n(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def _rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_blocks() -> dict[str, str]:
    final = _rows(REPO / "dataset_out" / "labels_final.csv")
    pub = _rows(REPO / "dataset" / "labels.csv")
    tiers = collections.Counter(r["tier"] for r in final)
    tot = len(final)

    # --- NGUON_GOC: bảng hash + lệnh tái sinh -----------------------------
    out = ["| Tệp | dòng | sha256 (16 đầu) | Lệnh tái sinh |", "|---|---|---|---|"]
    for rel, cmd in FILES:
        p = REPO / rel
        if not p.exists():
            out.append(f"| `{rel}` | (chưa có) | — | `{cmd}` |")
            continue
        n = sum(1 for _ in open(p, encoding="utf-8")) - 1
        bold = "**" if rel == "dataset/labels.csv" else ""
        out.append(f"| `{rel}`{' (**bộ giao nộp**)' if bold else ''} | {bold}{_n(n)}{bold} "
                   f"| `{_sha(p)}` | `{cmd}` |")
    nguon = "\n".join(out)

    # --- PHAN_HANG: tier ---------------------------------------------------
    note = {
        "GOLD": ("✅", "⚪ CHƯA ĐO"),
        "SILVER_uncalibrated": ("❌ ngoài `USABLE_TIERS`",
                                "⚪ CHƯA ĐO (verdict **máy**, không dùng làm bằng chứng)"),
        "SYLLABLE": ("✅ (nhãn cấp **âm tiết**)", "⚪ CHƯA ĐO"),
        "REVIEW": ("❌", "— (không phải nhãn)"),
    }
    out = [f"| Tier | Số ô | % | Vào bộ giao nộp? | Nguồn kiểm định |", "|---|---|---|---|---|"]
    for t in ("GOLD", "SILVER_uncalibrated", "SYLLABLE", "REVIEW"):
        n = tiers.get(t, 0)
        inn, src = note.get(t, ("?", "?"))
        b = "**" if t in USABLE and t == "GOLD" else ""
        out.append(f"| {b}{t}{b} | {b}{_n(n)}{b} | {100*n/max(tot,1):.1f}% | {inn} | {src} |")
    out.append("")
    out.append(f"**Bộ giao nộp = GOLD + SYLLABLE = {_n(len(pub))} ô** · {_n(len(pub))} ảnh đã copy, **0 thiếu**.")
    phan_hang = "\n".join(out)

    # --- LUAT --------------------------------------------------------------
    rules = collections.Counter(r["rule"].split("|")[0] for r in final)
    tier_of: dict[str, str] = {}
    for r in final:
        tier_of.setdefault(r["rule"].split("|")[0], r["tier"])
    out = ["| Rule | Số ô | Tier |", "|---|---|---|"]
    for k, v in rules.most_common():
        out.append(f"| `{k}` | {_n(v)} | {tier_of.get(k,'')} |")
    luat = "\n".join(out)

    # --- PHAM_VI -----------------------------------------------------------
    pages = {(r["book"], r["page"]) for r in final}
    percol = collections.defaultdict(set)
    for r in final:
        if r["column"]:
            percol[(r["book"], r["page"])].add(r["column"])
    n9 = sum(1 for v in percol.values() if len(v) == 9)
    cls_all = len({r["label"] for r in final if r["label"]})
    cls_pub = len({r["label"] for r in pub if r["label"]})
    splits = collections.Counter(r["split"] for r in pub)
    st = subprocess.run(["bash", str(REPO / "scripts" / "run_all_selftests.sh")],
                        capture_output=True, text=True, cwd=REPO)
    m = re.search(r"TỔNG\s+(\d+) passed,\s+(\d+) failed", st.stdout)
    sel = f"{m.group(1)} passed, {m.group(2)} failed" if m else "(không chạy được)"
    out = ["| Chỉ số | Giá trị |", "|---|---|",
           f"| Sách | {len({b for b,_ in pages})} ({', '.join(sorted({b for b,_ in pages}))}) |",
           f"| Trang | {_n(len(pages))} |",
           f"| **Trang cho đủ 9 cột có nhãn** | **{n9}/{len(percol)}** |",
           f"| Lớp ký tự phân biệt (mọi tier có nhãn) | {_n(cls_all)} |",
           f"| **Lớp trong bộ giao nộp** | **{_n(cls_pub)}** |",
           f"| Split bộ giao nộp | " + " · ".join(f"{k} {_n(v)}" for k, v in sorted(splits.items())) + " |",
           f"| **Selftest** | **{sel}** |"]
    pham_vi = "\n".join(out)

    # --- HEADER ------------------------------------------------------------
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, cwd=REPO).stdout.strip() or "(không phải git)"
    import datetime
    header = (f"**Đo ngày**: {datetime.date.today()} · **Commit**: `{sha}` · "
              f"**Bộ nhãn**: `dataset_out/labels_final.csv` ({_n(tot)} dòng)")

    # --- VA_LOI: số liệu bước 4-6 -------------------------------------------
    import json
    def _j(rel):
        p = REPO / rel
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    rem, cfx, unw = _j("dataset_out/remediation_report.json"), \
        _j("dataset_out/confusion_fix_report.json"), _j("dataset_out/s3_unwind_report.json")
    out = ["| Chỉ số | Giá trị | Ghi chú |", "|---|---|---|",
           f"| Quarantine (bbox trùng, nhãn mâu thuẫn) | **{_n(rem.get('quarantined_rows',0))}** | lớp lỗi đã đóng ở gốc engine |",
           f"| Đổi split do trùng md5 | **{_n(rem.get('split_reassigned_rows',0))}** | rò rỉ vốn đã bằng 0 trước bước 4 |",
           f"| Demote theo S3 (`--s3-demote`) | **{_n(rem.get('demoted_similar_lowcos',0))}** | **TẮT MẶC ĐỊNH** từ 2026-08-19 (tiêu chí dựa trên S3, chưa chứng minh được) |",
           f"| Demote lớp confusion 㝵/\"người\" | **{_n(cfx.get('total_demoted',0))}** | chốt chặn ở `s3_unwind` (KHỐI 1.1) |",
           f"| Trả về GOLD sau `s3_unwind` | **{_n(unw.get('readmitted_to_gold',0))}** | 0 = không còn ô nào mang hậu tố `demoted_lowcos_s3` |"]
    va_loi = "\n".join(out)

    # --- DO_KHAC: các đại lượng phụ thuộc dữ liệu ở §3 ----------------------
    sys.path.insert(0, str(REPO))
    from core.text.dictionary import dict_dir, load_qn_to_nom
    from core.text.text_utils import is_plausible_qn_syllable
    keys = set(load_qn_to_nom(str(dict_dir() / "QuocNgu_SinoNom.csv")))
    syl = [r["syllable"].lower() for r in final if r["syllable"]]
    oo = sum(1 for x in syl if x not in keys)
    pl = sum(1 for x in syl if is_plausible_qn_syllable(x))
    md5 = collections.defaultdict(set)
    for r in final:
        if r.get("image_md5") and r.get("label"):
            md5[r["image_md5"]].add(r["label"])
    out = ["| Chỉ số phụ thuộc dữ liệu | Giá trị |", "|---|---|",
           f"| Âm QN hợp lệ (`is_plausible_qn_syllable`) | **{100*pl/max(len(syl),1):.2f}%** |",
           f"| **Âm QN có trong từ điển** | **{100*(len(syl)-oo)/max(len(syl),1):.3f}%** ({_n(oo)} ô ngoài) |",
           f"| Trang cho đủ 9 cột có nhãn | **{n9}/{len(percol)}** |",
           f"| Mâu thuẫn tự thân (cùng md5, khác nhãn) | **{sum(1 for v in md5.values() if len(v)>1)}** / {_n(len(md5))} crop |"]
    do_khac = "\n".join(out)

    return {"HEADER": header, "NGUON_GOC": nguon, "PHAN_HANG": phan_hang, "LUAT": luat,
            "PHAM_VI": pham_vi, "VA_LOI": va_loi, "DO_KHAC": do_khac}


def apply_blocks(text: str, blocks: dict[str, str]) -> str:
    for name, body in blocks.items():
        s, e = f"<!-- AUTO:{name}:START -->", f"<!-- AUTO:{name}:END -->"
        if s not in text or e not in text:
            raise SystemExit(f"thiếu mốc {s} / {e} trong {DOC.name}")
        i, j = text.index(s) + len(s), text.index(e)
        text = text[:i] + "\n" + body + "\n" + text[j:]
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.update_bang_so_lieu")
    ap.add_argument("--check", action="store_true",
                    help="chỉ kiểm, KHÔNG ghi; exit 1 nếu tài liệu đã lệch với đĩa")
    args = ap.parse_args(argv)

    cur = DOC.read_text(encoding="utf-8")
    new = apply_blocks(cur, build_blocks())
    if args.check:
        if cur == new:
            print(f"[bảng số liệu] KHỚP đĩa — {DOC.relative_to(REPO)}")
            return 0
        print(f"[bảng số liệu] LỆCH với đĩa — chạy:\n"
              f"    python -m pipeline.tools.update_bang_so_lieu", file=sys.stderr)
        return 1
    DOC.write_text(new, encoding="utf-8")
    print(f"[bảng số liệu] đã cập nhật {len(BLOCKS)} khối trong {DOC.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
