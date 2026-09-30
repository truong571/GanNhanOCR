"""t02_build_stt.py — dựng lại STT (3 sách) trong HỘP CÁT với (a) nguồn đọc kim khác và/hoặc (b) box_decoder khác. 0 API.

Không ghi prepared/, dataset_out/, dataset/. Mã chạy = ẢNH CHỤP HEAD (measure_out/_tn9/code: git archive pipeline/core/config
+ liên kết tài nguyên) để không dính sửa đổi đang làm dở của agent khác. Dữ liệu = measure_out/_tn9/stt_prep/<đọc>/<Sách>/
(pages, pages_denoised, transcriptions, manifest.json = liên kết tới prepared/ ; detected/ = cache OCR SINH RIÊNG).

Nguồn đọc (--read):
  lt1          cache hiện hành (kim lang_type 1 = Hán) — tái lập đường sản xuất
  lt2          kim lang_type 2 (Nôm) nguyên trạng (prepared/<S>/kim_raw_lt2/<trang>_lt2.json, cùng định dạng)
  l1skel_l2    KHUNG lt1 (dòng, số chữ, hộp chữ giữ nguyên từng điểm) + CHỮ lt2 ở vị trí gióng được (Needleman–Wunsch
               theo dòng có IoU ≥ 0,5, như scripts/measure/stt_lt2_eval.align); vị trí không gióng giữ chữ lt1.
  twopass      ĐỀ XUẤT NGƯỜI DÙNG "đọc hai lượt" (lab/thu_nghiem_kim/tn7_kim_tham_so.merge_two_pass, dùng NGUYÊN hàm): khung =
               chữ lt1 (hộp chia đều); tại mỗi vị trí khung có chữ lt2 mà tâm rơi trong hộp khung nới 30 % (gần nhất, mỗi chữ
               lt2 dùng một lần) thì lấy chữ lt2, không thì giữ lt1; chữ lt2 KHÔNG vào khung nào được THÊM vào cột lt1 có
               dải x (nới 30 %) chứa tâm nó, sắp theo y (không cột nào chứa -> bỏ, có đếm).
Chuỗi lệnh = run_pipeline.sh step_build + step_remediate (build -> enrich -> census -> apply τ 0,62 -> confusion_fix);
rescue (bước 5) KHÔNG chạy ở đây (đo riêng).
  .venv/bin/python lab/thu_nghiem_kim/TN9_stt_in/t02_build_stt.py --read l1skel_l2 --box legacy
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402

CODE = T.OUT / "code"
PREP = T.OUT / "stt_prep"
BUILDS = T.OUT / "stt_builds"
PY = str(T.REPO / ".venv/bin/python")
BOOKS = ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]
LINK = ["pages", "pages_denoised", "transcriptions", "manifest.json"]


def merged_cache(c1: dict, c2: dict | None) -> tuple[dict, dict]:
    """Khung lt1 + chữ lt2 (vị trí gióng được). Trả (cache mới, thống kê)."""
    import stt_lt2_eval as SL
    from core.ocr.ocr_api import boxes_to_columns
    st = dict(lines=0, lines_matched=0, pos=0, pos_l2=0, pos_changed=0, fallback=0)
    out = copy.deepcopy(c1)
    if c2 is None:
        st["fallback"] = 1
        return out, st
    R1, R2 = SL.raw_lines(c1.get("boxes_raw")), SL.raw_lines(c2.get("boxes_raw"))
    new_raw = copy.deepcopy(c1.get("boxes_raw") or [])
    # chỉ số dòng hợp lệ trong boxes_raw (raw_lines bỏ dòng rỗng) -> ánh xạ lại
    valid_idx = [i for i, b in enumerate(c1.get("boxes_raw") or [])
                 if [ch for ch in (b.get("transcription") or "").strip() if ch.strip()]]
    for li, L in enumerate(R1):
        st["lines"] += 1
        st["pos"] += len(L["s"])
        oi = SL.best_line(L, R2)
        if oi is None:
            continue
        st["lines_matched"] += 1
        m = SL.align(L["s"], R2[oi]["s"])
        s = list(L["s"])
        for k, j in enumerate(m):
            if j is not None:
                st["pos_l2"] += 1
                if R2[oi]["s"][j] != s[k]:
                    st["pos_changed"] += 1
                s[k] = R2[oi]["s"][j]
        new_raw[valid_idx[li]]["transcription"] = "".join(s)
    # cột: giữ HÌNH HỌC cột cache lt1 từng điểm, thay chữ theo thứ tự (boxes_to_columns của khung mới)
    nc = boxes_to_columns(new_raw)
    oc = boxes_to_columns(c1.get("boxes_raw"))
    shape_new = [len(c) for c in nc]
    if shape_new != [len(c) for c in oc] or shape_new != [len(c) for c in c1["columns"]] or \
            [[x["char"] for x in c] for c in oc] != [[x["char"] for x in c] for c in c1["columns"]]:
        st["fallback"] = 1
        return out, st
    for col_out, col_new in zip(out["columns"], nc):
        for x, y in zip(col_out, col_new):
            x["char"] = y["char"]
    out["boxes_raw"] = new_raw
    out["tn9_read"] = "l1skel_l2"
    return out, st


def twopass_cache(c1: dict, c2: dict | None) -> tuple[dict, dict]:
    """Đề xuất đọc hai lượt: merge_two_pass(khung lt1, chữ lt2) -> cache cột (giữ hình học khung lt1, thêm chữ lt2 thừa)."""
    sys.path.insert(0, str(T.REPO / "lab/thu_nghiem_kim"))
    from tn7_kim_tham_so import merge_two_pass
    st = dict(skel=0, took_l2=0, changed=0, appended=0, dropped=0, fallback=0)
    out = copy.deepcopy(c1)
    if c2 is None:
        st["fallback"] = 1
        return out, st
    skel, pos = [], []
    for ci, col in enumerate(out["columns"]):
        for k, c in enumerate(col):
            b = c["bbox"]
            skel.append((c["char"], float(b[0]), float(b[1]), float(b[2]), float(b[3]))); pos.append((ci, k))
    nom = [(c["char"], float(c["bbox"][0]), float(c["bbox"][1]), float(c["bbox"][2]), float(c["bbox"][3]))
           for col in (c2.get("columns") or []) for c in col if c.get("bbox") and c.get("char")]
    merged = merge_two_pass(skel, nom)
    st["skel"] = len(skel)
    for (ci, k), m, s0 in zip(pos, merged[:len(skel)], skel):
        if m[0] != s0[0]:
            st["changed"] += 1
        out["columns"][ci][k]["char"] = m[0]
    # lượt 1 có chữ lt2 rơi vào hộp: merge_two_pass không trả cờ -> đếm lại theo luật của nó (tâm trong hộp nới 30 %)
    st["took_l2"] = len(nom) - len(merged[len(skel):])
    ext = []
    for col in out["columns"]:
        if col:
            x0 = min(c["bbox"][0] for c in col); x1 = max(c["bbox"][2] for c in col); w = x1 - x0
            ext.append((x0 - 0.3 * w, x1 + 0.3 * w, (x0 + x1) / 2))
        else:
            ext.append(None)
    for ch, a0, b0, a1, b1 in merged[len(skel):]:
        cx, cy = (a0 + a1) / 2, (b0 + b1) / 2
        cand = [(abs(cx - e[2]), ci) for ci, e in enumerate(ext) if e is not None and e[0] <= cx <= e[1]]
        if not cand:
            st["dropped"] += 1
            continue
        ci = min(cand)[1]
        out["columns"][ci].append({"char": ch, "y_center": cy, "bbox": [int(a0), int(b0), int(a1), int(b1)]})
        st["appended"] += 1
    for col in out["columns"]:
        col.sort(key=lambda c: c["y_center"])
    out["tn9_read"] = "twopass"
    return out, st


def prep_read(read: str) -> Path:
    base = PREP / read
    stats = {}
    for bk in BOOKS:
        src = T.REPO / "prepared" / bk
        dst = base / bk
        (dst / "detected").mkdir(parents=True, exist_ok=True)
        for x in LINK:
            if not (dst / x).exists():
                os.symlink(src / x, dst / x)
        n = dict(pages=0, lt2_missing=0)
        agg = {}
        for f in sorted((src / "detected").glob("page_*_ocr_cache.json")):
            page = f.name[: -len("_ocr_cache.json")]
            out = dst / "detected" / f.name
            if out.exists():
                continue
            c1 = json.load(open(f, encoding="utf-8"))
            f2 = src / "kim_raw_lt2" / f"{page}_lt2.json"
            c2 = json.load(open(f2, encoding="utf-8")) if f2.exists() else None
            if c2 is not None and (c2.get("image_hash") != c1.get("image_hash") or c2.get("coords_space") != "fullpage"):
                c2 = None
            n["pages"] += 1
            if c2 is None:
                n["lt2_missing"] += 1
            if read == "lt1":
                c = c1
            elif read == "lt2":
                if c2 is None:
                    c = c1
                else:
                    c = {k: v for k, v in c2.items() if k not in ("kim_params", "domain", "created", "lt1_cache", "lt1_n_columns")}
            elif read in ("l1skel_l2", "twopass"):
                c, s = (merged_cache if read == "l1skel_l2" else twopass_cache)(c1, c2)
                for k, v in s.items():
                    agg[k] = agg.get(k, 0) + v
            else:
                raise SystemExit(f"read lạ: {read}")
            json.dump(c, open(out, "w", encoding="utf-8"), ensure_ascii=False)
        n.update(agg)
        stats[bk] = n
    T.jdump(stats, base / "prep_stats.json")
    print(f"[t02] chuẩn bị đọc {read}: {stats}", flush=True)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", required=True, choices=["lt1", "lt2", "l1skel_l2", "twopass"])
    ap.add_argument("--box", default="legacy", choices=["legacy", "visual_dp", "visual_dp_hybrid"])
    ap.add_argument("--only-prep", action="store_true")
    ap.add_argument("--from-step", default="build")
    a = ap.parse_args()
    base = prep_read(a.read)
    if a.only_prep:
        return
    name = f"{a.read}__{a.box}"
    bd = BUILDS / name
    out = bd / "dataset_out"
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(CODE / "config/pipeline.yaml", encoding="utf-8"))
    for b in cfg["books"]:
        if a.box == "legacy":
            b.pop("box_decoder", None)
        else:
            b["box_decoder"] = a.box
    cfg["paths"]["data_dir"] = str(base)                 # tuyệt đối: REPO / "/abs" == "/abs"
    cfg["paths"]["output_dir"] = str(bd / "export")
    cfg.setdefault("run", {})["dataset_out"] = str(out)
    cpath = bd / "config.yaml"
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    O = str(out)
    steps = [
        ("build", [PY, "-m", "pipeline.align_engine.build_dataset", "--config", str(cpath), "--reseg", "detector",
                   "--qd01-cells", "none", "--force", "--out", O]),
        ("enrich", [PY, "-m", "pipeline.tools.enrich_crop_quality", "--labels", f"{O}/labels.csv", "--src-root", O]),
        ("census", [PY, "-m", "pipeline.remediation", "--labels", f"{O}/labels.csv", "--out", O, "census"]),
        ("apply", [PY, "-m", "pipeline.remediation", "--labels", f"{O}/labels.csv", "--out", O, "apply", "--tau", "0.62"]),
        ("confusion", [PY, "-m", "pipeline.remediation.confusion_fix", "--in", f"{O}/labels_remediated.csv", "--out",
                       f"{O}/labels_final.csv", "--fixes", str(CODE / "config/confusion_fixes.yaml"), "--measure"]),
    ]
    names = [s[0] for s in steps]
    steps = steps[names.index(a.from_step):]
    log = open(bd / "run.log", "a", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(CODE))
    for nm, cmd in steps:
        t0 = time.time()
        log.write(f"\n=== {nm}: {' '.join(cmd)}\n"); log.flush()
        r = subprocess.run(cmd, cwd=CODE, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(f"=== {nm}: rc {r.returncode} ({time.time() - t0:.0f} s)\n"); log.flush()
        print(f"[t02] {name} {nm}: rc {r.returncode} ({time.time() - t0:.0f} s)", flush=True)
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
