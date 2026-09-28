"""TN6 x03 — dựng lại MỘT bộ với box_decoder khác vào THƯ MỤC RIÊNG measure_out/_tn6/<Bộ>/<biến thể>/ (0 API: dùng cache
kim/QN đã có; KHÔNG ghi dataset/ hay prepared/<Bộ>/dataset_out). Chuỗi lệnh = run_pipeline.sh bước 2–4 (build → enrich →
remediation census/apply → confusion_fix → mechanism_gates) với config tạm (chỉ khác khoá box_decoder).

  .venv/bin/python lab/thu_nghiem_anh_chu/TN6_hop_anh/x03_run.py --book SachDungLyHoThan --variant visual_dp
"""
import argparse, subprocess, sys, time
from pathlib import Path
import yaml
REPO = Path(__file__).resolve().parents[3]
PY = str(REPO / ".venv/bin/python")
CONFIG = {"SachKinhThayCaBinh": "config/pipeline_SachKinhThayCaBinh.yaml", "SachDungLyHoThan": "config/pipeline_SachDungLyHoThan.yaml",
          "LucVanTien1916": "config/pipeline_LucVanTien1916.yaml", "TruyenKieu1872": "config/pipeline_TruyenKieu1872.yaml",
          "Chrestomathie1872": "config/pipeline_Chrestomathie1872.yaml", "LucVanTien1883": "config/pipeline_LucVanTien1883.yaml",
          "KimVanKieu1884": "config/pipeline_KimVanKieu1884_b1.yaml", "STT": "config/pipeline.yaml"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--variant", required=True, choices=["visual_dp", "visual_dp_hybrid", "legacy", "pitch"])
    ap.add_argument("--from", dest="start", default="build", choices=["build", "gates"])
    a = ap.parse_args()
    base = REPO / "measure_out/_tn6" / a.book / a.variant
    out = base / "dataset_out"
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(REPO / CONFIG[a.book], encoding="utf-8"))
    for b in (cfg["books"] if a.book == "STT" else cfg["books"][:1]):
        if a.variant == "legacy":
            b.pop("box_decoder", None)
        else:
            b["box_decoder"] = a.variant
    cfg.setdefault("run", {})["dataset_out"] = str(out.relative_to(REPO))
    cfg.setdefault("paths", {})["output_dir"] = str((base / "export").relative_to(REPO))
    cpath = base / "config.yaml"
    yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    C, O = str(cpath.relative_to(REPO)), str(out.relative_to(REPO))
    steps = [
        ("build", [PY, "-m", "pipeline.align_engine.build_dataset", "--config", C, "--reseg", "detector", "--qd01-cells", "none",
                   "--decisions", "none", "--use-s3", "--two-pass", "--box-rule", "syl_index", "--force", "--out", O]),
        ("enrich", [PY, "-m", "pipeline.tools.enrich_crop_quality", "--labels", f"{O}/labels.csv", "--src-root", O]),
        ("census", [PY, "-m", "pipeline.remediation", "--labels", f"{O}/labels.csv", "--out", O, "census"]),
        ("apply", [PY, "-m", "pipeline.remediation", "--labels", f"{O}/labels.csv", "--out", O, "apply", "--tau", "0.62"]),
        ("confusion", [PY, "-m", "pipeline.remediation.confusion_fix", "--in", f"{O}/labels_remediated.csv", "--out",
                       f"{O}/labels_final.csv", "--fixes", "config/confusion_fixes.yaml", "--measure"]),
        ("gates", [PY, "-m", "pipeline.remediation.mechanism_gates", "--in", f"{O}/labels_final.csv", "--out", f"{O}/labels_gated.csv",
                   "--config", C, "--book", a.book, "--report", f"{O}/mechanism_gates_report.json"]),
    ]
    if a.book in ("LucVanTien1883", "KimVanKieu1884"):   # sách có dị bản số hoá: cổng (d) --cross như run_pipeline.sh
        AP = [PY, "scripts/measure/auto_precision.py"]
        tr = f"prepared/{a.book}/transcriptions"
        steps = steps[:5] + [
            ("auto_precision", AP + ["--steps", "cross,gates", "--books", a.book, "--labels", f"{O}/labels_final.csv",
                                     "--trans", tr, "--out", f"{O}/auto_precision"]),
            ("gates", steps[5][1] + ["--cross", f"{O}/auto_precision/cross/{a.book}/cells.csv"]),
            ("auto_precision_gated", AP + ["--steps", "cross", "--books", a.book, "--labels", f"{O}/labels_gated.csv",
                                           "--trans", tr, "--out", f"{O}/auto_precision_gated"])]
    if a.book == "STT":       # đường STT của run_pipeline.sh (step_build + step_remediate; không cổng cơ chế)
        steps = [("build", [PY, "-m", "pipeline.align_engine.build_dataset", "--config", C, "--reseg", "detector",
                            "--qd01-cells", "none", "--force", "--out", O])] + steps[1:5]
    if a.start == "gates":
        steps = [st for st in steps if st[0] in ("auto_precision", "gates", "auto_precision_gated")]
    log = open(base / "run.log", "a", encoding="utf-8")
    for name, cmd in steps:
        t0 = time.time()
        log.write(f"\n=== {name}: {' '.join(cmd)}\n"); log.flush()
        r = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"=== {name}: rc {r.returncode} ({time.time() - t0:.0f} s)\n"); log.flush()
        print(f"{a.book}/{a.variant} {name}: rc {r.returncode} ({time.time() - t0:.0f} s)", flush=True)
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
