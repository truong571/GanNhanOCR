import sys, runpy
from pathlib import Path
SP = Path(__file__).parent
sys.argv = ["dp_vis3.py", "0.5"]
# nạp dp_vis3 nhưng chặn phần in cuối: chép mã tới trước vòng in
src = (SP / "dp_vis3.py").read_text().split('print(f"C0 (phát xạ')[0]
g = {"__file__": str(SP/"dp_vis3.py")}; exec(src, g)
pick, run, PROD, CALIB = g["pick"], g["run"], g["PROD"], g["CALIB"]
eq = pick("test") + pick("val")
print("cột", len(eq))
for kind in ("drop_char", "drop_syl", "split_char", "none"):
    out = []
    for name, M, lam, gv in (("PROD 0", PROD, 0, 0), ("PROD λ.02", PROD, 0.02, 8.0), ("PROD λ.04", PROD, 0.04, 8.0), ("PROD λ.08", PROD, 0.08, 8.0), ("PROD λ.15", PROD, 0.15, 8.0),
                             ("CALIB 0", CALIB, 0, 0), ("CALIB λ.1", CALIB, 0.1, 8.0), ("CALIB λ.25", CALIB, 0.25, 8.0), ("CALIB λ.25 khe4", CALIB, 0.25, 4.0), ("CALIB λ.25 khe16", CALIB, 0.25, 16.0)):
        n, w, gk, nc = run(eq, kind, M, lam, gv)
        out.append(f"{name} {w}/{n}={w/n:.2%} khe {gk/nc:.1%}")
    print(f"{kind:10s} | " + " | ".join(out))
