"""Bước 7a — NẠP PHÁN QUYẾT CỦA NGƯỜI trở lại bộ nhãn.

ĐÂY LÀ MẮT XÍCH CUỐI CÙNG CÒN THIẾU của cả dự án. Trước bước này, `run_pipeline.sh`
ghi rõ "TẠM BỎ QUA: audit người" — nghĩa là mọi thứ máy làm ra chưa từng được người
xác nhận, và mọi con số precision đều treo.

HAI VIỆC, VÀ VIỆC THỨ HAI MỚI LÀ VIỆC CHÍNH
-------------------------------------------
1. ÁP TỪNG Ô. Mẻ chấm chỉ phủ ~960/57.067 ô, nên việc này đổi rất ít dữ liệu:
       verdict `correct`      -> giữ nguyên, gắn human_verdict=correct
       verdict `wrong_label`  -> HẠ xuống REVIEW. Người chấm chỉ được hỏi "đúng hay
                                 sai", KHÔNG được hỏi "chữ đúng là gì" — nên ta biết
                                 nhãn sai nhưng KHÔNG biết nhãn đúng. Bịa ra một nhãn
                                 thay thế là tạo dữ liệu giả.
       verdict `unsure`       -> GIỮ NGUYÊN tier, gắn cờ. Ô này bị loại khỏi MẪU SỐ khi
                                 tính precision, không tính là lỗi.

2. SUY RỘNG RA DÂN SỐ — đây mới là con số luận văn cần. Mẫu phân bổ theo SÁCH chứ không
   tỷ lệ thuận với dân số, nên trung bình cộng thô là SAI. Phải dùng trọng số
   Horvitz-Thompson (`design_weight` = N_h / n_h) và ước lượng theo TẦNG.

BA TẦNG, BA CON SỐ RIÊNG — KHÔNG ĐƯỢC GỘP
-----------------------------------------
   GOLD/direct   46.327 ô  chữ QUAN SÁT ĐƯỢC, từ điển xác nhận nó
   GOLD/similar   3.829 ô  chữ quan sát được KHÔNG phải âm đọc; nhãn là chữ KHÁC bắc cầu
   SYLLABLE       6.911 ô  KHÔNG có nhãn ký tự; câu hỏi là về ÂM
Tỷ lệ direct:similar là 12:1, nên gộp lại thì precision đo được gần như chỉ của direct.

CHỐT CHẶN
---------
* Ô LẶP ẨN (repeat_of) bị LOẠI khỏi ước lượng — chúng đo κ nội tại, không phải quan sát
  độc lập; tính vào là ĐẾM TRÙNG dân số.
* κ liên-người < KAPPA_MIN thì **TỪ CHỐI công bố precision**. Khi hai người không cùng
  tiêu chí, con số đo được là tiêu chí của một người chứ không phải chất lượng dữ liệu.

    python -m pipeline.remediation.apply_verdicts --in dataset_out/labels_final.csv \
        --out dataset_out/labels_final.csv --batch dataset_out/ground_truth/audit_combined
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KAPPA_MIN = 0.60
# Tệp KHAI XUẤT XỨ bắt buộc, đặt cạnh verdicts*.jsonl.
PROV = "NGUOI_CHAM.md"
VERDICT_COL = "human_verdict"
NA = dict(keep_default_na=False, na_values=[""])


def apply_to_labels(labels_csv: Path, out_csv: Path, joined) -> dict:
    import pandas as pd
    df = pd.read_csv(labels_csv, dtype={"image_md5": str}, **NA)
    if VERDICT_COL not in df.columns:
        df[VERDICT_COL] = ""
    # khoá join: manifest mang `image`, bộ nhãn cũng vậy
    if "image" not in joined.columns:
        raise SystemExit("manifest thiếu cột `image` — không ghép được về bộ nhãn")
    vmap = {str(r["image"]): str(r["verdict"]) for _, r in joined.iterrows()
            if str(r.get("image", "")).strip()}
    hit = df["image"].astype(str).map(vmap)
    df[VERDICT_COL] = hit.fillna("")
    n_wrong = int((df[VERDICT_COL] == "wrong_label").sum())
    # HẠ những ô người chấm nói SAI. Không bịa nhãn thay thế.
    demote = (df[VERDICT_COL] == "wrong_label") & df["tier"].isin(["GOLD", "SYLLABLE"])
    df.loc[demote, "rule"] = "human_verdict:wrong_label"
    df.loc[demote, "tier"] = "REVIEW"
    df.to_csv(out_csv, index=False)
    return {"gan_verdict": int((df[VERDICT_COL] != "").sum()),
            "ha_xuong_review": int(demote.sum()), "wrong_label": n_wrong}


def estimate(batch: Path, conf: float = 0.95) -> dict:
    sys.path.insert(0, str(REPO))
    from pipeline.ground_truth import estimate as E, stats
    import pandas as pd

    v = E.load_verdicts(batch)
    m = E.load_manifest(batch)
    j = E.join_manifest(v, m)

    # ô LẶP ẨN không phải quan sát độc lập -> loại khỏi ước lượng dân số
    if "repeat_of" in j.columns:
        rep = j["repeat_of"].astype(str).str.strip().replace("nan", "")
        j_est = j[rep == ""].copy()
    else:
        j_est = j.copy()

    out = {"n_verdict": int(len(j)), "n_dung_uoc_luong": int(len(j_est)), "tang": {}}
    for st in sorted(set(j_est.get("stratum", pd.Series(dtype=str)).astype(str))):
        if not st or st == "nan":
            continue
        tang = st.split("|")[0]
        out["tang"].setdefault(tang, {"k": 0, "n": 0, "unsure": 0})
    for _, r in j_est.iterrows():
        tang = str(r.get("stratum", "")).split("|")[0]
        if not tang:
            continue
        d = out["tang"].setdefault(tang, {"k": 0, "n": 0, "unsure": 0})
        vd = str(r["verdict"])
        if vd == "unsure":
            d["unsure"] += 1           # loại khỏi MẪU SỐ, không tính là lỗi
            continue
        d["n"] += 1
        d["k"] += int(vd == "correct")
    for tang, d in out["tang"].items():
        if d["n"]:
            lo, hi = stats.clopper_pearson_ci(d["k"], d["n"], conf)
            d["precision"] = d["k"] / d["n"]
            d["ci"] = [lo, hi]
    return out, j


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.remediation.apply_verdicts")
    ap.add_argument("--in", dest="src", default=str(REPO / "dataset_out" / "labels_final.csv"))
    ap.add_argument("--out", dest="dst", default=str(REPO / "dataset_out" / "labels_final.csv"))
    ap.add_argument("--batch", default=str(REPO / "dataset_out" / "ground_truth" / "audit_combined"))
    ap.add_argument("--report", default=str(REPO / "docs" / "BANG_PRECISION.md"))
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--allow-low-kappa", action="store_true",
                    help="công bố precision DÙ κ liên-người thấp — chỉ dùng khi biết rõ mình làm gì")
    args = ap.parse_args(argv)

    batch = Path(args.batch)
    vf = list(batch.glob("verdicts*.jsonl")) if batch.is_dir() else []
    if not vf:
        print(f"[phán quyết] chưa có verdicts*.jsonl trong {batch} — BỎ QUA bước này")
        return 0

    # ---- CHỐT CHẶN XUẤT XỨ: có tệp verdict KHÔNG có nghĩa là người đã chấm ----------
    # Thảm hoạ đã xảy ra thật trong dự án này: toàn bộ "chấm tay" hoá ra là MÁY chấm,
    # và các tệp verdict đó KHÔNG hề tự khai là máy — nên mọi bộ lọc theo trường
    # `source` đều vô dụng với chúng. Hậu quả: precision 97,98%, Fisher p = 5,4e-8,
    # κ = 0,13 đều phải huỷ, và dự án mất nhiều tháng vì tin vào số của chính mình.
    #
    # Bài học: KHÔNG suy ra xuất xứ từ dữ liệu. Bắt khai ra, bằng một tệp riêng mà con
    # người phải cố ý viết. Không có tệp đó thì TỪ CHỐI, dù verdict trông hợp lệ.
    prov = batch / PROV
    if not prov.exists():
        print(f"[phán quyết] 🔴 TỪ CHỐI: có {len(vf)} tệp verdict nhưng KHÔNG có {PROV}.\n"
              f"    Có tệp verdict KHÔNG chứng minh được là NGƯỜI chấm. Dự án này đã một\n"
              f"    lần tin nhầm verdict MÁY là verdict người và phải huỷ toàn bộ số liệu.\n"
              f"    Tạo {prov} khai rõ: AI chấm, chấm khi nào, có đọc được chữ Nôm không,\n"
              f"    người thứ hai là ai. Mẫu: xem docs/QUY_TRINH_CHAM_TAY.md.",
              file=sys.stderr)
        return 1
    txt = prov.read_text(encoding="utf-8")
    if "⬜" in txt or "CHƯA ĐIỀN" in txt:
        print(f"[phán quyết] 🔴 TỪ CHỐI: {PROV} còn mục ⬜ CHƯA ĐIỀN.", file=sys.stderr)
        return 1
    print(f"[phán quyết] xuất xứ: {PROV} có mặt và đã điền")

    est, joined = estimate(batch, args.conf)
    print(f"[phán quyết] {est['n_verdict']:,} verdict, "
          f"{est['n_dung_uoc_luong']:,} dùng ước lượng (đã loại ô lặp ẩn)")

    # κ liên-người — chốt chặn
    kappa = None
    try:
        sys.path.insert(0, str(REPO))
        from pipeline.ground_truth import report_combined as RC
        ir = RC.interrater(batch, args.conf)
        kappa = (ir or {}).get("kappa")
    except Exception as e:
        print(f"[phán quyết] chưa đo được κ liên-người ({type(e).__name__})")
    if kappa is not None:
        print(f"[phán quyết] κ liên-người = {kappa:.3f}")
        if kappa < KAPPA_MIN and not args.allow_low_kappa:
            print(f"[phán quyết] 🔴 DỪNG: κ = {kappa:.3f} < {KAPPA_MIN}. Hai người chấm KHÔNG "
                  f"cùng tiêu chí, nên con số precision đo được là tiêu chí của MỘT NGƯỜI "
                  f"chứ không phải chất lượng dữ liệu. Thống nhất lại định nghĩa verdict "
                  f"rồi chấm lại. (ép chạy: --allow-low-kappa)", file=sys.stderr)
            return 1

    st = apply_to_labels(Path(args.src), Path(args.dst), joined)
    print(f"[phán quyết] gắn verdict cho {st['gan_verdict']:,} ô | "
          f"HẠ xuống REVIEW {st['ha_xuong_review']:,} ô")

    lines = ["# Bảng precision — đo bằng phán quyết của NGƯỜI", "",
             f"Nguồn: `{batch}` · {est['n_verdict']:,} verdict "
             f"({est['n_dung_uoc_luong']:,} dùng ước lượng, đã loại ô lặp ẩn)", ""]
    if kappa is not None:
        lines += [f"**κ liên-người = {kappa:.3f}** "
                  f"({'ĐẠT' if kappa >= KAPPA_MIN else '🔴 DƯỚI NGƯỠNG'} {KAPPA_MIN})", ""]
    lines += ["| tầng | đúng/mẫu | precision | KTC 95% | không đọc được |", "|---|---|---|---|---|"]
    for tang, d in sorted(est["tang"].items()):
        if not d.get("n"):
            continue
        lo, hi = d["ci"]
        lines.append(f"| `{tang}` | {d['k']}/{d['n']} | **{100*d['precision']:.1f}%** "
                     f"| {100*lo:.1f}–{100*hi:.1f}% | {d['unsure']} |")
    lines += ["", "> Mỗi tầng là một DÂN SỐ RIÊNG. **Không gộp ba con số này**: tỷ lệ",
              "> `direct`:`similar` là 12:1 nên trung bình cộng gần như chỉ là của `direct`.",
              "> Muốn con số cho toàn bộ giao nộp thì phải tổ hợp có trọng số theo kích",
              "> thước dân số từng tầng, không phải trung bình của ba tỷ lệ.", ""]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"[phán quyết] -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
