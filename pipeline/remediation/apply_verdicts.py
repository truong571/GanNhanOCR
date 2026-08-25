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
        --out dataset_out/labels_final.csv --batch dataset_out/human_audit/audit_combined
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KAPPA_MIN = 0.60
# Tệp KHAI XUẤT XỨ bắt buộc, đặt cạnh tệp verdict.
PROV = "NGUOI_CHAM.md"

# --- HAI ĐƯỜNG NẠP PHÁN QUYẾT -------------------------------------------------
# (A) ĐỘI NGOÀI chấm từ re-dataset/ — đường CHÍNH.
#     Họ nhận cả thư mục re-dataset/ (ảnh + labels.csv) và trả về MỘT tệp phẳng
#     `verdicts.csv` neo theo cột `image`. Họ tự làm giao diện; ta không áp đặt.
# (B) MẺ MẪU do make_combined_batch dựng — còn giữ, dùng khi chỉ chấm mẫu để ƯỚC LƯỢNG
#     precision có khoảng tin cậy (có manifest, stratum, design_weight).
#
# `image` là khoá đúng cho (A): duy nhất, có thật trong thư mục, người nhìn thấy được.
VERDICT_CSV = "verdicts.csv"
HOP_LE = {"dung", "sai", "khong_doc_duoc"}          # tiếng Việt, không dịch qua lại
MAP_VE_CHUAN = {"dung": "correct", "sai": "wrong_label", "khong_doc_duoc": "unsure",
                "correct": "correct", "wrong_label": "wrong_label", "unsure": "unsure"}
VERDICT_COL = "human_verdict"
NA = dict(keep_default_na=False, na_values=[""])


def doc_verdict_phang(f: Path) -> dict:
    """Đọc verdicts.csv của ĐỘI NGOÀI — neo theo cột `image`.

    Chấp nhận cả nhãn tiếng Việt (dung/sai/khong_doc_duoc) lẫn tên chuẩn nội bộ.
    Dòng sai định dạng là LỖI THẬT, không bỏ qua im lặng — một verdict đọc nhầm là một
    ô nhãn bị sửa sai, và ta sẽ không bao giờ biết.
    """
    import csv as _csv
    out, loi = {}, []
    with open(f, encoding="utf-8-sig", newline="") as fh:
        rd = _csv.DictReader(fh)
        cols = {c.strip().lower() for c in (rd.fieldnames or [])}
        for c in ("image", "verdict"):
            if c not in cols:
                raise SystemExit(f"{f.name}: THIẾU cột bắt buộc `{c}`. "
                                 f"Cột đang có: {sorted(cols)}")
        for i, r in enumerate(rd, 2):
            img = (r.get("image") or "").strip()
            v = (r.get("verdict") or "").strip().lower()
            if not img:
                continue
            if v not in MAP_VE_CHUAN:
                loi.append(f"    dòng {i}: verdict={v!r} không hợp lệ "
                           f"(phải là: {', '.join(sorted(HOP_LE))})")
                continue
            out[img] = MAP_VE_CHUAN[v]
    if loi:
        raise SystemExit(f"{f.name}: {len(loi)} dòng sai định dạng:\n" + "\n".join(loi[:10]))
    return out


def ap_dung_phang(labels_csv: Path, out_csv: Path, vmap: dict) -> dict:
    """Áp verdict phẳng vào bộ nhãn, và ĐỐI CHIẾU khoá — không khớp là LỖI THẬT."""
    import pandas as pd
    df = pd.read_csv(labels_csv, dtype={"image_md5": str}, **NA)
    if VERDICT_COL not in df.columns:
        df[VERDICT_COL] = ""
    co = set(df["image"].astype(str))
    la = [k for k in vmap if k not in co]
    if la:
        raise SystemExit(
            f"{len(la)} `image` trong verdict KHÔNG có trong bộ nhãn — gần như chắc chắn "
            f"đội chấm dùng bản re-dataset CŨ. Ví dụ: {la[:3]}\n"
            f"    Bộ nhãn hiện tại có {len(co):,} ảnh. Đối chiếu lại rồi chấm trên bản mới.")
    df[VERDICT_COL] = df["image"].astype(str).map(vmap).fillna("")
    demote = (df[VERDICT_COL] == "wrong_label") & df["tier"].isin(["GOLD", "SYLLABLE"])
    # GIỮ LUẬT GỐC. Nếu không, ô bị hạ sẽ mất dấu vết luật nào sinh ra nó, và bảng
    # precision theo luật sẽ báo 100% cho chính luật vừa gây ra lỗi — đúng lớp "con số
    # nói dối" mà dự án này đã dính nhiều lần.
    if "rule_goc" not in df.columns:
        df["rule_goc"] = ""
    df.loc[demote, "rule_goc"] = df.loc[demote, "rule"]
    df.loc[demote, "tier_goc"] = df.loc[demote, "tier"]
    df.loc[demote, "rule"] = "human_verdict:wrong_label"
    df.loc[demote, "tier"] = "REVIEW"
    df.to_csv(out_csv, index=False)
    return {"gan_verdict": int((df[VERDICT_COL] != "").sum()),
            "ha_xuong_review": int(demote.sum()),
            "phu": len(vmap) / max(len(co), 1)}


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


def bang_theo_luat(labels: Path, report: Path, conf: float, st: dict) -> None:
    """Precision theo TỪNG LUẬT — không có stratum/design_weight nên KHÔNG suy rộng được.

    Đường (A) là chấm TRỰC TIẾP trên bộ giao nộp, không phải mẫu xác suất. Nếu đội chấm
    HẾT thì đây là TỔNG ĐIỀU TRA, con số là chính xác chứ không cần khoảng tin cậy. Nếu
    họ chấm một phần thì con số CHỈ đúng cho phần đã chấm — và ta KHÔNG suy rộng, vì
    không biết phần đó có đại diện hay không (họ chọn theo tiêu chí gì, ta không nắm).
    """
    import csv as _csv, collections
    sys.path.insert(0, str(REPO))
    from pipeline.ground_truth import stats
    rows = list(_csv.DictReader(open(labels, encoding="utf-8")))
    agg = collections.defaultdict(lambda: {"dung": 0, "sai": 0, "khong_doc": 0})
    for r in rows:
        v = r.get(VERDICT_COL, "")
        if not v:
            continue
        # Ô bị hạ vì verdict sai đã đổi rule -> LẤY LẠI luật gốc, để lỗi được tính vào
        # ĐÚNG luật đã sinh ra nó. Không làm vậy thì luật gây lỗi lại báo 100%.
        rule = (r.get("rule_goc") or "").strip() or r.get("rule", "")
        tier = (r.get("tier_goc") or "").strip() or r.get("tier", "")
        key = ("GOLD/similar" if rule == "s1_inter_s2_similar" else
               "GOLD/direct" if rule == "s1_inter_s2_direct" else
               "SYLLABLE" if tier == "SYLLABLE" else rule)
        d = agg[key]
        d["dung" if v == "correct" else "sai" if v == "wrong_label" else "khong_doc"] += 1

    L = ["# Bảng precision — phán quyết của NGƯỜI (chấm trực tiếp trên bộ giao nộp)", "",
         f"Đã chấm **{st['gan_verdict']:,}** ô = **{100*st['phu']:.1f}%** bộ giao nộp.", ""]
    if st["phu"] < 0.999:
        L += ["> ⚠️ **Chưa chấm hết, nên KHÔNG suy rộng ra toàn bộ.** Con số dưới đây chỉ",
              "> đúng cho phần ĐÃ chấm. Muốn con số cho toàn bộ thì hoặc chấm hết, hoặc",
              "> chấm một MẪU NGẪU NHIÊN PHÂN TẦNG (`make_combined_batch --by-rule`) —",
              "> khi đó mới có khoảng tin cậy diễn giải được.", ""]
    else:
        L += ["> Đã chấm HẾT, nên đây là **tổng điều tra**: con số là chính xác, không",
              "> phải ước lượng.", ""]
    L += ["| luật | đúng | sai | không đọc được | precision | KTC 95% |", "|---|---|---|---|---|---|"]
    for k, d in sorted(agg.items()):
        n = d["dung"] + d["sai"]
        if not n:
            L.append(f"| `{k}` | {d['dung']} | {d['sai']} | {d['khong_doc']} | — | — |")
            continue
        lo, hi = stats.clopper_pearson_ci(d["dung"], n, conf)
        L.append(f"| `{k}` | {d['dung']} | {d['sai']} | {d['khong_doc']} "
                 f"| **{100*d['dung']/n:.1f}%** | {100*lo:.1f}–{100*hi:.1f}% |")
    L += ["", "> Ô *không đọc được* bị loại khỏi MẪU SỐ, KHÔNG tính là lỗi.",
          "> Mỗi luật là một dân số riêng — **đừng gộp**: tỷ lệ `direct`:`similar` là 12:1.", ""]
    report.write_text("\n".join(L), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.remediation.apply_verdicts")
    ap.add_argument("--in", dest="src", default=str(REPO / "dataset_out" / "labels_final.csv"))
    ap.add_argument("--out", dest="dst", default=str(REPO / "dataset_out" / "labels_final.csv"))
    ap.add_argument("--batch", default=str(REPO / "dataset_out" / "human_audit" / "audit_combined"))
    ap.add_argument("--report", default=str(REPO / "docs" / "BANG_PRECISION.md"))
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--redataset", default=str(REPO / "re-dataset"),
                    help="thư mục ĐỘI NGOÀI chấm; tìm verdicts.csv + NGUOI_CHAM.md ở đây")
    ap.add_argument("--allow-low-kappa", action="store_true",
                    help="công bố precision DÙ κ liên-người thấp — chỉ dùng khi biết rõ mình làm gì")
    args = ap.parse_args(argv)

    batch = Path(args.batch)
    # ---- ĐƯỜNG (A): tệp phẳng từ ĐỘI NGOÀI chấm trên re-dataset/ ------------------
    phang = Path(args.redataset) / VERDICT_CSV
    if phang.exists():
        prov = phang.parent / PROV
        if not prov.exists() or "⬜" in prov.read_text(encoding="utf-8"):
            print(f"[phán quyết] 🔴 TỪ CHỐI: có {phang.name} nhưng {PROV} thiếu hoặc chưa "
                  f"điền xong.\n    Có tệp verdict KHÔNG chứng minh được là NGƯỜI chấm. Dự "
                  f"án này đã một lần tin nhầm verdict MÁY là verdict người và phải huỷ "
                  f"toàn bộ số liệu.\n    Mẫu khai: docs/QUY_TRINH_CHAM_TAY.md",
                  file=sys.stderr)
            return 1
        vmap = doc_verdict_phang(phang)
        print(f"[phán quyết] đọc {len(vmap):,} verdict từ {phang}")
        st = ap_dung_phang(Path(args.src), Path(args.dst), vmap)
        print(f"[phán quyết] gắn cho {st['gan_verdict']:,} ô "
              f"({100*st['phu']:.1f}% bộ giao nộp) | HẠ xuống REVIEW {st['ha_xuong_review']:,} ô")
        bang_theo_luat(Path(args.dst), Path(args.report), args.conf, st)
        print(f"[phán quyết] -> {args.report}")
        return 0

    vf = list(batch.glob("verdicts*.jsonl")) if batch.is_dir() else []
    if not vf:
        print(f"[phán quyết] chưa có {phang} lẫn {batch}/verdicts*.jsonl — BỎ QUA bước này")
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
