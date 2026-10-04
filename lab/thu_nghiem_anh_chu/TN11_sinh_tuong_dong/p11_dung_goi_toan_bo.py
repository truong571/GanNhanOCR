"""TN11 p11 (04/10) — DỰNG GÓI KAGGLE ĐỦ 10 CUỐN (sinh ảnh glyph theo phong cách TỪNG cuốn) từ dữ liệu MỚI NHẤT. 0 API, CPU, tất định.

Nguồn (đều dựng từ nhãn pipeline lượt 01/10, KHÔNG dùng nhãn người để chọn chữ hay ảnh phong cách; xem w_hien_trang_du_lieu_* và
w_quy_mo_va_ngan_sach_* — hai khâu khảo sát đã sinh các tệp dưới measure_out/_tn11/full/):
  chữ ưu tiên    measure_out/_tn11/full/quy_mo_va_ngan_sach/work/{plans,phong}.pkl  -> S1(k=5) lọc phông, xếp ưu tiên giảm dần, mỗi cuốn một danh sách
  gán phông      cùng nguồn (chuỗi phông + quy tắc PUA + kiểm vẽ thật): font_of_<cuốn>.tsv — gói dùng ĐÚNG phông này cho từng chữ
  ảnh phong cách chọn từ measure_out/_tn11/full/hien_trang_du_lieu/neo_khuyen_nghi/<cuốn>.csv (crop chuẩn vuông 128 của ô neo tự động SẠCH hình học) -> nhị phân Otsu, nền trắng.
                 Mặc định --anchor-purity plain: chỉ ô neo chưa bị chon_chu chạm (không do chon_chu nâng/sửa nhãn). Lý do (phản biện 04/10): chon_chu học trên nhãn người của
                 sách Borg kia nên ô nó nâng lên GOLD không được dùng chọn ảnh phong cách; cùng quy tắc áp cho cả 10 cuốn.
Ra: <out>/ (thư mục gói Kaggle: plan.json, manifest.json, fonts.json, data/<cuốn>/style_*.png, code/, ckpt/, w_runner_khung_chay.py) + <out>.zip
    + <out>_bao_cao.json (số chữ/ảnh/việc, chữ không phông nào có, mã băm kế hoạch, kích thước, ước giờ GPU).

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p11_dung_goi_toan_bo.py [--refresh] [--styles 2] [--out measure_out/_tn11/full/kaggle_pack_<ngày>]
  --refresh  chạy lại khâu quy mô (w_quy_mo_va_ngan_sach_00_chay_het.py, ≈ 105 giây) trước khi dựng — dùng sau mỗi lần chạy lại pipeline.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import w_quy_mo_va_ngan_sach_lib as L  # noqa: E402

FULL = REPO / "measure_out" / "_tn11" / "full"
K_DEFAULT = 5
FONT_KEY = {                                   # khoá phông của khâu quy mô -> tên ngắn trong gói (None = bỏ: trùng tên tệp với NomNaTong chính)
    "NomNaTong-Regular.ttf [font_diffusion]": "NomNaTong",
    "HanaMinB.ttf": "HanaMinB",
    "HAN NOM A.ttf": "HanNomA",
    "NomNaTongLight2.ttf": "NomNaTongLight2",
    "HanaMinA.ttf": "HanaMinA",
    "PlangothicP2-Regular.ttf": "PlangothicP2",
    "PlangothicP1-Regular.ttf": "PlangothicP1",
}


def style_image(g: np.ndarray) -> np.ndarray:
    """== p02.style_image: crop xám -> ảnh phong cách (Otsu: mực đen / nền trắng), ô vuông lề 10 %, 128x128."""
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw < 128).mean() > 0.5:                 # nền tối (hiếm): đảo để mực luôn đen
        bw = 255 - bw
    h, w = bw.shape
    s = int(max(h, w) * 1.2)
    out = np.full((s, s), 255, np.uint8)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = bw
    return cv2.resize(out, (128, 128), interpolation=cv2.INTER_AREA)


def choose_styles(Rk: pd.DataFrame, n: int, plain: bool = True) -> pd.DataFrame:
    """Chọn ảnh phong cách ứng viên — bản chép w_hien_trang_du_lieu_05_tong_hop.py (mỗi chữ 1 ô; rộng/cao trong [0,8; 1,25] x trung vị; mực trong [p25, p75];
    mực lạc <= trung vị; chữ có >= 5 ô neo; xếp theo khoảng cách chuẩn hoá tới trung vị) trên tập neo; plain=True thêm lọc: loại ô do chon_chu chạm vào (bộ chọn đó học trên nhãn người của sách Borg kia)."""
    R = Rk.copy()
    for c in ("w", "h", "ink_pct", "stray_ink"):
        R[c] = pd.to_numeric(R[c], errors="coerce")
    if plain:
        R = R[(R.chon_chu.fillna("").astype(str) == "") & (~R.rule.fillna("").str.contains("chon_chu", regex=False))].copy()   # chưa bị chon_chu chạm (nâng/sửa nhãn)
    cc = R.label.value_counts()
    hm, wm = R.h.median(), R.w.median()
    q1, q3 = R.ink_pct.quantile(0.25), R.ink_pct.quantile(0.75)
    sm = R.stray_ink.median()
    S = R[(R.h.between(0.8 * hm, 1.25 * hm)) & (R.w.between(0.8 * wm, 1.25 * wm)) & (R.ink_pct.between(q1, q3)) & ((R.stray_ink <= sm) | R.stray_ink.isna())
          & (R.label.map(cc) >= 5)].copy()
    iqr = max(q3 - q1, 1e-6)
    S["diem"] = (S.h / hm - 1).abs() + (S.w / wm - 1).abs() + (S.ink_pct - R.ink_pct.median()).abs() / iqr
    S = S.sort_values(["diem", "image"]).drop_duplicates("label").head(n).copy()
    S.insert(0, "hang", range(1, len(S) + 1))
    return S


def qa_style(im: np.ndarray) -> dict:
    ink = (im < 128).astype(np.uint8)
    n_cc, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    big = [i for i in range(1, n_cc) if st[i, cv2.CC_STAT_AREA] >= 12]
    ys, xs = np.nonzero(ink)
    bb = (int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) if len(xs) else (0, 0)
    return dict(muc_pct=round(float(ink.mean()) * 100, 1), thanh_phan=len(big), bbox_w=bb[0], bbox_h=bb[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FULL / f"kaggle_pack_{time.strftime('%Y%m%d')}"))
    ap.add_argument("--styles", type=int, default=2, help="số ảnh phong cách mỗi cuốn đưa vào gói (kế hoạch mặc định chỉ dùng phong cách 0)")
    ap.add_argument("--passes", nargs="+", default=["base:0"])
    ap.add_argument("--shard", type=int, default=128)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--anchor-purity", choices=["plain", "all"], default="plain", help="plain: ảnh phong cách chỉ từ ô neo chưa bị chon_chu chạm (mặc định); all: như bản 20261004 (gồm ô do chon_chu nâng)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} không trống — chọn --out mới (bộ dựng không xoá thư mục có sẵn)")
    if a.refresh:
        subprocess.check_call([sys.executable, str(HERE / "w_quy_mo_va_ngan_sach_00_chay_het.py")], env=dict(__import__("os").environ, PYTHONDONTWRITEBYTECODE="1"))

    pl = pickle.load(open(L.OUT / "work" / "plans.pkl", "rb"))["plans"]
    ph = pickle.load(open(L.OUT / "work" / "phong.pkl", "rb"))
    assign = ph["assign"]
    spec_dir = FULL / "kaggle_spec"
    (spec_dir / "styles").mkdir(parents=True, exist_ok=True)
    spec = dict(fonts={}, books={})
    used_fonts = {}
    rep = dict(ngay=time.strftime("%Y-%m-%d %H:%M"), sach={}, ghi_chu=[])
    for b in L.ORDER:
        F = L.filtered(pl[b]["sach"], lambda c: c in ph["avail_cas"])
        cap = len(F["S1"][K_DEFAULT])
        chars = F["ranked"][:cap]
        assert set(chars) == F["S1"][K_DEFAULT], f"{b}: danh sách ≠ S1(k={K_DEFAULT}) lọc phông"
        (spec_dir / f"chars_{b}.txt").write_text("\n".join(chars) + "\n", encoding="utf-8")
        fo = {}
        with open(spec_dir / f"font_of_{b}.tsv", "w", encoding="utf-8") as f:
            for c in chars:
                k = assign[c]
                name = FONT_KEY.get(k)                 # None: phông trùng tên tệp (NomNaTong ở fonts/) -> chữ thành 'không phông' trong kế hoạch
                if name:
                    used_fonts[name] = L.FONT_FILES[k]
                    fo[c] = name
                f.write(f"{c}\t{name or ''}\n")
        # ảnh phong cách
        sc = FULL / "hien_trang_du_lieu" / "neo_khuyen_nghi" / f"{b}.csv"
        if not sc.exists():
            raise SystemExit(f"thiếu {sc}")
        Rk = pd.read_csv(sc, dtype=str, keep_default_na=False)
        Rk = Rk[(Rk.tier == "GOLD")]
        S = choose_styles(Rk, max(a.styles, 2), plain=(a.anchor_purity == "plain"))
        if len(S) < a.styles:
            raise SystemExit(f"{b}: chỉ chọn được {len(S)} ảnh phong cách")
        pngs, qa = [], []
        for j, r in enumerate(S.head(a.styles).itertuples()):
            src = REPO / str(r.crop_chuan_cache_128)
            g = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise SystemExit(f"{b}: không đọc được {src}")
            im = style_image(g)
            p = spec_dir / "styles" / f"{b}_s{j}.png"
            cv2.imwrite(str(p), im)
            pngs.append(str(p)); qa.append(dict(chu=r.label, nguon=str(r.crop_chuan_cache_128), rule=str(r.rule), chon_chu=str(r.chon_chu), **qa_style(im)))
        spec["books"][b] = dict(chars_file=str(spec_dir / f"chars_{b}.txt"), font_of_file=str(spec_dir / f"font_of_{b}.tsv"), style_pngs=pngs)
        rep["sach"][b] = dict(so_chu_danh_sach=len(chars), phong={n: sum(1 for c in chars if fo.get(c) == n) for n in sorted(set(fo.values()))},
                              khong_phong=sum(1 for c in chars if c not in fo), phong_cach=qa)
    spec["fonts"] = {n: p for n, p in used_fonts.items()}
    (spec_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")

    cmd = [sys.executable, str(HERE / "w_runner_khung_chay.py"), "pack", "--spec", str(spec_dir / "spec.json"), "--out", str(out),
           "--passes", *a.passes, "--shard", str(a.shard)]
    print("[p11]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    n_img = sum(len(v["chars"]) for v in plan["books"].values()) * len(a.passes)
    rep.update(plan_sha=plan["sha"], n_viec=len(plan["tasks"]), n_anh=int(n_img), gio_gpu_uoc=round(n_img * 2.7 / 3600, 1), gio_phien_2gpu_uoc=round(n_img * 2.7 / 3600 / 2, 1),
               khong_phong_tong=sum(len(v["uncovered"]) for v in plan["books"].values()),
               theo_sach_ke_hoach={b: dict(chu=len(v["chars"]), khong_phong=len(v["uncovered"])) for b, v in plan["books"].items()}, ckpt_sha256=man["ckpt_sha256"])
    # zip một gốc (Kaggle tự giải nén zip tải lên); safetensors/png không nén thêm
    zp = Path(str(out) + ".zip")
    with zipfile.ZipFile(zp, "w") as z:
        for f in sorted(out.rglob("*")):
            if f.is_file() and f.name != ".DS_Store":
                z.write(f, f.relative_to(out).as_posix(), compress_type=zipfile.ZIP_STORED if f.suffix in (".safetensors", ".png") else zipfile.ZIP_DEFLATED)
    rep["zip"] = dict(tep=str(zp), MB=round(zp.stat().st_size / 1e6, 1))
    Path(str(out) + "_bao_cao.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[p11] gói: {out} | {rep['n_viec']} việc | {rep['n_anh']} ảnh | ≈ {rep['gio_gpu_uoc']} giờ GPU = {rep['gio_phien_2gpu_uoc']} giờ phiên (2 GPU) | "
          f"chữ không phông: {rep['khong_phong_tong']} | zip {rep['zip']['MB']} MB | sha {plan['sha'][:12]}")


if __name__ == "__main__":
    main()
