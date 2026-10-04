"""TN11 "full" — hướng RUNNER/KAGGLE: đo bằng SCRIPT các ràng buộc và độ bền của runner Kaggle hiện tại (0 GPU, 0 API, chỉ CPU).

Không sửa tệp có sẵn. Đọc (chỉ đọc):
  · log chạy thật  measure_out/_tn11/kaggle/TN11/log_B34.txt, log_B18.txt   (thời gian/bước, giây/ảnh, cảnh báo use_fst)
  · gói gửi lên    measure_out/_tn11/kaggle/tn11_kaggle_data.zip           (manifest, chars.txt, kích thước ckpt)
  · kết quả về     measure_out/_tn11/kaggle/TN11/tn11_ket_qua.zip          (kích thước PNG, hiệu quả nén)
  · plan.pkl B34/B18 (danh sách chữ cần sinh), pairs.npz (CHỈ cột nhãn chữ L, không ảnh)
  · font_diffusion/fonts, fonts/  (bảng cmap bằng fontTools); header safetensors của ckpt (đếm tham số)
  · cột `syllable` của nhãn TỰ ĐỘNG hiện tại (labels_gated.csv / dataset/<Bộ>/labels.csv) -> hợp ứng viên R(âm) từng cuốn.
    KHÔNG đọc nhãn người; việc chọn chữ ở đây không dựa vào nhãn người.
Ghi duy nhất: measure_out/_tn11/full/runner/ngan_sach.json

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_runner_ngan_sach.py
"""
from __future__ import annotations

import io
import json
import math
import re
import struct
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
T11 = REPO / "measure_out" / "_tn11"
KG = T11 / "kaggle"
KRES = KG / "TN11" / "tn11_ket_qua.zip"
OUT = T11 / "full" / "runner"
KAGGLE_PY = REPO / "lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/kaggle/tn11_kaggle.py"
WRAP = REPO / "core/ranking/fontdiffusion_gen.py"
FD = REPO / "font_diffusion"
ORDER = ["stt2", "stt4", "stt11", "Chr", "L83", "KVK", "L16", "TK", "B18", "B34"]
LABELS = {  # chỉ để lấy cột syllable (nhãn TỰ ĐỘNG); STT lấy từ dataset_out/labels_final.csv (mọi ô, lọc theo cột book) như TN8 (tn8lib.BOOKS)
    "stt2": "dataset_out/labels_final.csv", "stt4": "dataset_out/labels_final.csv",
    "stt11": "dataset_out/labels_final.csv",
    "Chr": "prepared/Chrestomathie1872/dataset_out/labels_gated.csv", "L83": "prepared/LucVanTien1883/dataset_out/labels_gated.csv",
    "KVK": "prepared/KimVanKieu1884/dataset_out/labels_gated.csv", "L16": "prepared/LucVanTien1916/dataset_out/labels_gated.csv",
    "TK": "prepared/TruyenKieu1872/dataset_out/labels_gated.csv",
    "B18": "prepared/_auto/SachKinhThayCaBinh/dataset_out/labels_gated.csv",
    "B34": "prepared/_auto/SachDungLyHoThan/dataset_out/labels_gated.csv"}
# chuỗi phông dự phòng (thứ tự ưu tiên: gần nét Nôm trước, phông Hán tổng quát sau) — các tệp CÓ SẴN trong repo
FONT_CHAIN = [("NomNaTong", "font_diffusion/fonts/NomNaTong-Regular.ttf"), ("HanNomA", "font_diffusion/fonts/HAN NOM A.ttf"),
              ("HanNomB", "font_diffusion/fonts/HAN NOM B.ttf"), ("HanNomKhai", "font_diffusion/fonts/Han-Nom-Khai-Regular-300623.ttf"),
              ("HanNomMinh", "font_diffusion/fonts/Han-nom Minh 1.42.otf"), ("HanaMinA", "font_diffusion/fonts/HanaMinA.ttf"),
              ("HanaMinB", "font_diffusion/fonts/HanaMinB.ttf"), ("HanaMinC", "font_diffusion/fonts/HanaMinC.otf"),
              ("PlangothicP1", "fonts/PlangothicP1-Regular.ttf"), ("PlangothicP2", "fonts/PlangothicP2-Regular.ttf")]
GIB = 1024 ** 3


def find_lines(path: Path, patterns: dict) -> dict:
    """Bằng chứng mã: số dòng đầu tiên khớp mỗi mẫu (sinh bằng script, không gõ tay)."""
    L = path.read_text(encoding="utf-8").split("\n")
    out = {}
    for k, pat in patterns.items():
        rx = re.compile(pat)
        hit = next(((i + 1, s.strip()[:140]) for i, s in enumerate(L) if rx.search(s)), None)
        out[k] = dict(file=str(path.relative_to(REPO)), line=hit[0], text=hit[1]) if hit else None
    return out


# ------------------------------------------------------------------------------------------------ A. log chạy thật
def parse_log(p: Path, book: str) -> dict:
    t = p.read_text(encoding="utf-8", errors="replace")
    steps = [(int(a), float(b), int(c)) for a, b, c in re.findall(rf"\[{book}\] bước (\d+): loss ([\d.]+) \[(\d+)s\]", t)]
    gen = [(int(n), float(s)) for n, s, _ in re.findall(r"FontDiffusion: (\d+) images in ([\d.]+)s \(([\d.]+)s/img\)", t)]
    passes = [(m, int(j), int(n), int(s)) for m, j, n, s in re.findall(rf"\[{book}\] sinh (\S+) phong cách (\d+): (\d+) ảnh \[(\d+)s\]", t)]
    load = [float(x) for x in re.findall(r"FontDiffusion loaded in ([\d.]+)s", t)]
    miss = sorted(set(re.findall(r"Checkpoint for '(\w+)' not found", t)))
    d100 = np.diff([0] + [s[2] for s in steps])
    return dict(
        book=book, n_log_lines=t.count("\n"), steps_logged=len(steps), last_step=steps[-1][0], train_seconds=steps[-1][2],
        s_per_step=round(steps[-1][2] / steps[-1][0], 3), s_per_100steps_median=float(np.median(d100)), s_per_100steps_range=[int(d100.min()), int(d100.max())],
        loss_first_last=[round(steps[0][1], 4), round(steps[-1][1], 4)],
        gen_passes=len(gen), gen_images=sum(n for n, _ in gen), gen_seconds=round(sum(s for _, s in gen), 1),
        s_per_img=round(sum(s for _, s in gen) / sum(n for n, _ in gen), 3), s_per_img_range=[round(s / n, 2) for n, s in (min(gen, key=lambda x: x[1] / x[0]), max(gen, key=lambda x: x[1] / x[0]))],
        pass_table=[dict(model=m, style=j, images=n, seconds=s) for m, j, n, s in passes], model_load_seconds=load,
        fst_evidence=dict(
            loading_line_original_wrapper=t.count("Loading FontDiffusion pipeline on cuda"),
            loading_line_fixed_wrapper=t.count("use_fst=False, đường chuẩn"),
            model_class_with_fst_lines=len(re.findall(r"FontDiffuserModelDPMWithFST", t)),
            fst_modules_built_lines=t.count("FST module built successfully"),
            missing_checkpoint_components=miss,
            missing_checkpoint_warning_lines=len(re.findall(r"Checkpoint for '\w+' not found", t))),
        total_wall_seconds_est=steps[-1][2] + sum(s for *_, s in passes) + (load[0] if load else 0))


# ------------------------------------------------------------------------------------------------ B. phông / cmap
def cmap_set(p: Path) -> set:
    from fontTools.ttLib import TTFont
    f = TTFont(str(p), lazy=True)
    s = set()
    for st in f["cmap"].tables:
        s |= set(st.cmap.keys())
    f.close()
    return s


def time_is_char_in_font(p: Path, n=60) -> float:
    """Bản sao logic font_diffusion/src/tools/utils.py:104-110 (mở lại tệp phông MỖI chữ) — đo ms/chữ."""
    from fontTools.ttLib import TTFont
    chars = [chr(0x4E00 + i * 97) for i in range(n)]
    t0 = time.time()
    for c in chars:
        f = TTFont(str(p))
        any(ord(c) in st.cmap for st in f["cmap"].tables)
    return (time.time() - t0) / n * 1000


def lru_thrash(n_chars: int, maxsize=1024, passes=2) -> dict:
    """Mô phỏng lru_cache(maxsize=1024) của FontManager.is_char_in_font khi quét tuần tự n_chars chữ phân biệt, lặp `passes` lượt."""
    calls = {"n": 0}

    @lru_cache(maxsize=maxsize)
    def f(c):
        calls["n"] += 1
        return True
    for _ in range(passes):
        for i in range(n_chars):
            f(i)
    tot = n_chars * passes
    return dict(n_chars=n_chars, lookups=tot, recomputed=calls["n"], hit_rate=round(1 - calls["n"] / tot, 4))


# ------------------------------------------------------------------------------------------------ C. kích thước dữ liệu
def safetensors_bytes(p: Path) -> tuple[int, int]:
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    numel = sum(int(np.prod(v["shape"])) if v["shape"] else 1 for v in h.values())
    nbytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in h.values())
    return numel, nbytes


def universe_sizes() -> dict:
    """Hợp R(âm) (từ điển Dict/QuocNgu_SinoNom.csv) của mọi âm tiết có mặt trong nhãn TỰ ĐỘNG của cuốn — cận trên 'danh sách ứng viên đầy đủ'."""
    import pandas as pd
    from pipeline.gold_exact import common as C
    res = {}
    for b in ORDER:
        D = pd.read_csv(REPO / LABELS[b], dtype=str, keep_default_na=False, usecols=["syllable"] + (["book"] if b.startswith("stt") else []))
        if b.startswith("stt"):
            D = D[D.book == b]
        syl = sorted({s.strip().lower() for s in D.syllable if s.strip()})
        U = set()
        for s in syl:
            U |= C.R_of(s)
        res[b] = dict(rows=len(D), syllables=len(syl), union_R=len(U), chars=sorted(U))
    return res


def budget(name, per_book_n: dict, passes: int, ft_steps: int, ft_books: int, s_img: float, s_step: float, cmap_ms: float,
           gpus=2, session_h=11.0, quota_h=30.0, load_s=21.0) -> dict:
    """Mô hình thời gian: giây GPU = sinh (ảnh × lượt × s/ảnh) + tinh chỉnh (bước × s/bước) + nạp mô hình + lọc cmap (CPU, GPU rảnh)."""
    gen = sum(per_book_n.values()) * passes * s_img
    ft = ft_steps * s_step * ft_books
    cmap = sum(per_book_n.values()) * passes * cmap_ms / 1000
    ovh = load_s * len(per_book_n)
    gpu_s = gen + ft + cmap + ovh
    # (a) hàng đợi động theo lô nhỏ: ~ gpu_s / gpus ; (b) gán tĩnh theo cuốn (k % gpus) như tn11_kaggle.py:245
    per_book_s = {b: n * passes * s_img + (ft_steps * s_step if i < ft_books else 0) + n * passes * cmap_ms / 1000 + load_s
                  for i, (b, n) in enumerate(per_book_n.items())}
    load = [0.0] * gpus
    for k, b in enumerate(per_book_n):
        load[k % gpus] += per_book_s[b]
    wall_dyn = gpu_s / gpus / 3600
    wall_static = max(load) / 3600
    n_img = sum(per_book_n.values()) * passes
    return dict(kich_ban=name, anh_can_sinh=int(n_img), gio_GPU=round(gpu_s / 3600, 1), gio_dong_ho_T4x2_dong=round(wall_dyn, 1),
                gio_dong_ho_T4x2_tinh=round(wall_static, 1), lech_tinh_so_voi_dong=round(wall_static / max(wall_dyn, 1e-9), 3),
                so_ca_toi_thieu=int(math.ceil(wall_dyn / session_h)), so_tuan_han_muc=round(wall_dyn / quota_h, 2),
                ft_gio_GPU=round(ft / 3600, 1), gen_gio_GPU=round(gen / 3600, 1), cmap_gio_CPU_GPU_ranh=round(cmap / 3600, 2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = {"tao": time.strftime("%Y-%m-%d %H:%M:%S"), "luu_y": "mọi con số do script này tính; GPU/Kaggle KHÔNG chạy; tốc độ lấy từ log chạy thật 04/10 (T4, fp32, lô 32)"}

    # ---- A. log
    logd = KG / "TN11"
    A = {b: parse_log(logd / f"log_{b}.txt", b) for b in ("B34", "B18")}
    R["A_log_chay_that"] = A
    s_img = float(np.mean([A[b]["s_per_img"] for b in A]))
    s_step = float(np.mean([A[b]["s_per_step"] for b in A]))
    R["A_toc_do_chon"] = dict(s_per_img=round(s_img, 3), s_per_step=round(s_step, 3),
                              anh_moi_gio_moi_GPU=round(3600 / s_img, 0), anh_moi_ca_11h_2GPU=int(11 * 3600 * 2 / s_img), anh_moi_tuan_30h_2GPU=int(30 * 3600 * 2 / s_img))

    # ---- B. manifest gói + cmap
    with zipfile.ZipFile(KG / "tn11_kaggle_data.zip") as z:
        man = json.loads(z.read("tn11_manifest.json"))
        info = z.infolist()
        pack = dict(files=len(info), uncompressed_MB=round(sum(i.file_size for i in info) / 1e6, 1), zip_MB=round((KG / "tn11_kaggle_data.zip").stat().st_size / 1e6, 1),
                    top_level_entries=len({i.filename.split("/")[0] for i in info}),
                    has_fd_wrapper_fix=any(i.filename.endswith("fd_wrapper_fix.py") for i in info),
                    csv_or_label_files=[i.filename for i in info if re.search(r"\.(csv|tsv)$|label|human|truth", i.filename, re.I)])
        chars_txt = {b: len(z.read(f"data/{b}/chars.txt").decode("utf-8").split("\n")) for b in man["books"]}
        Ls = {}
        for b in man["books"]:
            Z = np.load(io.BytesIO(z.read(f"data/{b}/pairs.npz")))
            Ls[b] = Z["L"]
    R["B_goi_kaggle_hien_tai"] = dict(manifest_books=man["books"], chars_txt_lines=chars_txt, pack=pack)

    cm = {k: cmap_set(REPO / p) for k, p in FONT_CHAIN if (REPO / p).exists()}
    import pandas as pd
    miss_rows = {}
    for b in ("B34", "B18"):
        P = pd.read_pickle(T11 / f"p02_{b}" / "plan.pkl")
        ch = P["chars"]
        nn = sum(ord(c) in cm["NomNaTong"] for c in ch)
        logged = A[b]["pass_table"][0]["images"]
        cum, seen = {}, set()
        for k in cm:
            seen |= {c for c in ch if ord(c) in cm[k]}
            cum[k] = len(seen)
        miss_rows[b] = dict(chars_requested=len(ch), in_NomNaTong_cmap=nn, images_in_real_log=logged, khop_voi_log=(nn == logged),
                            silently_dropped=len(ch) - logged, dropped_pct=round((len(ch) - logged) / len(ch) * 100, 1), cumulative_coverage_by_chain=cum,
                            uncovered_by_all_fonts=len(ch) - len(seen))
    R["B_chu_bi_loai_am_tham"] = miss_rows
    R["B_cmap_ms_moi_chu"] = {k: round(time_is_char_in_font(REPO / p), 1) for k, p in FONT_CHAIN[:1] + [FONT_CHAIN[6]]}
    R["B_lru_cache_1024_quet_tuan_tu"] = [lru_thrash(n) for n in (800, 1019, 1025, 5000, 25000)]

    # ---- C. hợp ứng viên từng cuốn + phủ phông
    U = universe_sizes()
    cov = {}
    for b in ORDER:
        ch = U[b]["chars"]
        row = dict(rows=U[b]["rows"], syllables=U[b]["syllables"], union_R=U[b]["union_R"])
        row["in_NomNaTong"] = sum(ord(c) in cm["NomNaTong"] for c in ch)
        seen, cum = set(), {}
        for k in cm:
            seen |= {c for c in ch if ord(c) in cm[k]}
            cum[k] = len(seen)
        row["in_chain"] = len(seen)
        row["pct_NomNaTong"] = round(row["in_NomNaTong"] / row["union_R"] * 100, 1)
        row["pct_chain"] = round(row["in_chain"] / row["union_R"] * 100, 1)
        cov[b] = row
    R["C_hop_ung_vien_tung_cuon"] = cov
    R["C_tong"] = dict(union_R_cong_don=sum(r["union_R"] for r in cov.values()), nomnatong_cong_don=sum(r["in_NomNaTong"] for r in cov.values()),
                       chain_cong_don=sum(r["in_chain"] for r in cov.values()))

    # ---- D. đĩa / RAM
    ck = FD / "ckpt" / "PROD"
    nb = {k: safetensors_bytes(ck / f"{k}.safetensors") for k in ("unet", "style_encoder", "content_encoder")}
    train_state_B = nb["unet"][1] + nb["style_encoder"][1]            # state_dict unet+style_encoder (fp32)
    last_pt_B = train_state_B * 3                                      # + AdamW exp_avg + exp_avg_sq (2 × tham số huấn luyện)
    ft_pt_real = (T11 / "p04_B34" / "ft.pt").stat().st_size if (T11 / "p04_B34" / "ft.pt").exists() else None
    per_book_disk = 2 * train_state_B + last_pt_B                      # save_at [1000, 3000] + last.pt
    with zipfile.ZipFile(KRES) as zr:
        zi = zr.infolist()
        png = [i for i in zi if i.filename.lower().endswith(".png")]
        raw, comp = sum(i.file_size for i in png), sum(i.compress_size for i in png)
        zip_bytes = KRES.stat().st_size
        sample = [zr.read(i) for i in png[:600]]
    t0 = time.time()
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zz:
        for k, b in enumerate(sample):
            zz.writestr(f"{k}.png", b)
    t_defl = time.time() - t0
    raw_s = sum(len(b) for b in sample)
    R["D_dia"] = dict(
        params_train_MB=round(train_state_B / 1e6, 1), ft_pt_do_tham_so_MB=round(train_state_B / 1e6, 1), ft_pt_that_B34_MB=None if ft_pt_real is None else round(ft_pt_real / 1e6, 1),
        last_pt_MB=round(last_pt_B / 1e6, 1), moi_cuon_MB=round(per_book_disk / 1e6, 0), muoi_cuon_GB=round(per_book_disk * 10 / 1e9, 1),
        code_ckpt_goi_MB=pack["uncompressed_MB"], gioi_han_working_GiB=20, gioi_han_working_GB=round(20 * GIB / 1e9, 2),
        anh_PNG_trung_binh_KB=round(raw / len(png) / 1e3, 1), png_files=len(png), png_MB=round(raw / 1e6, 1), zip_MB=round(zip_bytes / 1e6, 1),
        zip_deflate_ratio=round(comp / raw, 4), zip_lon_hon_png_thuan_pct=round((zip_bytes / raw - 1) * 100, 1),
        deflate_giay_moi_MB_trong_RAM_may_nay=round(t_defl / (raw_s / 1e6), 3))
    # ước tính tổng đĩa /kaggle/working cho 10 cuốn theo công thức hiện tại (ft trong working + PNG + zip PNG nhân đôi)
    def work_gb(n_img):
        png_b = n_img * raw / len(png)
        return (per_book_disk * 10 + pack["uncompressed_MB"] * 1e6 + png_b + png_b * 1.0) / 1e9
    R["D_dia_10_cuon_cong_thuc_hien_tai"] = {f"{n}_anh": round(work_gb(n), 1) for n in (7390, 50000, 100000, 224000)}

    px = 3 * 96 * 96 * 4                                                # tgt float32 3×96×96 / cặp (tn11_kaggle.py:123; style_image_size=96)
    R["D_ram_tgt_float32"] = dict(byte_moi_cap=px, **{f"N{n}_GB_moi_tien_trinh": round(n * px / 1e9, 2) for n in (5461, 14988, 30000, 60000, 100000)},
                                  ram_T4x2_GB=29, ghi_chu="2 tiến trình song song (mỗi GPU một) -> nhân 2")

    # ---- E. vòng lặp chọn ảnh phong cách (tn11_kaggle.py:146-153)
    E = {}
    rng = np.random.default_rng(0)
    for b, L in Ls.items():
        u, inv, cnt = np.unique(L, return_inverse=True, return_counts=True)
        share = cnt / cnt.sum()
        exp_tries = float(np.sum(share / (1 - share))) if len(u) > 1 else float("inf")
        others = np.arange(len(L))
        tries = []
        t0 = time.time()
        for _ in range(4000):
            i = int(rng.integers(0, len(L)))
            k = 0
            while True:
                k += 1
                j = int(rng.choice(others))
                if L[j] != L[i]:
                    break
            tries.append(k)
        dt = (time.time() - t0) / 4000
        E[b] = dict(pairs=int(len(L)), classes=int(len(u)), max_class_share=round(float(share.max()), 4), expected_tries=round(exp_tries, 3),
                    mean_tries_measured=round(float(np.mean(tries)), 3), p999_tries=int(np.percentile(tries, 99.9)), max_tries=int(max(tries)),
                    us_per_sample=round(dt * 1e6, 1))
    # O(n) của bản p04 local (np.nonzero(L != L[i])[0][:5000]) trên L ghép lặp tới N
    base = Ls["B18"]
    On = {}
    for n in (15000, 60000, 120000, 240000):
        L = np.resize(base, n)
        t0 = time.time()
        for _ in range(200):
            i = int(rng.integers(0, n))
            rng.choice(np.nonzero(L != L[i])[0][:5000])
        On[n] = round((time.time() - t0) / 200 * 1e6, 1)
    # O(1) của runner Kaggle trên cùng N
    O1 = {}
    for n in (15000, 60000, 120000, 240000):
        L = np.resize(base, n)
        others = np.arange(n)
        t0 = time.time()
        for _ in range(2000):
            i = int(rng.integers(0, n))
            while True:
                j = int(rng.choice(others))
                if L[j] != L[i]:
                    break
        O1[n] = round((time.time() - t0) / 2000 * 1e6, 1)
    one_class = dict(classes=1, cap=100000, loops_before_cap="vô hạn (while True không có chặn)")
    R["E_vong_lap_phong_cach"] = dict(thuc_te=E, us_moi_mau_ban_p04_local_On=On, us_moi_mau_runner_kaggle_O1=O1,
                                      mau_moi_buoc=16, ghi_chu="16 mẫu/bước (lô 8 × tích luỹ 2); bước thật 3,44 s => vòng lặp < 0,1 % thời gian bước", truong_hop_xau=one_class,
                                      cong_thuc_ky_vong="E[lần thử] = Σ_l share_l / (1 − share_l); share_l → 1 thì tiến tới vô hạn")

    # ---- F. kịch bản ngân sách
    cmap_ms = R["B_cmap_ms_moi_chu"]["NomNaTong"]
    full = {b: cov[b]["union_R"] for b in ORDER}
    full_nn = {b: cov[b]["in_NomNaTong"] for b in ORDER}
    full_chain = {b: cov[b]["in_chain"] for b in ORDER}
    flat = lambda n: {b: n for b in ORDER}
    sc = [
        budget("A. công thức cũ cho 10 cuốn: ~1.000 chữ/cuốn, 5 lượt sinh, tinh chỉnh 3000 bước", flat(1000), 5, 3000, 10, s_img, s_step, cmap_ms),
        budget("B. chỉ sinh (mô hình gốc), 2 phong cách, 1.000 chữ/cuốn", flat(1000), 2, 0, 0, s_img, s_step, cmap_ms),
        budget("C. chỉ sinh, 2 phong cách, 3.000 chữ/cuốn", flat(3000), 2, 0, 0, s_img, s_step, cmap_ms),
        budget("D. chỉ sinh, 1 phong cách, 5.000 chữ/cuốn", flat(5000), 1, 0, 0, s_img, s_step, cmap_ms),
        budget("E. chỉ sinh, 1 phong cách, 10.000 chữ/cuốn", flat(10000), 1, 0, 0, s_img, s_step, cmap_ms),
        budget("F. hợp ứng viên R(âm) ĐẦY ĐỦ từng cuốn (phủ phông dự phòng), 1 lượt", full_chain, 1, 0, 0, s_img, s_step, cmap_ms),
        budget("G. như F nhưng chỉ phần phông NomNaTong vẽ được (bản hiện tại, bỏ âm thầm phần còn lại)", full_nn, 1, 0, 0, s_img, s_step, cmap_ms),
        budget("H. hợp ứng viên đầy đủ, 2 phong cách", full_chain, 2, 0, 0, s_img, s_step, cmap_ms),
    ]
    R["F_ngan_sach"] = sc

    # ---- G. HF / số tệp
    def files(n_img, shard):
        return dict(anh=n_img, tep_neu_1_anh_1_tep=n_img, so_tep_neu_shard=int(math.ceil(n_img / shard)), shard=shard,
                    vuot_100k_tep_khuyen_nghi_HF=n_img > 100_000, vuot_10k_muc_moi_thu_muc_neu_phang=n_img > 10_000)
    R["G_so_tep_HF"] = {f"{n}_anh": {f"shard{s}": files(n, s) for s in (128, 256)} for n in (7390, 50000, 224000, 448000)}
    R["G_log_dong"] = {f"{n}_anh": int(n / 32) + 4 for n in (7390, 224000)}
    # đầu ra Kaggle: nhiều báo cáo người dùng + nhân viên Kaggle (2020-09 / 2023-07, product-feedback 181143 & 420538) nói danh sách đầu ra/dataset-từ-đầu-ra bị cắt ở 500 tệp
    R["G2_dau_ra_Kaggle_500_tep"] = {f"{n}_anh": dict(tep_neu_lo128_trong_working=int(math.ceil(n / 128)) + 10, vuot_500=int(math.ceil(n / 128)) + 10 > 500,
                                                       tep_neu_bundle_moi_gio_ca_11h=2 * 11 + 10) for n in (7390, 50000, 100000, 224000)}
    shard_s = 128 * s_img
    R["G3_cua_so_mat_cong_khi_ca_chet"] = dict(runner_hien_tai_gio_toi_da=12.0, runner_hien_tai_ghi_chu="không đọc đầu ra ca trước (tn11_kaggle.py:38 find_input chỉ tìm tn11_manifest.json) -> mất toàn bộ ca nếu không tải về kịp",
                                               khung_moi_lo128_phut_moi_GPU=round(shard_s / 60, 1), khung_moi_lo128_giay=round(shard_s, 0))

    # ---- H. bằng chứng mã
    R["H_bang_chung_ma"] = {
        "tn11_kaggle.py": find_lines(KAGGLE_PY, dict(
            WORK_trong_working="^WORK = ", zip_cung_thu_muc_working="^OUTZIP = ", tgt_float32="tgt = torch.stack", resume_chi_tu_last_pt="if last.exists\\(\\):",
            last_pt_moi_500_buoc="if step % 500 == 0", torch_save_khong_nguyen_tu="torch.save\\(\\{\"unet\": m.unet.state_dict\\(\\), \"style_encoder\": m.style_encoder.state_dict\\(\\), \"opt\"",
            vong_chon_phong_cach="j = int\\(rng.choice\\(others\\)\\)", seed_mot_lan_moi_luot="torch.manual_seed\\(0\\)", gen_goi_ca_danh_sach="g.generate\\(chars",
            spawn_moi_cuon_cung_luc="procs.append\\(\\(b, subprocess.Popen", gan_gpu_tinh="CUDA_VISIBLE_DEVICES=str\\(k % ngpu\\)", nhanh_1_gpu_tuan_tu="for b in a.books:",
            zip_deflated="zipfile.ZIP_DEFLATED", oom_chi_bat_o_huan_luyen="except torch.cuda.OutOfMemoryError", pip_khong_ghim="subprocess.check_call\\(\\[sys.executable, \"-m\", \"pip\"",
            thoat_1_neu_loi="sys.exit\\(1\\)", khong_doc_dau_ra_ca_truoc="def find_input")),
        "fontdiffusion_gen.py": find_lines(WRAP, dict(use_fst_False_trong_args="args.use_fst = False", use_fst_True_khi_nap="args=self.args, use_fst=True", loc_chu_im_lang="available = self.font_manager.get_available_chars_for_font",
                                                      bo_nho_dem_theo_anh="cache_path.exists\\(\\)", chi_phong_dau="get_font_names\\(\\)\\[0\\]")),
        "sample_optimized.py": find_lines(FD / "inference/sample_optimized.py", dict(lru_cache_1024="@lru_cache\\(maxsize=1024\\)", model_DPM_chuan="model: FontDiffuserModelDPM = FontDiffuserModelDPM\\(", model_FST="model: FontDiffuserModelDPMWithFST = ")),
        "utils.py": find_lines(FD / "src/tools/utils.py", dict(mo_lai_phong_moi_chu="TTFont_font = TTFont\\(font_path\\)")),
        "build.py": find_lines(FD / "src/builders/build.py", dict(canh_bao_thieu_ckpt="not found in \\{args.ckpt_dir\\}")),
    }

    (OUT / "ngan_sach.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    # in tóm tắt ngắn
    print("toc_do:", json.dumps(R["A_toc_do_chon"], ensure_ascii=False))
    print("bi_loai_am_tham:", json.dumps({b: (v["chars_requested"], v["images_in_real_log"], v["khop_voi_log"], v["dropped_pct"], v["uncovered_by_all_fonts"]) for b, v in miss_rows.items()}))
    for s in sc:
        print(json.dumps(s, ensure_ascii=False))
    print("dia:", json.dumps({k: R["D_dia"][k] for k in ("last_pt_MB", "moi_cuon_MB", "muoi_cuon_GB", "zip_deflate_ratio", "ft_pt_that_B34_MB")}), json.dumps(R["D_dia_10_cuon_cong_thuc_hien_tai"]))
    print("ram_tgt:", json.dumps(R["D_ram_tgt_float32"]))
    print("vong_lap:", json.dumps({b: (v["max_class_share"], v["expected_tries"], v["mean_tries_measured"], v["us_per_sample"]) for b, v in E.items()}), On, O1)
    print("cmap_ms:", R["B_cmap_ms_moi_chu"], "lru:", R["B_lru_cache_1024_quet_tuan_tu"])
    print("C_tong:", R["C_tong"])
    print("ghi:", OUT / "ngan_sach.json")


if __name__ == "__main__":
    main()
