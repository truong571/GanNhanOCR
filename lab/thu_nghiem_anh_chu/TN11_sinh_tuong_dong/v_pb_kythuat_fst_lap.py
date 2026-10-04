"""v_pb_kythuat (04/10) — PHẢN BIỆN độc lập: lặp lại phép chẩn đoán "đường FST với trọng số NGẪU NHIÊN làm yếu đường phong cách" (CPU, 0 API).

Người kiểm chứng (v_kythuat_fst_random.py, 5 chữ 孟恒魂拪仁, một hạt khởi tạo FST = 0): chênh mực ảnh sinh giữa ảnh phong cách ĐẬM (mực 21 %) và MẢNH (0,8 %)
= +5,0 điểm ở đường TN11 (FontDiffuserModelDPMWithFST, 4 mô-đun thiếu ckpt) so với +17,9 ở đường chuẩn (FontDiffuserModelDPM). Cỡ mẫu nhỏ.
Để cố bác bỏ, script này lặp lại với: CHỮ KHÁC (天 心 國 水), HAI hạt nhiễu khởi đầu, HAI khởi tạo FST ngẫu nhiên khác nhau (hạt 11 và 22)
— nếu "FST yếu" chỉ do một lần khởi tạo xui, hai khởi tạo sẽ cho kết quả khác hẳn nhau; và đo thêm
  (a) d_fst_AB = |ảnh(FST hạt 11) − ảnh(FST hạt 22)| cùng (chữ, phong cách, nhiễu): độ lệch chỉ do khởi tạo ngẫu nhiên của 4 mô-đun,
  (b) d_nhieu = |ảnh(seed nhiễu 1) − ảnh(seed nhiễu 2)| cùng đường: mức chuẩn để so,
  (c) tương quan nội dung: Pearson giữa bản đồ mực của ảnh sinh và ảnh chữ phông (nội dung) — đường nào giữ nội dung tốt hơn,
  (d) chênh mực đậm−mảnh theo từng đường và từng seed nhiễu.
Ảnh phong cách: p02_B34/style_0.png (chỉ ĐỌC) + biến thể đậm (erode ×3) / mảnh (dilate ×3). Dùng đúng mã TN11: core/ranking/fontdiffusion_gen.py::_load_pipeline.
Ra: measure_out/_tn11/verify/pb_kythuat/fst_lap/{ket_qua.json, cache/*.npy}
    PYTORCH_ENABLE_MPS_FALLBACK=0 CUDA_VISIBLE_DEVICES= nice -n 10 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_pb_kythuat_fst_lap.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "font_diffusion"))
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_kythuat" / "fst_lap"
(OUT / "cache").mkdir(parents=True, exist_ok=True)
CK = REPO / "font_diffusion" / "ckpt" / "PROD"
P02 = REPO / "measure_out" / "_tn11" / "p02_B34"
CHARS = list("天心國水")
CFG = 2.0
DEV = "cpu"
torch.set_num_threads(3)


def gen_batch(pipe, args, content, style_pil, seed):
    from inference.sample_optimized import get_style_transform
    st = get_style_transform(args.style_image_size)(style_pil.convert("RGB"))
    B = len(content)
    bs = st[None, :].repeat(B, 1, 1, 1)
    g = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        ims = pipe.generate(content_images=content.to(DEV), style_images=bs.to(DEV), batch_size=B, order=args.order,
                            num_inference_step=args.num_inference_steps,
                            content_encoder_downsample_size=args.content_encoder_downsample_size,
                            t_start=args.t_start, t_end=args.t_end, dm_size=args.content_image_size,
                            algorithm_type=args.algorithm_type, skip_type=args.skip_type, method=args.method,
                            correcting_x0_fn=args.correcting_x0_fn, generator=g)
    return np.stack([np.asarray(i.convert("L"), np.float32) for i in ims])


SEEDS = {"FST_A": (1, 2), "FST_B": (1,), "STD": (1,)}   # rút gọn (04/10): chỉ FST_A có hai hạt nhiễu, B và STD một hạt — tiết kiệm ~35 phút CPU


def run_path(tag, pipe, args, content, pil, seeds=(1, 2)):
    R = {}
    for sty in ("dam", "manh"):
        for sd in seeds:
            f = OUT / "cache" / f"{tag}_{sty}_n{sd}.npy"
            if f.exists():
                R[(sty, sd)] = np.load(f)
                continue
            t0 = time.time()
            R[(sty, sd)] = gen_batch(pipe, args, content, pil[sty], sd)
            np.save(f, R[(sty, sd)])
            print(f"[{tag}] phong cách {sty} nhiễu {sd}: {len(content)} ảnh [{time.time() - t0:.0f}s]", flush=True)
    return R


def main():
    t00 = time.time()
    from core.ranking.fontdiffusion_gen import FontDiffusionGenerator
    from inference.sample_optimized import get_content_transform, load_fontdiffuser_pipeline
    from src.tools.utils import ttf2im

    def build_fst(seed):
        torch.manual_seed(seed)
        g = FontDiffusionGenerator(ckpt_dir=str(CK), phase1_ckpt_dir=str(CK), font_path=str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf"),
                                   cache_dir=str(OUT / "_tmp"), batch_size=len(CHARS), device=DEV)
        g._load_pipeline()
        assert type(g.pipe.model).__name__ == "FontDiffuserModelDPMWithFST"
        g.pipe.guidance_scale = CFG
        return g

    gA = build_fst(11)
    args = gA.args
    font = gA.font_manager.get_font(gA.font_manager.get_font_names()[0])
    ct = get_content_transform(args.content_image_size)
    content = torch.stack([ct(ttf2im(font=font, char=c)) for c in CHARS])
    # ảnh nội dung (bản đồ mực chuẩn) để đo tương quan nội dung
    cont_ink = np.stack([(np.asarray(ttf2im(font=font, char=c).convert("L").resize((96, 96)), np.float32) < 128).astype(np.float32) for c in CHARS])
    s0 = cv2.imread(str(P02 / "style_0.png"), cv2.IMREAD_GRAYSCALE)
    k3 = np.ones((3, 3), np.uint8)
    styles = {"dam": cv2.erode(s0, k3, iterations=3), "manh": cv2.dilate(s0, k3, iterations=3)}
    ink_style = {k: float((v < 128).mean()) for k, v in styles.items()}
    pil = {k: Image.fromarray(v).convert("RGB") for k, v in styles.items()}

    R = {}
    R["FST_A"] = run_path("FST_A", gA.pipe, args, content, pil, SEEDS["FST_A"])
    del gA
    gB = build_fst(22)
    R["FST_B"] = run_path("FST_B", gB.pipe, args, content, pil, SEEDS["FST_B"])
    del gB
    pipe_std = load_fontdiffuser_pipeline(args=args, use_fst=False)
    assert type(pipe_std.model).__name__ == "FontDiffuserModelDPM"
    pipe_std.guidance_scale = CFG
    R["STD"] = run_path("STD", pipe_std, args, content, pil, SEEDS["STD"])

    def ink(a):
        return float((a < 128).mean())

    def l1(a, b):
        return float(np.abs(a - b).mean())

    def pear(a, b):
        a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

    res = dict(thiet_lap=dict(chars="".join(CHARS), guidance=CFG, steps=args.num_inference_steps, device=DEV, fst_seeds=[11, 22], noise_seeds=[1, 2],
                              ink_anh_phong_cach=ink_style), seconds=round(time.time() - t00))
    for tag in ("STD", "FST_A", "FST_B"):
        r = {}
        sds = SEEDS[tag]
        for sd in sds:
            r[f"dmuc_dam_tru_manh_nhieu{sd}"] = round(ink(R[tag][("dam", sd)]) - ink(R[tag][("manh", sd)]), 4)
        r["dmuc_tb"] = round(float(np.mean([r[f"dmuc_dam_tru_manh_nhieu{sd}"] for sd in sds])), 4)
        r["muc"] = {f"{sty}_n{sd}": round(ink(R[tag][(sty, sd)]), 4) for sty in ("dam", "manh") for sd in sds}
        if len(sds) > 1:
            r["d_nhieu_dam"] = round(l1(R[tag][("dam", 1)], R[tag][("dam", 2)]), 2)
            r["d_nhieu_manh"] = round(l1(R[tag][("manh", 1)], R[tag][("manh", 2)]), 2)
        r["d_dam_manh_cung_nhieu"] = round(float(np.mean([l1(R[tag][("dam", sd)], R[tag][("manh", sd)]) for sd in sds])), 2)
        cc = []
        for key, A in R[tag].items():
            for j in range(len(CHARS)):
                g = 1.0 - cv2.resize(A[j] / 255.0, (96, 96), interpolation=cv2.INTER_AREA)   # mực = sáng đảo
                cc.append(pear(g, cont_ink[j]))
        r["tuong_quan_noi_dung_tb"] = round(float(np.mean(cc)), 4)
        res[tag] = r
    common = [k for k in R["FST_A"] if k in R["FST_B"] and k in R["STD"]]
    res["d_khoi_tao_FST_A_vs_B"] = round(float(np.mean([l1(R["FST_A"][k], R["FST_B"][k]) for k in common])), 2)
    res["d_FST_A_vs_STD"] = round(float(np.mean([l1(R["FST_A"][k], R["STD"][k]) for k in common])), 2)
    res["d_FST_B_vs_STD"] = round(float(np.mean([l1(R["FST_B"][k], R["STD"][k]) for k in common])), 2)
    res["d_nhieu_FST_A_seed1_vs_seed2"] = round(float(np.mean([l1(R["FST_A"][(sty, 1)], R["FST_A"][(sty, 2)]) for sty in ("dam", "manh")])), 2)
    # montage
    rows = []
    for tag in ("FST_A", "FST_B", "STD"):
        for sty in ("dam", "manh"):
            rows.append(np.concatenate([x.astype(np.uint8) for x in R[tag][(sty, 1)]], axis=1))
    cv2.imwrite(str(OUT / "montage_FSTA_FSTB_STD_dam_manh.png"), np.concatenate(rows, axis=0))
    (OUT / "ket_qua.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
