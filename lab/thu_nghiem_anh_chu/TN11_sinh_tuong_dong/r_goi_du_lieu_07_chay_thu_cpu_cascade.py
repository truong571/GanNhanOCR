"""r_goi_du_lieu_07_chay_thu_cpu_cascade.py — HƯỚNG 5 (tiếp): CHẠY THẬT TRÊN CPU đường mô hình thật của bộ chạy NẰM TRONG GÓI với các chữ dùng PHÔNG DỰ PHÒNG.

Ca thử đầu-cuối cũ (w_runner_kiem_that_cpu.py) dùng gói tí hon 2 cuốn x 2 chữ toàn NomNaTong — KHÔNG chạm HAN NOM A / HanaMin / Plangothic / NomNaTongLight2, chữ PUA mặt phẳng 15.
Ở đây: dựng gói MINI từ chính gói thật (liên kết mềm tới code/, ckpt/, data/ của gói; plan.json mini do make_plan của bộ chạy dựng từ cmap phông của gói) gồm 1 việc B18 (7 chữ:
2 chữ NomNaTong gồm 1 PUA mặt phẳng 15, và mỗi phông dự phòng một chữ, kể cả U+F094E của NomNaTongLight2) + 1 việc B34 (2 chữ), rồi chạy
`<gói>/w_runner_khung_chay.py run --device cpu` (bản bộ chạy trong gói), `validate`, và đọc ảnh trong tar. Ghi: review/goi_du_lieu/_chay_thu_cpu/ (xoá bản sao code+ckpt sau khi chạy).

    cd measure_out/_tn11/full/review/goi_du_lieu && PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 ../../../../../.venv/bin/python -B ../../../../../lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_goi_du_lieu_07_chay_thu_cpu_cascade.py
Ra: 07_chay_thu_cpu_cascade.json.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_goi_du_lieu_00_lib as Lb  # noqa: E402
import w_runner_khung_chay as R  # noqa: E402


def main():
    t0 = time.time()
    inv = Lb.Inv()
    res: dict = dict(tao_luc=time.strftime("%Y-%m-%d %H:%M:%S"))
    plan = json.loads((Lb.PACK / "plan.json").read_text(encoding="utf-8"))
    fn = plan["font_names"]
    W = Lb.OUT / "_chay_thu_cpu"
    shutil.rmtree(W, ignore_errors=True)
    (W / "run").mkdir(parents=True)
    # ---- chọn chữ từ kế hoạch thật: mỗi phông 1 chữ (ưu tiên chữ đầu danh sách B18), thêm 1 PUA mặt phẳng 15 của NomNaTong
    v = plan["books"]["B18"]
    by = {}
    for c, i in zip(v["chars"], v["font_idx"]):
        by.setdefault(fn[i], []).append(c)
    pick = []
    pick.append(by["NomNaTong"][0])
    pua = next(c for c in by["NomNaTong"] if Lb.is_pua(c))
    pick.append(pua)
    for n in ("HanaMinB", "HanNomA", "HanaMinA", "PlangothicP2", "NomNaTongLight2"):
        if n in by:
            pick.append(by[n][0])
    # PlangothicP2 không có ở B18: lấy từ cuốn khác (stt2), NomNaTongLight2 từ B18
    v2 = plan["books"]["stt2"]
    p2 = next((c for c, i in zip(v2["chars"], v2["font_idx"]) if fn[i] == "PlangothicP2"), None)
    if p2 and p2 not in pick:
        pick.append(p2)
    pick = list(dict.fromkeys(pick))
    fo = {}
    for b in ("B18", "stt2"):
        vv = plan["books"][b]
        for c, i in zip(vv["chars"], vv["font_idx"]):
            fo.setdefault(c, fn[i])
    # cmap phông của gói (như cmd_pack)
    fonts_json = json.loads((Lb.PACK / "fonts.json").read_text(encoding="utf-8"))
    cm = {n: Lb.cmap_union(Lb.PACK / "code" / "font_diffusion" / "fonts" / fonts_json[n]) for n in fn}
    books = {"B18": dict(chars=pick, styles=2, font_of={c: fo[c] for c in pick}), "B34": dict(chars=[by["NomNaTong"][1], by["HanNomA"][0]], styles=2)}
    books["B34"]["font_of"] = {c: fo[c] for c in books["B34"]["chars"]}
    mini = R.make_plan(books, [("base", 0)], 128, {n: cm[n] for n in fn})
    pack = W / "pack"
    pack.mkdir()
    for d in ("code", "ckpt", "data"):
        os.symlink(Lb.PACK / d, pack / d)
    shutil.copyfile(Lb.PACK / "fonts.json", pack / "fonts.json")
    man = json.loads((Lb.PACK / "manifest.json").read_text(encoding="utf-8")); man["plan_sha"] = mini["sha"]
    (pack / "manifest.json").write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    (pack / "plan.json").write_text(json.dumps(mini, ensure_ascii=False), encoding="utf-8")
    res["mini_ke_hoach"] = dict(sha=mini["sha"][:16], viec=[t["id"] for t in mini["tasks"]], B18=[dict(chu=f"U+{ord(c):05X}", phong=fo[c], PUA=Lb.is_pua(c)) for c in pick],
                                B34=[dict(chu=f"U+{ord(c):05X}", phong=books["B34"]["font_of"][c]) for c in books["B34"]["chars"]], uncovered={b: v_["uncovered"] for b, v_ in mini["books"].items()})
    inv.check("mini_ke_hoach_khong_chu_nao_bi_bo_(uncovered)", not any(v_["uncovered"] for v_ in mini["books"].values()), res["mini_ke_hoach"]["uncovered"])
    res["mini_phu_cac_phong"] = sorted({fo[c] for c in pick})
    inv.check("mini_ke_hoach_cham_du_6_phong_cua_goi", set(res["mini_phu_cac_phong"]) == set(fn), sorted(set(fn) - set(res["mini_phu_cac_phong"])))
    # ---- chạy thật bằng bộ chạy trong gói
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0", PYTHONDONTWRITEBYTECODE="1", SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", TMPDIR=str(W / "tmpdir"))
    (W / "tmpdir").mkdir()
    cmd = [sys.executable, "-B", str(Lb.PACK / "w_runner_khung_chay.py"), "run", "--pack", str(pack), "--out", str(W / "out"), "--tmp", str(W / "tmp"), "--workers", "1", "--budget-hours", "1",
           "--safety-s", "0", "--gen-bs", "2", "--device", "cpu", "--est-s", "60"]
    t1 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=W / "run", env=env)
    res["chay"] = dict(ma_thoat=p.returncode, giay=round(time.time() - t1, 1), duoi_log=(p.stdout + p.stderr).strip()[-600:])
    st = json.loads((W / "out" / "status" / "status.json").read_text(encoding="utf-8")) if (W / "out" / "status" / "status.json").exists() else {}
    res["status"] = st
    inv.check("chay_that_CPU_ma_thoat_0_va_complete", p.returncode == 0 and st.get("complete") is True, dict(code=p.returncode, stop=st.get("stop_reason")))
    pv = subprocess.run([sys.executable, "-B", str(Lb.PACK / "w_runner_khung_chay.py"), "validate", "--pack", str(pack), "--out", str(W / "out")], capture_output=True, text=True, env=env, cwd=W / "run")
    try:
        vr = json.loads(pv.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        vr = dict(loi=(pv.stdout + pv.stderr)[-300:])
    res["validate"] = vr
    inv.check("validate_n_errors_0_va_khong_chu_bi_bo_qua", vr.get("n_errors") == 0 and not vr.get("skipped_by_reason"), vr)
    # ---- đọc ảnh
    imgs = []
    for t in mini["tasks"]:
        tp = W / "out" / R.rel_path(t["id"])
        if not tp.exists():
            continue
        with tarfile.open(tp) as tf:
            meta = json.loads(tf.extractfile("_meta.json").read())
            for m in tf.getmembers():
                if m.name.startswith("U+"):
                    im = Image.open(io.BytesIO(tf.extractfile(m).read()))
                    a = np.asarray(im.convert("L"))
                    cp = int(m.name[2:-4], 16)
                    imgs.append(dict(task=t["id"], chu=f"U+{cp:05X}", phong=books[t["book"]]["font_of"][chr(cp)], size=list(im.size), mode=im.mode, muc_pct=round(float((a < 128).mean()) * 100, 1),
                                     nen_trung_vi=int(np.median(a)), khong_rong=bool((a < 128).any()), sha=Lb.sha256_bytes(im.tobytes())[:12]))
        res.setdefault("meta_tar", {})[t["id"]] = dict(generated=len(meta["generated"]), skipped=meta["skipped"], requested=len(meta["requested"]))
    res["anh"] = imgs
    n_exp = len(pick) + len(books["B34"]["chars"])
    inv.check(f"{n_exp}_anh_sinh_ra_deu_96x96_RGB_va_co_muc", len(imgs) == n_exp and all(i["size"] == [96, 96] and i["mode"] == "RGB" and i["khong_rong"] for i in imgs), [(i["chu"], i["size"], i["muc_pct"]) for i in imgs if not i["khong_rong"]])
    inv.check("anh_phong_du_phong_co_muc_trong_khoang_hop_ly_(5-80%)", all(5 <= i["muc_pct"] <= 80 for i in imgs), [(i["phong"], i["chu"], i["muc_pct"]) for i in imgs if not 5 <= i["muc_pct"] <= 80])
    # môi trường + dọn bản sao nặng
    envj = W / "out" / "status" / "env.json"
    res["env"] = json.loads(envj.read_text(encoding="utf-8")) if envj.exists() else None
    tj = []
    for f in sorted((W / "out" / "status").glob("tasks-*.jsonl")):
        tj += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    res["thoi_gian_viec_giay"] = tj
    shutil.rmtree(W / "tmp", ignore_errors=True)
    shutil.rmtree(W / "tmpdir", ignore_errors=True)
    os.unlink(pack / "code"); os.unlink(pack / "ckpt"); os.unlink(pack / "data")
    res["bat_bien"] = inv.summary(); res["bat_bien_chi_tiet"] = inv.rows; res["giay"] = round(time.time() - t0, 1)
    Lb.jdump(res, "07_chay_thu_cpu_cascade.json")
    print(f"[07] xong {res['giay']}s; bất biến {res['bat_bien']['dat']}/{res['bat_bien']['tong']} rớt={res['bat_bien']['rot']}")


if __name__ == "__main__":
    main()
