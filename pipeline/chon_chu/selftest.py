"""selftest.py — kiểm bước chọn chữ (pipeline.chon_chu), 0 API, dữ liệu GIẢ + config/tài sản thật (không đọc dataset_out).

    .venv/bin/python -m pipeline.chon_chu --selftest        # exit 0 = pass
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.chon_chu import __main__ as CM
from pipeline.chon_chu import model as M
from pipeline.chon_chu import policy as POL

REPO = Path(__file__).resolve().parents[2]
_ok, _fail = 0, []


def chk(name, cond, detail=""):
    global _ok
    if cond:
        _ok += 1
        print(f"  ok   {name}")
    else:
        _fail.append(name)
        print(f"  FAIL {name} {detail}")


def _F():
    """2 ô × 3 ứng viên; ô 0: ứng viên 'b' mạnh nhất mọi kênh; ô 1: có NaN."""
    rows = []
    for i in (0, 1):
        for c, s in (("a", 0.1), ("b", 0.9), ("c", 0.3)):
            r = dict(i=i, c=c, lp_syl=-1.0, lp_glob=1.0, n_self=3, n_hum=2, is_kim=int(c == "b"), is_lt2=0, inR=1)
            for f in M.SIMS:
                r[f] = s if not (i == 1 and c == "c") else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def test_model():
    print("[1] mô hình: đặc trưng, softmax, lưu/nạp, nhóm ô")
    F = _F()
    X, names = M.cand_matrix(F)
    chk("35 đặc trưng ứng viên", X.shape == (6, 35) and len(names) == 35)
    Xv, nv = M.cand_matrix(F, with_kim=False)
    chk("chỉ-ảnh bỏ 3 cột kim -> 32", Xv.shape[1] == 32 and not set(nv) & set(M.KIM_COLS))
    j = names.index("d_self")
    chk("tương đối: ứng viên tốt nhất có d > 0, còn lại < 0", X[1, j] > 0 and X[0, j] < 0 and X[2, j] < 0)
    jh = names.index("has_self")
    chk("NaN -> has = 0, giá trị thay bằng min trong ô", X[5, jh] == 0 and X[5, names.index("f_self")] == np.float32(0.1))
    m = M.CLogit(mu=X.mean(0), sd=X.std(0) + 1e-6, w=np.linspace(-1, 1, 35))
    p = m.predict(X, F.i.values)
    chk("softmax trong ô: tổng = 1", np.allclose(pd.Series(p).groupby(F.i.values).sum().values, 1.0))
    m2 = M.Logit(mu=np.zeros(3), sd=np.ones(3), w=np.array([1.0, -1.0, 0.5, 0.1]))
    z = M.pack({"in": (m, m2), "out": (m, m2)}, {"name": "x", "train_books": ["B34"]})
    f = Path(tempfile.mkdtemp()) / "c.npz"
    np.savez(f, **z)
    mods, meta = M.unpack(np.load(f))
    chk("lưu/nạp: tham số + meta nguyên vẹn", np.allclose(mods["in"][0].w, m.w) and np.allclose(mods["out"][1].w, m2.w)
        and meta["train_books"] == ["B34"])
    shutil.rmtree(f.parent, ignore_errors=True)
    D = pd.DataFrame(dict(
        tier=["GOLD", "REVIEW", "REVIEW", "SYLLABLE", "REVIEW", "REVIEW", "REVIEW", "GOLD_text_only", "QUARANTINE", "REVIEW"],
        rule=["s1_inter_s2_direct", "s1_inter_s2_direct|gate:qn_count_unfixed", "s1_inter_s2_direct|gate:crop_bad",
              "syl_ctx:bigram", "no_context", "not_plausible", "confusion_fix:x", "s1_inter_s2_direct", "s1_inter_s2_direct|quarantine_dup",
              "low_posterior"],
        gate_reason=["", "qn_count_unfixed", "crop_bad:blank", "", "", "", "", "box_low_conf:vdp_low", "", ""],
        label=["a", "a", "a", "", "", "", "a", "a", "a", ""]))
    chk("nhóm ô đúng", list(M.group_of(D)) == ["GOLD", "direct_qn", "direct_crop", "syl", "nocontext", "notplaus", "other",
                                                "textonly", "quarantine", "lowpost"], str(list(M.group_of(D))))


def _cfg():
    return yaml.safe_load((REPO / "config/chon_chu.yaml").read_text(encoding="utf-8"))


def test_policy():
    print("[2] chính sách: config, nâng theo nhóm/ngưỡng, L5, STT bảo thủ")
    cfg = _cfg()
    b18 = POL.book_cfg(cfg, "sachkinhthaycabinh")
    chk("config: tra không phân biệt hoa/thường + gắn nhóm của họ", b18 and b18["family"] == "hand" and "nocontext" in b18["promo_groups"])
    chk("config: sách lạ -> None", POL.book_cfg(cfg, "Khac") is None)
    bad = json.loads(json.dumps(cfg))
    bad["books"]["SachKinhThayCaBinh"]["verifiers"] = ["vft_T"]
    try:
        POL.book_cfg(bad, "SachKinhThayCaBinh"); chk("CẤM vft trên Borg", False)
    except ValueError:
        chk("CẤM vft trên Borg", True)
    bad = json.loads(json.dumps(cfg))
    bad["books"]["SachKinhThayCaBinh"]["human"] = ["B18"]
    try:
        POL.book_cfg(bad, "SachKinhThayCaBinh"); chk("CẤM nguyên mẫu người của chính sách (LOBO)", False)
    except ValueError:
        chk("CẤM nguyên mẫu người của chính sách (LOBO)", True)
    bad = json.loads(json.dumps(cfg))
    bad["books"]["LucVanTien1916"]["verifiers"] = ["vft_L"]
    try:
        POL.book_cfg(bad, "LucVanTien1916"); chk("CẤM vft_L trên L16 (vft_L học nhãn người L16)", False)
    except ValueError:
        chk("CẤM vft_L trên L16 (vft_L học nhãn người L16)", True)
    bad = json.loads(json.dumps(cfg))
    bad["books"]["TruyenKieu1872"]["khoa_la"] = 1
    try:
        POL.book_cfg(bad, "TruyenKieu1872"); chk("khoá lạ -> ValueError", False)
    except ValueError:
        chk("khoá lạ -> ValueError", True)
    stt = [POL.book_cfg(cfg, b) for b in POL.STT]
    chk("STT TẮT mặc định (cả 3)", all(s is not None and s["enabled"] is False for s in stt))
    # decide
    tier = np.array(["REVIEW", "REVIEW", "REVIEW", "GOLD", "GOLD", "REVIEW", "REVIEW", "GOLD_text_only", "REVIEW", "REVIEW"])
    grp = np.array(["direct_qn", "nocontext", "nocontext", "GOLD", "GOLD", "direct_other", "notplaus", "textonly", "syl", "direct_crop"])
    gate = np.array(["qn_count_unfixed", "", "", "", "", "cross_similar", "", "", "", "crop_bad:truncated"])
    part = np.array(["in", "out", "out", "in", "in", "in", "none", "in", "out", "in"])
    P = np.array([0.9, 0.7, 0.5, 0.95, 0.95, 0.99, 0.99, 0.99, 0.99, 0.2])
    top = np.array(["a", "b", "c", "x", "y", "d", "e", "f", "g", "h"], dtype=object)
    lab = np.array(["a", "", "", "x", "z", "d", "", "f", "", "h"], dtype=object)
    veq = lambda u, v: u == v  # noqa: E731
    bc = dict(b18, tau_in=0.5, tau_out=0.6)
    pro, rel, lev = POL.decide(tier, grp, gate, part, P, top, lab, bc, veq)
    chk("nâng: in ≥ τ_in, out ≥ τ_out", pro[0] and pro[1] and not pro[2], str(pro))
    chk("không đụng GOLD (trừ L5), cổng văn bản, not_plausible, text_only", not pro[3] and not pro[4] and not pro[5]
        and not pro[6] and not pro[7])
    chk("L1 dưới ngưỡng không nâng", not pro[9])
    chk("L5: GOLD top-1 ≠ nhãn, P ≥ 0,8 -> sửa; top-1 = nhãn -> không", rel[4] and not rel[3])
    chk("đòn bẩy ghi đúng", lev[0] == "L2" and lev[1] == "L4" and lev[4] == "L5" and lev[8] == "L4", str(lev))
    pr = dict(POL.book_cfg(cfg, "TruyenKieu1872"), tau_in=0.1)
    pro2, rel2, _ = POL.decide(tier, grp, gate, part, P, top, lab, pr, veq)
    chk("họ in: chỉ L1/confusion_fix, phần out TẮT, không L5", not pro2[0] and not pro2[1] and pro2[9] and not rel2.any())
    top_x = top.copy(); top_x[9] = "q"
    pro3, _, _ = POL.decide(tier, grp, gate, part, P, top_x, lab, pr, veq)
    chk("họ in: top-1 ≠ nhãn kim -> KHÔNG nâng (chỉ xác nhận, không đổi chữ)", not pro3[9])
    nl = POL.new_labels(np.array(["國", "b", "c"], dtype=object), np.array(["国", "a", ""], dtype=object),
                        lambda u, v: {u, v} == {"國", "国"})
    chk("nhãn mới: giữ nhãn cũ khi top-1 ≡ (V1+), đổi khi khác hẳn, ô chưa nhãn lấy top-1", list(nl) == ["国", "b", "c"])
    # STT bảo thủ
    R = {"a": {"甲", "乙"}, "b": {"丙"}}
    p3, l3 = POL.decide_stt_bao_thu(np.array(["SYLLABLE", "REVIEW", "SYLLABLE", "GOLD"]), np.array(["syl", "nocontext", "syl", "GOLD"]),
                                    np.array(["X", "X", "甲", "甲"], dtype=object), np.array(["甲", "甲", "乙", "乙"], dtype=object),
                                    np.array(["a", "a", "a", "a"], dtype=object), np.array(["甲", "乙", "乙", "乙"], dtype=object),
                                    lambda s: R.get(s, set()), veq)
    chk("STT bảo thủ: lt1 ∉ R ∧ lt2 ∈ R ∧ ảnh == lt2 -> nâng nhãn lt2; ảnh khác/lt1 ∈ R/GOLD -> không",
        list(p3) == [True, False, False, False] and l3[0] == "甲", str(p3))


def test_io(tmp: Path):
    print("[3] vào/ra: TẮT = không ghi gì; không cộng dồn (khôi phục bản trước)")
    L = pd.DataFrame(dict(image=["gold/x.png"], book=["b"], page=["p"], column=["1"], ocr_char=["a"], syllable=["s"], label=["a"],
                          unicode=["U+0061"], tier=["GOLD"], rule=["s1_inter_s2_direct"], bbox=["[0,0,1,1]"], image_md5=["m"]))
    f = tmp / "labels_gated.csv"; L.to_csv(f, index=False)
    h0 = hashlib.md5(f.read_bytes()).hexdigest()
    cfg = _cfg(); cfg["books"]["SachThanhTruyen2"]["enabled"] = False
    cp = tmp / "c.yaml"; cp.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with redirect_stdout(io.StringIO()):
        rep = CM.run(f, "SachThanhTruyen2", cp)
    chk("sách TẮT: tệp nhãn y hệt byte, không bản trước, không báo cáo",
        hashlib.md5(f.read_bytes()).hexdigest() == h0 and not CM.backup_path(f).exists() and rep["enabled"] is False
        and not (tmp / "chon_chu_report.json").exists())
    with redirect_stdout(io.StringIO()):
        L1, r1 = CM.load_input(f, f)
    chk("lần đầu: chép bản trước (byte)", CM.backup_path(f).exists() and not r1
        and hashlib.md5(CM.backup_path(f).read_bytes()).hexdigest() == h0)
    L2 = L.copy(); L2["rule"] = L2.rule + CM.MARK + "L2"; L2["chon_chu"] = "L2"; L2.to_csv(f, index=False)
    with redirect_stdout(io.StringIO()):
        L3, r3 = CM.load_input(f, f)
    chk("đã mang dấu: đọc lại bản trước, bỏ cột vết", r3 and (L3.rule == "s1_inter_s2_direct").all() and "chon_chu" not in L3.columns)
    CM.backup_path(f).unlink()
    try:
        with redirect_stdout(io.StringIO()):
            CM.load_input(f, f)
        chk("mang dấu mà thiếu bản trước -> lỗi", False)
    except RuntimeError:
        chk("mang dấu mà thiếu bản trước -> lỗi", True)


def test_real_config():
    print("[4] config + tài sản thật (models/chon_chu/MANIFEST.json)")
    cfg = _cfg()
    mdir = REPO / cfg.get("models_dir", "models/chon_chu")
    man = json.loads((mdir / "MANIFEST.json").read_text(encoding="utf-8"))
    ok_all = True
    for name in cfg["books"]:
        b = POL.book_cfg(cfg, name)
        ent = man["files"].get(f"chooser_{b['model']}.npz")
        own = POL.OWN.get(name)
        if ent is None or (own and own in set(ent.get("train_books", []))):
            ok_all = False
    chk("mọi sách: mô hình có trong MANIFEST và LOBO với sách có nhãn người", ok_all)
    npz = [n for n in man["files"] if n.endswith(".npz")]
    chk("chooser_*.npz đúng sha256 MANIFEST",
        all((mdir / n).exists() and CM.sha256(mdir / n) == man["files"][n]["sha256"] for n in npz))
    pts = [n for n in man["files"] if n.endswith(".pt")]
    have = [n for n in pts if (mdir / n).exists()]
    chk("hum_*.pt (gitignore) có -> đúng sha256", all(CM.sha256(mdir / n) == man["files"][n]["sha256"] for n in have),
        f"{len(have)}/{len(pts)} có trên đĩa")
    chk("ngưỡng Borg/IHR = LOBO của TN8 (summary_t12_b80)",
        abs(cfg["books"]["SachKinhThayCaBinh"]["tau_in"] - 0.132865) < 1e-6 and abs(cfg["books"]["TruyenKieu1872"]["tau_in"] - 0.743915) < 1e-6)


def test_geo(tmp: Path):
    print("[5] luật TN9 sách in: kim_geo + P2 qn_geo / P3 np_geo (01/10)")
    from pipeline.chon_chu import geo as GEO
    cache = {"columns": [[{"char": "甲", "bbox": [100, 0, 140, 40]}, {"char": "乙", "bbox": [100, 40, 140, 80]}],
                         [{"char": "丙", "bbox": [50, 0, 90, 40]}, {"char": "丁", "bbox": [60, 0, 95, 45]}]]}
    det = tmp / "detected"; det.mkdir()
    (det / "p1_ocr_cache.json").write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    L = pd.DataFrame(dict(page=["p1", "p1", "p1", "p2", "p1"],
                          bbox=["[105, 5, 135, 35]", "[105, 45, 135, 75]", "[200, 0, 220, 20]", "[105, 5, 135, 35]",
                                "[70, 10, 90, 40]"]))
    g = GEO.geo_for(L, lambda pg: det / f"{pg}_ocr_cache.json")
    chk("kim_geo: tâm trong hộp chia đều -> chữ đó; ngoài mọi hộp / trang không cache -> ''", list(g[:4]) == ["甲", "乙", "", ""],
        str(list(g)))
    chk("kim_geo: hai hộp cùng chứa tâm -> hộp có tâm gần nhất", g[4] == "丁", str(g[4]))
    cfg = _cfg()
    kvk = POL.book_cfg(cfg, "KimVanKieu1884"); l83 = POL.book_cfg(cfg, "LucVanTien1883")
    # 01/10: config/chon_chu.yaml TẮT qn_geo của L83 (dị bản 72,1 % < GOLD − 3 điểm, cổng TN9) -> chỉ KVK/Chr mang qn_geo
    chk("config: qn_geo KVK/Chr (L83 TẮT 01/10), np_geo CHỈ KVK, IHR/Borg/STT tắt",
        kvk["qn_geo"] and kvk["np_geo"] and not l83["qn_geo"] and not l83["np_geo"] and POL.book_cfg(cfg, "Chrestomathie1872")["qn_geo"]
        and not POL.book_cfg(cfg, "LucVanTien1916")["qn_geo"] and not POL.book_cfg(cfg, "SachKinhThayCaBinh")["qn_geo"])
    bad = json.loads(json.dumps(cfg)); bad["books"]["SachKinhThayCaBinh"]["qn_geo"] = True
    try:
        POL.book_cfg(bad, "SachKinhThayCaBinh"); chk("qn_geo cho họ viết tay -> ValueError", False)
    except ValueError:
        chk("qn_geo cho họ viết tay -> ValueError", True)
    bad = json.loads(json.dumps(cfg)); bad["books"]["KimVanKieu1884"]["np_geo"] = "yes"
    try:
        POL.book_cfg(bad, "KimVanKieu1884"); chk("np_geo sai kiểu -> ValueError", False)
    except ValueError:
        chk("np_geo sai kiểu -> ValueError", True)
    R = {"thành": {"城", "成"}}
    tier = np.array(["REVIEW"] * 7 + ["GOLD"])
    grp = np.array(["direct_qn", "direct_qn", "direct_qn", "direct_qn", "notplaus", "notplaus", "nocontext", "GOLD"])
    kim = np.array(["城", "城", "城", "X", "城", "", "城", "城"], dtype=object)
    syl = np.array(["thành", "thành", "thành", "thành", "khongdocx", "khongdocx", "thành", "thành"], dtype=object)
    geo = np.array(["城", "成", "城", "X", "城", "城", "城", "城"], dtype=object)
    already = np.array([False, False, True, False, False, False, False, False])
    veq = lambda u, v: u == v  # noqa: E731
    p2, p3 = POL.decide_geo(tier, grp, kim, syl, geo, already, kvk, lambda x: R.get(x, set()), veq)
    chk("P2: direct_qn ∧ kim ∈ R ∧ kim_geo ≡ kim -> nâng; geo khác / đã nâng / kim ∉ R -> không",
        list(p2) == [True, False, False, False, False, False, False, False], str(list(p2)))
    chk("P3: not_plausible có kim ∧ geo ≡ kim -> nâng (KVK); không kim -> không; no_context/GOLD không bao giờ",
        list(p3) == [False, False, False, False, True, False, False, False], str(list(p3)))
    q2, q3 = POL.decide_geo(tier, grp, kim, syl, geo, already, l83, lambda x: R.get(x, set()), veq)
    chk("L83 (qn_geo TẮT 01/10): P2 TẮT, P3 TẮT", not q2.any() and not q3.any())
    # ghi vào bảng nhãn qua run() với process_book giả (không cần mô hình/ảnh)
    Lb = pd.DataFrame(dict(image=["gold/a.png", "", "gold/c.png"], book=["kvk"] * 3, page=["p1"] * 3, column=["1"] * 3,
                           ocr_char=["城", "城", "甲"], syllable=["thành", "khongdocx", "giáp"], label=["城", "", "甲"],
                           unicode=["U+57CE", "", "U+7532"], tier=["REVIEW", "REVIEW", "GOLD"],
                           rule=["s1_inter_s2_direct|gate:qn_count_unfixed", "not_plausible", "s1_inter_s2_direct"],
                           bbox=["[0,0,1,1]"] * 3, image_md5=["m1", "", "m3"], label_level=["", "", "char"],
                           label_canonical=["城", "", "甲"], nom_idx=["0", "1", "2"], syl_idx=["0", "1", "2"]))
    f = tmp / "labels_gated.csv"; Lb.to_csv(f, index=False)

    def fake(D, book, bcfg, cfg_, crop_root, log=print, dump_pred=None):
        n = len(D)
        return dict(promote=np.array([True, True, False]), relabel=np.zeros(n, bool),
                    lever=np.array(["qn_geo", "np_geo", ""], dtype=object), P=np.array([0.3, np.nan, np.nan]),
                    top1=np.array(["城", "城", ""], dtype=object),
                    wrote={1: dict(image="gold/chon_chu/x.png", image_md5="abcdef012345")}, rep={"promote": 2, "relabel": 0,
                    "theo_don_bay": {}, "crop_moi": 1, "seconds": 0}, blank_syl=np.array([False, True, False]))
    real = CM.process_book
    CM.process_book = fake
    try:
        with redirect_stdout(io.StringIO()):
            CM.run(f, "KimVanKieu1884", REPO / "config/chon_chu.yaml")
    finally:
        CM.process_book = real
    O = pd.read_csv(f, dtype=str, keep_default_na=False)
    chk("qn_geo: GOLD, nhãn kim giữ nguyên, rule += |chon_chu:qn_geo, âm giữ",
        O.tier[0] == "GOLD" and O.label[0] == "城" and O.rule[0].endswith("|chon_chu:qn_geo") and O.syllable[0] == "thành")
    chk("np_geo: GOLD, nhãn = kim, ÂM ĐỂ TRỐNG, âm gốc trong vết, ảnh mới + md5, unicode",
        O.tier[1] == "GOLD" and O.label[1] == "城" and O.syllable[1] == "" and O.rule[1] == "not_plausible|chon_chu:np_geo"
        and O.chon_chu_truoc[1] == "REVIEW||âm:khongdocx" and O.image[1] == "gold/chon_chu/x.png"
        and O.image_md5[1] == "abcdef012345" and O.unicode[1] == "U+57CE" and O.label_level[1] == "char", str(O.iloc[1].to_dict()))
    chk("ô không quyết giữ nguyên, 12 cột giao nộp còn đủ",
        O.tier[2] == "GOLD" and O.rule[2] == "s1_inter_s2_direct" and all(c in O.columns for c in
        ["image", "book", "page", "column", "ocr_char", "syllable", "label", "unicode", "tier", "rule", "bbox", "image_md5"]))


def main() -> int:
    print("=" * 64 + "\nCHON_CHU SELFTEST\n" + "=" * 64)
    tmp = Path(tempfile.mkdtemp(prefix="chon_chu_st_"))
    try:
        test_model()
        test_policy()
        test_io(tmp)
        test_real_config()
        test_geo(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("=" * 64 + f"\nRESULT: {_ok} passed, {len(_fail)} failed" + (f" {_fail}" if _fail else "") + "\n" + "=" * 64)
    return 0 if not _fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
