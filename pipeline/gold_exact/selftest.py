"""selftest.py — kiểm nhanh gói gold_exact (vài giây, 0 API, không đọc dataset/): tài sản + sha256, PARAMS đóng băng, crop chuẩn
trên trang tổng hợp, core_loss, thứ tự luật chính sách, V1+, đặc trưng viss, M-OCR theo thứ tự cổng, đo IHR, chặn --out trong dataset/;
(27/09) core_loss KHÔNG là cổng theo mặc định (gate false/vắng) + tái lập run1 khi gate true, phân rã ok = lai − luật A − M-OCR,
config thật (gate false, sha định nghĩa = bản đăng ký), khối tài liệu, publish trên thư mục tạm (dọn đúng đầu ra, md5 crop,
bản lọc theo bộ, CHECKSUMS giữ dòng labels.csv, labels.csv đổi -> PublishError).
(28/09, rà soát N1–N6) TA không nhìn GT IHR (has_gt lật -> cùng kết quả; tài sản không có cột GT), kiểm TOÀN BỘ MANIFEST lúc
khởi động (tệp hỏng -> AssetError liệt kê), ảnh thiếu -> MissingImageError (không ảnh trắng), dọn bản lọc theo bộ (N2), N3 tập
GOLD bộ nguồn ≠ bản lọc -> PublishError, verify_crops, CSV %.17g tái lập đúng bit, is_inside (dataset_out ≠ dataset), cache
hỏng -> tính lại, khối tài liệu bộ nguồn đổi theo tệp có/không. In dòng `RESULT:` cho run_all_selftests.
Chạy: .venv/bin/python -m pipeline.gold_exact --selftest   (hoặc python -m pipeline.gold_exact.selftest)
"""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pandas as pd

PARAMS_MD5 = "0e6a2cf3d2000a26c55998554b47c714"   # md5(json PARAMS) = cclib_v2.PARAMS; đổi PARAMS = FAIL


def _params_md5():
    from .crop_chuan import PARAMS
    return hashlib.md5(json.dumps(PARAMS, sort_keys=True).encode()).hexdigest()


def _page():
    """Trang tổng hợp 400×200: một cột 3 chữ (khối mực) + nền giấy xám nhạt."""
    G = np.full((400, 200), 235, np.uint8)
    boxes = []
    for k, y in enumerate((60, 180, 300)):
        cv2.rectangle(G, (80, y), (120, y + 40), 20, -1)
        cv2.rectangle(G, (92, y + 10), (108, y + 30), 235, -1)      # lỗ trong chữ
        boxes.append([78, y - 2, 122, y + 42])
    return G, boxes


def run() -> int:
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)
    from . import crop_chuan as CC, policy as POL, signals_geom as SG, signals_img as SI, signals_text as ST
    from .common import Assets, AssetError, set_lexicon, var_eq_plus, uid_path, REPO
    # ---- tài sản
    try:
        A = Assets(); set_lexicon(A)
        for n in A.man["files"]:
            A.path(n)
        for n in A.man["external"]:
            A.ext(n)
        check("assets_sha256", True)
        check("assets_invariants_pass", not A.man.get("invariants_fail"))
        check("hand_thr_00015", abs(A.load("hand_tables_T_Kinh.pt")["thr"]["0.00015"] - 0.99433) < 5e-6)
    except AssetError as e:
        print("AssetError:", e); check("assets_sha256", False)
    # ---- PARAMS đóng băng (giá trị cclib_v2)
    P = CC.PARAMS
    check("params_md5", _params_md5() == PARAMS_MD5)
    check("params_frozen", (P["virt"], P["cap"], P["seam_pen"], P["win_ext"], P["line_v"], P["two_ext"], P["margin"])
          == (0.55, 0.8, 20.0, 0.35, 1.2, 1.35, 0.10))
    # ---- crop chuẩn trên trang tổng hợp
    G, boxes = _page()
    src = cv2.cvtColor(G, cv2.COLOR_GRAY2BGR)
    r1 = CC.canon(G, src, boxes[1], boxes, [], None)
    r2 = CC.canon(G, src, boxes[1], boxes, [], None)
    check("canon_ok", r1["status"] == "ok" and r1["flags"] == "")
    check("canon_deterministic", hashlib.md5(r1["sq_img"].tobytes()).hexdigest() == hashlib.md5(r2["sq_img"].tobytes()).hexdigest())
    check("canon_square", r1["sq_img"].shape[0] == r1["sq_img"].shape[1] and r1["sq128"].shape[:2] == (128, 128))
    check("canon_ink_center", abs(r1["ink_cy"] - 200.5) < 2 and abs(r1["ink_cx"] - 100.5) < 2)
    oc = CC.old_crop(src, G, boxes[1], boxes[0], boxes[2])
    n_core, kept, loss = SG.core_loss_of(G, boxes[1], r1, oc)
    check("core_loss_zero_when_kept", n_core > 0 and loss == 0.0)
    wx0, wy0, M = r1["M_win"]
    r3 = dict(r1); r3["M_win"] = (wx0, wy0, np.zeros_like(M))
    check("core_loss_one_when_dropped", SG.core_loss_of(G, boxes[1], r3, oc)[2] == 1.0)
    check("uid_path", uid_path("A/b/page_0001/c1/n2/s3") == "A/b/page_0001/c1_n2_s3.png")
    # ---- V1+
    check("v1plus_same", var_eq_plus("國", "國"))
    check("v1plus_simp", var_eq_plus("國", "国"))
    check("v1plus_diff", not var_eq_plus("國", "人"))
    # ---- chính sách: thứ tự luật
    S8 = np.array(["L16", "L16", "L16", "L16", "TK", "stt2", "KVK"], dtype=object)
    z = np.zeros(7, bool)
    sig = {k: z.copy() for k in ("int_foreign", "rescue", "similar", "weak_text", "mocr", "f_blank", "f_cut", "dup_bbox",
                                 "ov_heavy", "f_two", "f_ink", "bleed_new", "trunc_new", "tall_new", "cnt", "bc", "core_loss_flag")}
    sig["one_char_ok"] = ~z; sig["simg"] = ~z; sig["lobo_nh"] = np.full(7, 5.0)
    sig["ta"] = np.array(["na", "na", "na", "na", "attested", "na", "contradicted"], dtype=object)
    sig["rescue"][0] = True; sig["bc"][0] = True                 # review thắng text_only
    sig["core_loss_flag"][1] = True; sig["simg"][1] = False      # (cổng bật) text_only thắng uncertified
    sig["simg"][2] = False
    sig["core_loss_flag"][3] = True                              # ô sạch CHỈ mang cờ core_loss
    cfg_on = dict(thresholds=dict(stt_n_hum_min=3), core_loss=dict(gate=True))     # = lượt run1 (policy 2026-09-27)
    dec, why, M_ = POL.decide(S8, sig, cfg_on)
    check("policy_order_gate_on", list(dec) == ["review", "text_only", "uncertified", "text_only", "ok", "ok", "uncertified"])
    check("policy_reason_core_gate_on", why[1] == "core_loss_crop_chuan_cat_vao_chu" and why[3] == why[1]
          and why[6] == "U_di_ban_nguoi_chong_nhan")
    check("policy_lai_mask", bool(M_["lai"][3]) and not bool(M_["lai"][6]))
    # QUYẾT ĐỊNH 27/09: core_loss KHÔNG là cổng (mặc định, vắng khoá = false) -> ô chỉ mang cờ core_loss là ok
    for nm, cfg in (("gate_false", dict(thresholds=dict(stt_n_hum_min=3), core_loss=dict(gate=False))),
                    ("gate_absent", dict(thresholds=dict(stt_n_hum_min=3)))):
        d2, w2, M2 = POL.decide(S8, sig, cfg)
        check(f"policy_core_loss_not_gate_{nm}", list(d2) == ["review", "uncertified", "uncertified", "ok", "ok", "ok", "uncertified"]
              and "core_loss_crop_chuan_cat_vao_chu" not in set(w2))
        check(f"policy_decomposition_{nm}", POL.decomposition(d2 == "ok", M2["lai"], M2["AINT"], sig["mocr"], sig["core_loss_flag"],
                                                               POL.core_loss_gate(cfg)))
    check("policy_decomposition_gate_on", POL.decomposition(dec == "ok", M_["lai"], M_["AINT"], sig["mocr"], sig["core_loss_flag"], True)
          and not POL.decomposition(dec == "ok", M_["lai"], M_["AINT"], sig["mocr"], sig["core_loss_flag"], False))
    # cấu hình thật: cổng tắt, định nghĩa core_loss_of trùng bản đăng ký, policy_version đã tăng
    from .common import load_cfg
    import inspect as _insp
    rc = load_cfg()
    check("config_core_loss_gate_false", POL.core_loss_gate(rc) is False and str(rc["version"]) != "2026-09-27")
    check("config_core_loss_prereg_sha", hashlib.sha256(_insp.getsource(SG.core_loss_of).encode()).hexdigest()
          == rc["core_loss"]["prereg"]["code_sha256"])
    check("evidence", list(POL.evidence(["L16", "KVK", "stt4"])) == ["do_tren_nhan_nguoi", "uoc_luong", "suy_doan"])
    th = dict(C5_T=[0.98744, 0.09307], C5_L=[0.95369, 0.13467], C2_T=0.98835, C2_L=0.9745, stt_n_hum_min=3)
    s = dict(p_wood_T=np.array([0.99, 0.98, 0.99, 0.99]), p_wood_L=np.array([0.99, 0.99, 0.96, 0.96]),
             viss_T=np.array([0.1, 0.1, np.nan, 0.2]), viss_L=np.array([np.nan, np.nan, 0.2, 0.2]),
             viss_X=np.array([np.nan, np.nan, np.nan, 0.14]), lobo_cert=np.array([np.nan] * 4), lobo_nh=np.array([np.nan] * 4))
    check("simg_thresholds", list(POL.simg(np.array(["L16", "L16", "TK", "KVK"], dtype=object), s, th)) == [True, False, True, True])
    # ---- viss: max thứ hai theo nhóm
    g = np.array([0, 0, 0, 1, 2, 2]); v = np.array([0.2, 0.5, 0.5, 0.1, 0.3, 0.1])
    mx, s2 = SI._second_max(v, g)
    check("viss_second_max", list(mx) == [0.5, 0.5, 0.5, 0.1, 0.3, 0.3] and list(s2) == [0.5, 0.5, 0.5, -1.0, 0.1, 0.1])
    # ---- text_attested / M-OCR theo thứ tự cổng
    check("ta_classify", ST.classify("國", ["國"], "quốc")[0] == "attested" and ST.classify("國", [], "quốc")[0] == "unattestable")
    Gm = pd.DataFrame(dict(book_set=["KimVanKieu1884"] * 3 + ["LucVanTien1916"]))
    rules = dict(rescue=np.array([1, 0, 0, 0], bool), similar=np.zeros(4, bool), weak_text=np.zeros(4, bool))
    mg = ST.mocr_gate(Gm, np.ones(4, bool), np.zeros(4, bool), rules, np.array(["", "", "blank", ""], dtype=object),
                      np.zeros(4, bool), np.zeros(4), np.zeros(4), np.zeros(4),
                      {"KimVanKieu1884": {"kim": "0.5dx", "vis": -0.5, "mocr": True}, "LucVanTien1916": {"kim": 0.6, "vis": -1.0, "mocr": False}})
    check("mocr_gate_order", list(mg) == [False, True, False, False])
    # ---- đo IHR
    from .eval_ihr import measure
    H = pd.DataFrame(dict(book_set=["LucVanTien1916"] * 4 + ["TruyenKieu1872"] * 2, page=["p1", "p1", "p2", "p2", "p1", "p2"],
                          gt_char=["a", "b", "c", "", "e", "f"], y_lab=[True, False, True, False, True, True],
                          y_str=[True, False, False, False, True, True], slot_new=["1", "1", "1", "1", "", "1"]))
    r = measure(np.array([1, 1, 1, 1, 1, 1], bool), H, "slot_new", 200, 1)
    check("ihr_measure", r["L16"]["n_eval"] == 3 and abs(r["L16"]["both_pt"] - 2 / 3) < 1e-9 and abs(r["TK"]["both_pt"] - 0.5) < 1e-9
          and abs(r["L16"]["strict_pt"] - 1 / 3) < 1e-9)
    # ---- CLI chặn --out trong dataset/
    from .__main__ import main as cli
    try:
        cli(["--out", str(REPO / "dataset/_x"), "--all-dir", str(REPO / "dataset/_ALL")])
        check("cli_blocks_dataset_out", False)
    except SystemExit as e:
        check("cli_blocks_dataset_out", "dataset" in str(e))
    # ---- N6: TA không phụ thuộc GT người IHR
    _ta_tests(check, ST)
    # ---- N1/N5/nhỏ: MANIFEST toàn bộ, ảnh thiếu, cache hỏng, is_inside
    _robust_tests(check)
    _borg_tests(check, POL)
    # ---- khối tài liệu README/DATASHEET: mức chắc trùng common.EVIDENCE, không đụng chuỗi tầng text-only của bộ giao nộp
    from . import doc_text as DT
    from .common import EVIDENCE
    inv_doc = {s: lv for lv, ss in DT.EVIDENCE_DOC.items() for s in ss}
    check("doc_evidence_eq_common", inv_doc == EVIDENCE)
    check("doc_block_no_tier_string", all("GOLD_text_only" not in DT.block_md(w) for w in ("all", "book"))
          and "GOLD_text_only" not in DT.datasheet_line() and all(k in DT.block_md() for k, _ in DT.TRANG_THAI)
          and all("GOLD_text_only" not in DT.book_section(p) + DT.datasheet_block(p) for p in (True, False)))
    check("doc_limits_lt2_and_range", all(k in DT.block_md(w) for w in ("all", "book") for k in ("lt2", "1.343–6.670", "9.200–12.137"))
          and "không có** `gold_exact.csv`" in DT.book_section(False) and "`uncertified`" in DT.book_section(True))
    # ---- publish (bản giao vào dataset/_ALL + bộ nguồn) trên thư mục tạm
    _publish_tests(check)
    print(json.dumps(dict(passed=ok, failed=fail, params_md5=_params_md5()), ensure_ascii=False))
    print(f"RESULT: {ok} passed, {len(fail)} failed")
    return 0 if not fail else 1


def _publish_tests(check):
    import tempfile
    from pathlib import Path
    from . import publish as PUB
    from .__main__ import csv_frame, CSV_COLS
    from .common import uid_path
    from pipeline.tools.merge_datasets import _checksums
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root, cache = td / "dataset", td / "cache"
        alld = root / "_ALL"
        alld.mkdir(parents=True)
        (root / "BoA").mkdir()
        (root / "BoA" / "labels.csv").write_text("image,tier\ngold/0.png,GOLD\ngold/1.png,GOLD\nsyllable/z.png,SYLLABLE\n",
                                                encoding="utf-8")
        (root / "BoA" / "README.md").write_text("# BoA\n\n" + __import__("pipeline.gold_exact.doc_text", fromlist=["x"])
                                               .book_section(False) + "\n## Sau\n", encoding="utf-8")
        (root / "BoC").mkdir(); (root / "BoC" / "labels.csv").write_text("image,tier\n", encoding="utf-8")
        (root / "BoC" / "gold_exact.csv").write_text("cell_uid,gold_exact,crop_chuan\nq,ok,../_ALL/crops_chuan/BoC/q.png\n",
                                                   encoding="utf-8")
        uids = ["BoA/boa/page_0001/c1/n0/s0", "BoA/boa/page_0001/c1/n1/s1", "BoB/bob/page_0002/c2/n0/s0"]
        L = pd.DataFrame(dict(cell_uid=uids, image=[f"crops/{u.split('/')[0]}/gold/{i}.png" for i, u in enumerate(uids)],
                              tier="GOLD"))
        L.to_csv(alld / "labels.csv", index=False)
        _checksums(alld)
        (alld / "VIEC_KHAC.txt").write_text("giữ", encoding="utf-8")
        (alld / "crops_chuan").mkdir(); (alld / "crops_chuan" / "cu.png").write_bytes(b"old")
        check("pub_check_labels_ok", len(PUB.check_labels(alld)) == 64)
        gone = PUB.clean(alld, root, ["BoA", "BoB"])
        check("pub_clean_only_outputs", "crops_chuan/" in gone and (alld / "VIEC_KHAC.txt").exists()
              and (alld / "labels.csv").exists() and not (alld / "crops_chuan").exists())
        check("pub_clean_per_book_N2", "BoC/gold_exact.csv" in gone and not (root / "BoC" / "gold_exact.csv").exists()
              and (root / "BoC" / "labels.csv").exists())
        md = {}
        for i, u in enumerate(uids):
            for sub, b in (("sq", bytes([1, i])), ("sq128", bytes([2, i]))):
                f = cache / sub / uid_path(u); f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(b)
            md[u] = hashlib.md5(bytes([1, i])).hexdigest()
        try:
            PUB.export_crops(uids[:1], {uids[0]: "sai"}, cache / "sq", cache / "sq128", [])
            check("pub_md5_mismatch_raises", False)
        except PUB.PublishError:
            check("pub_md5_mismatch_raises", True)
        m1, m2 = PUB.export_crops(uids[:2], md, cache / "sq", cache / "sq128", [alld])
        check("pub_export_crops", m1[uids[1]] == md[uids[1]] and (alld / "crops_chuan_128" / uid_path(uids[1])).exists()
              and m2[uids[0]] == hashlib.md5(bytes([2, 0])).hexdigest())
        # khung X/G tối thiểu -> csv_frame (bool -> 0/1, đường crop chỉ cho ô ok)
        X = pd.DataFrame({c: [False] * 3 for c in CSV_COLS})
        X["cell_uid"] = uids; X["book_set"] = [u.split("/")[0] for u in uids]; X["gold_exact"] = ["ok", "ok", "text_only"]
        X["status"] = ["ok", "ok", "ok"]; X["core_loss_flag"] = [True, False, False]
        G = pd.DataFrame(dict(image=L.image, img_md5=["a", "b", "c"]))
        X["p_wood_T"] = [0.98744 - 2.0 ** -40, 1 / 3, float("nan")]
        P = csv_frame(X, G, m1, m2, "vX", "f" * 64, "mps")
        check("pub_csv_frame", list(P.columns) == CSV_COLS and P.core_loss_flag.tolist() == [1, 0, 0]
              and P.crop_chuan.tolist()[2] == "" and P.crop_chuan_md5.tolist()[0] == md[uids[0]]
              and P.config_sha16.iloc[0] == "f" * 16 and P.policy_version.iloc[0] == "vX" and set(P.device) == {"mps"})
        PUB._write_csv(P, td / "rt.csv")
        from .common import to_float
        rt = to_float(pd.read_csv(td / "rt.csv", dtype=str, keep_default_na=False).p_wood_T)
        check("pub_csv_float_roundtrip_exact", rt[0] == 0.98744 - 2.0 ** -40 and rt[0] < 0.98744 and rt[1] == 1 / 3 and np.isnan(rt[2]))
        vc = PUB.verify_crops(P, alld)
        check("pub_verify_crops", vc["PASS"] and vc["n"] == 4)
        (alld / P.crop_chuan.iloc[0]).write_bytes(b"hong")
        check("pub_verify_crops_detects_bad", PUB.verify_crops(P, alld)["md5_bad"] == 1)
        (alld / P.crop_chuan.iloc[0]).write_bytes(bytes([1, 0]))
        mdp = td / "GOLD_EXACT.md"; mdp.write_text("# x", encoding="utf-8")
        sha0 = PUB.check_labels(alld)
        # N3: tập GOLD của labels.csv bộ nguồn lệch bản lọc -> PublishError, KHÔNG ghi gì
        lab_ok = (root / "BoA" / "labels.csv").read_text(encoding="utf-8")
        (root / "BoA" / "labels.csv").write_text(lab_ok + "gold/9.png,GOLD\n", encoding="utf-8")
        try:
            PUB.publish(P, alld, mdp, root, log=lambda *a: None)
            check("pub_N3_gold_set_mismatch_raises", False)
        except PUB.PublishError:
            check("pub_N3_gold_set_mismatch_raises", not (root / "BoA" / "gold_exact.csv").exists()
                  and not (alld / "gold_exact.csv").exists())
        (root / "BoA" / "labels.csv").write_text(lab_ok, encoding="utf-8")
        r = PUB.publish(P, alld, mdp, root, log=lambda *a: None)
        B = pd.read_csv(root / "BoA" / "gold_exact.csv", dtype=str, keep_default_na=False)
        check("pub_per_book", r["per_book"].get("BoA", {}).get("rows") == 2 and "BoB" not in r["per_book"]
              and B.image.tolist()[0] == "gold/0.png" and B.crop_chuan.tolist()[0].startswith("../_ALL/crops_chuan/")
              and r["per_book"]["BoA"]["gold_set_eq_labels"])
        rd_ = (root / "BoA" / "README.md").read_text(encoding="utf-8")
        check("pub_per_book_readme_present", "`uncertified`" in rd_ and "không có** `gold_exact.csv`" not in rd_ and "## Sau" in rd_)
        PUB.clean(alld, root, ["BoA"])
        rd_ = (root / "BoA" / "README.md").read_text(encoding="utf-8")
        check("pub_clean_per_book_readme_absent", not (root / "BoA" / "gold_exact.csv").exists()
              and "không có** `gold_exact.csv`" in rd_ and "## Sau" in rd_)
        _checksums(alld)
        m1, m2 = PUB.export_crops(uids[:2], md, cache / "sq", cache / "sq128", [alld])
        r = PUB.publish(P, alld, mdp, root, log=lambda *a: None)
        ck = (alld / "CHECKSUMS.txt").read_text(encoding="utf-8")
        check("pub_checksums", PUB.checksums_labels_sha(alld) == sha0 and " gold_exact.csv" in ck and " GOLD_EXACT.md" in ck
              and r["n_png"] == {"crops_chuan": 2, "crops_chuan_128": 2})
        (alld / "labels.csv").write_text("cell_uid\nđã sửa\n", encoding="utf-8")
        try:
            PUB.check_labels(alld)
            check("pub_labels_changed_raises", False)
        except PUB.PublishError:
            check("pub_labels_changed_raises", True)



def _ta_tests(check, ST):
    """N6: quyết định TA của TK chỉ dựa văn bản người dị bản; khoá GT (has_gt) nếu có cũng bị bỏ qua; tài sản không chứa GT."""
    from .common import Assets
    G = pd.DataFrame(dict(cell_uid=["t1", "t2", "t3", "k1"], book_set=["TruyenKieu1872"] * 3 + ["KimVanKieu1884"],
                          label=["國", "國", "國", "國"], syllable=["quốc"] * 4))
    z = dict(cell_uid=["t1", "t2", "t3", "k1"], book=["TruyenKieu1872"] * 3 + ["KimVanKieu1884"], syllable=["quốc"] * 4,
             ref71=["", "", "", "國"], ref72=["", "", "", ""], ref16=["", "", "", ""], rev_h71=["國", "國", "", ""],
             convmap={"L16": [], "TK": []})
    ta0, _ = ST.text_attested(G, None, dict(z))
    ta1, _ = ST.text_attested(G, None, dict(z, has_gt=[0, 1, 0, 0]))
    ta2, _ = ST.text_attested(G, None, dict(z, has_gt=[1, 0, 1, 1]))
    check("ta_tk_no_gt_dependency", list(ta0) == ["attested", "attested", "unattestable", "attested"]
          and list(ta0) == list(ta1) == list(ta2))
    try:
        zz = Assets().load("ta_refs.pt")
        check("ta_refs_asset_has_no_ihr_gt", "has_gt" not in zz and "gt_char" not in zz)
        cov = ST.ta_coverage(G, z)
        check("ta_coverage", cov["TruyenKieu1872"]["with_ref"] == 2 and cov["KimVanKieu1884"]["frac"] == 1.0)
    except Exception as e:  # noqa: BLE001
        print("ta asset:", e); check("ta_refs_asset_has_no_ihr_gt", False)


def _robust_tests(check):
    import tempfile
    from pathlib import Path
    from .common import Assets, AssetError, EmbCache, REPO, is_inside, load_pickle_safe, dump_pickle_atomic, sha256_file
    from . import signals_img as SI
    check("is_inside_dataset_out_not_blocked", not is_inside(REPO / "dataset_out/x", REPO / "dataset")
          and is_inside(REPO / "dataset/_x", REPO / "dataset") and is_inside(REPO / "dataset", REPO / "dataset"))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # ảnh thiếu / rỗng -> MissingImageError (không ảnh trắng)
        (td / "rong.png").write_bytes(b"")
        for nm, f in (("missing", lambda: SI.read_crop(td / "khong_co.png")), ("empty", lambda: SI.read_crop(td / "rong.png")),
                      ("prep64_none", lambda: SI.prep64(None))):
            try:
                f(); check(f"n5_{nm}_raises", False)
            except SI.MissingImageError:
                check(f"n5_{nm}_raises", True)
        check("n5_check_images", SI.check_images([td / "khong_co.png", td / "rong.png"]) == [str(td / "khong_co.png"),
                                                                                             str(td / "rong.png")])
        # MANIFEST toàn bộ: một tệp hỏng + một tệp thiếu -> AssetError liệt kê CẢ HAI trước khi tính
        (td / "a.pt").write_bytes(b"aaaa"); (td / "b.pt").write_bytes(b"bbbb")
        man = dict(files={n: dict(sha256=sha256_file(td / n), bytes=4) for n in ("a.pt", "b.pt")}, external={})
        man["files"]["c.pt"] = dict(sha256="0" * 64, bytes=4)
        (td / "MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")
        (td / "b.pt").write_bytes(b"bbbX")
        try:
            Assets(td).verify_all(); check("assets_verify_all_raises", False)
        except AssetError as e:
            check("assets_verify_all_raises", "b.pt" in str(e) and "c.pt" in str(e) and str(e).startswith("2 "))
        # cache hỏng -> tính lại (không ném)
        (td / "enc_0123456789abcdef.npz").write_bytes(b"khong phai npz")
        c = EmbCache(td, "enc", "0123456789abcdef" * 4)
        check("embcache_corrupt_recovers", c.d == {} and c.missing(["k"]) == ["k"])
        (td / "p.pkl").write_bytes(b"\x80\x04hong")
        check("pickle_corrupt_none", load_pickle_safe(td / "p.pkl", lambda *a: None) is None)
        dump_pickle_atomic(dict(key="k", out=1), td / "q" / "p.pkl")
        check("pickle_atomic_roundtrip", load_pickle_safe(td / "q" / "p.pkl") == dict(key="k", out=1))


def _borg_tests(check, POL):
    """27/09: Borg chép tay (B18/B34) — LOBO-sách của bộ kiểm chữ viết tay, thư mục _auto, mức chắc, lý do riêng, 8 bộ cũ không đổi."""
    import numpy as np
    from .common import (AB, AUTO_PREP, BORG8, EVIDENCE, HAND_TRAIN_BOOK, HAND_VARIANT, REPO, Assets, page_dir, prep_root, set8_of,
                         sets_in, SETS8)
    inv_ab = {v: k for k, v in AB.items()}
    # LOBO: sách đang chấm KHÔNG bao giờ là sách học của biến thể
    check("borg_lobo_variant_other_book", all(HAND_TRAIN_BOOK[HAND_VARIANT[s]] != inv_ab[s] for s in BORG8))
    check("stt_variant_kinh_unchanged", all(HAND_VARIANT[s] == "Kinh" for s in ("stt2", "stt4", "stt11")))
    try:
        man = Assets().man
        inv = man.get("invariants", {})
        ok_ = True
        for s in BORG8:
            var = HAND_VARIANT[s]
            for t in "TL":
                pb = inv.get(f"hand_{t}_{var}_proto_books"); cb = inv.get(f"hand_{t}_{var}_cal_books")
                ok_ &= (f"hand_model_{t}_{var}.pt" in man["files"] and f"hand_tables_{t}_{var}.pt" in man["files"]
                        and pb == [HAND_TRAIN_BOOK[var]] and cb == [HAND_TRAIN_BOOK[var]] and inv_ab[s] not in (pb or []) + (cb or []))
        check("borg_lobo_assets_proto_cal_other_book", ok_)
    except Exception as e:  # noqa: BLE001
        print("borg assets:", e); check("borg_lobo_assets_proto_cal_other_book", False)
    check("borg_page_dir_auto", page_dir("SachKinhThayCaBinh", "sachkinhthaycabinh") == REPO / "prepared/_auto/SachKinhThayCaBinh"
          and prep_root("SachDungLyHoThan") == REPO / "prepared/_auto/SachDungLyHoThan"
          and page_dir("LucVanTien1916", "lucvantien1916") == REPO / "prepared/LucVanTien1916"
          and page_dir("SachThanhTruyen", "stt4") == REPO / "prepared/SachThanhTruyen4")
    check("borg_set8_evidence", list(set8_of(["SachKinhThayCaBinh", "SachDungLyHoThan"], ["x", "y"])) == ["B18", "B34"]
          and all(EVIDENCE[s] == "do_tren_nhan_nguoi" for s in BORG8) and set(AUTO_PREP) == {"SachKinhThayCaBinh", "SachDungLyHoThan"})
    check("sets_in_old_unchanged", sets_in(np.array(SETS8)) == SETS8 and sets_in(np.array(["B34", "stt2"])) == SETS8 + ["B34"])
    # chính sách: Borg dùng lobo_* (không p_wood), lý do riêng; STT/in giữ lý do cũ
    S8 = np.array(["B18", "B18", "B34", "stt2", "L16"], dtype=object)
    s = dict(p_wood_T=np.array([np.nan, np.nan, np.nan, np.nan, 0.999]), p_wood_L=np.full(5, np.nan), viss_T=np.array([0, 0, 0, 0, 0.5]),
             viss_L=np.zeros(5), viss_X=np.zeros(5), lobo_cert=np.array([1, 0, 1, 0, np.nan]), lobo_nh=np.array([5, 5, 1, 5, np.nan]))
    th = dict(C5_T=[0.98744, 0.09307], C5_L=[0.95369, 0.13467], C2_T=0.98835, C2_L=0.9745, stt_q=0.00015, stt_n_hum_min=3)
    si = POL.simg(S8, s, th)
    check("borg_simg_hand_only", list(si) == [True, False, False, False, True])
    n = len(S8)
    z = np.zeros(n, bool)
    sig = {k: z for k in ("int_foreign", "rescue", "similar", "weak_text", "mocr", "f_blank", "f_cut", "dup_bbox", "ov_heavy", "f_two",
                          "f_ink", "bleed_new", "trunc_new", "tall_new", "cnt", "bc", "core_loss_flag")}
    sig.update(one_char_ok=np.ones(n, bool), ta=np.array(["na"] * n, dtype=object), lobo_nh=s["lobo_nh"], simg=si)
    dec, why, _ = POL.decide(S8, sig, dict(thresholds=th, core_loss=dict(gate=False)))
    check("borg_decide_reasons", list(dec) == ["ok", "uncertified", "uncertified", "uncertified", "ok"]
          and why[1] == "U_Borg_bo_kiem_viet_tay_LOBO_khong_chung_nhan" and why[2] == "U_Borg_thieu_nguyen_mau_nguoi_LOBO"
          and why[3] == "U_STT_bo_kiem_viet_tay_khong_chung_nhan")


if __name__ == "__main__":   # cuối tệp: mọi hàm phụ (_ta_tests, _robust_tests, _borg_tests) đã được định nghĩa
    raise SystemExit(run())
