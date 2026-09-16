"""Selftest B-2 (visual_emission) — kiểm: trung tính, kích thước ma trận, fold đúng trang,
cut() == lab, cost_ij/gap, DP tương thích.

    .venv/bin/python -m pipeline.align_engine.visual_emission_selftest
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipeline.align_engine import anchor_align as aa
from pipeline.align_engine.visual_emission import (
    CAP,
    FOLD_FORMULA,
    GAP_VIS,
    LAMBDA,
    N_FOLDS,
    NEUTRAL_P,
    SZ,
    VisualEmitter,
    build_net,
    cut_box,
    load_page_gray,
    page_fold,
)


def selftest() -> int:
    """Kiểm: trung tính, kích thước ma trận, fold đúng trang, cut() == lab, cost_ij/gap, DP tương thích.
    Không cần mô hình thật: dựng mạng ngẫu nhiên (seed) và ghi 5 fold .pt vào thư mục tạm."""
    passed = failed = 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name}")

    print("[visual_emission selftest]")
    # 1. fold: công thức md5(book|page) % 5 == KhoiB/v3/train_oof_cnn_v3.py
    sys.path.insert(0, str(REPO / "KhoiB/v3"))
    try:
        import train_oof_cnn_v3 as T3
        check("fold_formula == train_oof_cnn_v3.FOLD_FORMULA", T3.FOLD_FORMULA == FOLD_FORMULA)
        pages = [("stt2", f"page_{i:04d}") for i in range(10, 60)] + [("stt11", "page_0010"), ("stt4", "page_0078")]
        check("page_fold == train_oof_cnn_v3.page_fold trên 52 trang",
              all(page_fold(b, p) == T3.page_fold(b, p) for b, p in pages))
        check("page_fold phủ đủ 5 fold", set(page_fold(b, p) for b, p in pages) == set(range(5)))
        # state_dict tương thích với build_net của B-1'
        k_ours = list(build_net(7).state_dict().keys()); k_b1 = list(T3.build_net(7).state_dict().keys())
        check("build_net state_dict keys == train_oof_cnn_v3.build_net", k_ours == k_b1)
    except ImportError as e:
        print(f"  (bỏ qua đối chiếu train_oof_cnn_v3: {e})")
    check("page_fold tất định + trong [0,5)", 0 <= page_fold("stt2", "page_0100") < 5
          and page_fold("stt2", "page_0100") == page_fold("stt2", "page_0100"))
    check("page_fold: 'book|page' ≠ 'book_page' (không nhầm công thức cũ KhoiB/train_oof_cnn.py)",
          any(page_fold(b, p) != int(hashlib.md5(f"{b}_{p}".encode()).hexdigest(), 16) % 5 for b, p in
              [("stt2", f"page_{i:04d}") for i in range(10, 40)]))

    # 2. cut_box == lab cut() (byte) trên hộp thật nếu có ảnh + lab
    lab = REPO / "lab/gan_nhan_2026-09-13"
    png = REPO / "prepared/SachThanhTruyen2/pages/page_0100.png"
    if lab.exists() and png.exists():
        sys.path.insert(0, str(lab))
        try:
            import thi_giac_am_tiet as TG
            gray = load_page_gray(png)
            rng = np.random.default_rng(0)
            same = True
            for _ in range(40):
                x1 = int(rng.integers(50, gray.shape[1] - 120)); y1 = int(rng.integers(50, gray.shape[0] - 120))
                bb = [x1, y1, x1 + int(rng.integers(20, 90)), y1 + int(rng.integers(20, 90))]
                a, b = cut_box(gray, bb), TG.cut(gray, bb)
                same &= (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b))
            check("cut_box == lab thi_giac_am_tiet.cut byte-identical (40 hộp ngẫu nhiên trang thật)", same)
            check("cut_box trả (64,64) uint8", cut_box(gray, [100, 100, 160, 170]).shape == (SZ, SZ))
        except ImportError as e:
            print(f"  (bỏ qua đối chiếu lab cut: {e})")
    check("cut_box hộp rỗng -> None", cut_box(np.full((50, 50), 255, np.uint8), [10, 10, 10, 30]) is None)

    # 3. thiếu mô hình -> available False, reason rõ, không crash; strict -> ném
    with tempfile.TemporaryDirectory() as td:
        em0 = VisualEmitter(models_dir=Path(td) / "khong_co")
        check("thiếu mô hình: available == False + reason nêu fold0..4.pt",
              not em0.available and "fold0..4.pt" in em0.reason)
        try:
            VisualEmitter(models_dir=Path(td) / "khong_co", strict=True)
            check("strict thiếu mô hình -> FileNotFoundError", False)
        except FileNotFoundError:
            check("strict thiếu mô hình -> FileNotFoundError", True)

        # 4. mô hình giả (ngẫu nhiên, seed) 5 fold với 6 lớp
        classes = ["an", "ba", "cha", "mẹ", "người", "trời"]
        torch.manual_seed(0)
        md = Path(td) / "models"; md.mkdir()
        for k in range(N_FOLDS):
            net = build_net(len(classes))
            torch.save({"state": net.state_dict(), "classes": classes, "fold": k, "fold_formula": FOLD_FORMULA}, md / f"fold{k}.pt")
        em = VisualEmitter(models_dir=md, device=torch.device("cpu"))
        check("mô hình giả: available", em.available)
        gray = np.full((300, 200), 255, np.uint8)
        rng = np.random.default_rng(1)
        gray[rng.random(gray.shape) < 0.15] = 0
        boxes = [[20, 20 + 40 * i, 80, 55 + 40 * i] for i in range(6)]
        syls = ["an", "xyz", "Người", "ba", "khongco"]
        logP, info = em.emission(gray, boxes, syls, "stt2", "page_0100")
        check("kích thước logP = (n_box, n_syl)", logP.shape == (6, 5))
        check("fold đúng trang (models fold == page_fold)", info["fold"] == page_fold("stt2", "page_0100"))
        net_used = em._nets[info["fold"]][0]
        check("chỉ nạp 1 mô hình cho 1 trang, cache theo fold", len(em._nets) == 1 and net_used is em.model_for_page("stt2", "page_0100")[0])
        check("âm ∉ lớp -> trung tính log 0,5 (cột 1 và 4)",
              np.allclose(logP[:, 1], math.log(NEUTRAL_P)) and np.allclose(logP[:, 4], math.log(NEUTRAL_P)))
        check("âm ∈ lớp (không phân biệt hoa/thường) -> logP ≤ 0, không trung tính đồng loạt",
              (logP[:, [0, 2, 3]] <= 1e-6).all() and not np.allclose(logP[:, 2], math.log(NEUTRAL_P)))
        check("logP âm ∈ lớp = LP[:, cid] (đúng cột lớp)", np.allclose(logP[:, 2], info["LP"][:, info["cid"]["người"]]))
        check("exp(LP) mỗi hàng tổng 1", np.allclose(np.exp(info["LP"]).sum(1), 1.0, atol=1e-4))
        check("syl_in_classes = [1,0,1,1,0]", info["syl_in_classes"].tolist() == [True, False, True, True, False])
        # hộp không cắt được -> hàng trung tính, NaN trong LP
        logP2, info2 = em.emission(gray, boxes[:2] + [[50, 50, 50, 50]], syls, "stt2", "page_0100")
        check("hộp rỗng -> hàng logP trung tính + LP NaN + argmax ''",
              np.allclose(logP2[2], math.log(NEUTRAL_P)) and np.isnan(info2["LP"][2, 0]) and info2["argmax"][2] == "")
        check("đếm n_cut_fail", em.n_cut_fail == 1 and em.n_boxes == 9)
        lp_sub = em.emission_from_LP(info["LP"], info["cid"], syls, rows=[0, 2, 5])
        check("emission_from_LP(rows) = hàng tương ứng của logP đầy đủ", lp_sub.shape == (3, 5)
              and np.allclose(lp_sub, logP[[0, 2, 5]]))
        # trang khác fold -> mô hình khác
        other = next((b, p) for b, p in [("stt2", f"page_{i:04d}") for i in range(10, 80)] if page_fold(b, p) != info["fold"])
        _ = em.emission(gray, boxes[:1], syls, *other)
        check("trang fold khác -> nạp mô hình fold khác (2 mô hình trong cache)", len(em._nets) == 2)
        # fold sai trong .pt -> ném
        ck = torch.load(md / "fold0.pt"); ck["fold"] = 3; torch.save(ck, md / "fold0.pt")
        em_bad = VisualEmitter(models_dir=md, device=torch.device("cpu"))
        pg0 = next((b, p) for b, p in [("stt2", f"page_{i:04d}") for i in range(10, 80)] if page_fold(b, p) == 0)
        try:
            em_bad.model_for_page(*pg0); check("fold0.pt ghi fold=3 -> RuntimeError", False)
        except RuntimeError:
            check("fold0.pt ghi fold=3 -> RuntimeError", True)

        # 5. chi phí: cost_ij = text + λ·min(−logP, cap); khe = 8,6 + λ·8
        V = em.cost_matrix(logP)
        check("cost_matrix = λ·min(−logP, CAP) (trung tính = λ·log2 ≈ 0,173)",
              np.allclose(V[:, 1], LAMBDA * math.log(2)) and (V <= LAMBDA * CAP + 1e-6).all())
        cij = em.make_cost_ij(lambda c, s: 6.7, logP)
        check("cost_ij(i,j) = text + ảnh", abs(cij(0, 1, "人", "xyz") - (6.7 + LAMBDA * math.log(2))) < 1e-6)
        cd, ci = em.gap_costs()
        check("khe = COST_DEL + λ·8 = 10,6", abs(cd - (aa.COST_DEL + 0.25 * 8.0)) < 1e-9 and abs(ci - cd) < 1e-9)
        em_cap = VisualEmitter(models_dir=md, device=torch.device("cpu"), cap=0.1)
        check("cap ghi đè: cost ≤ λ·cap", (em_cap.cost_matrix(logP) <= 0.25 * 0.1 + 1e-6).all())

        # 6. DP tương thích: cost_ij mọi ô = text (ảnh 0) -> ops == realign_column mặc định;
        #    ảnh rất mạnh kéo đường ghép đổi; posterior cùng cost: argmax hàng == Viterbi
        qn = {"an": ["安"], "ba": ["巴"], "xa": ["車"], "tam": ["三"]}
        chars = ["安", "巴", "車", "三"]; s_ = ["an", "ba", "xa", "tam"]
        ops0 = aa.realign_column(chars, s_, qn)
        ops1 = aa.realign_column(chars, s_, qn, cost_ij=lambda i, j, c, s: aa.substitution_cost(c, s, qn))
        check("cost_ij == text -> ops y hệt mặc định", ops0 == ops1)
        # ảnh: hộp 1 nói 'xa' rất chắc, hộp 2 nói 'ba' -> ghép chéo (1↔2, 2↔1) không thể (đơn điệu)
        # -> kiểm bằng trường hợp thiếu 1 chữ: chars thiếu 巴; ảnh kéo khe đúng vị trí
        c_drop = ["安", "車", "三"]
        lp = np.full((3, 4), math.log(0.5), np.float32)
        lp[0, 0] = lp[2, 3] = math.log(0.99)                     # hộp 0 = 'an', hộp 2 = 'tam'
        lp[1, 2] = math.log(0.99); lp[1, 1] = math.log(1e-4)    # hộp 1 = 'xa', chắc chắn không 'ba'
        emx = VisualEmitter(models_dir=md, device=torch.device("cpu"), lam=1.0)
        cij2 = emx.make_cost_ij(lambda c, s: aa.COST_NODICT, lp)   # văn bản mù (mọi cặp 5,1)
        cd2, ci2 = emx.gap_costs()
        ops_v = aa.realign_column(c_drop, s_, {}, cost_ij=cij2, cost_del=cd2, cost_ins=ci2)
        inss = [o["syl_idx"] for o in ops_v if o["op"] == "ins"]
        m_v = {(o["nom_idx"], o["syl_idx"]) for o in ops_v if o["op"] == "match"}
        check("văn bản mù + ảnh: khe đúng j=1 và (1,2) ghép", inss == [1] and (1, 2) in m_v)
        post = aa.posterior_matches(c_drop, s_, {}, cost_ij=cij2, cost_del=cd2, cost_ins=ci2)
        row1 = {j: p for (i, j), p in post.items() if i == 1}
        check("posterior cùng cost: argmax hàng 1 == Viterbi (j=2), Σp ≤ 1",
              max(row1, key=row1.get) == 2 and sum(row1.values()) <= 1 + 1e-9)
        check("posterior_matches mặc định (không cost_ij) không đổi",
              aa.posterior_matches(chars, s_, qn) == aa.posterior_matches(chars, s_, qn, cost_ij=None))
        # cost_del/ins ghi đè có tác dụng: khe rất đắt -> không khe (ghép hết dù lệch)
        ops_nogap = aa.realign_column(c_drop, s_, {}, cost_ij=cij2, cost_del=1e6, cost_ins=1e6)
        check("cost_ins=1e6: vẫn phải có khe vì m≠n (ins bắt buộc 1)", sum(o["op"] == "ins" for o in ops_nogap) == 1)
        # emitter mặc định (lam 0.25) tương đương dp_vis5 CAP/λ/gap
        check("hằng mặc định = dp_vis5 (λ 0,25 · CAP 12 · gap 8 · C0 0,5)",
              (LAMBDA, CAP, GAP_VIS, NEUTRAL_P) == (0.25, 12.0, 8.0, 0.5))
        _ = json.dumps({"c": Counter()})  # noqa: F841
    print(f"RESULT: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
