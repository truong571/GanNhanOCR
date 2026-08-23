"""Điểm NGHĨA HÁN cho cặp (chữ, âm) — kênh xác nhận MỘT CHIỀU, không phải cổng lỗi.

VÌ SAO CHỈ MỘT CHIỀU
--------------------
Chữ Nôm phần lớn được chọn theo ÂM, không theo nghĩa. Từ điển cho 20 chữ đọc âm "người":
仉 (họ mẹ Mạnh Tử), 倘/儻 (nếu như), 匕 (cái thìa), 命 (số mệnh), 昆 (anh cả), 皚 (trắng xoá)
— không chữ nào nghĩa là "người". Nên "hồ sơ nghĩa" của một âm tiết phần lớn là nhiễu.

Đo được trên chuẩn 2.002 cặp (âm khớp-âm-tiết: giữ nguyên âm, chỉ đổi chữ, nên hồ sơ nghĩa
giống hệt nhau):

    điểm nghĩa                  AUC 0,742
    đối chứng độ dài định nghĩa AUC 0,432   <- dưới ngẫu nhiên, không phải giả tạo
    đối chứng tổng IDF          AUC 0,421

    ngưỡng > 0,02  ->  độ chính xác  98,9%   (giữ 724/2002 cặp đúng)
    ngưỡng > 0,05  ->  độ chính xác 100,0%   (giữ 303/2002 cặp đúng)

NHƯNG 47,7% cặp ĐÚNG cũng cho điểm 0. Kiểm chứng ngoài:

    㝵 "người"  điểm 0,0   <- lớp lỗi hệ thống đã chứng minh
    主 "chúa"   điểm 0,0   <- cặp Công giáo lõi, chắc chắn đúng

Hai thứ ngược nhau cùng một điểm. Vì vậy:

    ĐIỂM CAO  = bằng chứng xác nhận, dùng được
    ĐIỂM 0    = KHÔNG nói lên gì, TUYỆT ĐỐI không dùng để hạ cấp hay loại nhãn

Xem docs/NGHIEN_CUU_UNIHAN_KDEFINITION_2026-08-22.md cho toàn bộ phép đo.

    python -m pipeline.tools.sem_score            # in bảng tóm tắt
    python -m pipeline.tools.sem_score --bench    # chạy lại chuẩn đánh giá + đối chứng
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UNIHAN = REPO / "dict" / "unihan_kdefinition_full.csv"
QN_DICT = REPO / "dict" / "QuocNgu_SinoNom.csv"
DEFAULT_LABELS = REPO / "dataset_out" / "labels_final.csv"

# Ngưỡng đã hiệu chuẩn trên chuẩn 2.002 cặp — xem docstring.
TAU_CONFIRM = 0.05   # độ chính xác 100%
TAU_LIKELY = 0.02    # độ chính xác 98,9%

_STOP = set(
    "a an the of to in on and or but with for from by as is are be been being it its "
    "his her their that this these those same kind such used use also not no one two three".split()
)


def _toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2 and w not in _STOP}


def load_meanings(path: Path = UNIHAN) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            ch = r.get("Character")
            if ch:
                out[ch] = _toks(r.get("kDefinition", ""))
    return out


class SemScorer:
    """Chấm điểm quan hệ nghĩa giữa một chữ và tập chữ mà từ điển cho là đọc âm ấy."""

    def __init__(self, meanings: dict[str, set[str]], qn_to_nom: dict[str, list[str]]):
        self.uni = meanings
        self.qn = qn_to_nom
        df: collections.Counter = collections.Counter()
        for t in meanings.values():
            df.update(t)
        n = max(len(meanings), 1)
        self.idf = {w: math.log(n / (1 + c)) for w, c in df.items()}

    def _profile(self, syllable: str, exclude: str | None) -> collections.Counter:
        bag: collections.Counter = collections.Counter()
        for c in self.qn.get(syllable, []):
            if c == exclude or c not in self.uni:
                continue
            for w in self.uni[c]:
                bag[w] += 1
        return bag

    def score(self, char: str, syllable: str) -> float | None:
        """None = không chấm được (chữ không có nghĩa Hán, hoặc âm không có chữ nào có nghĩa)."""
        t = self.uni.get(char)
        if not t:
            return None
        # LEAVE-ONE-OUT: loại chính chữ đang chấm khỏi hồ sơ, nếu không sẽ tự khớp với mình.
        bag = self._profile(syllable, exclude=char)
        if not bag:
            return None
        num = sum(self.idf.get(w, 0.0) * min(bag[w], 3) for w in t if w in bag)
        da = math.sqrt(sum(self.idf.get(w, 0.0) ** 2 for w in t))
        db = math.sqrt(sum((self.idf.get(w, 0.0) * min(bag[w], 3)) ** 2 for w in bag))
        return num / (da * db) if da and db else 0.0

    def verdict(self, char: str, syllable: str) -> tuple[float | None, str]:
        v = self.score(char, syllable)
        if v is None:
            return None, "khong_cham_duoc"
        if v > TAU_CONFIRM:
            return v, "xac_nhan"
        if v > TAU_LIKELY:
            return v, "co_kha_nang"
        return v, "khong_ket_luan"   # KHÔNG phải "sai" — xem docstring


def _auc(pos: list[float], neg: list[float]) -> float:
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks: dict[int, float] = {}
    j = 0
    while j < len(allv):
        k = j
        while k < len(allv) and allv[k][0] == allv[j][0]:
            k += 1
        for m in range(j, k):
            ranks[m] = (j + 1 + k) / 2
        j = k
    sp = sum(ranks[i] for i, (_, y) in enumerate(allv) if y == 1)
    return (sp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.sem_score")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--bench", action="store_true", help="chạy chuẩn đánh giá + hai đối chứng")
    args = ap.parse_args(argv)

    import sys
    sys.path.insert(0, str(REPO))
    from core.text.dictionary import load_qn_to_nom

    sc = SemScorer(load_meanings(), load_qn_to_nom(str(QN_DICT)))
    with open(args.labels, encoding="utf-8") as fh:
        lab = list(csv.DictReader(fh))

    if args.bench:
        random.seed(0)
        corpus = [x["label"] for x in lab if x["label"] and x["label"] in sc.uni]
        pos, neg, plen, nlen = [], [], [], []
        seen: set[tuple[str, str]] = set()
        for x in lab:
            if x["tier"] != "GOLD":
                continue
            c, s = x["label"], (x["syllable"] or "").lower()
            if not c or not s or c not in sc.uni or c not in sc.qn.get(s, []) or (c, s) in seen:
                continue
            seen.add((c, s))
            v = sc.score(c, s)
            if v is None:
                continue
            for _ in range(30):
                c2 = random.choice(corpus)
                if c2 == c or c2 in sc.qn.get(s, []):
                    continue
                v2 = sc.score(c2, s)
                if v2 is None:
                    continue
                pos.append(v); neg.append(v2)
                plen.append(len(sc.uni[c])); nlen.append(len(sc.uni[c2]))
                break
        print(f"[chuẩn] n={len(pos)} cặp khớp-âm-tiết")
        print(f"  điểm nghĩa                  AUC = {_auc(pos, neg):.4f}")
        print(f"  đối chứng độ dài định nghĩa AUC = {_auc(plen, nlen):.4f}")
        print(f"  cặp đúng cho điểm 0: {sum(1 for v in pos if v == 0) / len(pos) * 100:.1f}%"
              f" | cặp sai cho điểm 0: {sum(1 for v in neg if v == 0) / len(neg) * 100:.1f}%")
        return 0

    groups = {
        "s1_inter_s2_similar (GOLD yếu nhất)":
            ([x for x in lab if x["tier"] == "GOLD" and "s1_inter_s2_similar" in x["rule"]], "label"),
        "SYLLABLE (chưa nguồn nào xác nhận)":
            ([x for x in lab if x["tier"] == "SYLLABLE"], "ocr_char"),
        "no_s1_inter_s2 (REVIEW)":
            ([x for x in lab if x["rule"].startswith("no_s1_inter_s2")], "ocr_char"),
    }
    for name, (rows, key) in groups.items():
        pair: collections.Counter = collections.Counter()
        for x in rows:
            c, s = x[key], (x["syllable"] or "").lower()
            if c and s:
                pair[(c, s)] += 1
        conf = likely = n_conf = n_likely = 0
        for (c, s), n in pair.items():
            v, tag = sc.verdict(c, s)
            if tag == "xac_nhan":
                conf += 1; n_conf += n
            elif tag == "co_kha_nang":
                likely += 1; n_likely += n
        print(f"\n{name}: {len(pair)} cặp / {sum(pair.values())} ô")
        print(f"   xac_nhan     (>{TAU_CONFIRM}): {conf:4} cặp / {n_conf:5} ô")
        print(f"   co_kha_nang  (>{TAU_LIKELY}): {likely:4} cặp / {n_likely:5} ô")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
