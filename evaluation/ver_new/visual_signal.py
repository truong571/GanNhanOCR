"""S3 — visual glyph-match signal (TRAINED Nôm embedder + FontDiffusion glyphs).

The third independent signal. For a Nôm crop it ranks candidate characters by
cosine of a NÔM-TRAINED embedding (evaluation/ver_new/nom_classifier, ResNet
+ ArcFace) against the FontDiffusion reference glyph of each candidate. NB the FD
glyph is a FontDiffusion *handwritten-style* render conditioned on the woodblock
style (gannhanocr-fd, see kaggle_diffusion/README.md) — NOT a clean print font. This
REPLACES DINOv2, which was proven non-discriminative on chữ-Nôm
(REPORT_dinov2_unsuitable.md: cosine 0.91 between different chars, retrieval 0%).
The trained encoder: T2 separation +0.29, T3 retrieval 76.5% (DINOv2: +0.01, 0%).

Returns consensus.S3 (top_char / cosine / margin / top_in_dict) -> decide_label
populates the SILVER tier. Checkpoint auto-found at nom-embed/best.pt (train via
nom_classifier/ on Kaggle, download best.pt there).
"""
from __future__ import annotations

import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.ver_new.consensus import S3, TAU_SILVER, DELTA_SILVER
from evaluation.ver_new.bbox_fix import tighten_box
from evaluation.ver_new.nom_classifier.infer import NomEncoder

# Bước 1 — reference bank. Each candidate is represented not by ONE synthetic
# glyph but by a per-class REFERENCE BANK in priority order:
#   "crop"    real GOLD train-split crops (same woodblock domain -> ZERO gap)
#   "simfont" a generated glyph in a font similar to the crops (small gap)  [optional]
#   "fd"      the FontDiffusion glyph (covers the long tail / 0-crop classes)
# A crop is scored against a candidate by the trimmed top-k cosine to its bank.
PROTO_K = 8        # max real-crop references kept per class
PROTO_TOPK = 3     # score = mean of the top-k reference cosines (robust to 1 bad ref)


def _is_cjk(ch: str) -> bool:
    if not ch or len(ch) != 1:
        return False
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0x20000 <= o <= 0x2A6DF or 0x2A700 <= o <= 0x2EBEF
            or 0xF900 <= o <= 0xFAFF)


def _find_ckpt(repo: Path) -> str:
    for c in [repo / "nom-embed" / "best.pt",
              repo / "evaluation" / "ver_new" / "nom_classifier" / "checkpoints" / "best.pt",
              repo / "nom-embed" / "last.pt"]:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "Nôm embedder checkpoint not found (nom-embed/best.pt). Train it first via "
        "evaluation/ver_new/nom_classifier (Kaggle) and place best.pt at nom-embed/.")


class VisualS3:
    def __init__(self, repo: Path, font_path: str | None = None, fd_dir: str = "",
                 cache_dir: str | None = None, ckpt: str | None = None,
                 proto_index: str | None = None, simfont_dir: str = ""):
        repo = Path(repo)
        self.repo = repo
        self.enc_ckpt = ckpt or _find_ckpt(repo)
        self.enc = NomEncoder(self.enc_ckpt)
        # head_rescue #4: the ArcFace HEAD is a 2nd, independent visual classifier
        # (1591-way). label->index map lets decide() add a head vote per candidate.
        self.lab2idx = ({lab: i for i, lab in self.enc.classes.items()}
                        if getattr(self.enc, "classes", None) else {})
        self.fd_index = self._build_fd_index(Path(fd_dir))
        # optional: glyphs in a font SIMILAR to the crops (smaller domain gap than FD)
        self.simfont_index = self._build_fd_index(Path(simfont_dir)) if simfont_dir else {}
        self._page_cache: dict[str, Image.Image] = {}
        self._ref_cache: dict[str, np.ndarray] = {}
        self._sim_cache: dict[str, np.ndarray] = {}
        self.n_fd = 0
        self.n_font = 0   # no font fallback with the trained encoder
        # real-crop prototypes per class (the zero-domain-gap reference)
        self.proto = self._load_or_build_protos(repo, proto_index)
        # Bước 2: per-tier cosine->P(match) calibration + open-set operating point
        self.calib = self._load_calibration(repo)
        cal = (f"calibrated@P≥{self.calib['target_precision']:.2f} "
               f"(τ={self.calib['tau_p']:.2f},δ={self.calib['delta_p']:.2f})"
               if self.calib else f"UNcalibrated (τ/δ fallback {TAU_SILVER}/{DELTA_SILVER})")
        print(f"  S3 = trained Nôm embedder on {self.enc.device} | FD glyphs "
              f"{len(self.fd_index)} | crop-protos {len(self.proto)} classes"
              + (f" | simfont {len(self.simfont_index)}" if self.simfont_index else "")
              + f" | {cal}", flush=True)

    def _load_calibration(self, repo: Path):
        """Load s3_calibration.json (per-tier isotonic + operating point)."""
        p = Path(__file__).resolve().parent / "s3_calibration.json"
        if not p.exists():
            return None
        try:
            d = json.load(open(p, encoding="utf-8"))
            for t in d.get("tiers", {}).values():     # pre-convert knots to arrays
                t["x"] = np.asarray(t["x"], float)
                t["p"] = np.asarray(t["p"], float)
            return d
        except Exception:
            return None

    def _p_tier(self, tier: str, cos: float) -> float | None:
        """Calibrated P(match) for a raw cosine on a given reference tier."""
        if not self.calib:
            return None
        t = self.calib["tiers"].get(tier)
        if t is None or len(t["x"]) == 0:
            return None
        return float(np.interp(cos, t["x"], t["p"]))

    def tier_cosines(self, crop_emb, char: str) -> dict:
        """Best RAW cosine of the crop to each reference tier of `char`
        ({'crop','simfont','fd'} -> cosine). Used by compute() and calibrate_s3."""
        out = {}
        for tier, embs in self._ref_bank(char).items():
            out[tier] = max(self.enc.cosine_raw(crop_emb, e) for e in embs)
        return out

    @staticmethod
    def _build_fd_index(fd_dir: Path) -> dict[str, str]:
        idx: dict[str, str] = {}
        if not fd_dir.exists():
            return idx
        for png in fd_dir.rglob("U+*.png"):
            try:
                idx[chr(int(png.stem.replace("U+", ""), 16))] = str(png)
            except ValueError:
                pass
        return idx

    def _page(self, page_png: str) -> Image.Image:
        img = self._page_cache.get(page_png)
        if img is None:
            img = Image.open(page_png).convert("RGB")
            self._page_cache[page_png] = img
        return img

    def _ref_emb(self, char: str):
        """Reference embedding = trained-encoder embedding of the FD glyph."""
        if char in self._ref_cache:
            return self._ref_cache[char]
        p = self.fd_index.get(char)
        if not p:
            return None
        e = self.enc.embed_path(p)
        if e is not None:
            self.n_fd += 1
            self._ref_cache[char] = e
        return e

    def _sim_emb(self, char: str):
        """Reference embedding = encoder embedding of the SIMILAR-FONT glyph (if any)."""
        if char in self._sim_cache:
            return self._sim_cache[char]
        p = self.simfont_index.get(char)
        if not p:
            return None
        e = self.enc.embed_path(p)
        if e is not None:
            self._sim_cache[char] = e
        return e

    def _load_or_build_protos(self, repo: Path, proto_index: str | None) -> dict:
        """Per-class real-crop prototype = stack of <=PROTO_K embeddings of the
        candidate's GOLD TRAIN-split crops (index.csv). Zero domain gap; the
        single biggest cheap lift. Cached to nom-embed/s3_proto_cache.pkl with a
        signature (index mtime + ckpt mtime + K) so it rebuilds only when stale.

        NOTE on leakage: prototypes use split=='train' GOLD crops ONLY. At label
        time S3 scores UN-confirmed pairs (never GOLD), so a crop is never in its
        own prototype. The train-only restriction also keeps a held-out eval
        (Bước 3) honest. In production this is legitimate use of confirmed
        examples to label unconfirmed ones.
        """
        idx_csv = Path(proto_index) if proto_index else (
            repo / "evaluation" / "ver_new" / "nom_classifier" / "index.csv")
        if not idx_csv.exists():
            print(f"  [S3] proto index not found ({idx_csv.name}); crop-protos OFF "
                  "(FD-only references).", flush=True)
            return {}
        try:
            sig = f"{idx_csv.stat().st_mtime_ns}|{Path(self.enc_ckpt).stat().st_mtime_ns}|{PROTO_K}"
        except OSError:
            sig = "nosig"
        cache = Path(__file__).resolve().parent / "s3_proto_cache.pkl"
        if cache.exists():
            try:
                d = pickle.load(open(cache, "rb"))
                if d.get("__sig__") == sig:
                    return {k: v for k, v in d.items() if k != "__sig__"}
            except Exception:
                pass
        by: dict[str, list[str]] = defaultdict(list)
        for r in csv.DictReader(open(idx_csv, encoding="utf-8")):
            if r.get("source") == "crop" and r.get("split") == "train" and r.get("label"):
                by[r["label"]].append(str(repo / r["path"]))
        print(f"  [S3] building crop prototypes for {len(by)} classes (<= {PROTO_K} crops each) ...",
              flush=True)
        proto: dict[str, np.ndarray] = {}
        for ch, paths in by.items():
            embs = [e for p in paths[:PROTO_K] if (e := self.enc.embed_path(p)) is not None]
            if embs:
                proto[ch] = np.stack(embs)
        try:
            pickle.dump({**proto, "__sig__": sig}, open(cache, "wb"))
        except Exception:
            pass
        return proto

    def _ref_bank(self, char: str) -> dict:
        """Reference bank for one candidate, keyed by domain tier (best first)."""
        bank: dict[str, np.ndarray] = {}
        if char in self.proto:                      # real crops (zero gap)
            bank["crop"] = self.proto[char]
        e = self._sim_emb(char)                     # similar-font glyph (small gap)
        if e is not None:
            bank["simfont"] = e[None]
        e = self._ref_emb(char)                     # FD glyph (coverage)
        if e is not None:
            bank["fd"] = e[None]
        return bank

    def _head_scores(self, crop_emb, cands):
        """(top_char, margin) of the ArcFace HEAD over candidates ∩ vocab.
        margin = top1 − top2 head cosine-logit = head's CONFIDENCE in its pick (a
        boolean head_agree was too loose: 80% GOLD-test; the margin gate recovers
        precision). Returns (None, 0.0) if no head."""
        if not getattr(self.enc, "has_head", False) or not self.lab2idx:
            return None, 0.0
        lg = self.enc.logits(crop_emb)
        if lg is None:
            return None, 0.0
        sc = sorted(((c, float(lg[self.lab2idx[c]])) for c in cands if c in self.lab2idx),
                    key=lambda t: -t[1])
        if not sc:
            return None, 0.0
        margin = sc[0][1] - (sc[1][1] if len(sc) > 1 else -1.0)
        return sc[0][0], margin

    def decide(self, crop_emb, cands: list[str], guard: bool = True) -> dict:
        """Core cross-candidate scoring, shared by compute() and the eval harnesses.

        Calibrated path: each candidate's score = best per-tier P(match). Plus an
        OPEN-SET fairness guard (the crop-bias fix): if another candidate beats the
        winner on the GLYPH tier by > glyph_guard_margin (i.e. the winner won ONLY
        via its real-crop prototype), the winner may be a crop-backed distractor
        stealing a crop-less / unseen true char -> reject (abstain to REVIEW) rather
        than assert a wrong char. guard=False disables it (for the ablation).
        Returns {top_char, p_match, p_margin, reject, glyph_winner, glyph_contra}.
        """
        banks = {c: self._ref_bank(c) for c in cands}
        head_top, head_margin = self._head_scores(crop_emb, cands)   # #4: independent head vote + confidence
        TIERS = ("crop", "simfont", "fd")
        if self.calib:
            scored, glyph_cos = {}, {}
            for c in cands:
                ps = []
                for t, embs in banks[c].items():
                    cos = max(self.enc.cosine_raw(crop_emb, e) for e in embs)
                    if t in ("fd", "simfont"):          # glyph-tier cosine (every char has one)
                        glyph_cos[c] = max(glyph_cos.get(c, -1.0), cos)
                    p = self._p_tier(t, cos)
                    if p is not None:
                        ps.append(p)
                scored[c] = max(ps) if ps else 0.0
            ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
            top_char, p_top = ranked[0]
            p_run = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = p_top - p_run
            gw, glyph_contra = None, False
            if glyph_cos:
                gw = max(glyph_cos, key=glyph_cos.get)
                gmarg = self.calib.get("glyph_guard_margin", 0.10)
                if gw != top_char and glyph_cos[gw] - glyph_cos.get(top_char, -1.0) > gmarg:
                    glyph_contra = True               # winner won only via crop; glyph disagrees
            reject = (p_top < self.calib["tau_p"]) or (margin < self.calib["delta_p"]) \
                or (guard and glyph_contra)
            return {"top_char": top_char, "p_match": p_top, "p_margin": margin,
                    "reject": bool(reject), "glyph_winner": gw, "glyph_contra": glyph_contra,
                    "head_top": head_top, "head_margin": head_margin,
                    "head_agree": bool(head_top is not None and head_top == top_char)}

        # FALLBACK (no calibration): shared-tier remapped cosine + TAU/DELTA gate.
        shared = next((t for t in TIERS if all(banks[c].get(t) is not None for c in cands)), None)
        scored = {}
        for c in cands:
            embs = banks[c].get(shared) if shared is not None else None
            if embs is None:
                embs = next((banks[c][t] for t in TIERS if banks[c].get(t) is not None), None)
            if embs is None or len(embs) == 0:
                scored[c] = 0.0
                continue
            sims = sorted((self.enc.cosine(crop_emb, e) for e in embs), reverse=True)
            scored[c] = float(np.mean(sims[:PROTO_TOPK]))
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        top_char, top = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        reject = not (top >= TAU_SILVER and (top - runner) >= DELTA_SILVER)
        return {"top_char": top_char, "p_match": top, "p_margin": top - runner,
                "reject": bool(reject), "glyph_winner": None, "glyph_contra": False,
                "head_top": head_top, "head_margin": head_margin,
                "head_agree": bool(head_top is not None and head_top == top_char)}

    def compute(self, page_png: str, bbox, ocr_char: str | None,
                s2_candidates: list[str]) -> S3 | None:
        if bbox is None:
            return None
        x1, y1, x2, y2 = (int(v) for v in bbox)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = self._page(page_png).crop((x1, y1, x2, y2))
        gray = np.asarray(crop.convert("L"))
        ink = (gray < 128).mean()                # reject blank / solid crops
        if ink < 0.03 or ink > 0.97:
            return None
        # Match the encoder's TRAINING framing: it was trained on tighten_box
        # crops (build_dataset.save_crop), but the bbox here is the LOOSE OCR box
        # (ruling lines, neighbour slivers, whitespace). Tighten to the ink before
        # embedding so query and reference live in the same framing.
        tb = tighten_box(gray)
        if tb is not None:
            a, c, b, d = tb
            if b - a >= 8 and d - c >= 8:
                gray = gray[c:d, a:b]
        crop_emb = self.enc.embed_gray(gray)

        cands: list[str] = []
        for c in ([ocr_char] if ocr_char else []) + list(s2_candidates):
            if _is_cjk(c) and c not in cands:
                cands.append(c)
        if not cands:
            return None

        dec = self.decide(crop_emb, cands)
        return S3(top_char=dec["top_char"], cosine=round(dec["p_match"], 4),
                  margin=round(dec["p_margin"], 4),
                  top_in_dict=dec["top_char"] in set(s2_candidates),
                  p_match=round(dec["p_match"], 4), p_margin=round(dec["p_margin"], 4),
                  reject=bool(dec["reject"]),
                  head_top=dec.get("head_top") or "", head_agree=bool(dec.get("head_agree")),
                  head_margin=round(float(dec.get("head_margin") or 0.0), 4))
