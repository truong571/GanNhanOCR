"""Reference S3 scoring v2 — the P0 + P1 inference fix (drop into the repo).

The trained head fixes the ENCODER; this fixes how it is SCORED. Two changes vs
the current pipeline/align_engine/visual_signal.py:decide():

  P0 — stop `max over tiers of P(match)`. That bug let a noisy real-crop proto
       (calibrated ceiling 0.98) beat a strong glyph match (ceiling 0.49) → −18
       retrieval points. Score candidates by the HEAD LOGIT (cosine to the class
       centroid), which measured 78.1% top-1 vs the buggy 60.2%.
  P1 — add an open-set gate so the ranker becomes an error DETECTOR: Energy score
       (Liu NeurIPS'20) / MLS (Vaze ICLR'22) over ALL classes says "is this a
       known glyph at all", and a CONFORMAL threshold gives a distribution-free
       false-alarm guarantee (calibrate on human verdicts, GĐ0).

This module is framework-free (numpy). Wire it by replacing the body of
`VisualS3.decide` with `score_v2(...)`, passing the head logits from
NomEncoder.logits(crop_emb) and the candidate→class-index map.
"""
from __future__ import annotations

import numpy as np


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def score_v2(head_logits, class_of, candidates, s=30.0,
             conformal_tau=None, energy_floor=None):
    """Score a crop over its candidate chars using the trained head.

    head_logits : np.ndarray (n_classes,) cosine logits in [-1,1] from
                  NomEncoder.logits(crop_emb) — the P0 primary signal.
    class_of    : dict char -> class index in the head (None if char not a head class).
    candidates  : list[str]  = {ocr_char} ∪ dict_readings (S2), order irrelevant.
    conformal_tau : accept only if the winner's calibrated p >= tau (P1 FAR guarantee).
                    Pass the value from a held-out human-verdict calibration; None -> use
                    energy_floor only.
    energy_floor : reject as open-set (miscut/garbage) if Energy < floor. None -> skip.

    Returns dict {top_char, p_match, margin, reject, head_top, energy, mls}.
    """
    # candidate-restricted head logits (P0)
    cand_idx = [(c, class_of.get(c)) for c in candidates]
    scored = [(c, float(head_logits[i])) for c, i in cand_idx if i is not None]
    # global open-set signals over ALL classes (P1) — candidate-independent
    mls = float(head_logits.max())
    energy = float(np.log(np.exp(head_logits * s).sum()) / s)   # scaled logsumexp
    head_top = None
    if scored:
        head_top = max(scored, key=lambda t: t[1])[0]

    if not scored:                                              # no candidate is a known glyph
        return {"top_char": None, "p_match": 0.0, "margin": 0.0, "reject": True,
                "head_top": head_top, "energy": energy, "mls": mls}

    scored.sort(key=lambda t: t[1], reverse=True)
    top_char, top_logit = scored[0]
    runner = scored[1][1] if len(scored) > 1 else -1.0
    # map cosine-logit -> pseudo-probability among candidates (calibrate for real use)
    ps = _softmax(np.array([s * v for _, v in scored]))
    p_top, margin = float(ps[0]), float(ps[0] - (ps[1] if len(ps) > 1 else 0.0))

    reject = False
    if conformal_tau is not None and p_top < conformal_tau:
        reject = True
    if energy_floor is not None and energy < energy_floor:      # open-set: not a glyph
        reject = True
    if top_logit - runner < 0.0:
        reject = True
    return {"top_char": top_char, "p_match": p_top, "margin": margin, "reject": reject,
            "head_top": head_top, "energy": energy, "mls": mls}


def calibrate_conformal(scores_correct, scores_wrong, target_far=0.05):
    """Pick tau so FAR(wrong accepted) <= target on held-out HUMAN verdicts.

    scores_correct/scores_wrong : winner p_match on rows the human marked right/wrong.
    Returns tau (accept p>=tau) and the achieved (recall, far). Distribution-free at
    the chosen quantile (conformal). Use the returned tau as score_v2(conformal_tau=).
    """
    w = np.sort(np.asarray(scores_wrong))[::-1]
    if len(w) == 0:
        return 0.5, (1.0, 0.0)
    k = int(np.floor(target_far * (len(w) + 1))) - 1
    tau = float(w[max(0, k)]) + 1e-6
    c = np.asarray(scores_correct)
    recall = float((c >= tau).mean()) if len(c) else 0.0
    far = float((np.asarray(scores_wrong) >= tau).mean())
    return tau, (recall, far)
