"""Blind multiple-choice verifier logic for the qwen channel (report §07).

Free-form VLM reading of ancient glyphs is unreliable and VLMs are sycophantic — they
agree with a label shown in the prompt (arXiv:2410.11302). So the qwen channel NEVER
sees kim's proposed label as "the answer". Instead it is shown the crop next to a
lineup of rendered candidates (kim's label + S3-nearest look-alikes + "none of the
above"), the option order is randomised across K reads to cancel position bias
(arXiv:2410.15393), and the verdict is the majority over reorderings — inconsistent or
"none" reads become an ABSTAIN, never a GOLD overrule.

This module is the pure, testable logic (lineup construction, response parsing, verdict
aggregation). The actual image+API call is an injectable hook so it runs without network.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NONE_OPTION", "build_lineup", "parse_choice", "aggregate_verdict",
           "VerifierOutcome", "verify"]

NONE_OPTION = "∅"          # "none of the above / cannot read"


def build_lineup(true_label: str, distractors: list[str], rng: np.random.Generator):
    """Return (options, correct_index). Options = shuffled candidates + NONE.

    The caller must render each option glyph; the prompt shows them WITHOUT marking
    which is kim's label (blinding). `correct_index` is kept private for scoring only.
    Duplicate distractors and any equal to true_label are dropped.
    """
    seen = {true_label}
    opts = [true_label]
    for d in distractors:
        if d and d not in seen:
            seen.add(d)
            opts.append(d)
    order = rng.permutation(len(opts)).tolist()
    shuffled = [opts[i] for i in order]
    shuffled.append(NONE_OPTION)
    correct_index = shuffled.index(true_label)
    return shuffled, correct_index


def parse_choice(response_text: str, options: list[str]) -> str | None:
    """Extract the chosen option from a model reply.

    Accepts an option index (1-based letter/number) or a literal glyph match. Returns
    the chosen option string, NONE_OPTION, or None if unparseable.
    """
    if not response_text:
        return None
    t = response_text.strip()
    # literal glyph match (most reliable)
    for o in options:
        if o and o in t:
            return o
    # letter A/B/C... or number 1/2/3...
    for k, o in enumerate(options):
        letter = chr(ord("A") + k)
        if letter == t.upper()[:1] or t[:2].rstrip(".)") == str(k + 1):
            return o
    return None


@dataclass
class VerifierOutcome:
    verdict: str          # 'confirm' | 'disagree' | 'abstain'
    agree_score: float    # fraction of reads that picked kim's label (0..1)
    picked: str | None    # the consistently-picked non-kim label, if disagree
    n_reads: int


def aggregate_verdict(choices: list[str | None], kim_label: str,
                      min_consistency: float = 0.67) -> VerifierOutcome:
    """Majority over K randomised reads.

      confirm   >= min_consistency of reads picked kim's label
      disagree  >= min_consistency picked ONE specific other real label (not NONE)
      abstain   otherwise (NONE, unparseable, or inconsistent)
    """
    n = len(choices)
    if n == 0:
        return VerifierOutcome("abstain", 0.0, None, 0)
    valid = [c for c in choices if c is not None]
    agree = sum(1 for c in valid if c == kim_label)
    agree_score = agree / n
    if agree_score >= min_consistency:
        return VerifierOutcome("confirm", agree_score, None, n)
    # any consistent non-kim, non-NONE label?
    others = [c for c in valid if c not in (kim_label, NONE_OPTION)]
    if others:
        vals, cnts = np.unique(others, return_counts=True)
        top, topc = vals[np.argmax(cnts)], cnts.max()
        if topc / n >= min_consistency:
            return VerifierOutcome("disagree", agree_score, str(top), n)
    return VerifierOutcome("abstain", agree_score, None, n)


def verify(true_label: str, distractors: list[str], ask, k_reads: int = 3,
           seed: int = 0) -> VerifierOutcome:
    """Run K blind reorderings through `ask(options) -> response_text` and aggregate.

    `ask` is any callable (real VLM API, or a stub in tests) taking the shuffled option
    list and returning the model's raw reply. No kim label is ever passed to `ask`.
    """
    rng = np.random.default_rng(seed)
    choices = []
    for _ in range(k_reads):
        options, _ = build_lineup(true_label, distractors, rng)
        resp = ask(options)
        choices.append(parse_choice(resp, options))
    return aggregate_verdict(choices, true_label)
