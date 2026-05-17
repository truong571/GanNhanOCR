"""Send the suspect bigrams to Gemini for confirmation.

Reads suspects.json from bigram_finder, packs all unique pairs into ONE prompt,
asks Gemini to confirm/reject each fix. Output is a verified fix_map.json.

Runs in 3 modes:
  --dry-run        : print the prompt that WOULD be sent, no API call
  (default)        : call Gemini if GEMINI_API_KEY set in .env, else dry-run
  --mock           : skip API call, use a hardcoded mock response (for testing)

API key: get a free one at https://aistudio.google.com/app/apikey
Add to .env:  GEMINI_API_KEY=your_key_here

Usage:
    .venv/bin/python evaluation/test_llm_postfix/llm_fixer.py --dry-run
    .venv/bin/python evaluation/test_llm_postfix/llm_fixer.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
SUSPECTS_PATH = OUT / "suspects.json"
FIX_MAP_PATH = OUT / "fix_map.json"
PROMPT_PATH = OUT / "prompt_sent.txt"


SYSTEM_PROMPT = """Bạn là chuyên gia ngôn ngữ tiếng Việt, am hiểu văn Công giáo thế kỷ 17-19.

Nhiệm vụ: kiểm tra các cặp bigram tiếng Việt nghi ngờ là lỗi OCR.
Với mỗi cặp, quyết định:
  - "FIX"    : cặp WRONG đúng là lỗi → trả lại version CORRECT
  - "KEEP"   : cặp WRONG thật ra là tiếng Việt hợp lệ → giữ nguyên
  - "OTHER"  : cả 2 cặp đều sai, trả version đúng

QUY TẮC NGHIÊM NGẶT:
1. GIỮ NGUYÊN tên thánh, địa danh phiên âm (Pha-lan-sa, Ghê-rê, Phê-rô, I-na-xu).
2. GIỮ NGUYÊN từ cổ Công giáo (bổn đạo, đứng tinh, vít vồ, vít vô, ấy thì).
3. Văn TK19 dùng "chăng" (= không), "rằng" (= that), "liền" (= immediately).
   Nhưng "răng" (= tooth), "chăng" (= perhaps), "liên" (= continuous) cũng valid.
   PHẢI xét ngữ cảnh trước (prev word) để quyết định.
4. Phân biệt:
   - "nói rằng / quở rằng" — quote marker → "rằng"
   - "đau răng / nhổ răng" — tooth → "răng"
   - "chẳng có / chẳng phải" — negation → "chẳng"
   - "chăng có lẽ / lẽ nào chăng" — interrogative → "chăng"

Output: JSON map {id: action} với action ∈ {"FIX", "KEEP", "OTHER:correct_word"}.
"""


def build_prompt(suspects: list[dict]) -> str:
    """Pack suspects into a single LLM prompt."""
    items_str = json.dumps({
        str(i): {"prev": s["prev"], "wrong": s["wrong"],
                 "suggested": s["suggested"]}
        for i, s in enumerate(suspects)
    }, ensure_ascii=False, indent=2)

    return f"""{SYSTEM_PROMPT}

Các cặp cần kiểm tra (prev = từ đứng trước, wrong = từ nghi ngờ, suggested = từ thay thế đề xuất):

{items_str}

Trả về JSON map với key = id, value:
  - "FIX"               : confirm suggested là đúng
  - "KEEP"              : wrong thực ra là valid trong văn cảnh này
  - "OTHER:abc"         : cả 2 đều sai, dùng "abc"

CHỈ trả JSON, không giải thích."""


def call_gemini(prompt: str, model_name: str = "gemini-2.0-flash-exp") -> str:
    """Send the prompt to Gemini, return raw text response."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: pip install google-generativeai", file=sys.stderr)
        raise

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY in .env "
            "(get free key at https://aistudio.google.com/app/apikey)"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    print(f"  calling Gemini ({model_name})...", flush=True)
    t0 = time.time()
    response = model.generate_content(prompt)
    print(f"  done in {time.time()-t0:.1f}s")
    return response.text


def parse_response(text: str) -> dict[str, str]:
    """Extract JSON map from LLM response (strip code fences if any)."""
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ```
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"!! failed to parse: {e}\nraw:\n{text[:500]}", file=sys.stderr)
        return {}


MOCK_RESPONSE = """{
  "0": "FIX", "1": "FIX", "2": "FIX", "3": "FIX", "4": "FIX",
  "5": "FIX", "6": "FIX", "7": "FIX", "8": "FIX",
  "9": "KEEP",
  "10": "FIX", "11": "FIX", "12": "FIX", "13": "FIX", "14": "FIX"
}"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Print prompt, don't call API")
    p.add_argument("--mock", action="store_true",
                   help="Use mock response instead of real API call")
    p.add_argument("--model", default="gemini-2.5-flash")
    args = p.parse_args()

    if not SUSPECTS_PATH.exists():
        print(f"!! {SUSPECTS_PATH} not found. Run bigram_finder.py first.")
        return

    # Load .env (so GEMINI_API_KEY shows up in os.environ)
    env_path = REPO / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip("'").strip('"')
                os.environ.setdefault(k.strip(), v)

    suspects = json.loads(SUSPECTS_PATH.read_text())
    print(f"Loaded {len(suspects)} suspects from {SUSPECTS_PATH.name}")

    prompt = build_prompt(suspects)
    PROMPT_PATH.write_text(prompt, encoding="utf-8")
    print(f"Prompt size: {len(prompt):,} chars (~{len(prompt)//4:,} tokens)")
    print(f"Wrote full prompt to {PROMPT_PATH}")
    print()

    if args.dry_run:
        print("--- DRY RUN — first 800 chars of prompt ---")
        print(prompt[:800])
        print("... (truncated)")
        return

    if args.mock:
        print("--- MOCK MODE — using canned response ---")
        raw = MOCK_RESPONSE
    else:
        if not os.environ.get("GEMINI_API_KEY"):
            print("!! GEMINI_API_KEY not set in .env — falling back to --dry-run")
            print(f"Get free key: https://aistudio.google.com/app/apikey")
            print(f"Add to .env: GEMINI_API_KEY=your_key_here")
            print()
            print("--- DRY RUN — first 800 chars of prompt ---")
            print(prompt[:800])
            return
        raw = call_gemini(prompt, args.model)

    print()
    print("--- raw LLM response ---")
    print(raw[:600])
    print("---")
    actions = parse_response(raw)
    print(f"\nParsed {len(actions)} actions from response.")

    # Build fix_map: {wrong_bigram: correct_bigram}
    fix_map: dict[str, str] = {}
    stats = {"FIX": 0, "KEEP": 0, "OTHER": 0, "INVALID": 0}
    for i, s in enumerate(suspects):
        action = actions.get(str(i), "INVALID")
        wrong_bigram = s["bigram_wrong"]
        if action == "FIX":
            fix_map[wrong_bigram] = s["bigram_correct"]
            stats["FIX"] += 1
        elif action == "KEEP":
            stats["KEEP"] += 1
        elif isinstance(action, str) and action.startswith("OTHER:"):
            other_word = action.split(":", 1)[1].strip()
            fix_map[wrong_bigram] = f"{s['prev']} {other_word}"
            stats["OTHER"] += 1
        else:
            stats["INVALID"] += 1

    FIX_MAP_PATH.write_text(json.dumps(fix_map, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print()
    print(f"Stats: {stats}")
    print(f"Wrote {len(fix_map)} confirmed fixes to {FIX_MAP_PATH.name}")
    print()
    print("Next step: apply_fixes.py to apply these to transcriptions/.")


if __name__ == "__main__":
    main()
