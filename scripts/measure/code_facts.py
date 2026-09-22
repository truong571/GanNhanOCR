#!/usr/bin/env python3
"""code_facts.py — sinh docs/PIPELINE_FACTS.json bằng grep/AST có chủ đích (KHÔNG LLM).

Mục đích: agent nghiên cứu KHÔNG đọc lại 10 tệp nguồn; chỉ đọc tệp sự kiện này.
Chạy lại 0 token khi mã đổi. Mỗi mục có "how" = cách rút (lệnh grep / AST).

Mục:
  (a) column_pins   — mọi chỗ ghim số cột 9 / expected_cols / n_columns trong pipeline/ core/
  (b) cli           — argparse thật của từng bước (cờ, mặc định, action, choices) + dòng gọi
                      trong run_pipeline.sh
  (c) json_keys     — khoá JSON engine ĐỌC từ prepared/<book>/detected/*_ocr_cache.json,
                      transcriptions/page_*.json và *_qn_ocr_cache.json (đối chiếu với khoá GHI)
  (d) config_books  — schema books trong config/pipeline.yaml + mọi nơi đọc book['pdf']
  (e) checkpoints   — checkpoint mô hình (.pt/.h5/...) được tham chiếu + trạng thái tồn tại
  (f) git           — tệp pipeline/ core/ có diff chưa commit; Guest Mode ocr_api ĐÃ commit (65f7ca9,
                      21/09); dataset_out/ STT tracked phải SẠCH (đã khôi phục 22/09 sau sự cố apply không --out)

Chỉ phụ thuộc stdlib (+ PyYAML nếu có; nếu không thì đọc yaml bằng regex tối giản).
CLI:
  .venv/bin/python scripts/measure/code_facts.py [--out docs/PIPELINE_FACTS.json]
        [--summary measure_out/code_facts/summary.json] [--limit N] [--check]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("pipeline", "core")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
GENERATOR = "scripts/measure/code_facts.py"

# ----------------------------------------------------------------------------- helpers

def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def py_files(dirs=SCAN_DIRS) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        out.extend(p for p in (REPO / d).rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out)                       # thứ tự cố định -> idempotent


_SRC_CACHE: dict[Path, tuple[list[str], ast.AST | None]] = {}


def load(p: Path) -> tuple[list[str], ast.AST | None]:
    if p not in _SRC_CACHE:
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # SyntaxWarning escape-sequence của mã nguồn
                tree = ast.parse(text, filename=str(p))
            for node in ast.walk(tree):            # gắn parent để phân loại ngữ cảnh
                for ch in ast.iter_child_nodes(node):
                    ch._parent = node             # type: ignore[attr-defined]
        except SyntaxError:
            tree = None
        _SRC_CACHE[p] = (lines, tree)
    return _SRC_CACHE[p]


def line_of(p: Path, n: int) -> str:
    lines, _ = load(p)
    return lines[n - 1].strip() if 0 < n <= len(lines) else ""


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60)
        return r.stdout
    except Exception as e:                       # noqa: BLE001
        return f"<git error: {e}>"


def is_lab_or_test(relpath: str) -> bool:
    return ("/lab/" in relpath or "selftest" in relpath or relpath.endswith("_test.py")
            or "/tests/" in relpath)


# ----------------------------------------------------------------------------- (a) column pins

COL_CTX = re.compile(r"col|c[oộ]t|line|qn|numbered|expected|total|max_lines|N=9", re.I)
COL_WINDOW = re.compile(r"column|c[oộ]t|expected\s*=\s*9|1-9", re.I)  # ngữ cảnh chú thích phía trên
CTX_WINDOW = 8
NON_COL_CALL = re.compile(r"^(cv2|np|numpy|torch|nn|F|plt|scipy|random)\.")   # thư viện: 9 không phải số cột
PIN_IDENTS = {"expected_cols", "n_columns", "n_expected", "total_columns", "EXPECTED",
              "NUM_COLS", "N_COLS", "max_lines", "expected_columns", "num_columns"}
KNOWN_PINS = [                                   # (file, line, gợi ý) — đối chiếu bắt buộc (cập nhật 22/09 vòng 2)
    # align_production.py không còn ghim 9: n_columns lấy từ BookLayout (fb345a29b1); mặc định STT nằm ở book_layout.py
    ("pipeline/align_engine/book_layout.py", 83, "DEFAULT_N_COLUMNS = 9"),   # 22/09 I5: docstring thêm khoá box_decoder (43 -> 50); detector_ckpt/detector_resize (50 -> 61); 23/09 vòng 5: kim_lang_type/kim_ocr_id/kim_font_type/tier_dp (61 -> 83)
    ("pipeline/step2_align.py", 62, "_get_qn_lines(n_columns=9)"),
    ("pipeline/step2_align.py", 154, "detect_nom_columns_v3(..., 9)  # CLI riêng, vẫn ghim 9"),
    ("pipeline/step2_align.py", 161, "len(qn_lines) == 9"),
    ("core/align/parser_v5.py", 136, "max_lines: int = 9"),
    ("core/pdf/pdf_parser.py", 128, "total: int = 9"),
    ("core/pdf/pdf_parser.py", 179, "total_columns: int = 9"),
    ("core/pdf/pdf_parser.py", 243, "total: int = 9"),
    ("core/image/column_detector.py", 11, "n_expected: int = 9"),
]


def _param_name_for_default(fn: ast.FunctionDef | ast.AsyncFunctionDef, const: ast.Constant) -> str | None:
    a = fn.args
    pos = a.posonlyargs + a.args
    for i, d in enumerate(a.defaults):
        if d is const:
            return pos[len(pos) - len(a.defaults) + i].arg
    for kwarg, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is const:
            return kwarg.arg
    return None


def classify_nine(node: ast.Constant) -> tuple[str, str] | None:
    """Trả (category, detail) cho hằng int 9 theo nút cha; None = bỏ qua."""
    par = getattr(node, "_parent", None)
    if par is None:
        return None
    if isinstance(par, ast.arguments):
        fn = getattr(par, "_parent", None)
        name = _param_name_for_default(fn, node) if fn is not None else None
        return "default_param", f"{getattr(fn, 'name', '?')}({name}=9)"
    if isinstance(par, ast.Compare):
        return "compare", ast.unparse(par)
    if isinstance(par, ast.Call) and node in par.args:
        return "call_arg", ast.unparse(par.func) + f"(… arg#{par.args.index(node)}=9)"
    if isinstance(par, ast.keyword):
        call = getattr(par, "_parent", None)
        fname = ast.unparse(call.func) if isinstance(call, ast.Call) else "?"
        return "call_kwarg", f"{fname}({par.arg}=9)"
    if isinstance(par, (ast.Assign, ast.AnnAssign)) and getattr(par, "value", None) is node:
        tg = [ast.unparse(t) for t in par.targets] if isinstance(par, ast.Assign) else [ast.unparse(par.target)]
        return "assign", "=".join(tg) + " = 9"
    if isinstance(par, ast.Subscript):
        return None                               # x[9] — chỉ số, không phải số cột
    return "other", type(par).__name__


def section_column_pins(limit: int) -> dict:
    hits: list[dict] = []
    discarded = Counter()
    discarded_examples: list[str] = []
    ident_refs: list[dict] = []
    for p in py_files():
        rp = rel(p)
        lines, tree = load(p)
        if tree is None:
            continue
        seen_ident_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 9:
                cls = classify_nine(node)
                if cls is None:
                    discarded["subscript_index"] += 1
                    continue
                cat, detail = cls
                code = line_of(p, node.lineno)
                ctx = None
                if COL_CTX.search(code) or COL_CTX.search(detail) or (cat == "compare" and "len(" in detail):
                    ctx = "line"                                           # len(x) == 9 = đếm cột/dòng
                elif cat in ("compare", "assign") and COL_WINDOW.search(
                        "\n".join(lines[max(0, node.lineno - 1 - CTX_WINDOW):node.lineno - 1])):
                    ctx = f"window{CTX_WINDOW}"                            # chú thích/docstring phía trên
                if NON_COL_CALL.match(detail):                             # cv2.bilateralFilter(d=9)…
                    ctx = None
                if cat == "other" or ctx is None:
                    if cat != "other":
                        discarded_examples.append(f"{rp}:{node.lineno} {detail[:50]}")
                    discarded[f"{cat}_no_column_context"] += 1
                    continue
                hits.append({"file": rp, "line": node.lineno, "category": cat, "context": ctx,
                             "detail": detail, "code": code[:160],
                             "lab_or_test": is_lab_or_test(rp)})
            elif isinstance(node, ast.Name) and node.id in PIN_IDENTS:
                if node.lineno not in seen_ident_lines:
                    seen_ident_lines.add(node.lineno)
                    ident_refs.append({"file": rp, "line": node.lineno, "ident": node.id,
                                       "code": line_of(p, node.lineno)[:160]})
            elif isinstance(node, ast.arg) and node.arg in PIN_IDENTS:
                if node.lineno not in seen_ident_lines:
                    seen_ident_lines.add(node.lineno)
                    ident_refs.append({"file": rp, "line": node.lineno, "ident": node.arg,
                                       "code": line_of(p, node.lineno)[:160]})
    hits.sort(key=lambda h: (h["file"], h["line"]))
    ident_refs.sort(key=lambda h: (h["file"], h["line"]))

    by_loc = {(h["file"], h["line"]): h for h in hits}
    known_check = []
    for f, ln, hint in KNOWN_PINS:
        h = by_loc.get((f, ln))
        found = h is not None
        near = None
        if not found:                             # tìm dòng gần nhất cùng tệp (mã có thể dời)
            cands = [x for x in hits if x["file"] == f]
            if cands:
                near = min(cands, key=lambda x: abs(x["line"] - ln))
        known_check.append({"file": f, "line": ln, "hint": hint, "found": found,
                            "code": (h or near or {}).get("code"),
                            "nearest_line": None if found else (near or {}).get("line")})

    # ghim trong YAML cấu hình (nếu có)
    yaml_pins = []
    ycfg = REPO / "config" / "pipeline.yaml"
    if ycfg.exists():
        for i, l in enumerate(ycfg.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"expected_cols|n_columns|num_columns|columns\s*:\s*9", l):
                yaml_pins.append({"file": rel(ycfg), "line": i, "code": l.strip()[:160]})

    prod_hits = [h for h in hits if not h["lab_or_test"]]
    return {
        "how": ("AST: mọi ast.Constant int 9 trong pipeline/ core/ (bỏ __pycache__), phân loại theo "
                "nút cha (default_param / compare / call_arg / call_kwarg / assign); giữ khi dòng mã "
                f"hoặc ngữ cảnh khớp /{COL_CTX.pattern}/i, hoặc so sánh len(x)==9, hoặc (compare/assign) {CTX_WINDOW} dòng phía trên "
                f"khớp /{COL_WINDOW.pattern}/i (context=window); bỏ chỉ số x[9], gọi thư viện cv2/np, hằng không ngữ cảnh cột. "
                f"Định danh: ast.Name/ast.arg ∈ {sorted(PIN_IDENTS)}. YAML: regex trên config/pipeline.yaml."),
        "n_hits": len(hits), "n_hits_prod": len(prod_hits),
        "by_category": dict(sorted(Counter(h["category"] for h in hits).items())),
        "by_file": dict(sorted(Counter(h["file"] for h in hits).items())),
        "discarded": dict(sorted(discarded.items())),
        "discarded_examples": discarded_examples,
        "known_pins_check": known_check,
        "hits": hits[:limit] if limit else hits,
        "identifier_refs": ident_refs[:limit] if limit else ident_refs,
        "n_identifier_refs": len(ident_refs),
        "yaml_pins": yaml_pins,
    }


# ----------------------------------------------------------------------------- (b) CLI

CLI_FILES = [
    ("pipeline.step0_setup", "pipeline/step0_setup.py"),
    ("pipeline.step1_extract", "pipeline/step1_extract.py"),
    ("pipeline.step2_align", "pipeline/step2_align.py"),
    ("pipeline.align_engine.build_dataset", "pipeline/align_engine/build_dataset.py"),
    ("pipeline.remediation", "pipeline/remediation/cli.py"),
    ("pipeline.remediation.confusion_fix", "pipeline/remediation/confusion_fix.py"),
    ("pipeline.remediation.mechanism_gates", "pipeline/remediation/mechanism_gates.py"),   # B4' 22/09 (lithograph)
    ("pipeline.remediation.self_training_rescue", "pipeline/remediation/self_training_rescue.py"),
    ("pipeline.export_final_dataset", "pipeline/export_final_dataset.py"),
    ("pipeline.tools.enrich_crop_quality", "pipeline/tools/enrich_crop_quality.py"),
    ("pipeline.tools.ingest_lithograph_book", "pipeline/tools/ingest_lithograph_book.py"),   # adapter thạch bản (B1' 22/09: --verses/--dict-boost)
]

_PATH_ROOTS = {"REPO": "<REPO>", "ROOT": "<REPO>", "REPO_ROOT": "<REPO>", "repo": "<REPO>", "_TRAIN_CROP": "<REPO>/train_crop"}


def _module_to_path(mod: str, level: int, from_file: Path) -> Path | None:
    if level:
        base = from_file.parent
        for _ in range(level - 1):
            base = base.parent
        parts = mod.split(".") if mod else []
        cand = base.joinpath(*parts)
    else:
        cand = REPO.joinpath(*mod.split("."))
    for c in (cand.with_suffix(".py"), cand / "__init__.py"):
        if c.exists():
            return c
    return None


def _module_assign(p: Path, name: str) -> ast.AST | None:
    _, tree = load(p)
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.value
    return None


def _import_alias(p: Path, alias: str) -> Path | None:
    """`import a.b as alias` / `from a import b as alias` -> đường dẫn tệp module."""
    _, tree = load(p)
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if (a.asname or a.name.split(".")[-1]) == alias:
                    return _module_to_path(a.name, 0, p)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if (a.asname or a.name) == alias:
                    return _module_to_path(f"{node.module}.{a.name}" if node.module else a.name, node.level, p)
    return None


def eval_expr(node: ast.AST | None, p: Path, depth: int = 0):
    """Ước lượng tĩnh giá trị biểu thức mặc định (đường dẫn, hằng module). Trả None nếu chịu."""
    if node is None or depth > 6:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:                             # noqa: BLE001
        pass
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return rel(p)
        if node.id in _PATH_ROOTS:
            return _PATH_ROOTS[node.id]
        v = _module_assign(p, node.id)
        return eval_expr(v, p, depth + 1) if v is not None else None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        mp = _import_alias(p, node.value.id)
        if mp is not None:
            v = _module_assign(mp, node.attr)
            return eval_expr(v, mp, depth + 1) if v is not None else None
        base = eval_expr(node.value, p, depth + 1)
        if isinstance(base, str) and node.attr == "parent":
            if base.startswith("<"):
                return base + "/.."
            par = str(Path(base).parent)
            return "<REPO>" if par == "." else par
        return None
    if isinstance(node, ast.Attribute):                 # a.b.parent (chuỗi)
        base = eval_expr(node.value, p, depth + 1)
        if isinstance(base, str) and node.attr == "parent":
            if base.startswith("<"):
                return base + "/.."
            par = str(Path(base).parent)
            return "<REPO>" if par == "." else par
        return None
    if isinstance(node, (ast.Tuple, ast.List)):
        vals = [eval_expr(e, p, depth + 1) for e in node.elts]
        return vals if all(v is not None for v in vals) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        l, r = eval_expr(node.left, p, depth + 1), eval_expr(node.right, p, depth + 1)
        if isinstance(l, str) and isinstance(r, str):
            return f"{l}/{r}"
        return None
    if isinstance(node, ast.Call):
        fn = ast.unparse(node.func)
        if fn in ("str", "Path", "pathlib.Path") and len(node.args) == 1:
            return eval_expr(node.args[0], p, depth + 1)
        if fn in ("list", "tuple", "sorted", "set") and len(node.args) == 1:
            inner = eval_expr(node.args[0], p, depth + 1)
            if isinstance(inner, dict):
                return sorted(inner.keys())
            if isinstance(inner, (list, tuple, set)):
                return list(inner) if fn != "sorted" else sorted(inner)
            return None
        if isinstance(node.func, ast.Attribute) and node.func.attr == "resolve":
            return eval_expr(node.func.value, p, depth + 1)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
            return None
    return None


def _kw_value(kw: ast.keyword, p: Path) -> dict:
    src = ast.unparse(kw.value)
    out = {"src": src}
    try:
        out["resolved"] = ast.literal_eval(kw.value)      # kể cả None/False
        return out
    except Exception:                                     # noqa: BLE001
        pass
    val = eval_expr(kw.value, p)
    if val is not None:
        out["resolved"] = val
    return out


def section_cli(limit: int) -> dict:
    run_sh = REPO / "run_pipeline.sh"
    sh_lines = run_sh.read_text(encoding="utf-8").splitlines() if run_sh.exists() else []

    def sh_invocations(mod: str, f: str) -> list[dict]:
        out = []
        for i, l in enumerate(sh_lines, 1):
            if re.search(rf"-m\s+{re.escape(mod)}(\s|$|\")", l) or re.search(rf"\s{re.escape(f)}(\s|$|\")", l):
                block, j = [l.rstrip()], i
                while block[-1].endswith("\\") and j < len(sh_lines):
                    block.append(sh_lines[j].rstrip()); j += 1
                out.append({"line": i, "cmd": " ".join(b.strip().rstrip("\\").strip() for b in block)[:300]})
        return out

    steps = []
    for mod, f in CLI_FILES:
        p = REPO / f
        entry = {"module": mod, "file": f, "invocation": f"python -m {mod}", "exists": p.exists()}
        if not p.exists():
            steps.append(entry); continue
        _, tree = load(p)
        if tree is None:
            entry["error"] = "SyntaxError"; steps.append(entry); continue
        parsers: dict[str, dict] = {}                 # var -> {"kind": "main"|"sub", "name": ...}
        args: list[dict] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = ast.unparse(node.value.func)
                tgt = ast.unparse(node.targets[0]) if node.targets else "?"
                if fn.endswith("ArgumentParser"):
                    meta = {kw.arg: eval_expr(kw.value, p) for kw in node.value.keywords
                            if kw.arg in ("prog", "description")}
                    parsers[tgt] = {"kind": "main", "name": None, **meta}
                elif fn.endswith(".add_parser") and node.value.args:
                    parsers[tgt] = {"kind": "sub", "name": eval_expr(node.value.args[0], p)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "add_argument":
                recv = ast.unparse(node.func.value)
                flags = [eval_expr(a, p) for a in node.args]
                a: dict = {"flags": flags}
                sub = parsers.get(recv, {})
                if sub.get("kind") == "sub":
                    a["subcommand"] = sub.get("name")
                for kw in node.keywords:
                    if kw.arg in ("default", "action", "type", "choices", "dest", "nargs", "required"):
                        v = _kw_value(kw, p)
                        a[kw.arg] = v.get("resolved", v["src"]) if kw.arg != "default" else v
                    elif kw.arg == "help":
                        h = eval_expr(kw.value, p)
                        a["help"] = (h if isinstance(h, str) else ast.unparse(kw.value))[:140]
                a["line"] = node.lineno
                args.append(a)
        args.sort(key=lambda x: x["line"])
        main_meta = next((v for v in parsers.values() if v["kind"] == "main"), {})
        entry.update({
            "prog": main_meta.get("prog"), "description": (main_meta.get("description") or "")[:120] or None,
            "subcommands": sorted({v["name"] for v in parsers.values() if v["kind"] == "sub" and v["name"]}),
            "n_arguments": len(args),
            "arguments": args[:limit] if limit else args,
            "run_pipeline_sh": sh_invocations(mod, f),
        })
        if mod == "pipeline.remediation":
            entry["main_module"] = "pipeline/remediation/__main__.py" if (REPO / "pipeline/remediation/__main__.py").exists() else None
        steps.append(entry)
    return {
        "how": ("AST: ast.Call .add_argument trong từng tệp CLI; cờ = literal vị trí; keyword default/action/"
                "type/choices/dest/help lấy literal, ước lượng tĩnh (REPO / 'x' -> '<REPO>/x', hằng module, "
                "import alias) hoặc giữ mã nguồn `src`. Subparser: gán biến từ .add_parser(name). "
                "Dòng gọi thật: regex `-m <module>` hoặc `<file.py>` trong run_pipeline.sh (nối dòng \\)."),
        "n_steps": len(steps), "n_arguments_total": sum(s.get("n_arguments", 0) for s in steps),
        "run_pipeline_sh_exists": run_sh.exists(),
        "steps": steps,
    }


# ----------------------------------------------------------------------------- (c) JSON keys

READER_FILES = [
    "pipeline/align_engine/align_production.py", "pipeline/align_engine/build_dataset.py",
    "pipeline/align_engine/bbox_fix.py", "pipeline/align_engine/visual_emission.py",
    "pipeline/step2_align.py", "pipeline/step1_extract.py", "pipeline/check_ocr_columns.py",
    "core/align/nom_detect_v3.py", "core/align/run_full.py", "core/align/export_dataset_v4.py",
    "core/align/parser_v2.py", "core/ocr/ocr_api.py", "core/ocr/qn_ocr.py",
]


def _dict_literal_keys(p: Path, anchor_regex: str) -> tuple[list[str], int | None]:
    """Khoá của ast.Dict literal đầu tiên có dòng khớp anchor (vd `cache_data = {`)."""
    lines, tree = load(p)
    if tree is None:
        return [], None
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and re.search(anchor_regex, line_of(p, node.lineno)):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if keys:
                return keys, node.lineno
    return [], None


def _reader_accesses(keys: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for f in READER_FILES:
        p = REPO / f
        if not p.exists():
            continue
        _, tree = load(p)
        if tree is None:
            continue
        seen = set()
        for node in ast.walk(tree):
            key = recv = kind = None
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                    and isinstance(node.slice.value, str) and node.slice.value in keys:
                par = getattr(node, "_parent", None)
                if isinstance(par, (ast.Assign, ast.AugAssign)) and getattr(par, "targets", [None])[0] is node:
                    kind = "write"                # x['k'] = ... (ghi, không phải đọc)
                else:
                    kind = "subscript"
                key, recv = node.slice.value, ast.unparse(node.value)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "get" and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str) and node.args[0].value in keys:
                key, recv, kind = node.args[0].value, ast.unparse(node.func.value), "get"
                if len(node.args) > 1:
                    kind = f"get(default={ast.unparse(node.args[1])[:30]})"
            if key is None or kind == "write":
                continue
            sig = (f, node.lineno, key)
            if sig in seen:
                continue
            seen.add(sig)
            out[key].append({"file": f, "line": node.lineno, "recv": recv[:60], "access": kind})
    return out


def section_json_keys(limit: int) -> dict:
    ocr_api = REPO / "core/ocr/ocr_api.py"
    step1 = REPO / "pipeline/step1_extract.py"
    qn_ocr = REPO / "core/ocr/qn_ocr.py"
    pdfp = REPO / "core/pdf/pdf_parser.py"

    cache_keys, cache_ln = _dict_literal_keys(ocr_api, r"cache_data\s*=")
    char_keys, char_ln = _dict_literal_keys(ocr_api, r"chars_with_pos\.append|\{\s*$")
    # per-char dict nằm trong boxes_to_columns: tìm dict có khoá "char" và "bbox"
    _, tree = load(ocr_api)
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                ks = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                if "char" in ks and "bbox" in ks:
                    char_keys, char_ln = ks, node.lineno
                    break
    trans_keys, trans_ln = [], None
    _, t1 = load(step1)
    if t1 is not None:
        for node in ast.walk(t1):
            if isinstance(node, ast.Dict):
                ks = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                if "book_page" in ks and "columns" in ks:
                    trans_keys, trans_ln = ks, node.lineno
                    break
    col_keys, col_ln = [], None
    _, tp = load(pdfp)
    if tp is not None:
        for node in ast.walk(tp):
            if isinstance(node, ast.Dict):
                ks = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                if "syllables" in ks and "column" in ks:
                    col_keys, col_ln = ks, node.lineno
                    break
    qn_keys, qn_ln = [], None
    _, tq = load(qn_ocr)
    if tq is not None:
        for node in ast.walk(tq):
            if isinstance(node, ast.Dict):
                ks = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                if "text" in ks:
                    qn_keys, qn_ln = ks, node.lineno
                    break

    schemas = {
        "ocr_cache": {"path_pattern": "prepared/<book>/detected/<page>_ocr_cache.json",
                      "writer": {"file": rel(ocr_api), "line": cache_ln, "keys": cache_keys},
                      "per_char_writer": {"file": rel(ocr_api), "line": char_ln, "keys": char_keys}},
        "transcription": {"path_pattern": "prepared/<book>/transcriptions/<page>.json",
                          "writer": {"file": rel(step1), "line": trans_ln, "keys": trans_keys},
                          "per_column_writer": {"file": rel(pdfp), "line": col_ln, "keys": col_keys}},
        "qn_ocr_cache": {"path_pattern": "prepared/<book>/transcriptions/<page>_qn_ocr_cache.json",
                         "writer": {"file": rel(qn_ocr), "line": qn_ln, "keys": qn_keys}},
    }
    all_keys = set(cache_keys) | set(char_keys) | set(trans_keys) | set(col_keys) | set(qn_keys)
    readers = _reader_accesses(all_keys)
    key_table = []
    for k in sorted(all_keys):
        owners = [n for n, s in (("ocr_cache", cache_keys), ("ocr_cache.columns[][]", char_keys),
                                 ("transcription", trans_keys), ("transcription.columns[]", col_keys),
                                 ("qn_ocr_cache", qn_keys)) if k in s]
        rs = readers.get(k, [])
        key_table.append({"key": k, "written_by": owners, "n_reads": len(rs),
                          "reader_files": sorted({r["file"] for r in rs}),
                          "reads": rs[:limit] if limit else rs[:12]})
    # đường dẫn tệp mà engine mở
    path_sites = []
    for f in READER_FILES:
        p = REPO / f
        if not p.exists():
            continue
        for i, l in enumerate(load(p)[0], 1):
            if re.search(r"_ocr_cache\.json|transcriptions.*\.json|\"transcriptions\"|'transcriptions'", l):
                path_sites.append({"file": f, "line": i, "code": l.strip()[:150]})
    # chỗ engine gọi json.load: gán về schema theo 3 dòng phía trên (đường dẫn mở)
    load_sites = []
    for f in ("pipeline/align_engine/align_production.py", "pipeline/align_engine/build_dataset.py",
              "pipeline/step2_align.py", "core/ocr/ocr_api.py"):
        p = REPO / f
        if not p.exists():
            continue
        src_lines, _ = load(p)
        for i, l in enumerate(src_lines, 1):
            if re.search(r"json\.loads?\(", l):
                ctx = "\n".join(src_lines[max(0, i - 4):i])
                fn_ctx = "\n".join(src_lines[max(0, i - 12):i])          # docstring/def phía trên
                schema = ("qn_ocr_cache" if "_qn_ocr_cache" in ctx else
                          "ocr_cache" if ("_ocr_cache" in ctx or "ocr_path" in ctx
                                          or ("cache_path" in ctx and "ocr_api" in f)
                                          or re.search(r"cached OCR|OCR cache|_ocr_cache", fn_ctx)) else
                          "transcription" if "transcriptions" in ctx else "other")
                load_sites.append({"file": f, "line": i, "schema": schema, "code": l.strip()[:120]})
    engine_loads = Counter(x["schema"] for x in load_sites if "ocr_api" not in x["file"])
    return {
        "json_load_sites": load_sites,
        "engine_json_load_by_schema": dict(sorted(engine_loads.items())),
        "note_transcription_json": ("engine (align_production/build_dataset/step2_align) KHÔNG json.load transcriptions/page_*.json; "
                                    "chỉ glob tên tệp để liệt kê trang; QN lấy từ *_qn_ocr_cache.json['text'] qua parse_v5/parse_numbered_lines "
                                    "(fallback transcriptions/<page>.txt)") if engine_loads.get("transcription", 0) == 0 else
                                   "engine CÓ json.load transcriptions/page_*.json",
        "how": ("Khoá GHI: ast.Dict literal tại chỗ json.dump (ocr_api cache_data; per-char dict có 'char'+'bbox'; "
                "step1_extract dict có 'book_page'+'columns'; pdf_parser dict có 'column'+'syllables'; qn_ocr dict có 'text'). "
                "Khoá ĐỌC: ast.Subscript hằng chuỗi và .get('k') trong READER_FILES, giao với tập khoá ghi; bỏ phép gán x['k']=…. "
                "Lưu ý 'bbox'/'char'/'column' cũng dùng cho dict nội bộ nên n_reads là cận trên. json_load_sites: regex json.load( "
                "trong engine, gán schema theo 4 dòng phía trên."),
        "reader_files": READER_FILES,
        "schemas": schemas,
        "keys": key_table,
        "never_read": sorted(k for k in all_keys if not readers.get(k)),
        "file_open_sites": path_sites[:limit] if limit else path_sites,
    }


# ----------------------------------------------------------------------------- (d) config books

def _load_yaml(p: Path):
    try:
        import yaml                                # type: ignore
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:                             # noqa: BLE001
        return None


def section_config_books() -> dict:
    ycfg = REPO / "config" / "pipeline.yaml"
    cfg = _load_yaml(ycfg) if ycfg.exists() else None
    books = []
    keyset: Counter = Counter()
    if isinstance(cfg, dict) and isinstance(cfg.get("books"), list):
        for b in cfg["books"]:
            if isinstance(b, dict):
                keyset.update(b.keys())
                pdf = b.get("pdf")
                books.append({"name": b.get("name"), "pdf": pdf,
                              "pdf_exists": (REPO / pdf).exists() if isinstance(pdf, str) else None,
                              "reocr": b.get("reocr"),
                              "prepared_dir_exists": (REPO / str((cfg.get("paths") or {}).get("data_dir", "prepared")) / str(b.get("name"))).exists(),
                              "extra_keys": sorted(set(b) - {"name", "pdf", "reocr"})})
    else:                                          # không có PyYAML: regex tối giản
        cur = None
        for l in ycfg.read_text(encoding="utf-8").splitlines() if ycfg.exists() else []:
            m = re.match(r"\s*-\s*name:\s*(\S+)", l)
            if m:
                cur = {"name": m.group(1)}; books.append(cur); keyset["name"] += 1; continue
            m = re.match(r"\s+(\w+):\s*(\S+)", l)
            if m and cur is not None and m.group(1) in ("pdf", "reocr"):
                cur[m.group(1)] = m.group(2); keyset[m.group(1)] += 1
    # nơi đọc khoá của book: vòng for X in config["books"] + hàm nhận book_cfg
    readers = []
    for p in py_files():
        rp = rel(p)
        _, tree = load(p)
        if tree is None:
            continue
        book_vars: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                it = ast.unparse(node.iter)
                if re.search(r"\[['\"]books['\"]\]|\.get\(['\"]books['\"]", it):
                    book_vars.add(node.target.id)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for a in node.args.args:
                    if a.arg in ("book_cfg", "book_config"):
                        book_vars.add(a.arg)
        if not book_vars:
            continue
        grew = True                                   # bí danh: book_cfg = b (điểm bất động)
        while grew:
            grew = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) \
                        and node.value.id in book_vars:
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id not in book_vars:
                            book_vars.add(t.id); grew = True
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                    and node.value.id in book_vars and isinstance(node.slice, ast.Constant) \
                    and isinstance(node.slice.value, str):
                readers.append({"file": rp, "line": node.lineno, "key": node.slice.value,
                                "access": "subscript", "keyerror_if_missing": True,
                                "code": line_of(p, node.lineno)[:140]})
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "get" and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id in book_vars and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                readers.append({"file": rp, "line": node.lineno, "key": node.args[0].value,
                                "access": "get", "keyerror_if_missing": False,
                                "code": line_of(p, node.lineno)[:140]})
    readers.sort(key=lambda r: (r["file"], r["line"]))
    required = sorted({r["key"] for r in readers if r["keyerror_if_missing"]})
    optional = sorted({r["key"] for r in readers if not r["keyerror_if_missing"]} - set(required))
    yaml_names = {b["name"] for b in books}
    data_root = REPO / "data"; prep_root = REPO / str(((cfg or {}).get("paths") or {}).get("data_dir", "prepared"))
    data_books = []
    for d in sorted(x for x in data_root.iterdir() if x.is_dir()) if data_root.exists() else []:
        pdfs = sorted(x.name for x in d.glob("*.pdf"))
        pages = d / "pages"
        exts = Counter(x.suffix.lower() for x in pages.iterdir() if x.suffix.lower() in IMG_EXTS) if pages.is_dir() else Counter()
        n_pages = sum(exts.values())
        prep = prep_root / d.name
        det = prep / "detected"
        data_books.append({
            "dir": d.name, "in_yaml": d.name in yaml_names, "pdfs": pdfs[:4], "n_pages": n_pages,
            "page_exts": dict(sorted(exts.items())),
            "has_quocngu_pages": (d / "quocngu_pages").is_dir(), "has_nom_pages": (d / "nom_pages").is_dir(),
            "prepared_exists": prep.is_dir(),
            "n_ocr_cache": sum(1 for _ in det.glob("*_ocr_cache.json")) if det.is_dir() else 0,
            "n_qn_cache": sum(1 for _ in (prep / "transcriptions").glob("*_qn_ocr_cache.json")) if (prep / "transcriptions").is_dir() else 0,
        })
    books_missing_required = [b["name"] for b in books
                              if any(k not in (b["extra_keys"] + ["name", "pdf", "reocr"]) or b.get(k) is None
                                     for k in required if k in ("name", "pdf"))]
    return {
        "data_books": data_books,
        "data_books_not_in_yaml": [b["dir"] for b in data_books if not b["in_yaml"]],
        "books_missing_required": books_missing_required,
        "how": ("YAML: yaml.safe_load(config/pipeline.yaml)['books'] (fallback regex). Reader: AST vòng "
                "`for X in config['books']`, tham số hàm `book_cfg`, bí danh `Y = X` -> X['k'] (KeyError nếu thiếu) / X.get('k'). "
                "data_books: liệt kê data/*/ (pdf, pages/*.{png,jpg,...}, quocngu_pages, nom_pages) và prepared/<dir>/ (số *_ocr_cache.json, *_qn_ocr_cache.json)."),
        "yaml": rel(ycfg), "yaml_loaded": cfg is not None,
        "n_books": len(books), "books": books,
        "keys_present_in_yaml": dict(sorted(keyset.items())),
        "required_keys": required, "optional_keys": optional,
        "pdf_readers": [r for r in readers if r["key"] == "pdf"],
        "all_key_readers": readers,
        "paths": {k: v for k, v in ((cfg or {}).get("paths") or {}).items()} if isinstance(cfg, dict) else None,
    }


# ----------------------------------------------------------------------------- (e) checkpoints

CKPT_RE = re.compile(r"\.(?:pt|pth|h5|ckpt|keras|onnx|safetensors)$")
CKPT_DIR_HINTS = ("train_crop", "nom-embed", "pipeline/align_engine/nom_classifier/checkpoints",
                  "pipeline/align_engine/char_detector", "ArcFace", "KhoiB/v3/models",
                  "KhoiB/v3/p_visual_oof_v3_results/models")


def _outer_path_expr(node: ast.AST) -> ast.AST:
    """Leo lên qua chuỗi BinOp '/' để lấy biểu thức đường dẫn đầy đủ chứa literal."""
    cur = node
    while True:
        par = getattr(cur, "_parent", None)
        if isinstance(par, ast.BinOp) and isinstance(par.op, ast.Div):
            cur = par
        else:
            return cur


def _real(pathstr: str) -> Path | None:
    if pathstr.startswith("<REPO>"):
        return REPO / pathstr[len("<REPO>"):].lstrip("/")
    if pathstr.startswith("<"):
        return None
    return REPO / pathstr


def section_checkpoints(limit: int) -> dict:
    refs: dict[str, dict] = {}
    env_overrides: list[dict] = []
    for p in py_files():
        rp = rel(p)
        _, tree = load(p)
        if tree is None:
            continue
        # thư mục ứng viên trong cùng tệp (tuple/list module-level các Path) cho f-string fold{k}.pt
        local_dirs: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                v = eval_expr(node.value, p)
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    local_dirs.extend(v)
                elif isinstance(v, str) and v.startswith("<REPO>"):
                    local_dirs.append(v)
        for node in ast.walk(tree):
            lit = None; is_fstr = False
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and CKPT_RE.search(node.value):
                if isinstance(getattr(node, "_parent", None), ast.JoinedStr):
                    continue                          # mảnh của f-string, xử lý ở JoinedStr
                lit = node.value
                if " " in lit or ".." in lit or "không" in lit:
                    continue                          # chuỗi thông báo, không phải đường dẫn
            elif isinstance(node, ast.JoinedStr):
                txt = "".join(v.value if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}"
                              for v in node.values)
                if CKPT_RE.search(txt):
                    lit, is_fstr = txt, True
            if lit is None:
                continue
            outer = _outer_path_expr(node)
            full = None if is_fstr else eval_expr(outer, p)
            unresolved = (not is_fstr and full is None)
            if unresolved:
                full = ast.unparse(outer)
            key = full if isinstance(full, str) else lit
            if key.startswith("<REPO>/"):
                key = key[len("<REPO>/"):]            # gộp '<REPO>/x' và 'x'
            e = refs.setdefault(key, {"path": key, "literal": lit, "fstring": is_fstr, "referenced_in": [],
                                      "src": ast.unparse(outer)[:100]})
            e["referenced_in"].append(f"{rp}:{node.lineno}")
            if is_fstr:
                pat = re.sub(r"\{[^}]*\}", "*", lit)
                found = []
                for d in local_dirs + [f"<REPO>/{h}" for h in CKPT_DIR_HINTS]:
                    rd = _real(d)
                    if rd and rd.is_dir():
                        found += [rel(x) for x in sorted(rd.glob(pat))]
                e["glob_pattern"] = pat
                e["matches"] = sorted(set(found))[:10]
                e["exists"] = bool(found)
            else:
                rd = None if unresolved else (_real(full) if isinstance(full, str) else None)
                e["resolved"] = rel(rd) if rd else None
                e["unresolved_expr"] = unresolved
                e["exists"] = bool(rd and rd.exists())
                e["size_mb"] = round(rd.stat().st_size / 1e6, 1) if rd and rd.exists() else None
        for node in ast.walk(tree):                  # os.environ.get("X_CKPT") — ghi đè bằng env
            if isinstance(node, ast.Call) and ast.unparse(node.func) in ("os.environ.get", "os.getenv") \
                    and node.args and isinstance(node.args[0], ast.Constant) \
                    and re.search(r"CKPT|MODEL|WEIGHT", str(node.args[0].value)):
                env_overrides.append({"env": node.args[0].value, "file": rp, "line": node.lineno})
    items = sorted(refs.values(), key=lambda e: e["path"])
    for e in items:
        e["referenced_in"] = sorted(set(e["referenced_in"]))[:limit or 20]
    known_dirs = {}
    for d in CKPT_DIR_HINTS:
        dp = REPO / d
        if dp.exists():
            known_dirs[d] = sorted(x.name for x in dp.iterdir() if CKPT_RE.search(x.name))[:20]
    return {
        "how": (f"AST: ast.Constant/JoinedStr khớp /{CKPT_RE.pattern}/ trong pipeline/ core/; leo qua chuỗi BinOp '/' "
                "rồi ước lượng tĩnh (REPO, __file__, Path().parent, hằng module) -> đường dẫn; kiểm tồn tại + kích thước. "
                "f-string (fold{k}.pt) -> glob trong tuple/list Path cùng tệp (vd DEFAULT_MODEL_DIRS) và CKPT_DIR_HINTS. "
                "env_overrides: os.environ.get('*CKPT*')."),
        "n_paths": len(items), "n_exists": sum(1 for i in items if i.get("exists")),
        "missing": [i["path"] for i in items if not i.get("exists")],
        "items": items, "env_overrides": env_overrides, "known_dirs": known_dirs,
    }


# ----------------------------------------------------------------------------- (f) git

def section_git() -> dict:
    head = git("rev-parse", "--short", "HEAD").strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    status = git("status", "--porcelain", "--", "pipeline", "core").splitlines()
    numstat = {}
    for l in git("diff", "--numstat", "--", "pipeline", "core").splitlines():
        parts = l.split("\t")
        if len(parts) == 3:
            numstat[parts[2]] = {"ins": int(parts[0]) if parts[0].isdigit() else None,
                                 "del": int(parts[1]) if parts[1].isdigit() else None}
    dirty = []
    for l in status:
        st, f = l[:2], l[3:]
        dirty.append({"file": f, "status": st.strip() or "?", **numstat.get(f, {})})
    diff_ocr = git("diff", "--", "core/ocr/ocr_api.py")
    added = [l[1:] for l in diff_ocr.splitlines() if l.startswith("+") and not l.startswith("+++")]
    guest = {
        "file": "core/ocr/ocr_api.py",
        "has_uncommitted_diff": bool(diff_ocr.strip()),
        "guest_mode_lines_added": sum(1 for l in added if re.search(r"guest", l, re.I)),
        "functions_touched": sorted({m.group(1) for m in re.finditer(r"^@@.*?def (\w+)", diff_ocr, re.M)}),
        "added_defs": sorted({m.group(1) for l in added for m in [re.match(r"\s*def (\w+)", l)] if m}),
        "committed_version_has_guest": bool(re.search(r"guest", git("show", "HEAD:core/ocr/ocr_api.py"), re.I)),
        "worktree_has_guest": bool(re.search(r"guest", (REPO / "core/ocr/ocr_api.py").read_text(encoding="utf-8", errors="replace"), re.I)),
    }
    ds_tracked = sorted(git("ls-files", "--", "dataset_out").splitlines())
    ds_status = git("status", "--porcelain", "--", "dataset_out").splitlines()
    ds_deleted = sorted(l[3:] for l in ds_status if l.startswith(" D") or l.startswith("D "))
    ds_modified = sorted(l[3:] for l in ds_status if l[:2].strip() and l[:2].strip() not in ("D", "??"))
    ds_dir = REPO / "dataset_out"
    dataset_out = {
        "dir_exists": ds_dir.exists(),
        "n_tracked": len(ds_tracked), "n_deleted_in_worktree": len(ds_deleted),
        "tracked_files": ds_tracked, "deleted_files": ds_deleted,
        "all_tracked_deleted": bool(ds_tracked) and set(ds_tracked) == set(ds_deleted),
        "n_modified_in_worktree": len(ds_modified), "modified_files": ds_modified,
        "tracked_clean": bool(ds_tracked) and not ds_deleted and not ds_modified,
        "n_files_on_disk": sum(1 for _ in ds_dir.rglob("*") if _.is_file()) if ds_dir.exists() else 0,
    }
    untracked = [l[3:] for l in git("status", "--porcelain", "--untracked-files=all", "--", "pipeline", "core").splitlines()
                 if l.startswith("??")]
    return {
        "how": ("git rev-parse HEAD; git status --porcelain -- pipeline core; git diff --numstat; git diff -- "
                "core/ocr/ocr_api.py (đếm dòng + chứa 'guest'); git ls-files/status -- dataset_out; kiểm thư mục."),
        "head": head, "branch": branch,
        "n_dirty_pipeline_core": len(dirty), "dirty_pipeline_core": dirty,
        "untracked_pipeline_core": untracked,
        "ocr_api_guest_mode": guest,
        "dataset_out": dataset_out,
    }


# ----------------------------------------------------------------------------- invariants + main

def build_invariants(S: dict) -> list[dict]:
    inv = []

    def add(name, expected, observed):
        inv.append({"name": name, "expected": expected, "observed": observed, "pass": expected == observed})

    kc = S["column_pins"]["known_pins_check"]
    add("known_pins_all_found", len(KNOWN_PINS), sum(1 for k in kc if k["found"]))
    add("column_pins_prod_nonzero", True, S["column_pins"]["n_hits_prod"] > 0)
    cli = S["cli"]
    add("cli_all_files_exist", len(CLI_FILES), sum(1 for s in cli["steps"] if s["exists"]))
    add("cli_each_step_has_args", len(CLI_FILES), sum(1 for s in cli["steps"] if s.get("n_arguments", 0) > 0))
    add("remediation_has_census_apply", ["apply", "census"],
        next((s["subcommands"] for s in cli["steps"] if s["module"] == "pipeline.remediation"), []))
    add("run_pipeline_sh_calls_build_dataset", True,
        any(s["module"] == "pipeline.align_engine.build_dataset" and s.get("run_pipeline_sh") for s in cli["steps"]))
    jk = S["json_keys"]["schemas"]
    add("ocr_cache_writer_keys_include_core", True,
        {"columns", "coords_space", "image_hash", "pixel_hash", "framed", "frame_pad"} <= set(jk["ocr_cache"]["writer"]["keys"]))
    add("ocr_cache_per_char_keys", ["bbox", "char", "y_center"], sorted(jk["ocr_cache"]["per_char_writer"]["keys"]))
    add("transcription_writer_has_columns", True, "columns" in jk["transcription"]["writer"]["keys"])
    add("coords_space_is_read_by_engine", True,
        any(k["key"] == "coords_space" and k["n_reads"] > 0 for k in S["json_keys"]["keys"]))
    cb = S["config_books"]
    add("config_books_n", 3, cb["n_books"])
    add("config_books_required_keys", ["name", "pdf"], cb["required_keys"])
    add("config_books_all_have_required_keys", [], cb["books_missing_required"])
    lvt = next((b for b in cb["data_books"] if b["dir"] == "LucVanTien1883"), None)
    if lvt:                                          # sự thật đã đo: LVT1883 = 105 trang
        add("data_LucVanTien1883_pages_105", 105, lvt["n_pages"])
    add("config_books_pdf_read_in_step0_and_step1", ["pipeline/step0_setup.py", "pipeline/step1_extract.py"],
        sorted({r["file"] for r in cb["pdf_readers"]}))
    ck = S["checkpoints"]
    det = [i for i in ck["items"] if i["literal"] == "detector_r34.best.pt" and i.get("resolved") == "train_crop/detector_r34.best.pt"]
    add("detector_r34_ckpt_exists", True, bool(det and det[0]["exists"]))
    g = S["git"]
    # 22/09: Guest Mode đã commit (65f7ca9) → kỳ vọng bản HEAD **và** bản worktree đều có guest.
    # 23/09 (vòng 5): bỏ mệnh đề "ocr_api.py không có diff" khỏi BẤT BIẾN. Mệnh đề ấy sinh ra khi
    # Guest Mode chỉ tồn tại trong worktree (chưa commit) — nó bắt đúng lỗi ĐÓ. Nay guest đã ở
    # trong git, còn ocr_api.py vẫn được sửa hợp lệ (vòng 5: tham số kim ocr_id/lang_type/
    # reading_direction/font_type cho recognize) nên "có diff" KHÔNG còn là dấu hiệu hỏng.
    # Trạng thái diff vẫn được GHI trong summary.git (ocr_api_guest_mode.has_uncommitted_diff).
    add("ocr_api_guest_mode_committed", True,
        g["ocr_api_guest_mode"]["committed_version_has_guest"] and g["ocr_api_guest_mode"]["worktree_has_guest"])
    # 22/09: dataset_out/ STT (10 tệp tracked) phải sạch — `remediation apply` không --out từng ghi đè (đã khôi phục)
    add("dataset_out_tracked_clean", True, g["dataset_out"]["tracked_clean"])
    return inv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPO / "docs" / "PIPELINE_FACTS.json"),
                    help="tệp sự kiện đầy đủ (mặc định docs/PIPELINE_FACTS.json)")
    ap.add_argument("--summary", default=str(REPO / "measure_out" / "code_facts" / "summary.json"),
                    help="tóm tắt ≤ 8 KB")
    ap.add_argument("--limit", type=int, default=0, help="giới hạn số mục liệt kê mỗi bảng (0 = tất cả)")
    ap.add_argument("--check", action="store_true", help="exit 1 nếu có bất biến FAIL")
    ap.add_argument("--book", default=None, help="(không dùng; giữ đồng nhất CLI bộ measure)")
    ap.add_argument("--workers", type=int, default=1, help="(không dùng)")
    args = ap.parse_args(argv)

    S = {
        "column_pins": section_column_pins(args.limit),
        "cli": section_cli(args.limit),
        "json_keys": section_json_keys(args.limit),
        "config_books": section_config_books(),
        "checkpoints": section_checkpoints(args.limit),
        "git": section_git(),
    }
    inv = build_invariants(S)
    counts = {
        "column_pins": S["column_pins"]["n_hits"], "column_pins_prod": S["column_pins"]["n_hits_prod"],
        "identifier_refs": S["column_pins"]["n_identifier_refs"],
        "known_pins_found": f"{sum(1 for k in S['column_pins']['known_pins_check'] if k['found'])}/{len(KNOWN_PINS)}",
        "cli_steps": S["cli"]["n_steps"], "cli_arguments": S["cli"]["n_arguments_total"],
        "json_keys": len(S["json_keys"]["keys"]), "json_keys_never_read": len(S["json_keys"]["never_read"]),
        "config_books": S["config_books"]["n_books"], "pdf_readers": len(S["config_books"]["pdf_readers"]),
        "checkpoint_paths": S["checkpoints"]["n_paths"], "checkpoint_exists": S["checkpoints"]["n_exists"],
        "git_dirty_pipeline_core": S["git"]["n_dirty_pipeline_core"],
        "dataset_out_deleted": S["git"]["dataset_out"]["n_deleted_in_worktree"],
        "invariants_pass": f"{sum(1 for i in inv if i['pass'])}/{len(inv)}",
    }
    doc = {
        "generated_by": GENERATOR, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": str(REPO), "git_head": S["git"]["head"],
        "note": "Sinh tự động bằng grep/AST; không có LLM. Agent đọc tệp này thay vì đọc mã nguồn. "
                "Chạy lại: .venv/bin/python scripts/measure/code_facts.py",
        "counts": counts, "invariants": inv, "sections": S,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    # tóm tắt ≤ 8 KB
    summary = {
        "generated_by": GENERATOR, "generated_at": doc["generated_at"], "git_head": S["git"]["head"],
        "facts_file": rel(out), "facts_bytes": out.stat().st_size,
        "counts": counts,
        "known_pins_not_found": [f"{x['file']}:{x['line']} -> nearest {x['nearest_line']}"
                                 for x in S["column_pins"]["known_pins_check"] if not x["found"]],
        "column_pins_prod": [f"{h['file']}:{h['line']} [{h['category']}] {h['detail'][:48]}"
                             for h in S["column_pins"]["hits"] if not h["lab_or_test"]][:40],
        "cli_steps": [{"module": s["module"], "n_args": s.get("n_arguments", 0),
                       "subcommands": s.get("subcommands") or None,
                       "in_run_pipeline_sh": len(s.get("run_pipeline_sh", []))} for s in S["cli"]["steps"]],
        "json_keys_never_read": S["json_keys"]["never_read"],
        "config_books": [b["name"] for b in S["config_books"]["books"]],
        "data_books_not_in_yaml": S["config_books"]["data_books_not_in_yaml"],
        "config_required_keys": S["config_books"]["required_keys"],
        "checkpoints_missing": S["checkpoints"]["missing"],
        "git": {"dirty_pipeline_core": [d["file"] for d in S["git"]["dirty_pipeline_core"]],
                "ocr_api_guest_mode_committed": S["git"]["ocr_api_guest_mode"]["committed_version_has_guest"],
                "dataset_out": {k: S["git"]["dataset_out"][k] for k in ("dir_exists", "n_tracked", "n_deleted_in_worktree")}},
        "invariants": inv,
    }
    sp = Path(args.summary); sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[code_facts] -> {rel(out)} ({out.stat().st_size/1024:.1f} KB) | {rel(sp)} ({sp.stat().st_size/1024:.1f} KB)")
    for k, v in counts.items():
        print(f"  {k:28} {v}")
    fails = [i for i in inv if not i["pass"]]
    for i in fails:
        print(f"  FAIL {i['name']}: expected={i['expected']!r} observed={i['observed']!r}")
    for k in S["column_pins"]["known_pins_check"]:
        if not k["found"]:
            print(f"  LỆCH known pin {k['file']}:{k['line']} -> gần nhất dòng {k['nearest_line']}")
    return 1 if (args.check and fails) else 0


if __name__ == "__main__":
    sys.exit(main())
