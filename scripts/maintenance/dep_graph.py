#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dep_graph.py — dựng ĐỒ THỊ PHỤ THUỘC của kho GanNhanOCR bằng máy (AST + grep), KHÔNG đoán.

Mục đích: tách tệp .py SỐNG (truy ngược được về một ĐIỂM VÀO THẬT) khỏi ứng viên CHẾT.

Cách làm
  1. Quét mọi .py ngoài .venv/.git/__pycache__ -> nút của đồ thị.
  2. AST từng tệp:
       - import tĩnh:  `import a.b`, `from a.b import c` (kể cả relative `from . import x`)
       - import động:  importlib.import_module("a.b"), __import__("a.b"),
                       chuỗi hằng dạng module giải được ra tệp thật,
                       f-string tiền tố "a.b.{...}"  -> nở ra CẢ gói a.b (bảo toàn)
       - gọi tiến trình con: mọi chuỗi hằng chứa ".py"; mọi cặp ("-m", "a.b") trong list/tuple/call
  3. Grep văn bản mọi .sh/.ipynb/.yaml/.yml/.toml/.cfg/.md-trong-docs? (mặc định .sh/.ipynb/.yaml/.yml):
       - `-m <module>` và `<đường/dẫn>.py`
  4. Lan truyền đóng gói (BFS) từ tập ĐIỂM VÀO THẬT -> SỐNG; phần còn lại -> ứng viên CHẾT.

Xuất (mặc định measure_out/maintenance/):
    dep_graph.json      nút + cạnh + lý do sống + cảnh báo cạnh không giải được
    alive.txt           danh sách tệp SỐNG (đã sắp)
    dead_candidates.txt danh sách ứng viên CHẾT (đã sắp)
    by_dir.tsv          bảng theo thư mục: sống / chết / tổng

Chạy:  .venv/bin/python scripts/maintenance/dep_graph.py [--root .] [--out measure_out/maintenance]
       thêm --check để chạy các bất biến (invariants) kiểm chéo.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

SKIP_DIR_NAMES = {".venv", ".git", "__pycache__", "node_modules", ".ipynb_checkpoints",
                  ".mypy_cache", ".pytest_cache", ".idea", ".vscode"}

# ---------------------------------------------------------------- ĐIỂM VÀO THẬT
# Mọi thứ "sống" phải truy ngược được về đây. Nguồn: CLAUDE.md + docs + run_pipeline.sh.
ENTRY_SHELL = [
    "run_pipeline.sh",                 # điểm vào chính (mọi nhánh cờ)
    "scripts/run_all_selftests.sh",    # runner selftest chốt trong docs
]
ENTRY_PY = [
    # bộ đo tái lập được (CLAUDE.md: KHÔNG chạy lại phép đo bằng LLM)
    "scripts/measure/measure.py",
    "scripts/measure/auto_precision.py",
    # selftest được CLAUDE.md/docs nêu đích danh
    "pipeline/phase1_engine_selftest.py",
    "pipeline/align_engine/book_layout_selftest.py",
    "pipeline/tools/ingest_lithograph_selftest.py",
    "pipeline/tools/ingest_prose_selftest.py",
    "pipeline/tools/ingest_ihr_selftest.py",
    "pipeline/remediation/selftest.py",
    "pipeline/remediation/mechanism_gates_selftest.py",
    "pipeline/tools/selftest.py",
    "pipeline/publish/selftest.py",
    "pipeline/ground_truth/selftest.py",
    # quy trình huấn luyện detector v2 còn dùng
    "lab/i5_detector_v2/make_bundle.py",
    "lab/i5_detector_v2/train_kaggle.py",
]
ENTRY_NOTEBOOK = [
    "lab/i5_detector_v2/kaggle_train_i5_v2.ipynb",
]
# Thư mục quét văn bản để lấy lệnh gọi (ngoài ENTRY_SHELL đã nêu)
TEXT_EXT = (".sh", ".ipynb", ".yaml", ".yml")

RE_DASH_M = re.compile(r"-m\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)")
RE_PY_PATH = re.compile(r"([A-Za-z_][\w./\-]*\.py)\b")
RE_MODULE = re.compile(r"^[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*$")


# ---------------------------------------------------------------- quét tệp
def iter_files(root: Path, exts=None):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for fn in filenames:
            p = Path(dirpath) / fn
            if exts is None or p.suffix in exts:
                yield p


class Graph:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.py_files: set[str] = set()          # rel posix path
        self.edges: dict[str, set[str]] = defaultdict(set)   # src -> {dst rel path}
        self.edge_reason: dict[tuple[str, str], str] = {}
        self.unresolved: list[dict] = []
        self._index()

    # --- chỉ mục module -> tệp
    def _index(self):
        for p in iter_files(self.root, {".py"}):
            self.py_files.add(p.relative_to(self.root).as_posix())

    def resolve_module(self, dotted: str) -> list[str]:
        """Trả về danh sách tệp .py ứng với tên module dạng chấm (có thể 0..2 tệp)."""
        if not dotted or not RE_MODULE.match(dotted):
            return []
        parts = dotted.split(".")
        out = []
        # Python nạp MỌI gói tổ tiên: a.b.c -> a/__init__.py, a/b/__init__.py
        for i in range(1, len(parts)):
            anc = "/".join(parts[:i]) + "/__init__.py"
            if anc in self.py_files:
                out.append(anc)
        cand_mod = "/".join(parts) + ".py"
        if cand_mod in self.py_files:
            out.append(cand_mod)
        base = "/".join(parts)
        for special in ("__init__.py", "__main__.py"):
            c = f"{base}/{special}"
            if c in self.py_files:
                out.append(c)
        return out

    def resolve_path(self, raw: str, from_file: str | None = None) -> list[str]:
        """Giải một đường dẫn .py (tương đối gốc kho, hoặc cạnh tệp gọi)."""
        raw = raw.strip().lstrip("./")
        if raw in self.py_files:
            return [raw]
        if from_file:
            cand = (Path(from_file).parent / raw).as_posix()
            cand = os.path.normpath(cand).replace(os.sep, "/")
            if cand in self.py_files:
                return [cand]
        # bỏ tiền tố biến shell / đường tuyệt đối: "$REPO_ROOT/pipeline/x.py" -> "pipeline/x.py"
        segs = raw.split("/")
        for i in range(1, len(segs)):
            tail = "/".join(segs[i:])
            if tail in self.py_files:
                return [tail]
        # khớp theo đuôi đường dẫn (ví dụ "$REPO_ROOT/pipeline/x.py")
        hits = [f for f in self.py_files if f.endswith("/" + raw) or f == raw]
        if len(hits) == 1:
            return hits
        return []

    def expand_package(self, dotted: str) -> list[str]:
        """Nở cả gói (dùng cho import động f-string: prefix biết, hậu tố không)."""
        base = "/".join(dotted.split("."))
        return sorted(f for f in self.py_files if f.startswith(base + "/"))

    def add(self, src: str, dst: str, reason: str):
        if dst == src:
            return
        self.edges[src].add(dst)
        self.edge_reason.setdefault((src, dst), reason)


# ---------------------------------------------------------------- AST
def pkg_of(rel: str) -> list[str]:
    """Gói chứa tệp, dạng list phần tử (dùng giải relative import)."""
    parts = rel.split("/")
    if parts[-1] == "__init__.py":
        return parts[:-1]
    return parts[:-1]


def const_str(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def scan_py(g: Graph, rel: str):
    src_path = g.root / rel
    try:
        text = src_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text, filename=rel)
    except SyntaxError as e:
        g.unresolved.append({"file": rel, "kind": "syntax_error", "detail": str(e)[:200]})
        return
    except Exception as e:  # noqa: BLE001
        g.unresolved.append({"file": rel, "kind": "read_error", "detail": str(e)[:200]})
        return

    my_pkg = pkg_of(rel)

    my_dir = "/".join(rel.split("/")[:-1])

    def link_module(dotted: str, reason: str):
        tgts = g.resolve_module(dotted)
        if not tgts and my_dir:
            # `sys.path.insert(0, Path(__file__).parent)` rồi `from model import X`:
            # tên trần giải theo THƯ MỤC CỦA CHÍNH TỆP (vd nom_classifier/infer.py -> model.py)
            dpath = "/".join(dotted.split("."))
            for special in (f"{my_dir}/{dpath}.py", f"{my_dir}/{dpath}/__init__.py",
                            f"{my_dir}/{dpath}/__main__.py"):
                if special in g.py_files:
                    tgts = [special]
                    reason = reason + ":sibling"
                    break
        for t in tgts:
            g.add(rel, t, reason)
        return bool(tgts)

    for node in ast.walk(tree):
        # import tĩnh -------------------------------------------------
        if isinstance(node, ast.Import):
            for a in node.names:
                link_module(a.name, "import")
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative
                base = my_pkg[: len(my_pkg) - (node.level - 1)] if node.level > 1 else my_pkg
                prefix = ".".join(base)
                mod = f"{prefix}.{node.module}" if node.module else prefix
            else:
                mod = node.module or ""
            if mod:
                link_module(mod, "from-import")
                for a in node.names:
                    if a.name != "*":
                        link_module(f"{mod}.{a.name}", "from-import-submodule")
        # gọi động ----------------------------------------------------
        elif isinstance(node, ast.Call):
            fname = ""
            f = node.func
            if isinstance(f, ast.Attribute):
                fname = f.attr
            elif isinstance(f, ast.Name):
                fname = f.id
            if fname in ("import_module", "__import__", "find_spec", "spec_from_file_location"):
                for arg in node.args:
                    s = const_str(arg)
                    if s:
                        if s.endswith(".py"):
                            for t in g.resolve_path(s, rel):
                                g.add(rel, t, f"dynamic:{fname}")
                        elif not link_module(s, f"dynamic:{fname}"):
                            g.unresolved.append({"file": rel, "kind": f"dynamic:{fname}", "detail": s})
                    elif isinstance(arg, ast.JoinedStr):
                        # f"pipeline.tools.{name}" -> nở CẢ gói (bảo toàn)
                        lead = ""
                        for v in arg.values:
                            s2 = const_str(v)
                            if s2 is None:
                                break
                            lead += s2
                        lead = lead.rstrip(".")
                        if lead and RE_MODULE.match(lead):
                            exp = g.expand_package(lead)
                            for t in exp:
                                g.add(rel, t, f"dynamic:{fname}:fstring-prefix")
                            g.unresolved.append({"file": rel, "kind": "dynamic-fstring",
                                                 "detail": lead + ".*", "expanded": len(exp)})

        # mọi chuỗi hằng -------------------------------------------------
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if s.endswith(".py"):
                for t in g.resolve_path(s, rel):
                    g.add(rel, t, "string:script-path")

    # cặp ("-m", "module") ở bất kỳ list/tuple/args nào ------------------
    for node in ast.walk(tree):
        seq = None
        if isinstance(node, (ast.List, ast.Tuple)):
            seq = node.elts
        elif isinstance(node, ast.Call):
            seq = node.args
        if not seq:
            continue
        vals = [const_str(e) for e in seq]
        for i, v in enumerate(vals[:-1]):
            if v == "-m" and vals[i + 1]:
                if not g.resolve_module(vals[i + 1]):
                    g.unresolved.append({"file": rel, "kind": "dash-m", "detail": vals[i + 1]})
                for t in g.resolve_module(vals[i + 1]):
                    g.add(rel, t, "subprocess:-m")


# ---------------------------------------------------------------- văn bản (.sh/.ipynb/.yaml)
def scan_text(g: Graph, rel: str):
    p = g.root / rel
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return
    if rel.endswith(".ipynb"):
        try:
            nb = json.loads(text)
            text = "\n".join(
                "".join(c.get("source", [])) for c in nb.get("cells", [])
            )
        except Exception:  # noqa: BLE001
            pass
    for m in RE_DASH_M.finditer(text):
        tgts = g.resolve_module(m.group(1))
        if not tgts:
            g.unresolved.append({"file": rel, "kind": "text:-m", "detail": m.group(1)})
        for t in tgts:
            g.add(rel, t, "shell:-m")
    for m in RE_PY_PATH.finditer(text):
        raw = m.group(1)
        for t in g.resolve_path(raw, rel):
            g.add(rel, t, "shell:script-path")
    # .sh/.ipynb: mảng module truyền qua biến (vd MODULES=( core.pdf.parser_selftest ... )
    # rồi `"$PY" -m "$m"`) — regex `-m` không bắt được, nên nhận MỌI token dạng module
    # giải được ra tệp thật. Chỉ áp cho .sh/.ipynb để không dính chuỗi trong .yaml.
    if rel.endswith((".sh", ".ipynb")):
        for tok in re.findall(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+", text):
            for t in g.resolve_module(tok):
                g.add(rel, t, "shell:bare-module-token")


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out) if args.out else root / "measure_out" / "maintenance"
    out_dir.mkdir(parents=True, exist_ok=True)

    g = Graph(root)

    text_files = []
    for p in iter_files(root, set(TEXT_EXT)):
        text_files.append(p.relative_to(root).as_posix())

    for rel in sorted(g.py_files):
        scan_py(g, rel)
    for rel in sorted(text_files):
        scan_text(g, rel)

    # ------- tập điểm vào
    entries: list[str] = []
    missing_entries: list[str] = []
    for e in ENTRY_SHELL + ENTRY_NOTEBOOK:
        (entries if (root / e).exists() else missing_entries).append(e)
    for e in ENTRY_PY:
        if e in g.py_files:
            entries.append(e)
        else:
            missing_entries.append(e)

    # ------- BFS
    alive: dict[str, dict] = {}
    q = deque()
    for e in entries:
        if e.endswith(".py"):
            alive[e] = {"via": "ENTRY", "from": None}
            q.append(e)
        else:
            for t in sorted(g.edges.get(e, ())):
                if t not in alive:
                    alive[t] = {"via": g.edge_reason.get((e, t), "shell"), "from": e}
                    q.append(t)
    while q:
        cur = q.popleft()
        for t in sorted(g.edges.get(cur, ())):
            if t not in alive:
                alive[t] = {"via": g.edge_reason.get((cur, t), "?"), "from": cur}
                q.append(t)

    dead = sorted(f for f in g.py_files if f not in alive)
    alive_files = sorted(alive)

    # ------- bảng theo thư mục
    def bucket(f: str) -> str:
        parts = f.split("/")
        return "/".join(parts[:2]) if len(parts) > 2 else (parts[0] if len(parts) > 1 else "<root>")

    table = defaultdict(lambda: [0, 0])
    for f in g.py_files:
        table[bucket(f)][0 if f in alive else 1] += 1
    rows = sorted(table.items(), key=lambda kv: (-(kv[1][0] + kv[1][1]), kv[0]))

    # ------- xuất
    (out_dir / "alive.txt").write_text("\n".join(alive_files) + "\n", encoding="utf-8")
    (out_dir / "dead_candidates.txt").write_text("\n".join(dead) + "\n", encoding="utf-8")
    with (out_dir / "by_dir.tsv").open("w", encoding="utf-8") as fh:
        fh.write("dir\talive\tdead\ttotal\n")
        for d, (a, dd) in rows:
            fh.write(f"{d}\t{a}\t{dd}\t{a + dd}\n")

    payload = {
        "generated_by": "scripts/maintenance/dep_graph.py",
        "root": str(root),
        "counts": {
            "py_files": len(g.py_files),
            "text_files_scanned": len(text_files),
            "edges": sum(len(v) for v in g.edges.values()),
            "alive": len(alive_files),
            "dead_candidates": len(dead),
            "unresolved": len(g.unresolved),
        },
        "entries": entries,
        "entries_missing": missing_entries,
        "alive": {f: alive[f] for f in alive_files},
        "dead_candidates": dead,
        "by_dir": [{"dir": d, "alive": a, "dead": dd, "total": a + dd} for d, (a, dd) in rows],
        "edges": {s: sorted(v) for s, v in sorted(g.edges.items()) if v},
        "edge_reasons": {f"{s}->{d}": r for (s, d), r in sorted(g.edge_reason.items())},
        "unresolved": g.unresolved,
    }
    (out_dir / "dep_graph.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # ------- in gọn
    print(f"py={len(g.py_files)} text={len(text_files)} edges={payload['counts']['edges']} "
          f"ALIVE={len(alive_files)} DEAD={len(dead)} unresolved={len(g.unresolved)}")
    if missing_entries:
        print("ENTRY THIẾU:", ", ".join(missing_entries))
    print(f"{'dir':<34}{'alive':>6}{'dead':>6}{'tot':>6}")
    for d, (a, dd) in rows:
        print(f"{d:<34}{a:>6}{dd:>6}{a + dd:>6}")

    # ------- bất biến kiểm chéo
    if args.check:
        fails = []
        must_alive_dirs = ["pipeline/align_engine/", "pipeline/remediation/", "pipeline/tools/"]
        # mọi mô-đun được run_pipeline.sh gọi trực tiếp phải SỐNG
        rp = "run_pipeline.sh"
        direct = sorted(g.edges.get(rp, ()))
        for t in direct:
            if t not in alive:
                fails.append(f"run_pipeline.sh gọi {t} nhưng KHÔNG sống")
        print(f"\nINV run_pipeline.sh gọi trực tiếp {len(direct)} tệp .py — "
              f"{'PASS' if not fails else 'FAIL'}")
        for d in must_alive_dirs:
            tot = [f for f in g.py_files if f.startswith(d)]
            al = [f for f in tot if f in alive]
            print(f"INV {d:<28} sống {len(al)}/{len(tot)}")
        if fails:
            for f in fails[:20]:
                print("  FAIL:", f)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
