"""Phase 1 verification: AST chunks + SQLite symbol graph with true line numbers.

Run directly:      .venv/bin/python test_phase1.py
Or with pytest:    .venv/bin/python -m pytest test_phase1.py -q

Every test builds a throwaway fixture repo on disk, runs the real
`indexer.build_graph` pipeline against it, and asserts on the resulting
chunks / SQLite nodes / SQLite edges.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

from code_parser import parse_source
from graph import GraphDB, node_id
from indexer import build_graph


# --------------------------------------------------------------------------- #
# Fixture content. Line numbers below are RELIED ON by the assertions.
# --------------------------------------------------------------------------- #
AUTH_PY = """\
import os
import hashlib
from sample_pkg.utils import helper


TOKEN_TTL = 3600


def hash_password(password):
    \"\"\"Hash a password using sha256.\"\"\"
    salted = password + "pepper"
    return hashlib.sha256(salted.encode()).hexdigest()


class TokenValidator:
    \"\"\"Validates signed auth tokens.\"\"\"

    algorithm = "HS256"

    def __init__(self, secret):
        self.secret = secret

    def validate(self, token):
        digest = hash_password(token)
        return helper(digest, self.secret)
"""
#  1 import os
#  2 import hashlib
#  3 from sample_pkg.utils import helper
#  6 TOKEN_TTL = 3600
#  9 def hash_password(password):        -> function  9..12
# 12     return hashlib.sha256(salted.encode())...   -> calls sha256, encode
# 15 class TokenValidator:               -> class    15..25
# 20     def __init__(self, secret):     -> method   20..21
# 23     def validate(self, token):      -> method   23..25
# 24         digest = hash_password(token)           -> call hash_password
# 25         return helper(digest, self.secret)      -> call helper

UTILS_PY = """\
def helper(a, b):
    return a == b
"""

APP_JS = """\
import { readFile } from "fs";

const loadConfig = (path) => readFile(path);

function boot() {
  return loadConfig("./config.json");
}

class Server {
  start() {
    boot();
  }
}
"""
#  1 import { readFile } from "fs";
#  3 const loadConfig = (path) => readFile(path);   -> function 3..3, calls readFile
#  5 function boot() {                              -> function 5..7
#  6   return loadConfig("./config.json");          -> call loadConfig
#  9 class Server {                                 -> class 9..13
# 10   start() {                                    -> method 10..12
# 11     boot();                                    -> call boot

README_MD = "\n".join(f"line {i}" for i in range(1, 41)) + "\n"  # 40 lines


def make_repo() -> str:
    root = tempfile.mkdtemp(prefix="phase1_repo_")
    os.makedirs(os.path.join(root, "sample_pkg"))
    _write(root, "sample_pkg/auth.py", AUTH_PY)
    _write(root, "sample_pkg/utils.py", UTILS_PY)
    _write(root, "app.js", APP_JS)
    _write(root, "README.md", README_MD)
    return root


def _write(root: str, rel: str, content: str) -> None:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)


def _db(root: str) -> str:
    return os.path.join(root, "graph.sqlite")


def chunk(pf, name):
    return next(c for c in pf.chunks if c.symbol_name == name)


# --------------------------------------------------------------------------- #
# 1. AST chunk boundaries + deterministic line numbers (Python)
# --------------------------------------------------------------------------- #
def test_python_ast_chunks_have_true_line_numbers():
    pf = parse_source("sample_pkg/auth.py", AUTH_PY.encode())

    hp = chunk(pf, "hash_password")
    assert (hp.symbol_type, hp.start_line, hp.end_line) == ("function", 9, 12), hp
    assert hp.code.startswith("def hash_password(password):")
    assert '"""Hash a password using sha256."""' in hp.code  # docstring retained

    cls = chunk(pf, "TokenValidator")
    assert (cls.symbol_type, cls.start_line, cls.end_line) == ("class", 15, 25), cls

    init = chunk(pf, "TokenValidator.__init__")
    assert (init.symbol_type, init.start_line, init.end_line, init.parent) == \
        ("method", 20, 21, "TokenValidator")

    val = chunk(pf, "TokenValidator.validate")
    assert (val.symbol_type, val.start_line, val.end_line, val.parent) == \
        ("method", 23, 25, "TokenValidator")

    mod = chunk(pf, "<module>")
    assert (mod.symbol_type, mod.start_line, mod.end_line) == ("module", 1, 6), mod
    print("  ok  python AST chunk line numbers")


# --------------------------------------------------------------------------- #
# 2. Symbol graph nodes carry accurate line numbers
# --------------------------------------------------------------------------- #
def test_graph_nodes_line_numbers():
    root = make_repo()
    build_graph(root, _db(root))
    conn = sqlite3.connect(_db(root))
    conn.row_factory = sqlite3.Row

    rows = {r["qualified_name"]: r for r in conn.execute(
        "SELECT * FROM nodes WHERE file_path = 'sample_pkg/auth.py' AND symbol_type != 'file'"
    )}
    assert (rows["hash_password"]["symbol_type"],
            rows["hash_password"]["start_line"],
            rows["hash_password"]["end_line"]) == ("function", 9, 12)
    assert (rows["TokenValidator"]["start_line"], rows["TokenValidator"]["end_line"]) == (15, 25)
    assert (rows["TokenValidator.__init__"]["symbol_type"],
            rows["TokenValidator.__init__"]["start_line"]) == ("method", 20)
    assert (rows["TokenValidator.validate"]["start_line"],
            rows["TokenValidator.validate"]["end_line"]) == (23, 25)

    helper = conn.execute(
        "SELECT * FROM nodes WHERE file_path = 'sample_pkg/utils.py' AND symbol_type != 'file'"
    ).fetchone()
    assert (helper["symbol_name"], helper["start_line"], helper["end_line"]) == ("helper", 1, 2)
    conn.close()
    print("  ok  graph nodes line numbers")


# --------------------------------------------------------------------------- #
# 3. CALLS edges: correct caller, callee and call-site line
# --------------------------------------------------------------------------- #
def test_calls_edges():
    root = make_repo()
    build_graph(root, _db(root))
    gdb = GraphDB(_db(root))

    val_calls = gdb.edges_from("sample_pkg/auth.py", "TokenValidator.validate", "CALLS")
    seen = {(e["dst_name"], e["line"]) for e in val_calls}
    assert ("hash_password", 24) in seen, seen
    assert ("helper", 25) in seen, seen

    hp_calls = {(e["dst_name"], e["line"]) for e in
                gdb.edges_from("sample_pkg/auth.py", "hash_password", "CALLS")}
    assert ("sha256", 12) in hp_calls, hp_calls
    assert ("encode", 12) in hp_calls, hp_calls

    # same-file + unique-global resolution
    resolved = {e["dst_name"]: e["dst_id"] for e in val_calls}
    assert resolved["hash_password"] == node_id("sample_pkg/auth.py", "hash_password")
    assert resolved["helper"] == node_id("sample_pkg/utils.py", "helper")
    # external calls stay unresolved
    assert all(e["dst_id"] is None for e in
               gdb.edges_from("sample_pkg/auth.py", "hash_password", "CALLS"))
    gdb.close()
    print("  ok  CALLS edges + resolution")


# --------------------------------------------------------------------------- #
# 4. IMPORTS edges with correct lines
# --------------------------------------------------------------------------- #
def test_imports_edges():
    root = make_repo()
    build_graph(root, _db(root))
    gdb = GraphDB(_db(root))

    imports = {(e["dst_name"], e["line"]) for e in gdb.import_edges("sample_pkg/auth.py")}
    assert ("os", 1) in imports, imports
    assert ("hashlib", 2) in imports, imports
    assert ("sample_pkg.utils.helper", 3) in imports, imports

    js_imports = {(e["dst_name"], e["line"]) for e in gdb.import_edges("app.js")}
    assert ("fs", 1) in js_imports, js_imports
    gdb.close()
    print("  ok  IMPORTS edges")


# --------------------------------------------------------------------------- #
# 5. JavaScript extraction (functions, methods, calls, arrow spans)
# --------------------------------------------------------------------------- #
def test_javascript_extraction():
    pf = parse_source("app.js", APP_JS.encode())
    names = {(c.symbol_name, c.symbol_type, c.start_line, c.end_line) for c in pf.chunks}
    assert ("loadConfig", "function", 3, 3) in names, names
    assert ("boot", "function", 5, 7) in names, names
    assert ("Server", "class", 9, 13) in names, names
    assert ("Server.start", "method", 10, 12) in names, names

    calls = {(c.caller, c.callee_name, c.line) for c in pf.calls}
    assert ("loadConfig", "readFile", 3) in calls, calls
    assert ("boot", "loadConfig", 6) in calls, calls
    assert ("Server.start", "boot", 11) in calls, calls
    print("  ok  javascript extraction")


# --------------------------------------------------------------------------- #
# 6. Non-code fallback keeps true line numbers
# --------------------------------------------------------------------------- #
def test_non_code_fallback():
    pf = parse_source("README.md", README_MD.encode())
    assert pf.language == "text"
    assert pf.symbols == [] and pf.calls == [] and pf.imports == []
    assert pf.chunks[0].symbol_type == "text"
    assert pf.chunks[0].start_line == 1
    assert pf.chunks[-1].end_line == 40
    for c in pf.chunks:
        assert 1 <= c.start_line <= c.end_line <= 40
    print("  ok  non-code fallback line numbers")


# --------------------------------------------------------------------------- #
# 7. Git-diff incremental indexing via SHA-256
# --------------------------------------------------------------------------- #
def test_incremental_indexing():
    root = make_repo()
    db = _db(root)

    first = build_graph(root, db)
    assert set(first["changed"]) == {
        "README.md", "app.js", "sample_pkg/auth.py", "sample_pkg/utils.py"}
    assert first["skipped"] == []

    # (a) nothing changed -> everything skipped, no re-parse
    second = build_graph(root, db)
    assert second["changed"] == []
    assert set(second["skipped"]) == set(first["changed"])

    # (b) touch one file -> only that file is re-parsed
    _write(root, "sample_pkg/utils.py", UTILS_PY + "\n\ndef helper2():\n    return 2\n")
    third = build_graph(root, db)
    assert third["changed"] == ["sample_pkg/utils.py"], third["changed"]
    assert set(third["skipped"]) == {"README.md", "app.js", "sample_pkg/auth.py"}

    gdb = GraphDB(db)
    assert gdb.get_node("sample_pkg/utils.py", "helper2") is not None
    # untouched file's nodes are intact
    assert gdb.get_node("sample_pkg/auth.py", "hash_password")["start_line"] == 9
    gdb.close()

    # (c) delete a file -> removed, its nodes drop, stale edges self-heal to NULL
    os.remove(os.path.join(root, "sample_pkg/utils.py"))
    fourth = build_graph(root, db)
    assert fourth["removed"] == ["sample_pkg/utils.py"], fourth["removed"]

    gdb = GraphDB(db)
    assert gdb.nodes_in("sample_pkg/utils.py") == []
    val_calls = gdb.edges_from("sample_pkg/auth.py", "TokenValidator.validate", "CALLS")
    helper_edge = next(e for e in val_calls if e["dst_name"] == "helper")
    assert helper_edge["dst_id"] is None  # target gone -> unresolved again
    gdb.close()
    print("  ok  incremental indexing (skip / re-parse / delete)")


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
TESTS = [
    test_python_ast_chunks_have_true_line_numbers,
    test_graph_nodes_line_numbers,
    test_calls_edges,
    test_imports_edges,
    test_javascript_extraction,
    test_non_code_fallback,
    test_incremental_indexing,
]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {exc!r}")
    total = len(TESTS)
    print(f"\n{total - failed}/{total} passed")
    raise SystemExit(1 if failed else 0)
