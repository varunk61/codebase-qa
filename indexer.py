"""Repo indexing orchestrator (Phase 1: AST parsing + symbol graph).

Pipeline
--------
    GitHub URL -> shallow clone -> iter source files
      -> for each CHANGED file (SHA-256 differs): AST parse -> write graph
      -> drop files that disappeared -> resolve CALLS edges -> commit

Embedding chunks into ChromaDB is done by `embed_chunks()`, which imports
sentence-transformers / chromadb lazily so this module (and the Phase 1
tests) can run without those heavy dependencies installed.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile

from code_parser import ParsedFile, language_for, parse_source, sha256_bytes, TEXT_EXTENSIONS
from graph import GraphDB

GRAPH_DIR = os.path.join(os.path.dirname(__file__), "graphs")

CODE_EXTENSIONS = {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}
INDEXABLE_EXTENSIONS = CODE_EXTENSIONS | TEXT_EXTENSIONS

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "dist", "build", "venv", ".venv",
    ".mypy_cache", ".pytest_cache", ".tox", "site-packages", "vendor", ".next",
}

MAX_REPO_SIZE_MB = 50
MAX_FILE_BYTES = 1_000_000


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def iter_source_files(repo_path: str):
    """Yield absolute paths of indexable files under `repo_path`."""
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in INDEXABLE_EXTENSIONS:
                yield os.path.join(root, fname)


# --------------------------------------------------------------------------- #
# Incremental graph build
# --------------------------------------------------------------------------- #
def build_graph(repo_path: str, db_path: str, gdb: GraphDB | None = None) -> dict:
    """Parse changed files and (re)write the symbol graph.

    Returns {"changed": [...], "skipped": [...], "removed": [...],
             "parsed": {path: ParsedFile for changed files}, "stats": {...}}.
    """
    owns_gdb = gdb is None
    gdb = gdb or GraphDB(db_path)

    changed, skipped, parsed = [], [], {}
    seen: set[str] = set()

    for abspath in iter_source_files(repo_path):
        rel = os.path.relpath(abspath, repo_path)
        try:
            with open(abspath, "rb") as fh:
                data = fh.read(MAX_FILE_BYTES + 1)
        except OSError:
            continue
        if len(data) > MAX_FILE_BYTES:
            continue

        seen.add(rel)
        sha = sha256_bytes(data)
        if not gdb.file_changed(rel, sha):
            skipped.append(rel)
            continue

        pf = parse_source(rel, data, sha)
        gdb.index_parsed_file(pf)
        changed.append(rel)
        parsed[rel] = pf

    removed = sorted(gdb.known_files() - seen)
    for rel in removed:
        gdb.remove_file(rel)

    gdb.resolve_edges()
    gdb.commit()
    stats = gdb.stats()
    if owns_gdb:
        gdb.close()

    return {
        "changed": sorted(changed),
        "skipped": sorted(skipped),
        "removed": removed,
        "parsed": parsed,
        "stats": stats,
    }


# --------------------------------------------------------------------------- #
# Repo id helpers
# --------------------------------------------------------------------------- #
def repo_id_from_url(github_url: str) -> str:
    parts = github_url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL")
    return f"{parts[-2]}_{parts[-1]}".lower().replace("-", "_").replace(".git", "")


def graph_path_for(repo_id: str) -> str:
    os.makedirs(GRAPH_DIR, exist_ok=True)
    return os.path.join(GRAPH_DIR, f"{repo_id}.sqlite")


# --------------------------------------------------------------------------- #
# Cloning support check
# --------------------------------------------------------------------------- #
def _load_git():
    """Import GitPython, raising an actionable error if it or system git is missing.

    Kept as a local import so the AST / retrieval / test paths never need it.
    """
    try:
        import git  # GitPython
    except ImportError as exc:
        raise RuntimeError(
            "Cloning a GitHub repo needs the 'GitPython' package, which is not "
            "installed.\n"
            "  Install it:   pip install GitPython\n"
            "  (or:          pip install -r requirements.txt)\n"
            "Already-indexed repos under graphs/ still work without it."
        ) from exc

    try:
        git.Git().version()  # shells out to the real `git` binary
    except Exception as exc:  # git.exc.GitCommandNotFound and friends
        raise RuntimeError(
            "GitPython is installed but the system 'git' executable was not found "
            "on PATH.\n"
            "  macOS:         xcode-select --install   (or: brew install git)\n"
            "  Debian/Ubuntu: sudo apt-get install git\n"
            "  Windows:       https://git-scm.com/download/win"
        ) from exc
    return git


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def index_repo(github_url: str, warm: bool = True) -> str:
    if not re.match(r"https://github\.com/[\w\-.]+/[\w\-.]+", github_url):
        raise ValueError("Please provide a valid GitHub URL (https://github.com/owner/repo)")

    git = _load_git()

    repo_id = repo_id_from_url(github_url)
    db_path = graph_path_for(repo_id)
    tmp_dir = tempfile.mkdtemp()

    try:
        print(f"[indexer] Cloning {github_url} ...")
        git.Repo.clone_from(github_url, tmp_dir, depth=1)

        total_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fnames in os.walk(tmp_dir) for f in fnames
        ) / (1024 * 1024)
        if total_mb > MAX_REPO_SIZE_MB:
            raise ValueError(f"Repo is {total_mb:.1f} MB - limit is {MAX_REPO_SIZE_MB} MB")

        result = build_graph(tmp_dir, db_path)
        print(f"[indexer] Graph: {result['stats']}  "
              f"(changed={len(result['changed'])}, skipped={len(result['skipped'])})")

        if warm:
            warm_embeddings(repo_id)
        return repo_id
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def warm_embeddings(repo_id: str) -> None:
    """Pre-compute + cache chunk embeddings into the graph DB so the first
    query is fast. Heavy deps imported lazily."""
    from retriever import HybridRetriever

    db_path = graph_path_for(repo_id)
    HybridRetriever.from_graph(db_path)   # side effect: fills chunks.embedding
    print(f"[indexer] Embeddings cached for '{repo_id}'")
