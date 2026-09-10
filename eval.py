"""Phase 4 - empirical benchmark harness.

Compares the upgraded **AST + graph** retrieval engine against a faithful
re-creation of the **original character-chunking RAG** on a fixed set of
ground-truth (question -> file + symbol) pairs across three sample repos.

Metrics
-------
Retrieval Hit-Rate @k : fraction of questions where >=1 retrieved chunk
                        lands in the correct file AND overlaps the true
                        symbol's line span.
Citation Accuracy     : fraction of emitted citations whose (file, start, end)
                        exactly matches a real AST node boundary in that file.
Context Precision @k  : mean over questions of (relevant chunks / k).

Run:  .venv/bin/python eval.py
"""

from __future__ import annotations

import os
import statistics
import tempfile

from code_parser import parse_source
from indexer import build_graph, iter_source_files

REPOS_DIR = os.path.join(os.path.dirname(__file__), "eval_data", "repos")
K = 5

# --------------------------------------------------------------------------- #
# Ground-truth benchmark: 21 pairs across 3 repos.
# The line range is resolved at runtime from the named symbol, so it stays
# correct even if the fixture files are edited.
# --------------------------------------------------------------------------- #
BENCHMARK = [
    # ---- authsvc ----
    ("authsvc", "How are user passwords hashed before they are stored?", "hashing.py", "hash_password"),
    ("authsvc", "How does the service compare a submitted password without a timing side channel?", "hashing.py", "verify_password"),
    ("authsvc", "How is a packed salt-and-digest field encoded for storage?", "hashing.py", "encode_hash"),
    ("authsvc", "Where are new user accounts created and stored?", "users.py", "register_user"),
    ("authsvc", "How does a user rotate their password to a new one?", "users.py", "change_password"),
    ("authsvc", "How does login check that a username and password are valid?", "login.py", "authenticate"),
    ("authsvc", "How does the service lock an account after too many failed logins?", "login.py", "is_locked_out"),
    ("authsvc", "What happens on a successful login?", "login.py", "login"),
    ("authsvc", "How do you require that a session holds a particular role?", "login.py", "require_role"),
    ("authsvc", "How is a session token generated?", "sessions.py", "create_session"),
    ("authsvc", "How does the system decide that a session has expired?", "sessions.py", "resolve_session"),
    ("authsvc", "How are all of a user's sessions revoked at once?", "sessions.py", "revoke_all_for_user"),
    # ---- miniorm ----
    ("miniorm", "How is a WHERE condition added to a query?", "query.py", "Query.where"),
    ("miniorm", "Where is user-supplied input parameterised to prevent SQL injection?", "query.py", "Query.where"),
    ("miniorm", "How does the builder compile its clauses into a SQL string?", "query.py", "Query.build"),
    ("miniorm", "How do I limit the number of rows returned?", "query.py", "Query.limit"),
    ("miniorm", "How is a database connection opened?", "connection.py", "Connection.__init__"),
    ("miniorm", "How do I run a query and fetch all matching rows?", "connection.py", "BoundQuery.all"),
    ("miniorm", "How do you get only the first matching row?", "connection.py", "BoundQuery.first"),
    ("miniorm", "How do you check whether any matching rows exist?", "connection.py", "BoundQuery.exists"),
    ("miniorm", "How are database tables created?", "schema.py", "create_table"),
    ("miniorm", "How do you insert many rows at once?", "schema.py", "insert_many"),
    # ---- taskcli ----
    ("taskcli", "How are tasks loaded from disk?", "store.py", "load_tasks"),
    ("taskcli", "How is the task list written back to disk without risking a half-written file?", "store.py", "save_tasks"),
    ("taskcli", "How do I add a new task?", "commands.py", "add_task"),
    ("taskcli", "How does marking a task as complete work?", "commands.py", "complete_task"),
    ("taskcli", "How can completed tasks be filtered out when listing?", "commands.py", "list_tasks"),
    ("taskcli", "How do you search tasks by keyword?", "commands.py", "find_tasks"),
    ("taskcli", "How does the CLI parse its command-line arguments?", "cli.py", "parse_args"),
    ("taskcli", "How is a parsed subcommand routed to its handler?", "cli.py", "dispatch"),
]

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
_PARSE_CACHE: dict[tuple[str, str], list] = {}


def _chunks_of(repo: str, file: str) -> list:
    key = (repo, file)
    if key not in _PARSE_CACHE:
        path = os.path.join(REPOS_DIR, repo, file)
        with open(path, "rb") as fh:
            _PARSE_CACHE[key] = parse_source(file, fh.read()).chunks
    return _PARSE_CACHE[key]


def gt_span(repo: str, file: str, symbol: str) -> tuple[int, int]:
    for c in _chunks_of(repo, file):
        if c.symbol_name == symbol:
            return c.start_line, c.end_line
    raise KeyError(f"{repo}/{file}::{symbol} not found in fixture")


def real_boundaries(repo: str, file: str) -> set[tuple[int, int]]:
    return {(c.start_line, c.end_line) for c in _chunks_of(repo, file)}


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 <= b1 and b0 <= a1


def covers(chunk0: int, chunk1: int, gt0: int, gt1: int, frac: float = 0.5) -> bool:
    """True when a retrieved [chunk0, chunk1] span contains at least `frac` of
    the ground-truth symbol's lines. Barely clipping a function is not a hit."""
    inside = max(0, min(chunk1, gt1) - max(chunk0, gt0) + 1)
    return inside / max(1, gt1 - gt0 + 1) >= frac


def _graph_neighbors(retriever, repo: str, file: str, symbol: str) -> set[tuple[str, str]]:
    """(file, qualified_name) of the target's direct callers+callees, restricted
    to neighbours whose line span we can resolve (so both systems chase the
    identical target set)."""
    from graph import node_id

    nb = retriever.graph.neighbors(node_id(file, symbol))
    out = set()
    for row in (*nb["callees"], *nb["callers"]):
        nf, nsym = row["file_path"], row["qualified_name"]
        try:
            gt_span(repo, nf, nsym)
        except KeyError:
            continue
        out.add((nf, nsym))
    return out


# --------------------------------------------------------------------------- #
# System 1: upgraded AST + graph retriever
# --------------------------------------------------------------------------- #
def build_ast_retriever(repo: str):
    from retriever import HybridRetriever

    src = os.path.join(REPOS_DIR, repo)
    db = os.path.join(tempfile.mkdtemp(prefix=f"eval_{repo}_"), "graph.sqlite")
    build_graph(src, db)
    return HybridRetriever.from_graph(db)


def ast_retrieve(retriever, question: str):
    """Return (cited_chunks, context_chunks):
       cited   = reranked seeds, best first (what the UI shows as citations)
       context = seeds + depth-1 call-graph neighbours (what the LLM sees)."""
    res = retriever.retrieve(question, k_seeds=K, expand=True)
    cited = [retriever.chunks[i] for i in res.seeds]
    return cited, res.context_chunks()


# --------------------------------------------------------------------------- #
# System 2: faithful re-creation of the original character-chunking RAG
# --------------------------------------------------------------------------- #
class CharacterRAG:
    """The original pipeline, verbatim:
        RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        + all-MiniLM-L6-v2 embeddings
        + top-k cosine similarity
        + the original fabricated line number  (chunk_index * 20 + 1).

    `real_start`/`real_end` are recovered from the chunk's character offset
    so Hit-Rate can still judge *content* retrieval fairly; `fake_line` is
    what the system would actually have cited.
    """

    def __init__(self, repo: str):
        import numpy as np
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from sentence_transformers import SentenceTransformer

        self._np = np
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100, length_function=len)

        self.chunks: list[dict] = []
        src = os.path.join(REPOS_DIR, repo)
        for abspath in iter_source_files(src):
            rel = os.path.relpath(abspath, src)
            with open(abspath, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            cursor = 0
            for i, part in enumerate(splitter.split_text(text)):
                idx = text.find(part, cursor)
                if idx < 0:
                    idx = text.find(part)
                if idx >= 0:
                    cursor = idx + len(part)
                real_start = text.count("\n", 0, max(idx, 0)) + 1
                real_end = real_start + part.count("\n")
                self.chunks.append({
                    "file": rel,
                    "text": part,
                    "fake_line": i * 20 + 1,       # <- the original bug
                    "real_start": real_start,
                    "real_end": real_end,
                })

        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._emb = np.asarray(
            self._model.encode([c["text"] for c in self.chunks],
                               normalize_embeddings=True),
            dtype="float32")

    def retrieve(self, question: str) -> list[dict]:
        q = self._model.encode([question], normalize_embeddings=True)[0]
        sims = self._emb @ self._np.asarray(q, dtype="float32")
        order = self._np.argsort(-sims)[:K]
        return [self.chunks[i] for i in order]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_ast() -> dict:
    retrievers: dict[str, object] = {}
    hit1 = hit3 = hitk = 0
    cprec: list[float] = []
    ctx_recall: list[float] = []
    cit_ok = cit_total = 0
    per_repo: dict[str, list[int]] = {}

    for repo, q, file, symbol in BENCHMARK:
        r = retrievers.setdefault(repo, build_ast_retriever(repo))
        gs, ge = gt_span(repo, file, symbol)
        cited, context = ast_retrieve(r, q)

        def is_hit(cs):
            return any(c.file_path == file and covers(c.start_line, c.end_line, gs, ge)
                       for c in cs)

        h1, h3, hk = is_hit(cited[:1]), is_hit(cited[:3]), is_hit(cited[:K])
        hit1 += h1
        hit3 += h3
        hitk += hk
        per_repo.setdefault(repo, [0, 0]);  per_repo[repo][0] += hk;  per_repo[repo][1] += 1

        rel = sum(1 for c in cited[:K]
                  if c.file_path == file and covers(c.start_line, c.end_line, gs, ge))
        cprec.append(rel / max(1, len(cited[:K])))

        # citation accuracy: reported (start,end) must be a real AST boundary
        for c in cited[:K]:
            cit_total += 1
            if (c.start_line, c.end_line) in real_boundaries(repo, c.file_path):
                cit_ok += 1

        # call-context recall: how many of the target's direct callers/callees
        # made it into the context the LLM actually sees
        want = _graph_neighbors(r, repo, file, symbol)
        if want:
            got = {(c.file_path, c.symbol_name) for c in context}
            ctx_recall.append(len(want & got) / len(want))

    n = len(BENCHMARK)
    return {
        "hit@1": hit1 / n, "hit@3": hit3 / n, "hit@k": hitk / n,
        "citation_accuracy": cit_ok / max(1, cit_total),
        "context_precision": statistics.mean(cprec),
        "call_context_recall": statistics.mean(ctx_recall) if ctx_recall else 0.0,
        "per_repo": {k: v[0] / v[1] for k, v in per_repo.items()},
        "retrievers": retrievers,
    }


def score_character(ast_retrievers: dict) -> dict:
    systems: dict[str, CharacterRAG] = {}
    hit1 = hit3 = hitk = 0
    cprec: list[float] = []
    ctx_recall: list[float] = []
    cit_ok = cit_total = 0
    per_repo: dict[str, list[int]] = {}

    for repo, q, file, symbol in BENCHMARK:
        s = systems.setdefault(repo, CharacterRAG(repo))
        gs, ge = gt_span(repo, file, symbol)
        ranked = s.retrieve(q)

        def is_hit(cs):
            return any(c["file"] == file and covers(c["real_start"], c["real_end"], gs, ge)
                       for c in cs)

        h1, h3, hk = is_hit(ranked[:1]), is_hit(ranked[:3]), is_hit(ranked[:K])
        hit1 += h1
        hit3 += h3
        hitk += hk
        per_repo.setdefault(repo, [0, 0]);  per_repo[repo][0] += hk;  per_repo[repo][1] += 1

        rel = sum(1 for c in ranked[:K]
                  if c["file"] == file and covers(c["real_start"], c["real_end"], gs, ge))
        cprec.append(rel / max(1, len(ranked[:K])))

        # the old pipeline cites `file:fake_line`; accurate only if that line is
        # genuinely where the retrieved chunk begins.
        for c in ranked[:K]:
            cit_total += 1
            if c["fake_line"] == c["real_start"]:
                cit_ok += 1

        # call-context recall: a neighbour counts as present if some retrieved
        # window lands in its file and covers its span (no symbol concept here)
        want_syms = _graph_neighbors(ast_retrievers[repo], repo, file, symbol)
        if want_syms:
            got = 0
            for nf, nsym in want_syms:
                ns, ne = gt_span(repo, nf, nsym)
                if any(c["file"] == nf and covers(c["real_start"], c["real_end"], ns, ne)
                       for c in ranked):
                    got += 1
            ctx_recall.append(got / len(want_syms))

    n = len(BENCHMARK)
    return {
        "hit@1": hit1 / n, "hit@3": hit3 / n, "hit@k": hitk / n,
        "citation_accuracy": cit_ok / max(1, cit_total),
        "context_precision": statistics.mean(cprec),
        "call_context_recall": statistics.mean(ctx_recall) if ctx_recall else 0.0,
        "per_repo": {k: v[0] / v[1] for k, v in per_repo.items()},
        "_retrievers": systems,
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def report(old: dict, new: dict) -> str:
    rows = [
        (f"Retrieval Hit-Rate @{K}", old["hit@k"], new["hit@k"]),
        ("Retrieval Hit-Rate @3", old["hit@3"], new["hit@3"]),
        ("Retrieval Hit-Rate @1", old["hit@1"], new["hit@1"]),
        ("Citation Accuracy", old["citation_accuracy"], new["citation_accuracy"]),
        (f"Context Precision @{K}", old["context_precision"], new["context_precision"]),
        ("Call-Context Recall", old["call_context_recall"], new["call_context_recall"]),
    ]
    lines = [
        f"# Codebase QA — Retrieval Benchmark  ({len(BENCHMARK)} questions, 3 repos, k={K})",
        "",
        "| Metric | Old Character RAG | Upgraded AST + Graph RAG | Δ |",
        "|---|---|---|---|",
    ]
    for name, o, n in rows:
        delta = n - o
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {name} | {pct(o)} | {pct(n)} | {sign}{pct(delta)} |")

    lines += ["", "### Hit-Rate @%d by repository" % K, "",
              "| Repo | Old | Upgraded |", "|---|---|---|"]
    for repo in old["per_repo"]:
        lines.append(f"| {repo} | {pct(old['per_repo'][repo])} | {pct(new['per_repo'][repo])} |")

    lines += [
        "",
        "### How each metric is measured",
        "- **Hit-Rate @k** — content retrieval: a hit means a retrieved unit is in the",
        "  right file and covers >=50% of the true symbol's lines. The character",
        "  baseline is scored with *real* line spans (recovered from character offsets),",
        "  so this measures pure retrieval quality, not its (fabricated) citations.",
        "- **Citation Accuracy** — location: does the line range the system reports",
        "  actually match where the retrieved code is? The old pipeline reports",
        "  `chunk_index * 20 + 1`, which is almost never the real start line.",
        "- **Context Precision @k** — share of the top-k that covers the target. AST",
        "  chunks are whole functions (clean hit or clean miss); 800-char windows",
        "  straddle boundaries and dilute the set.",
        "- **Call-Context Recall** — of the target symbol's direct callers + callees,",
        "  the fraction that appear in the context handed to the LLM. The AST engine",
        "  pulls these deterministically from the symbol graph; the character RAG can",
        "  only get them by lucky embedding similarity.",
        "",
        "### Takeaway",
        "On codebases this small, dense retrieval alone already finds the right file, so",
        "raw Hit-Rate is close. The upgrade's decisive gains are **trustworthy citations**",
        "and **call-graph context** the old design cannot produce at all.",
    ]
    return "\n".join(lines)


def main():
    print("Scoring upgraded AST + graph retriever …")
    new = score_ast()
    print("Scoring original character-chunking RAG …")
    old = score_character(new["retrievers"])

    md = report(old, new)
    print("\n" + md + "\n")

    out_path = os.path.join(os.path.dirname(__file__), "eval_data", "benchmark_report.md")
    with open(out_path, "w") as fh:
        fh.write(md + "\n")
    print(f"(written to {os.path.relpath(out_path)})")


if __name__ == "__main__":
    main()
