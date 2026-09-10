"""Phase 3 - 100% local answer generation via Ollama.

No external API clients. Retrieval is the Phase 2 hybrid engine
(BM25 + jina dense + RRF + cross-encoder + depth-1 graph expansion);
generation is a local Ollama model (`qwen2.5-coder`, falls back to `llama3.1`).

`answer_question()` returns everything the Streamlit UI needs:
    {
      "answer":    str,                # LLM text, with [path:start-end] citations
      "citations": [ {file,start_line,end_line,symbol_name,symbol_type,code,...} ],
      "graph":     {"nodes": [...], "edges": [...]},   # for the 2D visualiser
      "model":     str,
      "backends":  {...},
    }
"""

from __future__ import annotations

import os

import requests

from indexer import graph_path_for
from retriever import HybridRetriever

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
# Fast local model first; `qwen2.5-coder` still wins if the user pulls it.
PREFERRED_MODELS = ["qwen2.5-coder", "qwen2.5-coder:7b", "llama3.2", "llama3.1"]
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
MAX_CONTEXT_CHUNKS = 5           # cap prompt size for local generation
MAX_CHUNK_LINES = 70

_RETRIEVERS: dict[str, HybridRetriever] = {}   # repo_id -> retriever (process cache)
_WARMED: set[str] = set()


# --------------------------------------------------------------------------- #
# Ollama plumbing
# --------------------------------------------------------------------------- #
def ollama_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        return []


def pick_model(requested: str | None = None) -> str:
    available = ollama_models()
    if requested and (requested in available or f"{requested}:latest" in available):
        return requested
    for name in PREFERRED_MODELS:
        for a in available:
            if a == name or a.split(":")[0] == name:
                return a
    if available:
        return available[0]
    return DEFAULT_MODEL


def warm_model(model: str) -> None:
    """Force Ollama to load the model into memory (first-token latency can
    otherwise blow the request timeout on a cold model)."""
    if model in _WARMED:
        return
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": "ok", "stream": False,
                  "keep_alive": "10m", "options": {"num_predict": 1}},
            timeout=OLLAMA_TIMEOUT,
        )
        _WARMED.add(model)
    except requests.RequestException:
        pass


def ollama_chat(model: str, system: str, user: str, temperature: float = 0.1) -> str:
    warm_model(model)
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "keep_alive": "10m",
                "options": {"temperature": temperature, "num_predict": 400},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Start it with `ollama serve` "
            f"and pull a model, e.g. `ollama pull llama3.1`."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    return r.json()["message"]["content"].strip()


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a precise code assistant. You answer questions about a \
codebase using ONLY the code sections provided below. Each section has a header \
of the form [path:start-end].

Rules:
- Base every claim on the provided sections. If the answer is not in them, say
  "I couldn't find that in the indexed code."
- Cite the exact section you used inline, in square brackets, e.g. [src/auth/jwt.py:42-78].
- When a call graph is provided, use it to explain how functions call each other.
- Be concise. Prefer a short explanation followed by a bullet list of the
  relevant functions with their citations.
"""

_LANG_BY_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".go": "go", ".rb": "ruby", ".java": "java", ".rs": "rust",
    ".c": "c", ".cpp": "cpp", ".md": "markdown", ".json": "json", ".yaml": "yaml",
    ".yml": "yaml",
}


def lang_of(path: str) -> str:
    return _LANG_BY_EXT.get(os.path.splitext(path)[1].lower(), "")


def _cite(c) -> str:
    return f"{c.file_path}:{c.start_line}-{c.end_line}"


def _clip(code: str) -> str:
    lines = code.splitlines()
    if len(lines) <= MAX_CHUNK_LINES:
        return code
    return "\n".join(lines[:MAX_CHUNK_LINES]) + "\n# … (truncated)"


def build_prompt(question: str, result) -> str:
    parts = [f"Question: {question}", "", "=== CODE SECTIONS ==="]
    for c in result.context_chunks()[:MAX_CONTEXT_CHUNKS]:
        parts.append(f"\n[{_cite(c)}]  {c.symbol_name}  ({c.symbol_type})")
        parts.append("```" + lang_of(c.file_path))
        parts.append(_clip(c.code))
        parts.append("```")

    graph_lines = []
    for exp in result.expansions:
        seed = exp.seed
        calls = ", ".join(f"{x.symbol_name} [{_cite(x)}]" for x in exp.callees) or "-"
        callers = ", ".join(f"{x.symbol_name} [{_cite(x)}]" for x in exp.callers) or "-"
        ext = ", ".join(exp.unresolved_callees)
        line = f"- {seed.symbol_name} [{_cite(seed)}]  calls: {calls}   called by: {callers}"
        if ext:
            line += f"   (external: {ext})"
        graph_lines.append(line)
    if graph_lines:
        parts += ["", "=== CALL GRAPH (depth 1) ===", *graph_lines]

    parts += ["", "Answer the question now, citing [path:start-end] sections."]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Graph view for the visualiser
# --------------------------------------------------------------------------- #
def _node(nid, chunk, role):
    return {
        "id": nid,
        "label": chunk.symbol_name,
        "role": role,                       # seed | callee | caller | module
        "file": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "symbol_type": chunk.symbol_type,
    }


def build_graph_view(result, retriever) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def key(c):
        return f"{c.file_path}::{c.symbol_name}"

    for exp in result.expansions:
        sid = key(exp.seed)
        nodes[sid] = _node(sid, exp.seed, "seed")
        for callee in exp.callees:
            cid = key(callee)
            nodes.setdefault(cid, _node(cid, callee, "callee"))
            edges.append({"source": sid, "target": cid, "label": "CALLS"})
        for caller in exp.callers:
            rid = key(caller)
            nodes.setdefault(rid, _node(rid, caller, "caller"))
            edges.append({"source": rid, "target": sid, "label": "CALLS"})
        for ext in exp.unresolved_callees:
            eid = f"ext::{ext}"
            nodes.setdefault(eid, {"id": eid, "label": ext, "role": "external",
                                   "file": "", "start_line": 0, "end_line": 0,
                                   "symbol_type": "external"})
            edges.append({"source": sid, "target": eid, "label": "CALLS"})

    # import edges for the files that own the seed symbols
    seed_files = {exp.seed.file_path for exp in result.expansions}
    for fpath in seed_files:
        fid = f"{fpath}::<file>"
        for e in retriever.graph.import_edges(fpath):
            mid = f"mod::{e['dst_name']}"
            nodes.setdefault(fid, {"id": fid, "label": os.path.basename(fpath),
                                   "role": "file", "file": fpath, "start_line": 0,
                                   "end_line": 0, "symbol_type": "file"})
            nodes.setdefault(mid, {"id": mid, "label": e["dst_name"], "role": "module",
                                   "file": "", "start_line": 0, "end_line": 0,
                                   "symbol_type": "module"})
            edges.append({"source": fid, "target": mid, "label": "IMPORTS"})

    return {"nodes": list(nodes.values()), "edges": edges}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def get_retriever(repo_id: str) -> HybridRetriever:
    if repo_id not in _RETRIEVERS:
        db_path = graph_path_for(repo_id)
        if not os.path.exists(db_path):
            raise ValueError(f"Repo '{repo_id}' is not indexed yet.")
        _RETRIEVERS[repo_id] = HybridRetriever.from_graph(db_path)
    return _RETRIEVERS[repo_id]


def answer_question(repo_id: str, question: str, model: str | None = None,
                    generate: bool = True) -> dict:
    retr = get_retriever(repo_id)
    result = retr.retrieve(question)

    chosen = pick_model(model)
    answer = None
    if generate:
        answer = ollama_chat(chosen, SYSTEM_PROMPT, build_prompt(question, result))

    rerank_score = {i: s for i, s in result.reranked}
    citations = []
    for i in result.seeds:
        c = retr.chunks[i]
        citations.append({
            "file": c.file_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "symbol_name": c.symbol_name,
            "symbol_type": c.symbol_type,
            "language": lang_of(c.file_path),
            "code": c.code,
            "citation": _cite(c),
            "rerank_score": round(float(rerank_score.get(i, 0.0)), 4),
        })

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "graph": build_graph_view(result, retr),
        "model": chosen,
        "backends": result.backends,
    }


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    rid = sys.argv[1] if len(sys.argv) > 1 else "pallets_flask"
    q = sys.argv[2] if len(sys.argv) > 2 else "How does routing work?"
    out = answer_question(rid, q)
    print("model:", out["model"], "| backends:", out["backends"])
    print("\n" + out["answer"] + "\n")
    print("citations:")
    for c in out["citations"]:
        print(f"  [{c['citation']}] {c['symbol_name']} ({c['symbol_type']})  "
              f"score={c['rerank_score']}")
    print(f"\ngraph: {len(out['graph']['nodes'])} nodes, {len(out['graph']['edges'])} edges")
