# ◆ Codebase QA

Ask a natural-language question about a GitHub repository and get back an answer
with **real `file:line` citations** and a **call-graph** of the code the answer
came from — running **100% locally**, no API keys, no cloud calls.

> *"How does login check a username and password?"*
> → "`authenticate()` in `login.py:31-58` hashes the input with `verify_password`
> `[hashing.py:22-40]` and compares it to the stored record loaded by
> `get_user` `[users.py:12-19]`…"

---

## Who it's for

A developer dropped into an unfamiliar codebase — a new hire, an open-source
contributor, someone reviewing a dependency — who needs to know *how something
works and exactly where it lives*, without reading the whole repo or trusting an
LLM that has never seen it.

The hard requirement that shapes every design decision: **every claim must be
traceable to a specific line range**, and the tool must run on a laptop with no
external service and no secrets.

---

## What was broken

The first version of this project was a textbook RAG pipeline:

`RecursiveCharacterTextSplitter(800/100)` → `all-MiniLM-L6-v2` embeddings →
top-k cosine → Groq API for generation.

It worked as a demo and failed as a tool:

| Problem | Why it happened |
|---|---|
| **Citations were fabricated.** The UI showed `file:line`, but the line number was `chunk_index * 20 + 1` — almost never where the code actually was. | Character chunks have no concept of a line span. Measured citation accuracy: **32.7%**. |
| **Chunks straddled function boundaries.** An 800-char window would start mid-function and end mid-comment, so the model saw half of two things instead of one whole thing. | The splitter cuts on character count, not syntax. |
| **No relationships.** "What calls this?" / "what does this call?" could only be answered if the caller happened to be embedded near the callee. | There was no structure, just a bag of text windows. |
| **Needed a cloud key.** Generation went to Groq; nothing ran offline. | External API dependency. |

---

## What I built

A four-phase engine that replaces text windows with the repo's actual structure.

### Phase 1 — AST parsing + symbol graph
`code_parser.py`, `graph.py`, `indexer.py`

- Shallow-clone the repo (GitPython), walk source files, hash each file
  (SHA-256) so re-indexing only re-parses what changed.
- **tree-sitter** parses Python / JavaScript / TypeScript / TSX into chunks
  aligned to AST boundaries — whole functions, methods, classes — each with its
  **true 1-indexed line span**. Non-code files (`.md`, `.json`, `.yaml`, …) fall
  back to a line-window splitter so they stay searchable (no graph edges).
- Extract **symbols** (function / method / class), **CALLS** edges, and
  **IMPORTS** edges.
- Persist everything to a per-repo SQLite graph (`graphs/<repo_id>.sqlite`):
  `files`, `nodes`, `edges`, `chunks` (chunk bodies + cached embedding blobs).
- `resolve_edges()` links each call to a concrete definition — same file first,
  then a unique repo-wide match, else it's marked external (stdlib / third-party).

### Phase 2 — Hybrid retrieval + re-ranking
`retriever.py`

```
query ─┬─ BM25 over code-aware tokens (camelCase / snake_case split,
       │      symbol name ×3, signature line ×2)
       └─ dense: jina-embeddings-v2-base-code  (cosine, ChromaDB)
             │
             ├─ Reciprocal Rank Fusion (k = 60)
             ├─ cross-encoder re-rank (bge-reranker-base)
             └─ depth-1 call-graph expansion (callers + callees from the graph)
             ▼
      ordered context chunks → LLM
```

Model loading is lazy and **degrades gracefully**: if the transformer models
can't be downloaded, a hashed-n-gram embedder and a Jaccard re-ranker keep the
whole pipeline runnable. The result records which backend actually ran.

### Phase 3 — Local generation
`querier.py`

- A local **Ollama** model writes the answer (`qwen2.5-coder` preferred, falls
  back to `llama3.2` / `llama3.1`). No external API client anywhere in the path.
- The system prompt forces the model to answer **only** from the provided
  `[path:start-end]` sections, cite them inline, use the call graph to explain
  how functions connect, and say *"I couldn't find that in the indexed code"*
  when the context doesn't cover the question.
- Returns the answer, citations (with re-rank scores and code bodies), and a
  nodes/edges graph view for the visualiser.

### Phase 4 — Benchmark harness
`eval.py`, `eval_data/`

A reproducible comparison of the new engine against a faithful re-creation of
the old character-RAG (same splitter, same MiniLM embeddings, same fake line
number) — see [How I evaluated](#how-i-evaluated).

### UI
`app.py` — a Streamlit app: clone/index a repo from the sidebar, pick an Ollama
model, ask questions in a chat box. Answers render with citation chips,
expandable source cards (code + re-rank score), and an interactive PyVis 2D
call-graph. Custom dark "glass" styling; physics freezes after the graph settles.

---

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Parsing | `tree-sitter` 0.25 + Python / JavaScript / TypeScript grammars |
| Graph store | SQLite (stdlib), one DB per repo |
| Sparse retrieval | `rank-bm25` |
| Dense retrieval | `sentence-transformers` 3.0.1 · `jinaai/jina-embeddings-v2-base-code` |
| Vector index | `chromadb` (ephemeral, in-process; numpy flat fallback) |
| Re-ranker | `BAAI/bge-reranker-base` cross-encoder |
| Generation | **Ollama** local daemon over REST (`requests`) — `qwen2.5-coder` / `llama3.x` |
| Repo cloning | `GitPython` (+ system `git`) |
| UI | `streamlit`, `pyvis` |
| Eval baseline | `langchain-text-splitters` (only to reproduce the old pipeline) |
| Tests | `pytest` |
| Legacy (optional) | `fastapi` + `uvicorn` for `main.py` — the old HTTP backend, no longer used by `app.py` |

Pinned on purpose: `transformers==4.44.2` (the jina model's custom code breaks on
`transformers>=4.49`), `huggingface_hub<0.26`.

---

## Running it

**Prerequisites**

- Python 3.11+
- `git` on your `PATH` (used to clone target repos)
- [Ollama](https://ollama.com) running locally with a model pulled — optional;
  retrieval and the call-graph work without it, you just won't get a written
  answer:
  ```bash
  ollama pull qwen2.5-coder      # or: ollama pull llama3.1
  ```

**Install & launch**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # first run pulls torch + the jina/bge models
streamlit run app.py                      # opens http://localhost:8501
```

Then, in the sidebar: **Add a new index** → paste a GitHub URL → *Clone & index*,
or **Open index** to reuse one already built under `graphs/`.

**Command line**

```bash
python querier.py <repo_id> "How does routing work?"
# repo_id is derived from the URL: github.com/pallets/flask → pallets_flask
```

---

## Project layout

```
.
├── app.py            # Streamlit UI (index a repo, ask questions, view call-graph)
├── indexer.py        # clone → walk → incremental parse → build graph → cache embeddings
├── code_parser.py    # tree-sitter → AST chunks + symbols + CALLS/IMPORTS edges
├── graph.py          # SQLite schema + symbol-graph read/write + edge resolution
├── retriever.py      # BM25 + dense + RRF + cross-encoder + depth-1 graph expansion
├── querier.py        # prompt construction + local Ollama generation + graph view
├── eval.py           # Phase 4 benchmark: new engine vs. old character-RAG
├── eval_data/
│   ├── repos/        # 3 hand-built fixture repos (authsvc, miniorm, taskcli)
│   └── benchmark_report.md
├── test_phase1.py    # AST chunks + SQLite graph + true line numbers
├── test_phase3.py    # chunk persistence, embedding cache, end-to-end answer
├── main.py           # legacy FastAPI backend (/index, /ask) — not used by app.py
├── graphs/           # per-repo SQLite graph DBs (git-ignored, regenerated on index)
└── requirements.txt
```

---

## How I evaluated

`eval.py` runs **30 ground-truth `question → (file, symbol)` pairs** across 3
small hand-built repos (`authsvc`, `miniorm`, `taskcli`) through both engines and
scores them. A "hit" means a retrieved unit is in the right file and covers ≥50%
of the true symbol's lines. The character baseline is scored with its *real* line
spans (recovered from character offsets), so retrieval quality is judged fairly —
the fake citations are scored separately.

| Metric | Old Character RAG | AST + Graph RAG | Δ |
|---|---|---|---|
| Retrieval Hit-Rate @5 | 100.0% | 100.0% | +0.0% |
| Retrieval Hit-Rate @3 | 90.0% | 83.3% | −6.7% |
| Retrieval Hit-Rate @1 | 70.0% | 63.3% | −6.7% |
| **Citation Accuracy** | 32.7% | **100.0%** | **+67.3%** |
| Context Precision @5 | 20.0% | 24.7% | +4.7% |
| **Call-Context Recall** | 60.6% | **100.0%** | **+39.4%** |

**Reading of the results.** On repos this small, dense retrieval alone already
finds the right file, so raw hit-rate barely moves — and @1/@3 actually dip,
because whole-function chunks are coarser than 800-char windows for pinpoint
"where is X" questions. The decisive wins are the two things the old design
*could not do at all*: **citations that actually point at the code** (32.7% →
100%) and **deterministic caller/callee context** in the prompt (60.6% → 100%).

Regenerate with `python eval.py` (writes `eval_data/benchmark_report.md`).

---

## What I'd change before production

- **Answer quality isn't measured yet.** The benchmark scores retrieval and
  citations, not whether the generated answer is correct. Add answer-faithfulness
  / groundedness scoring, and test on a larger, real-world question set — 3
  fixture repos is enough to catch regressions, not enough to trust.
- **Retrieval @1/@3 regressed.** Whole-function chunks are too coarse for some
  questions. Sub-chunk long function bodies, and tune `k` / RRF `k` / re-rank
  depth against that bigger benchmark instead of the current fixed values.
- **Language coverage.** Only Python/JS/TS get an AST and a call graph;
  everything else is line-window text with no edges. Add Go/Java/Ruby/Rust
  grammars. Call resolution is name-based (same-file or globally-unique only), so
  overloaded or common method names collapse to "external" — needs proper
  scope/type resolution.
- **Scale & concurrency.** Single process, ephemeral in-memory Chroma rebuilt per
  session, 50 MB repo cap, no job queue, no per-user isolation. Move to a
  persistent vector store and a background indexing worker.
- **No conversation memory.** Each question retrieves independently; prior turns
  are displayed but never fed back. Add follow-up-question rewriting.
- **Private repos.** Only public, default-branch, shallow clones; no auth, no
  submodules.
- **Security / hygiene.**
  - `.env` still contains a legacy `GROQ_API_KEY` the local pipeline no longer
    uses — remove it.
  - The `origin` remote URL has a GitHub token embedded in `.git/config` — rotate
    it and switch to a credential helper.
  - Delete or clearly quarantine the dead paths: `main.py` (old FastAPI backend)
    and the stale `chroma_db/` directory from the v1 pipeline.
  - Pin transitive dependencies with a lockfile.
- **Test coverage.** `test_phase1.py` / `test_phase3.py` and the in-file tests in
  `retriever.py` cover parsing, the graph, and retrieval. Nothing covers
  `app.py` or prompt construction in `querier.py`, and there's no CI.

---

## Contributors

- Varun Kumar Andigari
