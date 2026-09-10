# Codebase QA — Retrieval Benchmark  (30 questions, 3 repos, k=5)

| Metric | Old Character RAG | Upgraded AST + Graph RAG | Δ |
|---|---|---|---|
| Retrieval Hit-Rate @5 | 100.0% | 100.0% | +0.0% |
| Retrieval Hit-Rate @3 | 90.0% | 83.3% | -6.7% |
| Retrieval Hit-Rate @1 | 70.0% | 63.3% | -6.7% |
| Citation Accuracy | 32.7% | 100.0% | +67.3% |
| Context Precision @5 | 20.0% | 24.7% | +4.7% |
| Call-Context Recall | 60.6% | 100.0% | +39.4% |

### Hit-Rate @5 by repository

| Repo | Old | Upgraded |
|---|---|---|
| authsvc | 100.0% | 100.0% |
| miniorm | 100.0% | 100.0% |
| taskcli | 100.0% | 100.0% |

### How each metric is measured
- **Hit-Rate @k** — content retrieval: a hit means a retrieved unit is in the
  right file and covers >=50% of the true symbol's lines. The character
  baseline is scored with *real* line spans (recovered from character offsets),
  so this measures pure retrieval quality, not its (fabricated) citations.
- **Citation Accuracy** — location: does the line range the system reports
  actually match where the retrieved code is? The old pipeline reports
  `chunk_index * 20 + 1`, which is almost never the real start line.
- **Context Precision @k** — share of the top-k that covers the target. AST
  chunks are whole functions (clean hit or clean miss); 800-char windows
  straddle boundaries and dilute the set.
- **Call-Context Recall** — of the target symbol's direct callers + callees,
  the fraction that appear in the context handed to the LLM. The AST engine
  pulls these deterministically from the symbol graph; the character RAG can
  only get them by lucky embedding similarity.

### Takeaway
On codebases this small, dense retrieval alone already finds the right file, so
raw Hit-Rate is close. The upgrade's decisive gains are **trustworthy citations**
and **call-graph context** the old design cannot produce at all.
