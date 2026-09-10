"""Phase 2 - Advanced hybrid retrieval & re-ranking engine.

Pipeline
--------
    query
      |-- sparse:  BM25 over symbol names + signatures + code identifiers
      |-- dense :  jinaai/jina-embeddings-v2-base-code  (ChromaDB, cosine)
      |
      +-- Reciprocal Rank Fusion (k = 60)
      +-- Cross-encoder re-rank  (BAAI/bge-reranker-base)
      +-- Graph neighbourhood expansion (depth-1 callers + callees via graph.py)
      |
      v
    RetrievalResult  ->  ordered context chunks for the LLM

Model loading is lazy and degrades gracefully: if the transformer models cannot
be downloaded/loaded, deterministic fallbacks keep the full pipeline runnable
(the result records which backend was used in `.backends`).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from code_parser import Chunk
from graph import GraphDB, node_id

EMBED_MODEL = "jinaai/jina-embeddings-v2-base-code"
RERANK_MODEL = "BAAI/bge-reranker-base"
RRF_K = 60


# --------------------------------------------------------------------------- #
# Code-aware tokenisation
# --------------------------------------------------------------------------- #
_NON_IDENT = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_STOP = {"the", "a", "an", "of", "to", "is", "in", "and", "or", "how", "does",
         "do", "what", "where", "which", "this", "that", "for", "with", "on"}


def tokenize_code(text: str) -> list[str]:
    """Split code / prose into lowercased identifier tokens, also breaking
    camelCase and snake_case into their parts."""
    out: list[str] = []
    for raw in _NON_IDENT.split(text):
        if not raw:
            continue
        low = raw.lower()
        if low not in _STOP:
            out.append(low)
        parts = _CAMEL.sub(" ", raw).split()
        if len(parts) > 1:
            out.extend(p.lower() for p in parts if p.lower() not in _STOP)
    return out


def _signature_line(code: str) -> str:
    for line in code.splitlines():
        s = line.lstrip()
        if s.startswith(("def ", "async def ", "class ", "function ", "export ",
                         "const ", "public ", "private ", "func ")):
            return line
    return code.splitlines()[0] if code else ""


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = RRF_K,
                           top_n: int | None = None) -> list[tuple[int, float]]:
    """RRF_Score(d) = sum over lists of 1 / (k + rank(d)), rank is 1-based."""
    scores: dict[int, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, doc in enumerate(lst, start=1):
            scores[doc] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:top_n] if top_n else fused


# --------------------------------------------------------------------------- #
# Pluggable models (real + deterministic fallback)
# --------------------------------------------------------------------------- #
class _HashEmbedder:
    """Offline fallback: hashed character 4-gram bag, L2-normalised."""
    dim = 512

    def encode(self, texts, normalize_embeddings=True, **_):
        import numpy as np
        mat = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            toks = tokenize_code(t) or [t]
            for tok in toks:
                for j in range(max(1, len(tok) - 3)):
                    g = tok[j:j + 4]
                    h = int(hashlib.md5(g.encode()).hexdigest(), 16) % self.dim
                    mat[i, h] += 1.0
            n = np.linalg.norm(mat[i]) or 1.0
            mat[i] /= n
        return mat


class _LexicalReranker:
    """Offline fallback: Jaccard token overlap between query and code."""

    def predict(self, pairs, **_):
        scores = []
        for q, d in pairs:
            qs, ds = set(tokenize_code(q)), set(tokenize_code(d))
            scores.append(len(qs & ds) / (len(qs | ds) or 1))
        return scores


@lru_cache(maxsize=2)
def load_embedder(name: str = EMBED_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name, trust_remote_code=True)
        return model, "sentence-transformers:" + name
    except Exception as exc:  # noqa: BLE001
        print(f"[retriever] embedder '{name}' unavailable ({exc.__class__.__name__}); "
              f"using hash fallback")
        return _HashEmbedder(), "hash-fallback"


@lru_cache(maxsize=2)
def load_reranker(name: str = RERANK_MODEL):
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(name), "cross-encoder:" + name
    except Exception as exc:  # noqa: BLE001
        print(f"[retriever] reranker '{name}' unavailable ({exc.__class__.__name__}); "
              f"using lexical fallback")
        return _LexicalReranker(), "lexical-fallback"


# --------------------------------------------------------------------------- #
# Dense index (ChromaDB, numpy fallback)
# --------------------------------------------------------------------------- #
class _DenseIndex:
    def __init__(self, embeddings):
        import numpy as np
        self._np = np
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.backend = "numpy-flat"
        self._col = None
        try:
            import uuid

            import chromadb
            client = chromadb.EphemeralClient()
            self._col = client.get_or_create_collection(
                f"retriever_{uuid.uuid4().hex[:8]}",
                metadata={"hnsw:space": "cosine"})
            self._col.add(
                ids=[str(i) for i in range(len(self.embeddings))],
                embeddings=self.embeddings.tolist(),
            )
            self.backend = "chromadb"
        except Exception as exc:  # noqa: BLE001
            print(f"[retriever] ChromaDB unavailable ({exc.__class__.__name__}); "
                  f"using numpy flat index")

    def search(self, query_vec, top_n: int) -> list[tuple[int, float]]:
        top_n = min(top_n, len(self.embeddings))
        if self._col is not None:
            res = self._col.query(query_embeddings=[list(map(float, query_vec))],
                                  n_results=top_n)
            ids = [int(x) for x in res["ids"][0]]
            dists = res["distances"][0]
            return [(i, 1.0 - d) for i, d in zip(ids, dists)]
        sims = self.embeddings @ self._np.asarray(query_vec, dtype="float32")
        order = self._np.argsort(-sims)[:top_n]
        return [(int(i), float(sims[i])) for i in order]


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class GraphExpansion:
    seed: Chunk
    callers: list[Chunk] = field(default_factory=list)
    callees: list[Chunk] = field(default_factory=list)
    unresolved_callees: list[str] = field(default_factory=list)
    external_callers: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    query: str
    sparse: list[tuple[int, float]]
    dense: list[tuple[int, float]]
    fused: list[tuple[int, float]]
    reranked: list[tuple[int, float]]
    seeds: list[int]
    expansions: list[GraphExpansion]
    backends: dict
    _chunks: list[Chunk]

    def chunk(self, idx: int) -> Chunk:
        return self._chunks[idx]

    def context_chunks(self) -> list[Chunk]:
        """Reranked seeds first, then depth-1 graph neighbours, de-duplicated."""
        seen, ordered = set(), []
        for i in self.seeds:
            c = self._chunks[i]
            key = (c.file_path, c.symbol_name, c.start_line)
            if key not in seen:
                seen.add(key)
                ordered.append(c)
        for exp in self.expansions:
            for c in (*exp.callees, *exp.callers):
                key = (c.file_path, c.symbol_name, c.start_line)
                if key not in seen:
                    seen.add(key)
                    ordered.append(c)
        return ordered


# --------------------------------------------------------------------------- #
# The retriever
# --------------------------------------------------------------------------- #
class HybridRetriever:
    def __init__(self, chunks: list[Chunk], graph_db_path: str,
                 embed_model: str = EMBED_MODEL, rerank_model: str = RERANK_MODEL,
                 rrf_k: int = RRF_K, precomputed_embeddings=None):
        if not chunks:
            raise ValueError("HybridRetriever needs at least one chunk")
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.graph = GraphDB(graph_db_path)
        self._by_key = {(c.file_path, c.symbol_name): c for c in chunks}

        # ---- sparse ----
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([self._bm25_doc(c) for c in chunks])

        # ---- dense ----
        self._embedder, self.embed_backend = load_embedder(embed_model)
        if precomputed_embeddings is not None:
            vecs = precomputed_embeddings
        else:
            vecs = self._embedder.encode([c.code for c in chunks],
                                         normalize_embeddings=True)
        self._dense = _DenseIndex(vecs)

        # ---- reranker (loaded lazily on first query) ----
        self._rerank_model_name = rerank_model
        self._reranker = None
        self.rerank_backend = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_graph(cls, graph_db_path: str, embed_model: str = EMBED_MODEL,
                   rerank_model: str = RERANK_MODEL, rrf_k: int = RRF_K):
        """Build a retriever from a persisted graph DB, caching chunk
        embeddings back into the `chunks` table so later queries are fast."""
        import numpy as np

        gdb = GraphDB(graph_db_path)
        rows = gdb.all_chunks()
        if not rows:
            gdb.close()
            raise ValueError(f"No chunks stored in {graph_db_path} - index the repo first")

        row_ids = [rid for rid, _ in rows]
        chunks = [c for _, c in rows]
        embedder, _ = load_embedder(embed_model)

        cached = gdb.load_embeddings()          # {row_id: float32 blob}
        missing = [i for i, rid in enumerate(row_ids) if rid not in cached]
        if missing:
            new_vecs = np.asarray(
                embedder.encode([chunks[i].code for i in missing],
                                normalize_embeddings=True), dtype="float32")
            gdb.save_embeddings({row_ids[i]: new_vecs[j].tobytes()
                                 for j, i in enumerate(missing)})
            cached = gdb.load_embeddings()

        dim = len(next(iter(cached.values()))) // 4  # float32 bytes -> length
        mat = np.zeros((len(chunks), dim), dtype="float32")
        for i, rid in enumerate(row_ids):
            mat[i] = np.frombuffer(cached[rid], dtype="float32")
        gdb.close()

        return cls(chunks, graph_db_path, embed_model=embed_model,
                   rerank_model=rerank_model, rrf_k=rrf_k,
                   precomputed_embeddings=mat)

    # ------------------------------------------------------------------ #
    def _bm25_doc(self, c: Chunk) -> list[str]:
        toks = tokenize_code(c.symbol_name) * 3          # symbol name weighted
        toks += tokenize_code(_signature_line(c.code)) * 2  # signature weighted
        toks += tokenize_code(c.code)                    # all identifiers
        return toks

    def sparse_search(self, query: str, top_n: int = 50) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize_code(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked[:top_n] if scores[i] > 0.0]

    def dense_search(self, query: str, top_n: int = 50) -> list[tuple[int, float]]:
        qv = self._embedder.encode([query], normalize_embeddings=True)[0]
        return self._dense.search(qv, top_n)

    def fuse(self, sparse, dense, top_n: int = 20) -> list[tuple[int, float]]:
        return reciprocal_rank_fusion(
            [[i for i, _ in sparse], [i for i, _ in dense]],
            k=self.rrf_k, top_n=top_n,
        )

    def rerank(self, query: str, candidate_idxs: list[int],
               top_n: int = 8) -> list[tuple[int, float]]:
        if not candidate_idxs:
            return []
        if self._reranker is None:
            self._reranker, self.rerank_backend = load_reranker(self._rerank_model_name)
        pairs = [[query, self.chunks[i].code] for i in candidate_idxs]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(candidate_idxs, map(float, scores)),
                        key=lambda kv: kv[1], reverse=True)
        return ranked[:top_n]

    # ------------------------------------------------------------------ #
    def expand_graph(self, seed_idxs: list[int]) -> list[GraphExpansion]:
        expansions = []
        for i in seed_idxs:
            c = self.chunks[i]
            if c.symbol_type in ("module", "text"):
                continue
            nb = self.graph.neighbors(node_id(c.file_path, c.symbol_name))
            expansions.append(GraphExpansion(
                seed=c,
                callees=self._nodes_to_chunks(nb["callees"]),
                callers=self._nodes_to_chunks(nb["callers"]),
                unresolved_callees=nb["unresolved_callees"],
                external_callers=[],
            ))
        return expansions

    def _nodes_to_chunks(self, rows) -> list[Chunk]:
        out = []
        for r in rows:
            hit = self._by_key.get((r["file_path"], r["qualified_name"]))
            if hit is not None:
                out.append(hit)
            else:  # symbol is in the graph but we have no chunk text for it
                out.append(Chunk(
                    file_path=r["file_path"], symbol_name=r["qualified_name"],
                    symbol_type=r["symbol_type"], start_line=r["start_line"] or 0,
                    end_line=r["end_line"] or 0, code="",
                ))
        return out

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k_sparse: int = 50, k_dense: int = 50,
                 k_fused: int = 20, k_seeds: int = 6,
                 expand: bool = True) -> RetrievalResult:
        sparse = self.sparse_search(query, k_sparse)
        dense = self.dense_search(query, k_dense)
        fused = self.fuse(sparse, dense, top_n=k_fused)
        reranked = self.rerank(query, [i for i, _ in fused], top_n=k_seeds)
        seeds = [i for i, _ in reranked]
        expansions = self.expand_graph(seeds) if expand else []
        return RetrievalResult(
            query=query, sparse=sparse, dense=dense, fused=fused,
            reranked=reranked, seeds=seeds, expansions=expansions,
            backends={
                "embed": self.embed_backend,
                "dense": self._dense.backend,
                "rerank": self.rerank_backend or "(not run)",
            },
            _chunks=self.chunks,
        )


# ========================================================================= #
# Test script
# ========================================================================= #
_FIXTURE = {
    "auth.py": '''\
import hashlib
from db import get_user


def hash_password(password):
    """Return the sha256 hex digest of a password."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_login(username, password):
    """Check a username/password pair against the stored user record."""
    user = get_user(username)
    if user is None:
        return False
    return user["pw_hash"] == hash_password(password)
''',
    "db.py": '''\
_USERS = {}


def get_user(username):
    """Look up a user record by username."""
    return _USERS.get(username)


def create_user(username, pw_hash):
    _USERS[username] = {"pw_hash": pw_hash}
''',
    "api.py": '''\
from auth import verify_login


def login_endpoint(request):
    """HTTP handler: authenticate the incoming request."""
    ok = verify_login(request["user"], request["password"])
    return {"status": 200 if ok else 401}


def health_endpoint(request):
    return {"status": 200}
''',
}


def _build_fixture():
    import os
    import tempfile
    from indexer import build_graph

    root = tempfile.mkdtemp(prefix="phase2_")
    for name, body in _FIXTURE.items():
        with open(os.path.join(root, name), "w") as fh:
            fh.write(body)
    db_path = os.path.join(root, "graph.sqlite")
    result = build_graph(root, db_path)
    chunks = [c for pf in result["parsed"].values() for c in pf.chunks]
    return chunks, db_path


def test_rrf_math():
    # list A ranks doc 7 first; list B ranks doc 7 third
    fused = dict(reciprocal_rank_fusion([[7, 3, 1], [3, 1, 7]], k=60))
    assert abs(fused[7] - (1 / 61 + 1 / 63)) < 1e-12
    assert abs(fused[3] - (1 / 62 + 1 / 61)) < 1e-12
    # doc 3 (ranks 2 and 1) beats doc 7 (ranks 1 and 3)
    assert fused[3] > fused[7]
    print("  ok  RRF score formula  (k=60)")


def test_hybrid_fusion(retr):
    q = "how does password login verification work"
    sparse = retr.sparse_search(q, 50)
    dense = retr.dense_search(q, 50)
    assert sparse, "BM25 returned no hits"
    assert dense, "dense search returned no hits"

    sparse_names = {retr.chunks[i].symbol_name for i, _ in sparse}
    assert "verify_login" in sparse_names, sparse_names           # keyword match
    dense_names = {retr.chunks[i].symbol_name for i, _ in dense}
    assert "hash_password" in dense_names, dense_names            # semantic match

    fused = retr.fuse(sparse, dense, top_n=10)
    assert fused and fused == sorted(fused, key=lambda kv: kv[1], reverse=True)
    fused_names = {retr.chunks[i].symbol_name for i, _ in fused}
    assert "verify_login" in fused_names and "hash_password" in fused_names, fused_names
    print(f"  ok  BM25+dense RRF fusion  (sparse={len(sparse)} dense={len(dense)} "
          f"-> fused top10={sorted(fused_names)[:6]}...)")
    return q, fused


def test_cross_encoder_rerank(retr, q, fused):
    reranked = retr.rerank(q, [i for i, _ in fused], top_n=6)
    assert reranked, "reranker returned nothing"
    scores = [s for _, s in reranked]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True), "rerank not sorted by score"
    top_names = [retr.chunks[i].symbol_name for i, _ in reranked]
    assert top_names[0] in {"verify_login", "hash_password", "login_endpoint"}, top_names
    print(f"  ok  cross-encoder rerank  backend={retr.rerank_backend}")
    print(f"        ranking: " +
          ", ".join(f"{n}={s:.3f}" for (n, (_, s)) in zip(top_names, reranked)))
    return [i for i, _ in reranked]


def test_graph_expansion(retr, seeds):
    expansions = retr.expand_graph(seeds)
    assert expansions, "no expansions produced"
    by_seed = {e.seed.symbol_name: e for e in expansions}
    assert "verify_login" in by_seed, list(by_seed)

    vexp = by_seed["verify_login"]
    callee_names = {c.symbol_name for c in vexp.callees}
    caller_names = {c.symbol_name for c in vexp.callers}
    assert {"get_user", "hash_password"} <= callee_names, callee_names
    assert "login_endpoint" in caller_names, caller_names
    for c in (*vexp.callees, *vexp.callers):
        assert c.start_line >= 1 and c.code, f"neighbour {c.symbol_name} missing body/line"

    print("  ok  depth-1 call-graph expansion")
    for e in expansions:
        print(f"        seed  {e.seed.symbol_name}  ({e.seed.file_path}:"
              f"{e.seed.start_line}-{e.seed.end_line})")
        for c in e.callees:
            print(f"          -> calls    {c.symbol_name:<16} {c.file_path}:{c.start_line}")
        for name in e.unresolved_callees:
            print(f"          -> calls    {name:<16} (external / stdlib)")
        for c in e.callers:
            print(f"          <- called by {c.symbol_name:<15} {c.file_path}:{c.start_line}")


def main():
    print("building fixture repo + symbol graph ...")
    chunks, db_path = _build_fixture()
    print(f"  {len(chunks)} chunks indexed\n")

    retr = HybridRetriever(chunks, db_path)
    print(f"backends: embed={retr.embed_backend}  dense={retr._dense.backend}\n")

    failed = 0
    try:
        test_rrf_math()
        q, fused = test_hybrid_fusion(retr)
        seeds = test_cross_encoder_rerank(retr, q, fused)
        test_graph_expansion(retr, seeds)
    except AssertionError as exc:
        failed += 1
        print(f"  FAIL  {exc}")

    print("\n--- end-to-end retrieve() ---")
    res = retr.retrieve("how are user passwords hashed and verified")
    print("backends:", res.backends)
    print("context chunks fed to LLM (seed + graph neighbours):")
    for c in res.context_chunks():
        print(f"  {c.symbol_type:<8} {c.symbol_name:<18} {c.file_path}:{c.start_line}-{c.end_line}")

    print("\nPASS" if not failed else "\nFAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
