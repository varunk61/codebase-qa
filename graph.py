"""Symbol graph storage + Git-diff incremental indexing (Phase 1).

SQLite schema
-------------
files : one row per indexed file, keyed by repo-relative path, holding the
        SHA-256 of the file's bytes. This is what makes re-indexing incremental:
        a file whose hash is unchanged is skipped entirely.

nodes : one row per symbol (function | method | class) plus one synthetic
        `<file>` node per file so module-level calls/imports have a source.

edges : CALLS and IMPORTS relationships. `dst_id` is resolved to a node id
        when the target can be located; otherwise it stays NULL (external
        symbol / third-party import).

chunks: retrieval units (one row per AST chunk) with their exact line span,
        code body, and an optional cached embedding blob. Persisting these
        lets the retriever rebuild BM25 + the dense index without re-parsing
        or (once cached) re-embedding.
"""

from __future__ import annotations

import os
import sqlite3
import time

from code_parser import Chunk, ParsedFile, SymbolDef

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path       TEXT PRIMARY KEY,
    sha256     TEXT NOT NULL,
    language   TEXT,
    indexed_at REAL
);

CREATE TABLE IF NOT EXISTS nodes (
    id             TEXT PRIMARY KEY,       -- "<path>::<qualified_name>"
    file_path      TEXT NOT NULL,
    symbol_name    TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    symbol_type    TEXT NOT NULL,          -- function | method | class | file
    start_line     INTEGER,
    end_line       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(symbol_name);

CREATE TABLE IF NOT EXISTS edges (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id    TEXT NOT NULL,
    dst_id    TEXT,                         -- resolved node id, or NULL if external
    dst_name  TEXT NOT NULL,
    edge_type TEXT NOT NULL,                -- CALLS | IMPORTS
    file_path TEXT NOT NULL,
    line      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_edges_src  ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_file ON edges(file_path);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_type TEXT NOT NULL,
    parent      TEXT,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    code        TEXT NOT NULL,
    embedding   BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
"""

FILE_NODE_SUFFIX = "::<file>"


def file_node_id(path: str) -> str:
    return f"{path}{FILE_NODE_SUFFIX}"


def node_id(path: str, qualified_name: str) -> str:
    return f"{path}::{qualified_name}"


class GraphDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ----------------------------------------------------------------- #
    # Incremental bookkeeping
    # ----------------------------------------------------------------- #
    def file_changed(self, path: str, sha256: str) -> bool:
        row = self.conn.execute(
            "SELECT sha256 FROM files WHERE path = ?", (path,)
        ).fetchone()
        return row is None or row["sha256"] != sha256

    def known_files(self) -> set[str]:
        return {r["path"] for r in self.conn.execute("SELECT path FROM files")}

    def remove_file(self, path: str) -> None:
        self.conn.execute("DELETE FROM nodes WHERE file_path = ?", (path,))
        self.conn.execute("DELETE FROM edges WHERE file_path = ?", (path,))
        self.conn.execute("DELETE FROM chunks WHERE file_path = ?", (path,))
        self.conn.execute("DELETE FROM files WHERE path = ?", (path,))

    # ----------------------------------------------------------------- #
    # Writing a parsed file into the graph
    # ----------------------------------------------------------------- #
    def index_parsed_file(self, pf: ParsedFile) -> None:
        """Replace everything the graph knows about this file."""
        self.remove_file(pf.file_path)

        self.conn.execute(
            "INSERT INTO files (path, sha256, language, indexed_at) VALUES (?, ?, ?, ?)",
            (pf.file_path, pf.sha256, pf.language, time.time()),
        )

        # synthetic file node (source for module-level calls / imports)
        self.conn.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (file_node_id(pf.file_path), pf.file_path, os.path.basename(pf.file_path),
             "<file>", "file", None, None),
        )

        for s in pf.symbols:
            self._insert_symbol(s)

        for c in pf.calls:
            src = (file_node_id(pf.file_path) if c.caller == "<module>"
                   else node_id(pf.file_path, c.caller))
            self.conn.execute(
                "INSERT INTO edges (src_id, dst_id, dst_name, edge_type, file_path, line) "
                "VALUES (?, NULL, ?, 'CALLS', ?, ?)",
                (src, c.callee_name, pf.file_path, c.line),
            )

        for imp in pf.imports:
            dst_name = f"{imp.module}.{imp.name}" if imp.name and imp.name != "*" else imp.module
            self.conn.execute(
                "INSERT INTO edges (src_id, dst_id, dst_name, edge_type, file_path, line) "
                "VALUES (?, NULL, ?, 'IMPORTS', ?, ?)",
                (file_node_id(pf.file_path), dst_name, pf.file_path, imp.line),
            )

        for ch in pf.chunks:
            self.conn.execute(
                "INSERT INTO chunks (file_path, symbol_name, symbol_type, parent, "
                "start_line, end_line, code, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (ch.file_path, ch.symbol_name, ch.symbol_type, ch.parent,
                 ch.start_line, ch.end_line, ch.code),
            )

    def _insert_symbol(self, s: SymbolDef) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (node_id(s.file_path, s.qualified_name), s.file_path, s.symbol_name,
             s.qualified_name, s.symbol_type, s.start_line, s.end_line),
        )

    # ----------------------------------------------------------------- #
    # Edge resolution (runs after every build so stale links self-heal)
    # ----------------------------------------------------------------- #
    def resolve_edges(self) -> None:
        """Point CALLS edges at a concrete node id when the callee can be located.

        Preference order: a definition in the same file, then a unique
        definition anywhere in the repo. Ambiguous or missing -> NULL (external).
        """
        by_name: dict[str, list[sqlite3.Row]] = {}
        for r in self.conn.execute(
            "SELECT id, file_path, symbol_name FROM nodes WHERE symbol_type != 'file'"
        ):
            by_name.setdefault(r["symbol_name"], []).append(r)

        for e in self.conn.execute(
            "SELECT id, dst_name, file_path FROM edges WHERE edge_type = 'CALLS'"
        ).fetchall():
            short = e["dst_name"].split(".")[-1]
            candidates = by_name.get(short, [])
            resolved = None
            same_file = [c for c in candidates if c["file_path"] == e["file_path"]]
            if len(same_file) == 1:
                resolved = same_file[0]["id"]
            elif len(candidates) == 1:
                resolved = candidates[0]["id"]
            self.conn.execute(
                "UPDATE edges SET dst_id = ? WHERE id = ?", (resolved, e["id"])
            )

    def commit(self) -> None:
        self.conn.commit()

    # ----------------------------------------------------------------- #
    # Read helpers (used by tests now, by retrieval in later phases)
    # ----------------------------------------------------------------- #
    def stats(self) -> dict:
        c = self.conn.execute
        return {
            "files": c("SELECT COUNT(*) FROM files").fetchone()[0],
            "nodes": c("SELECT COUNT(*) FROM nodes WHERE symbol_type != 'file'").fetchone()[0],
            "calls": c("SELECT COUNT(*) FROM edges WHERE edge_type = 'CALLS'").fetchone()[0],
            "imports": c("SELECT COUNT(*) FROM edges WHERE edge_type = 'IMPORTS'").fetchone()[0],
            "resolved_calls": c(
                "SELECT COUNT(*) FROM edges WHERE edge_type = 'CALLS' AND dst_id IS NOT NULL"
            ).fetchone()[0],
        }

    def get_node(self, path: str, qualified_name: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id(path, qualified_name),)
        ).fetchone()

    def nodes_in(self, path: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM nodes WHERE file_path = ? AND symbol_type != 'file' "
            "ORDER BY start_line", (path,)
        ).fetchall()

    def edges_from(self, path: str, qualified_name: str, edge_type: str = "CALLS") -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM edges WHERE src_id = ? AND edge_type = ? ORDER BY line",
            (node_id(path, qualified_name), edge_type),
        ).fetchall()

    def import_edges(self, path: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM edges WHERE file_path = ? AND edge_type = 'IMPORTS' ORDER BY line",
            (path,),
        ).fetchall()

    def callers_of(self, node_id_value: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM edges WHERE dst_id = ? AND edge_type = 'CALLS'", (node_id_value,)
        ).fetchall()

    def get_by_id(self, node_id_value: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id_value,)
        ).fetchone()

    # ----------------------------------------------------------------- #
    # Chunk store (feeds the retriever)
    # ----------------------------------------------------------------- #
    def all_chunks(self) -> list[tuple[int, Chunk]]:
        """Every stored chunk as (row_id, Chunk), ordered for stable indices."""
        out = []
        for r in self.conn.execute(
            "SELECT id, file_path, symbol_name, symbol_type, parent, "
            "start_line, end_line, code FROM chunks ORDER BY file_path, start_line, id"
        ):
            out.append((r["id"], Chunk(
                file_path=r["file_path"], symbol_name=r["symbol_name"],
                symbol_type=r["symbol_type"], start_line=r["start_line"],
                end_line=r["end_line"], code=r["code"], parent=r["parent"],
            )))
        return out

    def load_embeddings(self) -> dict[int, bytes]:
        return {r["id"]: r["embedding"] for r in self.conn.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL")}

    def save_embeddings(self, blobs: dict[int, bytes]) -> None:
        self.conn.executemany(
            "UPDATE chunks SET embedding = ? WHERE id = ?",
            [(v, k) for k, v in blobs.items()],
        )
        self.conn.commit()

    def callees_of(self, node_id_value: str, resolved_only: bool = False) -> list[sqlite3.Row]:
        """Outgoing CALLS edges from a node (by id)."""
        sql = "SELECT * FROM edges WHERE src_id = ? AND edge_type = 'CALLS'"
        if resolved_only:
            sql += " AND dst_id IS NOT NULL"
        return self.conn.execute(sql + " ORDER BY line", (node_id_value,)).fetchall()

    def neighbors(self, node_id_value: str) -> dict:
        """Depth-1 call-graph neighbourhood of a node.

        Returns {"callees": [Row(node), ...], "callers": [Row(node), ...],
                 "unresolved_callees": [dst_name, ...]}.
        """
        callees, unresolved = [], []
        for e in self.callees_of(node_id_value):
            if e["dst_id"]:
                n = self.get_by_id(e["dst_id"])
                if n is not None:
                    callees.append(n)
            else:
                unresolved.append(e["dst_name"])

        callers = []
        for e in self.callers_of(node_id_value):
            n = self.get_by_id(e["src_id"])
            if n is not None and n["symbol_type"] != "file":
                callers.append(n)

        return {
            "callees": _dedupe_nodes(callees),
            "callers": _dedupe_nodes(callers),
            "unresolved_callees": sorted(set(unresolved)),
        }


def _dedupe_nodes(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    seen, out = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out
