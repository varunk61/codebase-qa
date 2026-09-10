"""AST-based code parsing for the Codebase QA engine (Phase 1).

Turns a source file into:
  - `chunks`  : retrieval units aligned to AST boundaries, with true line numbers
  - `symbols` : function / method / class definitions (graph nodes)
  - `calls`   : call sites  (graph CALLS edges)
  - `imports` : import statements (graph IMPORTS edges)

Supported code languages: Python, JavaScript, TypeScript (+ TSX).
Everything else (.md, .json, .yaml, ...) falls back to a line-window splitter
so it still produces citable chunks, just without graph edges.

Tree-sitter 0.22+ API is used throughout:
    from tree_sitter import Language, Parser
    import tree_sitter_python
    Language(tree_sitter_python.language())
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

from tree_sitter import Language, Parser
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """A retrieval unit. Line numbers are 1-indexed and inclusive."""
    file_path: str
    symbol_name: str          # "hash_password", "TokenValidator.validate", "<module>", "lines 1-60"
    symbol_type: str          # function | method | class | module | text
    start_line: int
    end_line: int
    code: str
    parent: str | None = None  # enclosing class for methods, else None


@dataclass
class SymbolDef:
    """A definition that becomes a node in the symbol graph."""
    qualified_name: str       # "TokenValidator.validate"
    symbol_name: str          # "validate"
    symbol_type: str          # function | method | class
    file_path: str
    start_line: int
    end_line: int


@dataclass
class CallSite:
    """A call expression -> CALLS edge."""
    caller: str               # qualified name of the enclosing symbol, or "<module>"
    callee_name: str          # short name, e.g. "join"
    callee_full: str          # dotted text, e.g. "os.path.join"
    file_path: str
    line: int


@dataclass
class ImportRef:
    """An import statement -> IMPORTS edge."""
    file_path: str
    module: str               # "os", "sample_pkg.utils", "fs"
    name: str | None          # "helper" for `from x import helper`; None for plain import
    alias: str | None
    line: int


@dataclass
class ParsedFile:
    file_path: str
    language: str             # python | javascript | typescript | tsx | text
    sha256: str
    chunks: list[Chunk] = field(default_factory=list)
    symbols: list[SymbolDef] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    imports: list[ImportRef] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Language registry
# --------------------------------------------------------------------------- #
EXT_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}

_LANGUAGES: dict[str, Language] = {
    "python": Language(tree_sitter_python.language()),
    "javascript": Language(tree_sitter_javascript.language()),
    "typescript": Language(tree_sitter_typescript.language_typescript()),
    "tsx": Language(tree_sitter_typescript.language_tsx()),
}

_PARSERS: dict[str, Parser] = {name: Parser(lang) for name, lang in _LANGUAGES.items()}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def language_for(path: str) -> str | None:
    """Return the code language for a path, or None if it is not parseable code."""
    return EXT_LANGUAGE.get(os.path.splitext(path)[1].lower())


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def parse_source(file_path: str, data: bytes, sha256: str | None = None) -> ParsedFile:
    """Parse raw file bytes into a ParsedFile. `file_path` should be repo-relative."""
    sha = sha256 or sha256_bytes(data)
    text = data.decode("utf-8", errors="replace")
    lang = language_for(file_path)

    if lang is None:
        pf = ParsedFile(file_path=file_path, language="text", sha256=sha)
        pf.chunks = _text_chunks(file_path, text)
        return pf

    parser = _PARSERS[lang]
    tree = parser.parse(bytes(text, "utf-8"))
    src = bytes(text, "utf-8")

    pf = ParsedFile(file_path=file_path, language=lang, sha256=sha)
    if lang == "python":
        _extract_python(pf, tree.root_node, src)
    else:
        _extract_js_like(pf, tree.root_node, src)

    # Safety net: if the grammar produced nothing useful (e.g. a parse error left
    # an empty tree), still index the file as text so it stays searchable.
    if not pf.chunks:
        pf.chunks = _text_chunks(file_path, text)
    return pf


# --------------------------------------------------------------------------- #
# Shared tree helpers
# --------------------------------------------------------------------------- #
def _txt(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _lines(node) -> tuple[int, int]:
    """1-indexed inclusive (start_line, end_line) straight from the AST."""
    return node.start_point[0] + 1, node.end_point[0] + 1


def _walk(node):
    """Depth-first iterator over every node in the tree."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


# --------------------------------------------------------------------------- #
# Python extraction
# --------------------------------------------------------------------------- #
_PY_DEF_TYPES = {"function_definition", "class_definition"}


def _py_name(node) -> str | None:
    n = node.child_by_field_name("name")
    return n.text.decode("utf-8") if n is not None else None


def _py_qualified_name(def_node) -> str:
    """Dotted path built from every enclosing function/class name."""
    names: list[str] = []
    n = def_node
    while n is not None:
        if n.type in _PY_DEF_TYPES:
            nm = _py_name(n)
            if nm:
                names.append(nm)
        n = n.parent
    return ".".join(reversed(names))


def _py_immediate_scope(def_node):
    """Nearest enclosing function/class node above `def_node` (or None)."""
    n = def_node.parent
    while n is not None:
        if n.type in _PY_DEF_TYPES:
            return n
        n = n.parent
    return None


def _span_node(def_node):
    """Include leading decorators in the chunk span when present."""
    p = def_node.parent
    if p is not None and p.type == "decorated_definition":
        return p
    return def_node


def _extract_python(pf: ParsedFile, root, src: bytes) -> None:
    def_nodes = [n for n in _walk(root) if n.type in _PY_DEF_TYPES]

    for node in def_nodes:
        name = _py_name(node)
        if not name:
            continue
        qual = _py_qualified_name(node)
        scope = _py_immediate_scope(node)

        if node.type == "class_definition":
            symbol_type = "class"
            parent = None
        elif scope is not None and scope.type == "class_definition":
            symbol_type = "method"
            parent = _py_qualified_name(scope)
        else:
            symbol_type = "function"
            parent = None

        span = _span_node(node)
        start, end = _lines(span)

        pf.symbols.append(SymbolDef(
            qualified_name=qual, symbol_name=name, symbol_type=symbol_type,
            file_path=pf.file_path, start_line=start, end_line=end,
        ))
        pf.chunks.append(Chunk(
            file_path=pf.file_path, symbol_name=qual, symbol_type=symbol_type,
            start_line=start, end_line=end, code=_txt(span, src), parent=parent,
        ))

    _py_module_chunks(pf, root, src)
    _py_calls(pf, root, src)
    _py_imports(pf, root, src)


def _enclosing_qualified(node) -> str:
    """Qualified name of the nearest function/class that contains `node`."""
    n = node.parent
    while n is not None:
        if n.type in _PY_DEF_TYPES:
            return _py_qualified_name(n)
        n = n.parent
    return "<module>"


def _py_calls(pf: ParsedFile, root, src: bytes) -> None:
    for node in _walk(root):
        if node.type != "call":
            continue
        fn = node.child_by_field_name("function")
        if fn is None:
            continue
        full = _txt(fn, src)
        if fn.type == "identifier":
            short = full
        elif fn.type == "attribute":
            attr = fn.child_by_field_name("attribute")
            short = attr.text.decode("utf-8") if attr is not None else full.split(".")[-1]
        else:
            short = full.split(".")[-1]
        line = node.start_point[0] + 1
        pf.calls.append(CallSite(
            caller=_enclosing_qualified(node), callee_name=short,
            callee_full=full, file_path=pf.file_path, line=line,
        ))


def _py_imports(pf: ParsedFile, root, src: bytes) -> None:
    for node in _walk(root):
        line = node.start_point[0] + 1
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    pf.imports.append(ImportRef(pf.file_path, _txt(child, src), None, None, line))
                elif child.type == "aliased_import":
                    mod = child.child_by_field_name("name")
                    alias = child.child_by_field_name("alias")
                    pf.imports.append(ImportRef(
                        pf.file_path,
                        _txt(mod, src) if mod else _txt(child, src),
                        None,
                        alias.text.decode("utf-8") if alias is not None else None,
                        line,
                    ))
        elif node.type == "import_from_statement":
            mod_node = node.child_by_field_name("module_name")
            if mod_node is None:  # older grammars expose no field name
                mod_node = next((c for c in node.children if c.type in ("dotted_name", "relative_import")), None)
            module = _txt(mod_node, src) if mod_node is not None else ""
            mod_id = mod_node.id if mod_node is not None else None
            for child in node.children:
                if child.id == mod_id:
                    continue
                if child.type == "dotted_name":
                    pf.imports.append(ImportRef(pf.file_path, module, _txt(child, src), None, line))
                elif child.type == "aliased_import":
                    nm = child.child_by_field_name("name")
                    alias = child.child_by_field_name("alias")
                    pf.imports.append(ImportRef(
                        pf.file_path, module,
                        _txt(nm, src) if nm else None,
                        alias.text.decode("utf-8") if alias is not None else None,
                        line,
                    ))
                elif child.type == "wildcard_import":
                    pf.imports.append(ImportRef(pf.file_path, module, "*", None, line))


def _py_module_chunks(pf: ParsedFile, root, src: bytes) -> None:
    """Group top-level statements that are not defs into contiguous 'module' chunks."""
    covered = {"function_definition", "class_definition", "decorated_definition"}
    run: list = []

    def flush():
        if not run:
            return
        if all(c.type == "comment" for c in run):
            run.clear()
            return
        start = run[0].start_point[0] + 1
        end = run[-1].end_point[0] + 1
        code = src[run[0].start_byte:run[-1].end_byte].decode("utf-8", errors="replace")
        pf.chunks.append(Chunk(
            file_path=pf.file_path, symbol_name="<module>", symbol_type="module",
            start_line=start, end_line=end, code=code,
        ))
        run.clear()

    for child in root.children:
        if child.type in covered:
            flush()
        else:
            run.append(child)
    flush()


# --------------------------------------------------------------------------- #
# JavaScript / TypeScript extraction  (shared)
# --------------------------------------------------------------------------- #
_JS_FUNC_TYPES = {
    "function_declaration", "generator_function_declaration",
    "function", "function_expression", "arrow_function", "method_definition",
}
_JS_CLASS_TYPES = {"class_declaration", "class"}


def _js_name(node, src: bytes) -> str | None:
    n = node.child_by_field_name("name")
    if n is not None:
        return _txt(n, src)
    return None


def _js_def_name(node, src: bytes) -> str | None:
    """Name for a def node, resolving `const foo = () => {}` to `foo`."""
    direct = _js_name(node, src)
    if direct:
        return direct
    if node.parent is not None and node.parent.type == "variable_declarator":
        nm = node.parent.child_by_field_name("name")
        if nm is not None:
            return _txt(nm, src)
    return None


def _js_context(node, src: bytes) -> tuple[str, str | None]:
    """(qualified_name, parent_class) for a definition node.

    Scope search starts above the node's own name binding, so
    `const foo = () => {}` yields "foo", not "foo.foo".
    """
    name = _js_def_name(node, src)

    start = node
    if node.parent is not None and node.parent.type == "variable_declarator":
        start = node.parent
        if start.parent is not None and start.parent.type in ("lexical_declaration", "variable_declaration"):
            start = start.parent

    scopes: list[str] = []
    parent_class = None
    n = start.parent
    while n is not None:
        if n.type in _JS_CLASS_TYPES:
            cn = _js_name(n, src)
            if cn:
                scopes.append(cn)
                if parent_class is None:
                    parent_class = cn
        elif n.type in _JS_FUNC_TYPES:
            fn = _js_def_name(n, src)
            if fn:
                scopes.append(fn)
        n = n.parent
    scopes.reverse()
    qual = ".".join(scopes + ([name] if name else []))
    return qual, parent_class


def _extract_js_like(pf: ParsedFile, root, src: bytes) -> None:
    for node in _walk(root):
        if node.type in _JS_CLASS_TYPES:
            name = _js_name(node, src)
            if not name:
                continue
            qual, _ = _js_context(node, src)
            start, end = _lines(node)
            pf.symbols.append(SymbolDef(qual, name, "class", pf.file_path, start, end))
            pf.chunks.append(Chunk(pf.file_path, qual, "class", start, end, _txt(node, src)))

        elif node.type in _JS_FUNC_TYPES:
            # Skip arrow/function expressions that are not bound to a name.
            name = _js_def_name(node, src)
            if not name:
                continue
            qual, parent_class = _js_context(node, src)
            symbol_type = "method" if (parent_class and node.type == "method_definition") else "function"
            span = node
            if node.parent is not None and node.parent.type == "variable_declarator":
                # extend span to the whole `const foo = () => {...}` declaration
                gp = node.parent.parent
                if gp is not None and gp.type in ("lexical_declaration", "variable_declaration"):
                    span = gp
            start, end = _lines(span)
            pf.symbols.append(SymbolDef(qual, name, symbol_type, pf.file_path, start, end))
            pf.chunks.append(Chunk(
                pf.file_path, qual, symbol_type, start, end, _txt(span, src),
                parent=parent_class if symbol_type == "method" else None,
            ))

    _js_calls(pf, root, src)
    _js_imports(pf, root, src)


def _js_enclosing_qualified(node, src: bytes) -> str:
    n = node.parent
    while n is not None:
        if n.type in _JS_FUNC_TYPES or n.type in _JS_CLASS_TYPES:
            qual, _ = _js_context(n, src)
            if qual:
                return qual
        n = n.parent
    return "<module>"


def _js_calls(pf: ParsedFile, root, src: bytes) -> None:
    for node in _walk(root):
        if node.type != "call_expression":
            continue
        fn = node.child_by_field_name("function")
        if fn is None:
            continue
        full = _txt(fn, src)
        if fn.type == "identifier":
            short = full
        elif fn.type == "member_expression":
            prop = fn.child_by_field_name("property")
            short = _txt(prop, src) if prop is not None else full.split(".")[-1]
        else:
            short = full.split(".")[-1]
        if short == "require":
            continue  # handled as an import
        pf.calls.append(CallSite(
            caller=_js_enclosing_qualified(node, src), callee_name=short,
            callee_full=full, file_path=pf.file_path, line=node.start_point[0] + 1,
        ))


def _js_imports(pf: ParsedFile, root, src: bytes) -> None:
    for node in _walk(root):
        line = node.start_point[0] + 1
        if node.type == "import_statement":
            source = node.child_by_field_name("source")
            if source is not None:
                mod = _txt(source, src).strip("\"'`")
                pf.imports.append(ImportRef(pf.file_path, mod, None, None, line))
        elif node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None and fn.type == "identifier" and _txt(fn, src) == "require":
                args = node.child_by_field_name("arguments")
                if args is not None:
                    for a in args.children:
                        if a.type == "string":
                            pf.imports.append(ImportRef(
                                pf.file_path, _txt(a, src).strip("\"'`"), None, None, line,
                            ))


# --------------------------------------------------------------------------- #
# Non-code fallback: line-window splitter (keeps true line numbers)
# --------------------------------------------------------------------------- #
def _text_chunks(file_path: str, text: str, window: int = 60, overlap: int = 8) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []
    step = max(1, window - overlap)
    chunks: list[Chunk] = []
    i = 0
    while i < len(lines):
        start = i + 1
        end = min(len(lines), i + window)
        body = "\n".join(lines[i:end])
        if body.strip():
            chunks.append(Chunk(
                file_path=file_path, symbol_name=f"lines {start}-{end}",
                symbol_type="text", start_line=start, end_line=end, code=body,
            ))
        if end == len(lines):
            break
        i += step
    return chunks
