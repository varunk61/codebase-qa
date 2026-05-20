import os
import re
import shutil
import tempfile
import git
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma   = chromadb.PersistentClient(path="./chroma_db")

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rb", ".rs", ".cpp",
    ".c", ".cs", ".php", ".md", ".txt",
}

MAX_REPO_SIZE_MB = 50
BATCH_SIZE = 500


def _repo_id_from_url(url: str) -> str:
    parts = url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL")
    return f"{parts[-2]}_{parts[-1]}".lower().replace("-", "_")


def _collect_files(repo_path: str) -> list[dict]:
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in
                   ("node_modules", "__pycache__", ".git", "dist", "build", "venv")]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, fname)
                rel_path  = os.path.relpath(full_path, repo_path)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if content.strip():
                        files.append({"path": rel_path, "content": content})
                except Exception:
                    pass
    return files


def _chunk_files(files: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = []
    for file in files:
        parts = splitter.split_text(file["content"])
        for i, part in enumerate(parts):
            chunks.append({
                "id":   f"{file['path']}::chunk{i}",
                "text": part,
                "path": file["path"],
                "line": i * 20 + 1,
            })
    return chunks


def index_repo(github_url: str) -> str:
    if not re.match(r"https://github\.com/[\w\-\.]+/[\w\-\.]+", github_url):
        raise ValueError("Please provide a valid GitHub URL (https://github.com/owner/repo)")

    repo_id = _repo_id_from_url(github_url)
    tmp_dir = tempfile.mkdtemp()

    try:
        print(f"[indexer] Cloning {github_url} ...")
        git.Repo.clone_from(github_url, tmp_dir, depth=1)

        total_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fnames in os.walk(tmp_dir)
            for f in fnames
        ) / (1024 * 1024)
        if total_mb > MAX_REPO_SIZE_MB:
            raise ValueError(f"Repo is {total_mb:.1f} MB — limit is {MAX_REPO_SIZE_MB} MB")

        print(f"[indexer] Collecting files ...")
        files  = _collect_files(tmp_dir)
        print(f"[indexer] Found {len(files)} files")

        chunks = _chunk_files(files)
        print(f"[indexer] Created {len(chunks)} chunks")

        texts      = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

        try:
            chroma.delete_collection(repo_id)
        except Exception:
            pass

        collection = chroma.get_or_create_collection(repo_id)

        for i in range(0, len(chunks), BATCH_SIZE):
            batch     = chunks[i:i + BATCH_SIZE]
            batch_emb = embeddings[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_bat = (len(chunks) - 1) // BATCH_SIZE + 1
            print(f"[indexer] Uploading batch {batch_num}/{total_bat} ...")
            collection.upsert(
                ids        = [c["id"] for c in batch],
                documents  = [c["text"] for c in batch],
                embeddings = batch_emb,
                metadatas  = [{"path": c["path"], "line": c["line"]} for c in batch],
            )

        print(f"[indexer] Done! Collection '{repo_id}' ready.")
        return repo_id

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
