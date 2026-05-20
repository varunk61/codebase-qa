from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from indexer import index_repo
from querier import answer_question
import threading

app = FastAPI(title="Codebase QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# track indexing status per repo
indexing_status = {}  # repo_id -> "indexing" | "done" | "error: ..."

class IndexRequest(BaseModel):
    github_url: str

class QueryRequest(BaseModel):
    repo_id: str
    question: str

def _run_index(github_url: str, repo_id: str):
    """Runs in a background thread so the HTTP request returns immediately."""
    try:
        indexing_status[repo_id] = "indexing"
        index_repo(github_url)
        indexing_status[repo_id] = "done"
    except Exception as e:
        indexing_status[repo_id] = f"error: {str(e)}"

@app.get("/")
def root():
    return {"status": "Codebase QA is running"}

@app.post("/index")
def index(request: IndexRequest):
    """
    Starts indexing in background and returns immediately.
    Frontend should poll /status/{repo_id} to check progress.
    """
    # derive repo_id from URL
    parts = request.github_url.rstrip("/").split("/")
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL")
    repo_id = f"{parts[-2]}_{parts[-1]}".lower().replace("-", "_")

    # if already done, return immediately
    if indexing_status.get(repo_id) == "done":
        return {"repo_id": repo_id, "status": "done", "message": "Already indexed!"}

    # if currently indexing, don't start again
    if indexing_status.get(repo_id) == "indexing":
        return {"repo_id": repo_id, "status": "indexing", "message": "Already indexing..."}

    # start background thread
    thread = threading.Thread(target=_run_index, args=(request.github_url, repo_id))
    thread.daemon = True
    thread.start()

    return {"repo_id": repo_id, "status": "indexing", "message": "Indexing started in background!"}

@app.get("/status/{repo_id}")
def status(repo_id: str):
    """Frontend polls this to check if indexing is done."""
    s = indexing_status.get(repo_id, "not_started")
    return {"repo_id": repo_id, "status": s}

@app.post("/ask")
def ask(request: QueryRequest):
    # check indexing status first
    s = indexing_status.get(request.repo_id)
    if s == "indexing":
        raise HTTPException(status_code=400, detail="Still indexing! Please wait...")
    if s and s.startswith("error"):
        raise HTTPException(status_code=400, detail=s)
    try:
        result = answer_question(request.repo_id, request.question)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")