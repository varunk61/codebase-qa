# 🔍 Codebase QA — AI Documentation Assistant

Ask any question about a GitHub repository and get answers with **file:line citations**.

> "How does authentication work?" → "JWT is verified in `auth/jwt.py:42` using the HS256 algorithm…"

## Tech Stack (100% Free)
| Component | Tool |
|---|---|
| Backend | FastAPI (Python) |
| Embeddings | sentence-transformers (local, free) |
| Vector DB | ChromaDB (local, free) |
| LLM | Groq API (free tier) |
| Frontend | Streamlit |

---

## Setup (5 minutes)

### 1. Clone this repo
```bash
git clone https://github.com/YOUR_USERNAME/codebase-qa
cd codebase-qa
```

### 2. Get your free Groq API key
- Go to https://console.groq.com
- Sign up (free)
- Create an API key

### 3. Set up the backend
```bash
cd backend
pip install -r requirements.txt

# copy the example env file and add your key
cp .env.example .env
# open .env and paste your GROQ_API_KEY
```

### 4. Run the backend
```bash
cd backend
uvicorn main:app --reload
# → running at http://localhost:8000
```

### 5. Run the frontend (new terminal)
```bash
pip install streamlit
cd frontend
streamlit run app.py
# → opens at http://localhost:8501
```

---

## How it works

### Indexing pipeline (runs once per repo)
```
GitHub URL → clone → parse files → chunk code → embed → ChromaDB
```

### Query pipeline (runs on every question)
```
Question → embed → ChromaDB similarity search → top chunks → Groq LLM → answer + citations
```

---

## Deployment

### Backend (Render)
1. Push to GitHub
2. Go to render.com → New Web Service
3. Connect your repo → set `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add `GROQ_API_KEY` as environment variable
5. Deploy!

### Frontend (Streamlit Cloud)
1. Go to share.streamlit.io
2. Connect your GitHub repo
3. Set main file path: `frontend/app.py`
4. Update `API_URL` in `app.py` to your Render backend URL
5. Deploy!

---

## Project Structure
```
codebase-qa/
├── backend/
│   ├── main.py          # FastAPI app — /index and /ask endpoints
│   ├── indexer.py       # Clone → parse → chunk → embed → ChromaDB
│   ├── querier.py       # Embed question → search → Groq LLM → answer
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── app.py           # Streamlit UI
```
