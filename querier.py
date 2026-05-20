import os
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# reuse same embedder & chroma client (loaded once)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma   = chromadb.PersistentClient(path="./chroma_db")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

TOP_K = 6   # how many code chunks to retrieve per question


def answer_question(repo_id: str, question: str) -> dict:
    """
    1. Embed the question
    2. Search ChromaDB for top-k most similar chunks
    3. Build a prompt with those chunks as context
    4. Call Groq (free LLM) for an answer
    5. Return answer + source citations
    """
    # check the repo has been indexed
    try:
        collection = chroma.get_collection(repo_id)
    except Exception:
        raise ValueError(f"Repo '{repo_id}' not indexed yet. Please index it first.")

    # embed the question using the same model as indexing
    q_embedding = embedder.encode([question]).tolist()[0]

    # find the most relevant chunks
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # build the context block we'll inject into the LLM prompt
    context_parts = []
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        citation = f"{meta['path']}:{meta['line']}"
        context_parts.append(f"[{citation}]\n{chunk}")
    context = "\n\n---\n\n".join(context_parts)

    # the prompt — clear instructions so the LLM actually cites sources
    system_prompt = """You are an expert code assistant. You are given relevant code snippets
from a repository (with file:line citations) and a question about the codebase.

Rules:
- Answer clearly and concisely based ONLY on the provided code snippets
- Always mention which file and line the relevant code is in
- If the answer isn't in the provided snippets, say "I couldn't find that in the indexed code"
- Format file references like: `filename.py:42`
- Keep answers focused and practical
"""

    user_prompt = f"""Here are the most relevant code sections from the repository:

{context}

Question: {question}

Please answer the question and cite the specific files and line numbers."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # free, fast, very capable
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,   # low temp = more factual, less hallucination
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    # build clean citations list for the frontend
    citations = [
        {
            "file": meta["path"],
            "line": meta["line"],
            "relevance_score": round(1 - dist, 3),   # distance → similarity score
            "preview": chunk[:120] + "..." if len(chunk) > 120 else chunk,
        }
        for chunk, meta, dist in zip(chunks, metadatas, distances)
    ]

    return {
        "answer":    answer,
        "citations": citations,
        "question":  question,
    }
