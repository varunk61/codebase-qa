import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Codebase QA", page_icon="🔍", layout="centered")

st.title("🔍 Codebase QA")
st.caption("Ask any question about a GitHub repository — get answers with file:line citations.")

if "repo_id"  not in st.session_state: st.session_state.repo_id  = None
if "repo_url" not in st.session_state: st.session_state.repo_url = None
if "history"  not in st.session_state: st.session_state.history  = []

# ── Step 1: Index a repo ───────────────────────────────────────────────────────
st.subheader("Step 1 — Point it at a GitHub repo")

with st.form("index_form"):
    github_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/pallets/flask",
    )
    submitted = st.form_submit_button("Index Repo 🚀", use_container_width=True)

if submitted and github_url:
    try:
        res = requests.post(f"{API_URL}/index", json={"github_url": github_url}, timeout=10)
        if res.status_code == 200:
            data = res.json()
            st.session_state.repo_id  = data["repo_id"]
            st.session_state.repo_url = github_url
            st.session_state.history  = []

            if data["status"] == "done":
                st.success("✅ Already indexed! Ask questions below.")
            else:
                # poll until done
                progress_text = st.empty()
                bar = st.progress(0)
                dots = 0
                while True:
                    time.sleep(2)
                    dots = (dots + 1) % 4
                    poll = requests.get(f"{API_URL}/status/{data['repo_id']}", timeout=5).json()
                    s = poll["status"]

                    if s == "done":
                        bar.progress(100)
                        progress_text.success("✅ Indexed! Ask questions below.")
                        break
                    elif s.startswith("error"):
                        progress_text.error(f"❌ {s}")
                        st.session_state.repo_id = None
                        break
                    else:
                        # animate progress bar to show activity
                        bar.progress(min(95, (dots * 5) % 95 + 10))
                        progress_text.info(f"⏳ Indexing{'.' * (dots+1)} (big repos take 1-2 mins)")
        else:
            st.error(f"Error: {res.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure `uvicorn main:app --reload` is running.")

# ── Step 2: Ask questions ──────────────────────────────────────────────────────
if st.session_state.repo_id:
    st.divider()
    st.subheader("Step 2 — Ask questions about the repo")
    st.caption(f"📦 `{st.session_state.repo_url}`")

    st.markdown("**Try asking:**")
    example_cols = st.columns(3)
    examples = [
        "How does authentication work?",
        "Where is the database configured?",
        "What does the main entry point do?",
    ]
    for col, example in zip(example_cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state["prefill"] = example

    question = st.chat_input("Ask anything about the codebase…")

    if "prefill" in st.session_state:
        question = st.session_state.pop("prefill")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    res = requests.post(
                        f"{API_URL}/ask",
                        json={"repo_id": st.session_state.repo_id, "question": question},
                        timeout=60,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.markdown(data["answer"])

                        with st.expander(f"📎 Sources ({len(data['citations'])} files)"):
                            for c in data["citations"]:
                                score = c["relevance_score"]
                                filled = max(0, int(score * 10))
                                bar = "█" * filled + "░" * (10 - filled)
                                st.markdown(f"**`{c['file']}:{c['line']}`** — relevance: {bar} {score}")
                                st.caption(c["preview"])
                                st.divider()

                        st.session_state.history.append({
                            "question": question,
                            "answer":   data["answer"],
                        })
                    else:
                        st.error(f"Error: {res.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend.")

    if st.session_state.history:
        st.divider()
        st.subheader("Chat history")
        for item in reversed(st.session_state.history[:-1]):
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.write(item["answer"])

else:
    st.info("👆 Index a repo first to start asking questions.")

with st.sidebar:
    st.header("About")
    st.markdown("""
**Codebase QA** indexes any GitHub repo and lets you ask natural language questions about the code.

**How it works:**
1. Clones the repo
2. Splits code into chunks
3. Embeds chunks locally (free)
4. Stores in ChromaDB
5. On each question: finds relevant chunks → sends to Groq LLM → returns answer with citations

**Stack:**
- FastAPI backend
- sentence-transformers (local, free)
- ChromaDB (local, free)
- Groq LLM (free API)
- Streamlit frontend
    """)

    if st.session_state.repo_id:
        st.divider()
        if st.button("🗑️ Clear & index new repo"):
            st.session_state.repo_id  = None
            st.session_state.repo_url = None
            st.session_state.history  = []
            st.rerun()