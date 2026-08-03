"""
streamlit_app.py
Streamlit UI for the local RAG Document Q&A system.

Run with: streamlit run streamlit_app.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import requests
import streamlit as st

from config import settings
from rag import RAGEngine, OllamaConnectionError, EmbeddingError

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------- #
# Cached / session-persistent engine
# ---------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_engine() -> RAGEngine:
    return RAGEngine(settings)


engine = get_engine()

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question", "answer", "sources"}


def list_available_ollama_models() -> list[str]:
    """Query Ollama for locally installed models, falling back to the default."""
    try:
        resp = requests.get(f"{settings.ollama_host}/api/tags", timeout=3)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        names = [t["name"] for t in tags]
        return names or [settings.ollama_model]
    except Exception:
        return [settings.ollama_model]


# ---------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------- #

with st.sidebar:
    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF / TXT / MD / DOCX",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        saved = 0
        for uf in uploaded_files:
            dest = Path(settings.data_dir) / uf.name
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved += 1
        st.success(f"Saved {saved} file(s) to the data folder. Click 'Re-index' to embed them.")

    col_a, col_b = st.columns(2)
    with col_a:
        reindex_clicked = st.button("🔄 Re-index", use_container_width=True)
    with col_b:
        force_clicked = st.button("♻️ Full re-index", use_container_width=True)

    if reindex_clicked or force_clicked:
        with st.spinner("Indexing documents..."):
            try:
                summary = engine.index_documents(force_reindex=force_clicked)
                st.success(
                    f"Indexed {len(summary['indexed_files'])} new file(s), "
                    f"skipped {len(summary['skipped_files'])} duplicate(s)."
                )
                if summary["errors"]:
                    for err in summary["errors"]:
                        st.warning(f"{err['filename']}: {err['error']}")
            except (RuntimeError, EmbeddingError) as exc:
                st.error(f"Indexing failed: {exc}")

    stats = engine.collection_stats()
    st.caption(f"Vector store: **{stats['chunk_count']}** chunks indexed.")

    if st.button("🗑️ Delete database", use_container_width=True):
        engine.clear_database()
        st.warning("Vector database cleared. Re-index to rebuild it.")
        st.rerun()

    st.divider()
    st.header("⚙️ Model settings")

    available_models = list_available_ollama_models()
    default_index = (
        available_models.index(settings.ollama_model)
        if settings.ollama_model in available_models
        else 0
    )
    selected_model = st.selectbox("Ollama model", available_models, index=default_index)

    temperature = st.slider("Temperature", 0.0, 1.0, float(settings.temperature), 0.05)
    top_k = st.slider("Top-k retrieved chunks", 1, 10, int(settings.top_k))

    st.divider()
    st.header("✂️ Chunking (applies on next index)")
    st.caption(f"Current: chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}")
    st.caption("Edit config.yaml or .env to change these, then run a full re-index.")

    st.divider()
    if st.button("🧹 Clear chat history", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ---------------------------------------------------------------------- #
# Main chat area
# ---------------------------------------------------------------------- #

st.title("📄 RAG Document Q&A")
st.caption("Ask questions about your own documents. Everything runs locally via Ollama.")

if not engine.check_ollama_available():
    st.error(
        f"⚠️ Cannot reach Ollama at `{settings.ollama_host}`. "
        "Start it with `ollama serve` and make sure the model is pulled "
        f"(`ollama pull {selected_model}`)."
    )

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["sources"]:
            with st.expander("📚 Sources"):
                for s in turn["sources"]:
                    page_str = f", page {s['page']}" if s["page"] not in (None, "N/A") else ""
                    st.markdown(f"- **{s['filename']}**{page_str} (distance: {s['distance']:.3f})")

question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = engine.generate_answer(
                    question,
                    model=selected_model,
                    temperature=temperature,
                    top_k=top_k,
                    history=st.session_state.history,
                )
                st.write(result["answer"])
                if result["sources"]:
                    with st.expander("📚 Sources"):
                        for s in result["sources"]:
                            page_str = f", page {s['page']}" if s["page"] not in (None, "N/A") else ""
                            st.markdown(f"- **{s['filename']}**{page_str} (distance: {s['distance']:.3f})")

                st.session_state.history.append(
                    {"question": question, "answer": result["answer"], "sources": result["sources"]}
                )
            except OllamaConnectionError as exc:
                st.error(str(exc))
            except (RuntimeError, EmbeddingError) as exc:
                st.error(f"Something went wrong: {exc}")
