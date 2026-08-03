"""
rag.py
Core RAG engine: embedding generation, persistent vector storage (ChromaDB),
similarity-based retrieval, and answer generation via a local Ollama model.

This module intentionally knows nothing about Streamlit — it is a plain
Python API so it can be tested or reused (e.g. from a CLI) independently
of the UI layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import requests
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import Settings
from ingest import process_directory
from prompts import build_prompt
from utils import get_logger


class OllamaConnectionError(Exception):
    """Raised when the local Ollama server cannot be reached."""


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class RAGEngine:
    """
    Encapsulates the full RAG lifecycle: indexing documents into a
    persistent Chroma collection, and answering questions by retrieving
    relevant chunks and generating a grounded response with Ollama.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("rag", settings.logs_dir, settings.log_level)

        self._embedder: Optional[SentenceTransformer] = None  # lazy-loaded

        try:
            self._client = chromadb.PersistentClient(
                path=str(settings.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception as exc:
            self.logger.exception("Failed to initialize Chroma persistent client")
            raise RuntimeError(f"Vector database could not be opened: {exc}") from exc

        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazily load the sentence-transformer model (expensive to construct)."""
        if self._embedder is None:
            try:
                self._embedder = SentenceTransformer(self.settings.embedding_model)
            except Exception as exc:
                self.logger.exception("Failed to load embedding model")
                raise EmbeddingError(
                    f"Could not load embedding model '{self.settings.embedding_model}': {exc}"
                ) from exc
        return self._embedder

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            vectors = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return vectors.tolist()
        except Exception as exc:
            self.logger.exception("Embedding generation failed")
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def _existing_file_hashes(self) -> set:
        """Return the set of file_hash values already present in the collection."""
        try:
            data = self._collection.get(include=["metadatas"])
        except Exception:
            return set()
        hashes = set()
        for meta in data.get("metadatas", []) or []:
            if meta and "file_hash" in meta:
                hashes.add(meta["file_hash"])
        return hashes

    def index_documents(self, force_reindex: bool = False) -> Dict:
        """
        Walk the data directory, load/chunk all supported documents, and
        embed + store any chunks not already present (duplicate detection
        via content hash), unless force_reindex clears the collection first.

        Returns a summary dict: {"indexed_files": [...], "skipped_files": [...],
        "errors": [...], "new_chunks": int}
        """
        if force_reindex:
            self.clear_database()

        existing_hashes = self._existing_file_hashes()
        summary = {"indexed_files": [], "skipped_files": [], "errors": [], "new_chunks": 0}

        for result in process_directory(self.settings.data_dir, self.settings):
            if result["status"] == "error":
                summary["errors"].append({"filename": result["filename"], "error": result["error"]})
                continue

            chunks = result["chunks"]
            if not chunks:
                continue

            file_hash = chunks[0]["metadata"]["file_hash"]
            if file_hash in existing_hashes:
                summary["skipped_files"].append(result["filename"])
                continue

            self._store_chunks(chunks)
            summary["indexed_files"].append(result["filename"])
            summary["new_chunks"] += len(chunks)
            existing_hashes.add(file_hash)

        self.logger.info(
            f"Indexing complete: {len(summary['indexed_files'])} new file(s), "
            f"{len(summary['skipped_files'])} duplicate(s) skipped, "
            f"{len(summary['errors'])} error(s)."
        )
        return summary

    def _store_chunks(self, chunks: List[Dict]) -> None:
        """Embed and upsert a batch of chunks into the Chroma collection."""
        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        embeddings = self.embed_texts(texts)

        try:
            self._collection.upsert(
                ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
            )
        except Exception as exc:
            self.logger.exception("Failed to write chunks to Chroma")
            raise RuntimeError(f"Vector database write failed: {exc}") from exc

    def clear_database(self) -> None:
        """Delete and recreate the collection (used by 'Delete database' / force reindex)."""
        try:
            self._client.delete_collection(self.settings.collection_name)
        except Exception:
            pass  # collection may not exist yet
        self._collection = self._client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info("Vector database cleared.")

    def collection_stats(self) -> Dict:
        """Return basic stats about the current collection for the UI."""
        try:
            count = self._collection.count()
        except Exception:
            count = 0
        return {"chunk_count": count, "collection_name": self.settings.collection_name}

    # ------------------------------------------------------------------ #
    # Retrieval + generation
    # ------------------------------------------------------------------ #

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve the top-k most similar chunks to the question."""
        k = top_k or self.settings.top_k
        if self._collection.count() == 0:
            return []

        query_embedding = self.embed_texts([question])[0]
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            self.logger.exception("Chroma query failed")
            raise RuntimeError(f"Retrieval failed: {exc}") from exc

        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            chunks.append({"text": text, "metadata": meta, "distance": dist})
        return chunks

    def check_ollama_available(self) -> bool:
        """Ping the Ollama server to verify it's reachable before generating."""
        try:
            resp = requests.get(f"{self.settings.ollama_host}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def generate_answer(
        self,
        question: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Full RAG turn: retrieve relevant chunks, build a grounded prompt,
        and call Ollama to generate an answer.

        Returns {"answer": str, "sources": [...], "used_context": bool}.
        """
        if not self.check_ollama_available():
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.settings.ollama_host}. "
                "Make sure 'ollama serve' is running."
            )

        chunks = self.retrieve(question, top_k=top_k)
        if not chunks:
            return {
                "answer": (
                    "I don't have any indexed documents to answer from yet. "
                    "Please upload and index documents first."
                ),
                "sources": [],
                "used_context": False,
            }

        prompt = build_prompt(question, chunks, history=history)
        model_name = model or self.settings.ollama_model
        temp = temperature if temperature is not None else self.settings.temperature

        try:
            response = requests.post(
                f"{self.settings.ollama_host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temp},
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            self.logger.exception("Ollama generation request failed")
            raise OllamaConnectionError(f"Ollama generation failed: {exc}") from exc

        answer_text = data.get("response", "").strip()
        if not answer_text:
            answer_text = "The model returned an empty response. Try rephrasing your question."

        sources = [
            {
                "filename": c["metadata"].get("filename"),
                "page": c["metadata"].get("page"),
                "distance": c["distance"],
            }
            for c in chunks
        ]

        return {"answer": answer_text, "sources": sources, "used_context": True}
