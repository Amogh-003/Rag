"""
prompts.py
Prompt construction for the RAG pipeline. Keeping prompt templates in one
place makes it easy to iterate on prompt quality without touching
retrieval or generation logic.
"""

from __future__ import annotations

from typing import List, Dict


SYSTEM_INSTRUCTIONS = (
    "You are a precise, factual assistant that answers questions using ONLY "
    "the provided context extracted from the user's own documents. "
    "If the context does not contain enough information to answer confidently, "
    "say so explicitly instead of guessing. "
    "Always ground your answer in the given context and do not invent facts. "
    "When useful, refer to the source documents by name."
)


def format_context(chunks: List[Dict]) -> str:
    """
    Turn a list of retrieved chunk dicts (each with 'text' and metadata)
    into a numbered context block the LLM can cite by [Source N].
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "unknown")
        page = meta.get("page", None)
        page_str = f", page {page}" if page not in (None, "", "N/A") else ""
        lines.append(f"[Source {i}] ({filename}{page_str})\n{chunk['text']}")
    return "\n\n".join(lines)


def build_prompt(question: str, chunks: List[Dict], history: List[Dict] | None = None) -> str:
    """
    Build the final prompt sent to the LLM, combining system instructions,
    prior conversation turns (if any), retrieved context, and the new
    question.
    """
    context_block = format_context(chunks)

    history_block = ""
    if history:
        turns = []
        for turn in history[-4:]:  # keep only the last few turns to control prompt size
            turns.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
        history_block = "\n\n".join(turns)

    prompt_parts = [SYSTEM_INSTRUCTIONS]

    if history_block:
        prompt_parts.append(f"Previous conversation:\n{history_block}")

    prompt_parts.append(f"Context from documents:\n{context_block}")
    prompt_parts.append(
        f"Question: {question}\n\n"
        "Answer the question using the context above. Cite sources like "
        "[Source 1] where relevant. If the answer isn't in the context, "
        "say you don't have enough information."
    )

    return "\n\n---\n\n".join(prompt_parts)
