"""
app.py
Command-line entry point for the RAG system — useful for indexing
documents or asking one-off questions without launching Streamlit,
and for scripting/CI use.

Usage:
    python app.py index [--force]
    python app.py ask "What is the termination clause?"
    python app.py stats
"""

from __future__ import annotations

import argparse
import sys

from config import settings
from rag import RAGEngine, OllamaConnectionError, EmbeddingError


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG Document Q&A CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Index new documents in the data directory")
    index_parser = subparsers.choices["index"]
    index_parser.add_argument("--force", action="store_true", help="Clear and rebuild the whole index")

    ask_parser = subparsers.add_parser("ask", help="Ask a question against indexed documents")
    ask_parser.add_argument("question", type=str)

    subparsers.add_parser("stats", help="Show vector store statistics")

    args = parser.parse_args()
    engine = RAGEngine(settings)

    if args.command == "index":
        summary = engine.index_documents(force_reindex=args.force)
        print(f"Indexed files: {summary['indexed_files']}")
        print(f"Skipped (duplicate) files: {summary['skipped_files']}")
        print(f"New chunks: {summary['new_chunks']}")
        if summary["errors"]:
            print("Errors:")
            for err in summary["errors"]:
                print(f"  - {err['filename']}: {err['error']}")

    elif args.command == "ask":
        try:
            result = engine.generate_answer(args.question)
        except OllamaConnectionError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except EmbeddingError as exc:
            print(f"Embedding error: {exc}", file=sys.stderr)
            return 1

        print("\nAnswer:\n" + result["answer"])
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                page_str = f", page {s['page']}" if s["page"] not in (None, "N/A") else ""
                print(f"  - {s['filename']}{page_str}")

    elif args.command == "stats":
        stats = engine.collection_stats()
        print(f"Collection: {stats['collection_name']}")
        print(f"Chunks indexed: {stats['chunk_count']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
