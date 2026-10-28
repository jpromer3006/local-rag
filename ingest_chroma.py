#!/usr/bin/env python3
"""Ingest a directory of files into the ChromaDB collection used by app.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# Reuse core utilities from rag_core
from rag_core import (
    chunk_text,
    extract_text,
    file_sha256,
    iter_candidate_files,
    load_config,
    safe_print,
)

# ── Defaults (must match app.py) ──────────────────────────────────────────────
COLLECTION_NAME = "local_rag"
EMBED_MODEL     = "all-MiniLM-L6-v2"
DB_PATH         = str(Path(__file__).parent / "chroma_db")
CONFIG_PATH     = Path(__file__).parent / "rag_config.json"

ALLOWED_EXTENSIONS = [
    ".txt", ".md", ".csv", ".json", ".pdf", ".docx", ".pptx",
    ".xlsx", ".xls", ".py", ".js", ".ts", ".log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest files into ChromaDB for app.py")
    parser.add_argument(
        "--source", required=True, action="append",
        help="Source directory (repeatable). Use quotes for paths with spaces.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete and recreate the collection before ingesting.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Override chunk size (characters). Default from rag_config.json.",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=None,
        help="Override chunk overlap (characters). Default from rag_config.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = load_config(CONFIG_PATH)
    chunk_size    = args.chunk_size    or int(cfg.get("chunk_size",    1200))
    chunk_overlap = args.chunk_overlap or int(cfg.get("chunk_overlap", 180))

    source_paths = [Path(s).expanduser().resolve() for s in args.source]
    exclude_paths: list[Path] = []

    # ── ChromaDB client & collection ─────────────────────────────────────────
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    if args.rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            safe_print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    safe_print(f"Collection '{COLLECTION_NAME}' ready (UUID {collection.id}). "
               f"Current chunks: {collection.count()}")

    # ── Ingest ────────────────────────────────────────────────────────────────
    total_files   = 0
    total_chunks  = 0
    skipped_files = 0
    errors: list[dict] = []

    for path in iter_candidate_files(
        source_paths=source_paths,
        exclude_paths=exclude_paths,
        allowed_extensions=ALLOWED_EXTENSIONS,
        max_files=0,
    ):
        total_files += 1
        try:
            text = extract_text(path)
            if not text.strip():
                skipped_files += 1
                continue

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            if not chunks:
                skipped_files += 1
                continue

            digest = file_sha256(path)
            ids       = [f"{digest}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": path.name, "path": str(path), "chunk_idx": i}
                         for i in range(len(chunks))]

            # Upsert: delete old chunks for this file, then add new ones
            try:
                existing = collection.get(where={"path": str(path)})
                if existing["ids"]:
                    collection.delete(ids=existing["ids"])
            except Exception:
                pass

            collection.add(documents=chunks, ids=ids, metadatas=metadatas)
            total_chunks += len(chunks)
            safe_print(f"  + {path.name}  ({len(chunks)} chunks)")

        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
            safe_print(f"  ! ERROR {path.name}: {exc}")

    safe_print(
        f"\nDone. Files processed: {total_files}, "
        f"chunks added: {total_chunks}, "
        f"skipped (empty): {skipped_files}, "
        f"errors: {len(errors)}"
    )
    safe_print(f"Collection now has {collection.count()} total chunks.")
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        safe_print("Interrupted.")
        raise SystemExit(130)
