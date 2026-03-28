#!/usr/bin/env python3
"""Query local SQLite FTS index and answer with local Ollama model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_core import (
    build_prompt,
    db_stats,
    load_config,
    open_db,
    parse_common_args,
    resolve_paths,
    run_ollama,
    safe_print,
    search_chunks,
)


def parse_args() -> argparse.Namespace:
    parent = parse_common_args()
    parser = argparse.ArgumentParser(parents=[parent], description="Ask questions over local indexed docs")
    parser.add_argument("--question", help="Single question mode")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k")
    parser.add_argument("--model", default=None, help="Override ollama model")
    parser.add_argument("--interactive", action="store_true", help="Interactive loop mode")
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved chunk snippets")
    return parser.parse_args()


def answer_once(conn, model: str, question: str, top_k: int, show_sources: bool) -> int:
    chunks = search_chunks(conn, question, top_k=top_k)
    if not chunks:
        safe_print("No relevant chunks found. Try broader keywords.")
        return 0

    prompt = build_prompt(question, chunks)
    answer = run_ollama(model, prompt)

    safe_print("\nAnswer:\n")
    safe_print(answer)
    safe_print("\nSources:")
    for i, c in enumerate(chunks, start=1):
        safe_print(f"[{i}] {c['path']} (chunk {c['chunk_index']}, score={c['score']:.4f})")
        if show_sources:
            snippet = str(c["content"])[:280].replace("\n", " ")
            safe_print(f"    {snippet}...")
    return 0


def interactive_loop(conn, model: str, top_k: int, show_sources: bool) -> int:
    safe_print("Local RAG interactive mode. Type 'exit' to quit.")
    while True:
        q = input("\nQuestion> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            return 0
        try:
            answer_once(conn, model=model, question=q, top_k=top_k, show_sources=show_sources)
        except Exception as exc:
            safe_print(f"Error: {exc}")


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config).expanduser())
    _, _, db_path = resolve_paths(cfg)

    model = args.model or str(cfg.get("ollama_model", "llama3.1:8b"))
    top_k = args.top_k if args.top_k is not None else int(cfg.get("top_k", 6))

    conn = open_db(db_path)
    stats = db_stats(conn)
    safe_print(json.dumps({"database": str(db_path), **stats, "model": model}, indent=2))

    if args.interactive:
        code = interactive_loop(conn, model=model, top_k=top_k, show_sources=args.show_sources)
        conn.close()
        return code

    if args.question:
        code = answer_once(conn, model=model, question=args.question, top_k=top_k, show_sources=args.show_sources)
        conn.close()
        return code

    safe_print("Provide --question or use --interactive")
    conn.close()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
