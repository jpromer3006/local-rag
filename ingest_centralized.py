#!/usr/bin/env python3
"""Ingest centralized/local files into SQLite FTS index for local RAG."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from rag_core import (
    build_rel_path,
    category_for_path,
    chunk_text,
    db_stats,
    extract_text,
    file_sha256,
    iter_candidate_files,
    load_config,
    open_db,
    parse_common_args,
    resolve_paths,
    safe_print,
    upsert_file_chunks,
)


def parse_args() -> argparse.Namespace:
    parent = parse_common_args()
    parser = argparse.ArgumentParser(parents=[parent], description="Ingest files into local SQLite FTS index")
    parser.add_argument("--max-files", type=int, default=None, help="Override max files from config")
    parser.add_argument("--source", action="append", help="Add one source path override (repeatable)")
    parser.add_argument("--rebuild", action="store_true", help="Drop all indexed chunks before ingest")
    parser.add_argument("--report", default="ingest_report.json", help="Write JSON ingest report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config).expanduser())
    source_paths, exclude_paths, db_path = resolve_paths(cfg)

    if args.source:
        source_paths = [Path(s).expanduser() for s in args.source]

    allowed_extensions = cfg.get("allowed_extensions", [])
    chunk_size = int(cfg.get("chunk_size", 1200))
    chunk_overlap = int(cfg.get("chunk_overlap", 180))
    max_files = args.max_files if args.max_files is not None else int(cfg.get("max_files", 0))

    conn = open_db(db_path)

    if args.rebuild:
        conn.execute("DELETE FROM chunks")
        conn.commit()

    report = {
        "database": str(db_path),
        "source_paths": [str(p) for p in source_paths],
        "excluded_paths": [str(p) for p in exclude_paths],
        "processed_files": 0,
        "indexed_files": 0,
        "inserted_chunks": 0,
        "replaced_chunk_sets": 0,
        "errors": [],
    }

    for path in iter_candidate_files(
        source_paths=source_paths,
        exclude_paths=exclude_paths,
        allowed_extensions=allowed_extensions,
        max_files=max_files,
    ):
        report["processed_files"] += 1
        try:
            text = extract_text(path)
            if not text.strip():
                continue
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            if not chunks:
                continue
            digest = file_sha256(path)
            rel_path = build_rel_path(path, source_paths)
            st = path.stat()
            replaced, inserted = upsert_file_chunks(
                conn=conn,
                path=path,
                rel_path=rel_path,
                chunks=chunks,
                digest=digest,
                category=category_for_path(path),
                size=int(st.st_size),
                mtime=float(st.st_mtime),
            )
            report["indexed_files"] += 1
            report["inserted_chunks"] += inserted
            if replaced:
                report["replaced_chunk_sets"] += 1
            if report["processed_files"] % 100 == 0:
                safe_print(
                    f"Processed {report['processed_files']} files | "
                    f"Indexed {report['indexed_files']} | Chunks {report['inserted_chunks']}"
                )
        except Exception as exc:
            report["errors"].append({"path": str(path), "error": str(exc)})

    stats = db_stats(conn)
    report.update(stats)

    report_path = Path(args.report).expanduser()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    safe_print(json.dumps(report, indent=2))
    safe_print(f"Report written to: {report_path}")

    conn.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        safe_print("Interrupted by user.")
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
