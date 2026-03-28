# Local RAG for Centralized Files (Ollama + SQLite FTS)

This project now includes a terminal-first local RAG flow that indexes your centralized files and answers questions with `llama3.1` through Ollama.

## What Was Chosen and Why

### Strategy Review: FAISS vs SQLite FTS5 vs Topic Modeling

1. **FAISS (vector DB)**
- Pros: Strong semantic retrieval for paraphrased queries.
- Cons: Requires an embedding model and extra runtime dependencies; more setup and maintenance.

2. **SQLite FTS5 (chosen for this build)**
- Pros: Built into Python/SQLite, fast local keyword retrieval, easy to back up and inspect, zero external DB service.
- Cons: Less semantic than vector embeddings when wording differs heavily.

3. **Topic Modeling (LDA/NMF/BERTopic)**
- Good for corpus exploration and labeling themes.
- Not ideal as primary retrieval engine for question answering by itself.

### Decision

For your immediate local workflow and large mixed files, this build uses **SQLite FTS5** as the retrieval database and **Ollama `llama3.1:8b`** for answer generation. This is the fastest reliable setup with minimal moving parts.

If you want semantic retrieval later, we can add FAISS as a second-stage retriever.

## Files Added

- `rag_core.py`: shared indexing/retrieval/Ollama helpers
- `ingest_centralized.py`: incremental ingestion into SQLite FTS index
- `ask_rag.py`: query CLI (single question or interactive mode)
- `rag_config.json`: source paths, DB path, model, chunk settings
- `launch_local_rag.sh`: run ingest + interactive RAG in terminal

## Install

```bash
cd ~/LocalRAG
python3 -m pip install -r requirements.txt
```

## Ingest

```bash
cd ~/LocalRAG
python3 ingest_centralized.py --config rag_config.json --report latest_ingest_report.json
```

## Ask a Single Question

```bash
cd ~/LocalRAG
python3 ask_rag.py --config rag_config.json --question "What invoices mention Q4?" --show-sources
```

## Interactive Mode

```bash
cd ~/LocalRAG
python3 ask_rag.py --config rag_config.json --interactive --show-sources
```

## Desktop Launch (double-click)

A macOS launcher is created on your Desktop:
- `Launch Local RAG.command`

It runs:
1. model check (`llama3.1`)
2. incremental ingest
3. interactive RAG session

## Notes

- Default source preference is `~/CentralizedLibrary`, with fallback sources `~/Desktop` and `~/Documents`.
- You can edit `rag_config.json` anytime to include mounted volumes.
- Ingest is incremental by path and hash, replacing chunks when files change.
