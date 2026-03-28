#!/bin/zsh
set -euo pipefail

cd "$HOME/LocalRAG"

if [[ -d "venv" ]]; then
  source "venv/bin/activate"
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install from https://ollama.com/download"
  exit 1
fi

echo "Checking Ollama model availability..."
if ! ollama list | grep -q "llama3.1"; then
  echo "Model llama3.1 not found locally. Pulling now..."
  ollama pull llama3.1:8b
fi

echo "Indexing documents (incremental)..."
python3 ingest_centralized.py --config rag_config.json --report latest_ingest_report.json

echo "Launching interactive local RAG..."
python3 ask_rag.py --config rag_config.json --interactive --show-sources
