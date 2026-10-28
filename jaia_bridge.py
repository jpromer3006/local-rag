#!/usr/bin/env python3
"""JAIA Bridge — lightweight HTTP API for OpenClaw to query the ChromaDB RAG.

Exposes a /query endpoint on port 8502 that accepts a question,
retrieves top-k matching chunks from ChromaDB, and returns JSON results.

Usage:
    cd ~/LocalRAG && source venv/bin/activate && python jaia_bridge.py

Endpoints:
    POST /query   {"question": "...", "top_k": 5}   → JSON results
    GET  /health  → {"status": "ok", "chunks": <count>}
"""

from __future__ import annotations

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# --- Configuration -----------------------------------------------------------

COLLECTION_NAME = "local_rag"
EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = str(Path(__file__).parent / "chroma_db")
DEFAULT_TOP_K = 5
PORT = 8502


# --- ChromaDB setup ----------------------------------------------------------

def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# --- HTTP Handler ------------------------------------------------------------

class BridgeHandler(BaseHTTPRequestHandler):
    collection = None

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            count = self.collection.count() if self.collection else 0
            self._send_json(200, {"status": "ok", "chunks": count})
        else:
            self._send_json(404, {"error": "Not found. Use POST /query or GET /health"})

    def do_POST(self) -> None:
        if self.path != "/query":
            self._send_json(404, {"error": "Not found. Use POST /query"})
            return

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": "Empty request body"})
            return

        try:
            body = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        question = body.get("question", "").strip()
        if not question:
            self._send_json(400, {"error": "Missing 'question' field"})
            return

        top_k = int(body.get("top_k", DEFAULT_TOP_K))

        # Query ChromaDB
        if not self.collection or self.collection.count() == 0:
            self._send_json(200, {
                "question": question,
                "results": [],
                "message": "No documents indexed yet.",
            })
            return

        n = min(top_k, self.collection.count())
        results = self.collection.query(query_texts=[question], n_results=n)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
            formatted.append({
                "rank": i + 1,
                "source": meta.get("source", "unknown"),
                "chunk_idx": meta.get("chunk_idx", -1),
                "distance": round(dist, 4),
                "text": doc[:2000],
            })

        self._send_json(200, {
            "question": question,
            "top_k": n,
            "total_chunks": self.collection.count(),
            "results": formatted,
        })

    def log_message(self, format, *args) -> None:
        print(f"[JAIA Bridge] {args[0]}" if args else "")


# --- Main --------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("JAIA Bridge — ChromaDB RAG API for OpenClaw")
    print(f"Loading embedding model: {EMBED_MODEL}")
    print("=" * 60)

    collection = get_collection()
    BridgeHandler.collection = collection

    print(f"ChromaDB collection '{COLLECTION_NAME}': {collection.count()} chunks")
    print(f"Starting server on http://127.0.0.1:{PORT}")
    print(f"  POST /query  {{\"question\": \"...\", \"top_k\": 5}}")
    print(f"  GET  /health")
    print()

    server = HTTPServer(("127.0.0.1", PORT), BridgeHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down JAIA Bridge.")
        server.server_close()


if __name__ == "__main__":
    main()
