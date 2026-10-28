import streamlit as st
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

# --- Load .env FIRST --------------------------------------------------------
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")


# --- macOS Keychain helper ---------------------------------------------------
def _keychain_get(service: str) -> str:
    """Read a secret from the macOS Keychain. Returns '' on failure."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


# --- Optional OCR ---
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# --- Document Parsers ---
import pdfplumber
from pptx import Presentation
from PIL import Image as PILImage
import pandas as pd

# --- Vector Store ---
import chromadb
from chromadb.utils import embedding_functions

# --- Ollama ---
import ollama

# --- Claude API ---
import anthropic

# ─── Config from .env ─────────────────────────────────────────────────────────
COLLECTION_NAME = "local_rag"
CHUNK_SIZE       = 400   # words per chunk
CHUNK_OVERLAP    = 50    # word overlap between chunks
TOP_K            = 5     # number of chunks to retrieve
LLM_MODEL        = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
EMBED_MODEL      = "all-MiniLM-L6-v2"
DB_PATH          = str(Path(__file__).parent / "chroma_db")
DEFAULT_MODE     = os.getenv("DEFAULT_MODE", "deep")
def _resolve_docs_dir() -> Path:
    env_val = os.getenv("RAG_DOCS_DIR", "")
    if env_val:
        p = Path(env_val).expanduser()
        if p.exists():
            return p
        p_space = Path(str(p) + " ")
        if p_space.exists():
            return p_space
        return p
    return Path.home() / "Desktop" / "Germany "

DOCS_DIR         = _resolve_docs_dir()
PROCESSED_DIR    = Path(os.getenv("RAG_PROCESSED_DIR", os.path.expanduser("~/Desktop/Germany_processed")))
# Keys: prefer macOS Keychain (openclaw_*), fall back to .env
ELEVENLABS_KEY   = _keychain_get("openclaw_elevenlabs") or os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ANTHROPIC_KEY    = _keychain_get("openclaw_anthropic") or os.getenv("ANTHROPIC_API_KEY", "")

SUPPORTED_TYPES = {
    "pdf":  "PDF", "pptx": "PowerPoint", "ppt":  "PowerPoint",
    "png":  "Image", "jpg":  "Image", "jpeg": "Image",
    "bmp":  "Image", "tiff": "Image", "txt":  "Text",
    "md":   "Markdown", "csv":  "CSV", "xlsx": "Excel", "xls":  "Excel",
}

# ─── JAIA System Prompt ───────────────────────────────────────────────────────
JAIA_SYSTEM_PROMPT = (
    "You are JAIA, an expert financial analyst assistant for the AdminAI LLC "
    "investment team. You have access to financial data for NYSE-listed "
    "infrastructure and engineering companies. Answer questions with specific "
    "numbers, comparisons, and investment insights. When greeting users, "
    "introduce yourself as JAIA. Always cite which company and time period "
    "data comes from."
)

# Conversational keywords — skip RAG for these
CONVERSATIONAL_KEYWORDS = {
    "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
    "who are you", "what can you do", "help", "what are you",
    "good morning", "good afternoon", "good evening", "how are you",
    "nice to meet", "greetings", "sup", "yo", "hola",
}

SIMILARITY_THRESHOLD = 0.3  # ChromaDB cosine distance; lower = more similar


# ─── Vector Store Init ────────────────────────────────────────────────────────
@st.cache_resource
def _get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


def get_chroma_client():
    return chromadb.PersistentClient(path=DB_PATH)


def get_collection_safe():
    ef = _get_embedding_function()
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


# ─── Parsers ──────────────────────────────────────────────────────────────────
def parse_pdf(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n\n".join(text)

def parse_pptx(path: str) -> str:
    prs = Presentation(path)
    text = []
    for i, slide in enumerate(prs.slides):
        slide_text = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = para.text.strip()
                    if line:
                        slide_text.append(line)
        if slide_text:
            text.append(f"[Slide {i+1}]\n" + "\n".join(slide_text))
    return "\n\n".join(text)

def parse_image(path: str) -> str:
    if not OCR_AVAILABLE:
        return "[Image OCR unavailable — install tesseract: brew install tesseract]"
    img = PILImage.open(path)
    return pytesseract.image_to_string(img)

def parse_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_xlsx(path: str) -> str:
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception:
        return ""
    parts = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None)
        rows = []
        for _, row in df.iterrows():
            cells = [str(c) if pd.notna(c) else "" for c in row]
            if any(c.strip() for c in cells):
                rows.append("\t".join(cells))
        if rows:
            parts.append(f"[Sheet: {sheet}]\n" + "\n".join(rows))
    return "\n\n".join(parts)

def parse_xls(path: str) -> str:
    try:
        xl = pd.ExcelFile(path, engine="xlrd")
    except Exception:
        return ""
    parts = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None)
        rows = []
        for _, row in df.iterrows():
            cells = [str(c) if pd.notna(c) else "" for c in row]
            if any(c.strip() for c in cells):
                rows.append("\t".join(cells))
        if rows:
            parts.append(f"[Sheet: {sheet}]\n" + "\n".join(rows))
    return "\n\n".join(parts)

def parse_file(path: str, ext: str) -> str:
    ext = ext.lower().lstrip(".")
    if ext == "pdf":
        return parse_pdf(path)
    elif ext in ("pptx", "ppt"):
        return parse_pptx(path)
    elif ext in ("png", "jpg", "jpeg", "bmp", "tiff"):
        return parse_image(path)
    elif ext in ("txt", "md", "csv"):
        return parse_text(path)
    elif ext == "xlsx":
        return parse_xlsx(path)
    elif ext == "xls":
        return parse_xls(path)
    return ""


# ─── Chunking ─────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    chunks, step = [], size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ─── Conversation Detection ──────────────────────────────────────────────────
def is_conversational(query: str) -> bool:
    """Detect greetings and conversational messages that don't need RAG."""
    q = query.lower().strip().rstrip("!?.,:;")
    # Exact or near-exact match
    if q in CONVERSATIONAL_KEYWORDS:
        return True
    # Check if query starts with a conversational phrase
    for kw in CONVERSATIONAL_KEYWORDS:
        if q.startswith(kw):
            return True
    # Very short messages (1-3 words) that aren't financial terms
    words = q.split()
    if len(words) <= 3:
        financial_signals = {"revenue", "income", "profit", "loss", "debt",
                             "eps", "margin", "ratio", "compare", "stock",
                             "price", "dividend", "asset", "equity", "cash",
                             "ebitda", "capex", "growth", "valuation", "pe"}
        if not any(w in financial_signals for w in words):
            return True
    return False


# ─── RAG Retrieval ────────────────────────────────────────────────────────────
def retrieve(query: str, collection, top_k: int = TOP_K):
    n = min(top_k, collection.count())
    if n == 0:
        return [], [], []
    results = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    return docs, metas, distances


# ─── Generation: Quick (Ollama) vs Deep (Claude API) ─────────────────────────
def generate_ollama(query: str, docs: list[str]) -> str:
    context = "\n\n---\n\n".join(docs)
    user_msg = f"Context:\n{context}\n\nQuestion: {query}"
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": JAIA_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    return resp["message"]["content"]


def generate_claude(query: str, docs: list[str]) -> str:
    if not ANTHROPIC_KEY or ANTHROPIC_KEY == "your_key_here":
        return "Claude API key not configured. Add your key to ~/LocalRAG/.env"
    context = "\n\n---\n\n".join(docs)
    user_msg = f"Context:\n{context}\n\nQuestion: {query}"
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=JAIA_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text


def generate_conversational(query: str, mode: str) -> str:
    """Handle greetings/conversational messages without RAG context."""
    if mode == "deep" and ANTHROPIC_KEY and ANTHROPIC_KEY != "your_key_here":
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=JAIA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        return response.content[0].text
    else:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": JAIA_SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ],
        )
        return resp["message"]["content"]


# ─── ElevenLabs TTS ──────────────────────────────────────────────────────────
def text_to_speech(text: str) -> bytes | None:
    """Send text to ElevenLabs and return audio bytes, or None on failure."""
    if not ELEVENLABS_KEY or ELEVENLABS_KEY == "your_key_here":
        return None
    import urllib.request
    import json

    # Limit to 500 chars
    truncated = text[:500]
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE}"
    payload = json.dumps({
        "text": truncated,
        "model_id": "eleven_monolingual_v1",
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "xi-api-key": ELEVENLABS_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except Exception as e:
        st.warning(f"Voice generation failed: {e}")
        return None


# ─── Auto-preprocess uploaded XLS files ──────────────────────────────────────
def preprocess_and_ingest_file(filepath: Path, collection) -> int:
    """Run preprocess_xls.py on a single file, then ingest the output into ChromaDB."""
    script = Path(__file__).parent / "preprocess_xls.py"
    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [sys.executable, str(script), "--file", str(filepath), "--out-dir", str(out_dir)],
        capture_output=True, text=True, cwd=str(Path(__file__).parent),
    )

    # Find the processed .txt file
    chunks_added = 0
    for txt_file in out_dir.iterdir():
        if txt_file.suffix == ".txt" and txt_file.stat().st_mtime > filepath.stat().st_mtime - 60:
            text = txt_file.read_text(encoding="utf-8")
            if text.strip():
                chunks = chunk_text(text)
                collection.add(
                    documents=chunks,
                    ids=[str(uuid.uuid4()) for _ in chunks],
                    metadatas=[{"source": filepath.name, "chunk_idx": j, "processed": "true"}
                               for j in range(len(chunks))],
                )
                chunks_added = len(chunks)
    return chunks_added


# ─── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JAIA - Local RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {font-size:2rem; font-weight:700; color:#1a1a2e;}
    .sub-header  {color:#555; font-size:.9rem; margin-top:-10px;}
    .source-box  {background:#f7f7f7; border-left:3px solid #4a90e2;
                  padding:8px 12px; border-radius:4px; font-size:.8rem;
                  font-family:monospace; white-space:pre-wrap;}
    .badge       {display:inline-block; padding:2px 8px; border-radius:12px;
                  font-size:.75rem; font-weight:600;}
    .badge-green {background:#d4edda; color:#155724;}
    .badge-blue  {background:#cce5ff; color:#004085;}
    .badge-purple{background:#e8daef; color:#6c3483;}
</style>
""", unsafe_allow_html=True)

client, collection = get_collection_safe()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 JAIA — Local RAG")

    # Analysis mode toggle
    mode_options = ["deep", "quick"]
    mode_labels = {"deep": "Deep (Claude API)", "quick": "Quick (Ollama)"}
    default_idx = 0 if DEFAULT_MODE == "deep" else 1
    analysis_mode = st.radio(
        "Analysis Mode",
        mode_options,
        index=default_idx,
        format_func=lambda x: mode_labels[x],
        horizontal=True,
    )

    mode_badge = (
        '<span class="badge badge-purple">Claude API</span>'
        if analysis_mode == "deep"
        else f'<span class="badge badge-green">Ollama {LLM_MODEL}</span>'
    )
    st.markdown(
        f'{mode_badge} '
        f'<span class="badge badge-blue">{collection.count()} chunks</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### Upload Documents")
    uploaded = st.file_uploader(
        "PDF, PPTX, Images, Text, Excel",
        type=list(SUPPORTED_TYPES.keys()),
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("Index Documents", use_container_width=True, type="primary"):
            bar = st.progress(0, text="Starting...")
            total = len(uploaded)
            _, write_col = get_collection_safe()

            for i, f in enumerate(uploaded):
                bar.progress(i / total, text=f"Parsing {f.name}...")
                ext = Path(f.name).suffix
                is_excel = ext.lower().lstrip(".") in ("xls", "xlsx")

                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name

                try:
                    if is_excel:
                        # Also save to DOCS_DIR for future reference
                        save_path = DOCS_DIR / f.name
                        DOCS_DIR.mkdir(parents=True, exist_ok=True)
                        save_path.write_bytes(f.getvalue())

                        # Auto-preprocess and ingest
                        st.info(f"Indexing {f.name}...")
                        n = preprocess_and_ingest_file(Path(tmp_path), write_col)
                        if n > 0:
                            st.success(f"**{f.name}** -> {n} structured chunks")
                        else:
                            # Fallback: raw parse
                            text = parse_file(tmp_path, ext)
                            if text.strip():
                                chunks = chunk_text(text)
                                write_col.add(
                                    documents=chunks,
                                    ids=[str(uuid.uuid4()) for _ in chunks],
                                    metadatas=[{"source": f.name, "chunk_idx": j}
                                               for j in range(len(chunks))],
                                )
                                st.success(f"**{f.name}** -> {len(chunks)} raw chunks")
                            else:
                                st.warning(f"No text extracted from **{f.name}**")
                    else:
                        text = parse_file(tmp_path, ext)
                        if text.strip():
                            chunks = chunk_text(text)
                            write_col.add(
                                documents=chunks,
                                ids=[str(uuid.uuid4()) for _ in chunks],
                                metadatas=[{"source": f.name, "chunk_idx": j}
                                           for j in range(len(chunks))],
                            )
                            st.success(f"**{f.name}** -> {len(chunks)} chunks")
                        else:
                            st.warning(f"No text extracted from **{f.name}**")
                finally:
                    os.unlink(tmp_path)

                bar.progress((i + 1) / total, text=f"Done: {f.name}")

            bar.empty()
            st.rerun()

    st.divider()

    with st.expander("Settings"):
        TOP_K = st.slider("Chunks to retrieve", 1, 10, TOP_K)
        CHUNK_SIZE = st.slider("Chunk size (words)", 100, 800, CHUNK_SIZE, step=50)

    if collection.count() > 0:
        if st.button("Clear Knowledge Base", use_container_width=True):
            client.delete_collection(COLLECTION_NAME)
            _get_embedding_function.clear()
            st.success("Knowledge base cleared.")
            st.rerun()

    if not OCR_AVAILABLE:
        st.info("Image OCR off. Run: `brew install tesseract`")

# ── Main Chat ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🧠 JAIA — Financial RAG Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AdminAI LLC Investment Analysis | Powered by Claude + LLaMA 3.1</p>',
    unsafe_allow_html=True,
)

if collection.count() == 0:
    st.info("Upload financial documents in the sidebar to get started.", icon="📂")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("content"):
            # TTS button for past messages
            if st.button("🔊", key=f"tts_hist_{idx}", help="Listen to this response"):
                with st.spinner("Generating voice..."):
                    audio = text_to_speech(msg["content"])
                if audio:
                    st.audio(audio, format="audio/mpeg")
        if msg.get("sources"):
            with st.expander("Retrieved Sources"):
                for src in msg["sources"]:
                    label = src["meta"].get("source", "unknown")
                    cidx = src["meta"].get("chunk_idx", "?")
                    st.markdown(
                        f"**{label}** chunk {cidx}\n"
                        f'<div class="source-box">{src["text"][:300]}...</div>',
                        unsafe_allow_html=True,
                    )

# Input
if prompt := st.chat_input("Ask JAIA anything about your financial data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sources = []

        # --- STEP 5: Conversation detection ---
        if is_conversational(prompt):
            with st.spinner("Thinking..."):
                answer = generate_conversational(prompt, analysis_mode)
        elif collection.count() == 0:
            answer = "No documents indexed yet. Please upload financial files first."
        else:
            with st.spinner("Retrieving & analyzing..."):
                docs, metas, distances = retrieve(prompt, collection, TOP_K)

                # Filter by similarity threshold (cosine distance)
                filtered_docs = []
                filtered_metas = []
                for d, m, dist in zip(docs, metas, distances):
                    if dist <= (1 - SIMILARITY_THRESHOLD):
                        filtered_docs.append(d)
                        filtered_metas.append(m)

                if not filtered_docs:
                    # No relevant results — treat as conversational
                    answer = generate_conversational(prompt, analysis_mode)
                else:
                    if analysis_mode == "deep":
                        answer = generate_claude(prompt, filtered_docs)
                    else:
                        answer = generate_ollama(prompt, filtered_docs)
                    sources = [{"text": d, "meta": m}
                               for d, m in zip(filtered_docs, filtered_metas)]

        st.markdown(answer)

        # TTS button
        tts_key = f"tts_{len(st.session_state.messages)}"
        if st.button("🔊 Listen", key=tts_key, help="Listen to this response"):
            with st.spinner("Generating voice..."):
                audio = text_to_speech(answer)
            if audio:
                st.audio(audio, format="audio/mpeg")
            elif not ELEVENLABS_KEY or ELEVENLABS_KEY == "your_key_here":
                st.info("Add your ElevenLabs API key to ~/LocalRAG/.env to enable voice.")

        if sources:
            with st.expander("Retrieved Sources"):
                for src in sources:
                    label = src["meta"].get("source", "unknown")
                    cidx = src["meta"].get("chunk_idx", "?")
                    st.markdown(
                        f"**{label}** chunk {cidx}\n"
                        f'<div class="source-box">{src["text"][:300]}...</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
