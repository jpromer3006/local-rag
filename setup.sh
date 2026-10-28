#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Local RAG – One-time setup
#  Run once: bash ~/LocalRAG/setup.sh
# ─────────────────────────────────────────────────────────────

set -e
DIR="$HOME/LocalRAG"
cd "$DIR"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║       Local RAG – Setup              ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 1. Tesseract (optional, for image OCR)
if ! command -v tesseract &>/dev/null; then
    echo "📦 Installing Tesseract OCR via Homebrew…"
    if command -v brew &>/dev/null; then
        brew install tesseract
    else
        echo "⚠️  Homebrew not found – skipping Tesseract (image OCR will be disabled)."
    fi
else
    echo "✅ Tesseract already installed."
fi

# 2. Python virtual environment
echo ""
echo "🐍 Creating Python virtual environment…"
python3 -m venv "$DIR/venv"
source "$DIR/venv/bin/activate"

# 3. Python packages
echo ""
echo "📦 Installing Python packages…"
pip install --upgrade pip -q
pip install -r "$DIR/requirements.txt"

# 4. Make launcher executable
chmod +x "$HOME/Desktop/LocalRAG.command" 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  ✅  Setup complete!                  ║"
echo "║  Double-click LocalRAG on Desktop     ║"
echo "╚══════════════════════════════════════╝"
echo ""
