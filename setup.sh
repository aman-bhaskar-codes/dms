#!/bin/bash
set -e
echo "🚗 DMS V3 Agentic Setup"
echo "======================="

# Create directory tree
mkdir -p perception alerts calibration dashboard/pyqt dashboard/web \
         memory agents/tools voice_agent data/driver_profiles \
         data/reports data/chroma_db

# Init files
for d in perception alerts calibration dashboard dashboard/pyqt dashboard/web memory agents agents/tools voice_agent; do
  touch "$d/__init__.py"
done

# Virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies (this might take a few minutes)..."
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found!"
fi

echo ""
echo "✅ DMS V3 setup complete!"
echo ""
echo "Optional — Ollama AI Setup for Memory & Voice:"
echo "  curl -fsSL https://ollama.ai/install.sh | sh"
echo "  ollama pull llama3.2:3b"
echo "  ollama pull nomic-embed-text"
echo "  ollama pull phi4-mini"
echo ""
echo "Run:"
echo "  source venv/bin/activate"
echo "  python main.py"
