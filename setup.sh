#!/usr/bin/env bash
set -e

echo "🚗 Setting up DMS V4 Ecosystem (Ollama + FastAPI + Web UI)..."

if ! command -v uv &> /dev/null
then
    echo "📦 uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "📦 Creating virtual environment and installing dependencies via uv..."
uv venv
source .venv/bin/activate
uv pip install -e .

echo "🦙 Ensuring Ollama is running..."
# Quick check if Ollama is accessible
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running."
else
    echo "⚠️ Ollama is not running! Please start Ollama before running DMS."
fi

echo "📁 Checking data directories..."
mkdir -p data/driver_profiles data/reports

if [ ! -f .env ]; then
    echo "📝 Copying .env.example to .env..."
    cp .env.example .env
fi

echo "✅ Setup complete! Run the system with:"
echo "source .venv/bin/activate && python main.py"
