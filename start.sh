#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

echo "Starting LocalGravity server at http://127.0.0.1:${PORT}"
exec .venv/bin/uvicorn app.server:app --host "$HOST" --port "$PORT"
