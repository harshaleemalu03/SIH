#!/usr/bin/env bash
set -e
# Lunar Image Registration — runner
# Usage: ./run.sh [port]  (default 8000)
PORT=${1:-8000}
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if command -v uv >/dev/null 2>&1 && [ -f uv.lock ]; then
  echo "Starting with uv (deep matcher if torch installed) — http://127.0.0.1:$PORT"
  exec uv run uvicorn app:app --host 0.0.0.0 --port "$PORT" --reload
else
  echo "Starting with system python (SIFT fallback) — http://127.0.0.1:$PORT"
  exec python3 -m uvicorn app:app --host 0.0.0.0 --port "$PORT" --reload
fi
