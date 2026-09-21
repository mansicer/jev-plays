#!/usr/bin/env bash
# Start the local Craftax agent engine on http://localhost:8765
cd "$(dirname "$0")/.."
exec uv run --python .venv/bin/python -m uvicorn craftax_agent.server.app:app --port "${PORT:-8765}"
