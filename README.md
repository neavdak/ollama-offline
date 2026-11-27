# LocalGravity — Ollama + Internet Antigravity-Style Agent Server

A fast, server-based agentic workflow engine powered by your local Ollama models (`qwen3.5:9b`, `qwen2.5-coder:7b`, or any installed model) with live internet access, Antigravity-style task planning, streaming SSE execution, subagent delegation, and a Mission Control Web UI.

## Features

- **Antigravity-Style Workflow**:
  - **Live Task Plan** (`update_plan`) and **Live Focus** (`set_focus`) streamed in real time.
  - **Artifacts** (`create_artifact`) saved per run and viewable/downloadable in the UI or via API.
  - **Approval Gate**: Choose **Fast Flow** (`auto_approve: true`) or interactive approval for file writes and shell commands.
  - **Subagent Delegation**: In `deep` mode, `qwen3.5:9b` can delegate coding subtasks to `qwen2.5-coder:7b`.
- **Live Internet Connection**:
  - `web_search` via DuckDuckGo (`ddgs` + HTML fallback + Wikipedia fallback).
  - `fetch_url` with clean HTML-to-text extraction and SSRF protection.
- **Speed Optimization for Local Ollama**:
  - Automatically toggles `think: false` on `qwen3.5:9b` in **Fast** / **Auto** mode so tool calls happen without slow reasoning delays, while **Deep** mode enables `think: true`.
  - Uses `keep_alive: "15m"` to keep models warm in memory.
- **Server-First API + Web UI**:
  - Mission Control UI at `http://127.0.0.1:8080`
  - Synchronous automation endpoint `POST /api/workflows/run`
  - Asynchronous runs + SSE streaming at `POST /api/runs` and `GET /api/runs/{run_id}/stream`

## Quick Start

```bash
chmod +x start.sh
./start.sh
```

Then open [http://127.0.0.1:8080](http://127.0.0.1:8080).

## Server API Examples

### Synchronous Workflow Call (Server-to-Server / Curl)

```bash
curl -X POST http://127.0.0.1:8080/api/workflows/run \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Search the web for the latest FastAPI release version and summarize highlights.",
    "mode": "fast"
  }'
```

### Asynchronous Run + SSE Stream

```bash
RUN_ID=$(curl -s -X POST http://127.0.0.1:8080/api/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Create hello.py in workspace and run it","mode":"coder"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['run']['run_id'])")

curl -N "http://127.0.0.1:8080/api/runs/${RUN_ID}/stream"
```

## Configuration (Environment Variables)

| Variable | Default | Description |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `WORKSPACE_DIR` | `./workspace` | Root folder for agent file/shell tools |
| `DATA_DIR` | `./data` | Persisted run history & artifacts |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8080` | Server port |
| `PLANNER_MODEL` | `qwen3.5:9b` | Default planner / deep agent model |
| `CODER_MODEL` | `qwen2.5-coder:7b` | Default coder model |
| `FAST_MODEL` | `qwen3.5:9b` | Default fast-flow model |
| `OLLAMA_KEEP_ALIVE` | `15m` | Keep models hot in RAM/VRAM |
