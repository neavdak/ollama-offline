"""Unit and API tests for LocalGravity server, tools, and workflow engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.engine import choose_auto_strategy
from app.internet import _clean_html_to_text, _is_safe_public_url
from app.ollama_client import ModelInfo, extract_fallback_tool_calls
from app.server import create_app
from app.tools import ToolContext, execute_tool, resolve_workspace_path


class FakeOllamaClient:
    """Deterministic fake Ollama client for fast offline unit tests."""

    def __init__(self) -> None:
        self.models = [
            ModelInfo(
                name="qwen3.5:9b",
                size_bytes=7_600_000_000,
                supports_tools=True,
                supports_thinking=True,
                supports_vision=True,
                capabilities=["completion", "tools", "thinking", "vision"],
            ),
            ModelInfo(
                name="qwen2.5-coder:7b",
                size_bytes=4_700_000_000,
                supports_tools=True,
                supports_thinking=False,
                capabilities=["completion", "tools"],
            ),
        ]

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "version": "0.22.0-test", "running_models": []}

    async def list_models(self, refresh: bool = False) -> list[ModelInfo]:
        return self.models

    async def inspect_model(self, name: str, size_bytes: int = 0, details: dict[str, Any] | None = None) -> ModelInfo:
        for m in self.models:
            if m.name == name:
                return m
        return self.models[0]

    async def resolve_model(self, preferred: str, role: str = "general") -> ModelInfo:
        for m in self.models:
            if m.name == preferred:
                return m
        return self.models[1] if role == "coder" else self.models[0]

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        think: bool | None = None,
        temperature: float = 0.3,
        num_ctx: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if not has_tool_result:
            call = {
                "function": {
                    "name": "write_file",
                    "arguments": {"path": "hello.txt", "content": "Hello from LocalGravity"},
                }
            }
            yield {
                "type": "done",
                "message": {"role": "assistant", "content": "", "tool_calls": [call]},
                "thinking": "",
                "tool_calls": [call],
                "meta": {"eval_count": 12},
            }
            return

        text = f"Created hello.txt using {model} (think={think})."
        yield {"type": "content", "delta": text}
        yield {
            "type": "done",
            "message": {"role": "assistant", "content": text},
            "thinking": "",
            "tool_calls": [],
            "meta": {"eval_count": 15},
        }


def make_settings(tmp_path: Path) -> Settings:
    ws = tmp_path / "workspace"
    data = tmp_path / "data"
    ws.mkdir(parents=True, exist_ok=True)
    (data / "runs").mkdir(parents=True, exist_ok=True)
    return Settings(
        ollama_base_url="http://127.0.0.1:11434",
        workspace_dir=ws,
        data_dir=data,
        host="127.0.0.1",
        port=8080,
        keep_alive="5m",
        max_steps=8,
        preferred_planner="qwen3.5:9b",
        preferred_coder="qwen2.5-coder:7b",
        preferred_fast="qwen3.5:9b",
        command_timeout=10,
    )


def test_auto_router_picks_coder_and_fast(tmp_path: Path) -> None:
    cfg = make_settings(tmp_path)
    models = ["qwen3.5:9b", "qwen2.5-coder:7b"]

    mode, model, think = choose_auto_strategy("Write a Python function to parse CSV", models, cfg)
    assert mode == "coder"
    assert model == "qwen2.5-coder:7b"
    assert think is False

    mode, model, think = choose_auto_strategy("What is the weather in Tokyo?", models, cfg)
    assert mode == "fast"
    assert model == "qwen3.5:9b"
    assert think is False

    mode, model, think = choose_auto_strategy("Research and compare microservices architecture options", models, cfg)
    assert mode == "deep"
    assert model == "qwen3.5:9b"
    assert think is True


def test_workspace_path_blocks_escape(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    safe = resolve_workspace_path(ws, "sub/file.py")
    assert str(safe).startswith(str(ws.resolve()))

    with pytest.raises(PermissionError):
        resolve_workspace_path(ws, "../../etc/passwd")


def test_ssrf_and_html_extraction() -> None:
    ok, _ = _is_safe_public_url("http://127.0.0.1:8080/secret")
    assert ok is False
    ok_file, _ = _is_safe_public_url("file:///etc/passwd")
    assert ok_file is False

    title, text = _clean_html_to_text(
        "<html><head><title>Doc</title><script>bad()</script></head>"
        "<body><main><h1>Title</h1><p>Clean body text.</p></main></body></html>"
    )
    assert title == "Doc"
    assert "Title" in text
    assert "Clean body text." in text
    assert "bad()" not in text


def test_fallback_tool_call_parser() -> None:
    raw = '<tool_call>{"name": "web_search", "arguments": {"query": "ollama"}}</tool_call>'
    calls = extract_fallback_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "web_search"
    assert calls[0]["function"]["arguments"]["query"] == "ollama"


@pytest.mark.asyncio
async def test_workspace_tools_execution(tmp_path: Path) -> None:
    cfg = make_settings(tmp_path)
    events: list[str] = []

    async def on_focus(s: str) -> None:
        events.append(f"focus:{s}")

    async def on_plan(steps: list[dict[str, str]]) -> None:
        events.append(f"plan:{len(steps)}")

    async def on_artifact(a: dict[str, Any]) -> None:
        events.append(f"artifact:{a['filename']}")

    ctx = ToolContext(
        workspace_dir=cfg.workspace_dir,
        artifacts_dir=cfg.data_dir / "artifacts",
        command_timeout=10,
        on_focus=on_focus,
        on_plan=on_plan,
        on_artifact=on_artifact,
    )

    w = await execute_tool("write_file", {"path": "sample.py", "content": "print('ok')\n"}, ctx)
    assert w["ok"] is True

    e = await execute_tool("edit_file", {"path": "sample.py", "target": "ok", "replacement": "verified"}, ctx)
    assert e["ok"] is True

    r = await execute_tool("read_file", {"path": "sample.py"}, ctx)
    assert "verified" in r["content"]

    cmd = await execute_tool("run_command", {"command": "python3 sample.py"}, ctx)
    assert cmd["exit_code"] == 0
    assert "verified" in cmd["stdout"]

    art = await execute_tool("create_artifact", {"filename": "out.md", "title": "Out", "content": "# Done"}, ctx)
    assert art["ok"] is True
    assert "artifact:out.md" in events


def test_server_sync_workflow_and_ui(tmp_path: Path) -> None:
    cfg = make_settings(tmp_path)
    app = create_app(app_settings=cfg, ollama_client=FakeOllamaClient())  # type: ignore[arg-type]
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert "LocalGravity" in home.text

    models = client.get("/api/models")
    assert models.status_code == 200
    assert len(models.json()["models"]) == 2

    res = client.post(
        "/api/workflows/run",
        json={"prompt": "Create a hello.txt file in Python project", "mode": "coder"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert data["model"] == "qwen2.5-coder:7b"
    assert (cfg.workspace_dir / "hello.txt").read_text(encoding="utf-8") == "Hello from LocalGravity"
