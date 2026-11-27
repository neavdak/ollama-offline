"""FastAPI server exposing REST + SSE endpoints and Mission Control Web UI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import Settings, settings as default_settings
from app.engine import WorkflowEngine
from app.logging_utils import get_recent_logs, setup_logging, subscribe_logs, unsubscribe_logs
from app.ollama_client import OllamaClient

STATIC_DIR = Path(__file__).resolve().parent / "static"


class CreateRunRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    mode: str = Field(default="auto", pattern="^(auto|fast|deep|coder)$")
    model: str = Field(default="auto")
    auto_approve: bool = Field(default=True)
    workspace_dir: str | None = Field(default=None)


class FollowupRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ApprovalRequest(BaseModel):
    approval_id: str = Field(..., min_length=1)
    approved: bool


class SyncWorkflowRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    mode: str = Field(default="fast", pattern="^(auto|fast|deep|coder)$")
    model: str = Field(default="auto")
    workspace_dir: str | None = Field(default=None)
    timeout_sec: int = Field(default=180, ge=5, le=600)


def create_app(
    app_settings: Settings | None = None,
    ollama_client: OllamaClient | None = None,
) -> FastAPI:
    cfg = app_settings or default_settings
    setup_logging(cfg.data_dir)
    ollama = ollama_client or OllamaClient(base_url=cfg.ollama_base_url, keep_alive=cfg.keep_alive)
    engine = WorkflowEngine(settings=cfg, ollama=ollama)

    app = FastAPI(title="LocalGravity Agent Server", version="1.0.0")
    app.state.settings = cfg
    app.state.ollama = ollama
    app.state.engine = engine

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        ollama_health = await ollama.health()
        return {
            "server": "ok",
            "workspace_dir": str(cfg.workspace_dir),
            "ollama_base_url": cfg.ollama_base_url,
            "log_file": str(cfg.data_dir / "logs" / "server.log"),
            "ollama": ollama_health,
        }

    @app.get("/api/logs")
    async def get_logs(limit: int = 120) -> dict[str, Any]:
        return {"logs": get_recent_logs(limit=max(1, min(limit, 400)))}

    @app.get("/api/logs/stream")
    async def stream_logs() -> StreamingResponse:
        q = subscribe_logs()

        async def log_generator():
            try:
                snapshot = {"logs": get_recent_logs(100)}
                yield f"event: snapshot\ndata: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                while True:
                    try:
                        entry = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"event: log\ndata: {json.dumps(entry, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
            finally:
                unsubscribe_logs(q)

        return StreamingResponse(
            log_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.get("/api/models")
    async def list_models(refresh: bool = False) -> dict[str, Any]:
        try:
            models = await ollama.list_models(refresh=refresh)
            return {
                "models": [m.to_dict() for m in models],
                "defaults": {
                    "planner": cfg.preferred_planner,
                    "coder": cfg.preferred_coder,
                    "fast": cfg.preferred_fast,
                },
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to reach Ollama: {exc}") from exc

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        return {"runs": engine.list_runs()}

    @app.post("/api/runs")
    async def create_run(req: CreateRunRequest) -> dict[str, Any]:
        try:
            run = await engine.start_run(
                prompt=req.prompt,
                mode=req.mode,
                model=req.model,
                auto_approve=req.auto_approve,
                workspace_dir=req.workspace_dir,
            )
            return {"run": run.summary()}
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        run = engine.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"run": run.detail()}

    @app.post("/api/runs/{run_id}/message")
    async def send_message(run_id: str, req: FollowupRequest) -> dict[str, Any]:
        try:
            run = await engine.send_followup(run_id, req.message)
            return {"run": run.summary()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/approve")
    async def approve_tool(run_id: str, req: ApprovalRequest) -> dict[str, Any]:
        ok = await engine.resolve_approval(run_id, req.approval_id, req.approved)
        if not ok:
            raise HTTPException(status_code=404, detail="Pending approval not found")
        return {"ok": True}

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> dict[str, Any]:
        ok = await engine.cancel_run(run_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"ok": True}

    @app.get("/api/runs/{run_id}/artifacts/{filename}")
    async def get_artifact(run_id: str, filename: str) -> PlainTextResponse:
        safe_name = Path(filename).name
        artifact_path = cfg.data_dir / "runs" / run_id / "artifacts" / safe_name
        if not artifact_path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return PlainTextResponse(artifact_path.read_text(encoding="utf-8"))

    @app.get("/api/runs/{run_id}/stream")
    async def stream_run(run_id: str) -> StreamingResponse:
        run = engine.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        queue = engine.subscribe(run_id)

        async def event_generator():
            try:
                # Replay snapshot first so reconnecting clients stay consistent
                snapshot = {"type": "snapshot", "data": run.detail()}
                yield f"event: snapshot\ndata: {json.dumps(snapshot, ensure_ascii=False)}\n\n"

                if run.status in ("completed", "failed", "cancelled"):
                    return

                while True:
                    try:
                        evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"event: {evt['type']}\ndata: {json.dumps(evt, ensure_ascii=False)}\n\n"
                        if evt["type"] in ("done", "error") or (
                            evt["type"] == "status" and evt["data"].get("status") == "cancelled"
                        ):
                            break
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
                        if run.status in ("completed", "failed", "cancelled"):
                            break
            finally:
                engine.unsubscribe(run_id, queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.post("/api/workflows/run")
    async def run_sync_workflow(req: SyncWorkflowRequest) -> dict[str, Any]:
        """Headless server-to-server workflow endpoint: executes and returns final JSON."""
        try:
            run = await engine.run_to_completion(
                prompt=req.prompt,
                mode=req.mode,
                model=req.model,
                workspace_dir=req.workspace_dir,
                timeout_sec=req.timeout_sec,
            )
            return {
                "run_id": run.run_id,
                "status": run.status,
                "mode": run.mode,
                "model": run.effective_model,
                "final_answer": run.final_answer,
                "plan": run.plan,
                "artifacts": run.artifacts,
                "metrics": run.metrics,
                "error": run.error,
            }
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Workflow execution timed out") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
