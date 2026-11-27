"""Antigravity-style server-based agentic workflow engine for local Ollama models."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings
from app.ollama_client import OllamaClient
from app.tools import MUTATING_TOOLS, ToolContext, execute_tool, get_tool_schemas, summarize_tool_result

logger = logging.getLogger("localgravity.engine")

_CODER_HINTS = (
    "code",
    "python",
    "javascript",
    "typescript",
    "react",
    "fastapi",
    "function",
    "bug",
    "debug",
    "refactor",
    "script",
    "test",
    "pytest",
    "sql",
    "dockerfile",
    "html",
    "css",
    "api",
    "fix ",
    "implement ",
    "write a ",
    "build a ",
)

_DEEP_HINTS = (
    "research",
    "compare",
    "analyze",
    "analysis",
    "architecture",
    "plan",
    "report",
    "deep dive",
    "comprehensive",
    "evaluate",
    "strategy",
)


def choose_auto_strategy(prompt: str, available_models: list[str], settings: Settings) -> tuple[str, str, bool]:
    """Pick (effective_mode, model_name, think_flag) quickly from the prompt."""
    lower = prompt.lower()
    is_deep = any(k in lower for k in _DEEP_HINTS) or len(prompt) > 360
    is_coder = any(k in lower for k in _CODER_HINTS)

    if is_deep:
        model = settings.preferred_planner if settings.preferred_planner in available_models else available_models[0]
        return "deep", model, True
    if is_coder:
        if settings.preferred_coder in available_models:
            return "coder", settings.preferred_coder, False
        for m in available_models:
            if "coder" in m.lower():
                return "coder", m, False
    model = settings.preferred_fast if settings.preferred_fast in available_models else available_models[0]
    return "fast", model, False


def build_system_prompt(mode: str, workspace_dir: Path, available_models: list[str]) -> str:
    models_str = ", ".join(available_models) if available_models else "local ollama models"
    base = f"""You are LocalGravity, a fast, pragmatic Antigravity-style autonomous agent running on a local server with live internet access and workspace tools.
Available local Ollama models: {models_str}.
Workspace root: {workspace_dir}

Core workflow rules:
1. For current facts, documentation, news, packages, prices, or anything post-training, ALWAYS use `web_search` and `fetch_url` instead of guessing.
2. Cite the exact URLs and facts you observed in tool outputs. Never fabricate file contents, command outputs, or URLs.
3. For tasks with 3+ distinct steps, call `update_plan` early and keep step statuses updated (`pending` -> `in_progress` -> `done`).
4. Call `set_focus` briefly when switching phases so the user sees live progress.
5. Keep tool arguments precise and concise. Read before editing files. Run tests/commands to verify code changes when applicable.
6. Use `create_artifact` when producing a substantial report, architecture spec, or deliverable document.
"""
    if mode == "fast":
        return (
            base
            + "\nMode: FAST FLOW. Prioritize speed and direct action. Minimize unnecessary steps, search or execute tools immediately, and give a crisp, accurate answer."
        )
    if mode == "coder":
        return (
            base
            + "\nMode: CODER FLOW. You are an expert software engineer. Inspect workspace files first, write clean modular code, look up docs online if needed, and verify by running commands/tests."
        )
    return (
        base
        + "\nMode: DEEP AGENT. Plan carefully with `update_plan`. You may use `delegate_subagent` (e.g. role='coder' for code writing/testing) to offload specialized subtasks, and save key deliverables via `create_artifact`."
    )


@dataclass
class PendingApproval:
    approval_id: str
    tool_name: str
    arguments: dict[str, Any]
    event: asyncio.Event = field(default_factory=asyncio.Event)
    approved: bool = False


@dataclass
class TaskRun:
    run_id: str
    prompt: str
    mode: str
    requested_model: str
    effective_model: str
    think: bool
    auto_approve: bool
    workspace_dir: str
    status: str = "queued"  # queued | running | waiting_approval | completed | failed | cancelled
    focus: str = "Starting workflow..."
    plan: list[dict[str, str]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "prompt": self.prompt,
            "mode": self.mode,
            "requested_model": self.requested_model,
            "effective_model": self.effective_model,
            "think": self.think,
            "auto_approve": self.auto_approve,
            "workspace_dir": self.workspace_dir,
            "status": self.status,
            "focus": self.focus,
            "plan": self.plan,
            "artifacts": [
                {"filename": a["filename"], "title": a["title"], "bytes": a["bytes"]} for a in self.artifacts
            ],
            "final_answer": self.final_answer,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metrics": self.metrics,
        }

    def detail(self) -> dict[str, Any]:
        data = self.summary()
        data["artifacts"] = self.artifacts
        data["events"] = self.events
        return data


class WorkflowEngine:
    def __init__(self, settings: Settings, ollama: OllamaClient) -> None:
        self.settings = settings
        self.ollama = ollama
        self.runs: dict[str, TaskRun] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._approvals: dict[str, PendingApproval] = {}
        self._load_persisted_runs()

    def _run_file(self, run_id: str) -> Path:
        return self.settings.data_dir / "runs" / f"{run_id}.json"

    def _artifacts_dir(self, run_id: str) -> Path:
        return self.settings.data_dir / "runs" / run_id / "artifacts"

    def _load_persisted_runs(self) -> None:
        runs_dir = self.settings.data_dir / "runs"
        if not runs_dir.exists():
            return
        for path in sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                status = raw.get("status", "completed")
                if status in ("running", "queued", "waiting_approval"):
                    status = "failed"
                    raw["error"] = "Server restarted while run was active"
                run = TaskRun(
                    run_id=raw["run_id"],
                    prompt=raw["prompt"],
                    mode=raw.get("mode", "fast"),
                    requested_model=raw.get("requested_model", "auto"),
                    effective_model=raw.get("effective_model", ""),
                    think=bool(raw.get("think", False)),
                    auto_approve=bool(raw.get("auto_approve", True)),
                    workspace_dir=raw.get("workspace_dir", str(self.settings.workspace_dir)),
                    status=status,
                    focus=raw.get("focus", ""),
                    plan=raw.get("plan", []),
                    artifacts=raw.get("artifacts", []),
                    events=raw.get("events", []),
                    messages=raw.get("messages", []),
                    final_answer=raw.get("final_answer", ""),
                    error=raw.get("error", ""),
                    created_at=float(raw.get("created_at", time.time())),
                    updated_at=float(raw.get("updated_at", time.time())),
                    metrics=raw.get("metrics", {}),
                )
                self.runs[run.run_id] = run
            except Exception:
                continue

    def _persist(self, run: TaskRun) -> None:
        run.updated_at = time.time()
        payload = run.detail()
        payload["messages"] = run.messages
        self._run_file(run.run_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _emit(self, run: TaskRun, event_type: str, data: dict[str, Any], persist_event: bool = True) -> None:
        evt = {
            "id": uuid.uuid4().hex[:10],
            "type": event_type,
            "ts": time.time(),
            "data": data,
        }
        if persist_event:
            run.events.append(evt)
        for q in list(self._subscribers.get(run.run_id, set())):
            await q.put(evt)

    def subscribe(self, run_id: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.setdefault(run_id, set()).add(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        subs = self._subscribers.get(run_id)
        if subs:
            subs.discard(q)

    def list_runs(self) -> list[dict[str, Any]]:
        ordered = sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)
        return [r.summary() for r in ordered]

    def get_run(self, run_id: str) -> TaskRun | None:
        return self.runs.get(run_id)

    async def start_run(
        self,
        prompt: str,
        mode: str = "auto",
        model: str = "auto",
        auto_approve: bool = True,
        workspace_dir: str | None = None,
    ) -> TaskRun:
        models = await self.ollama.list_models()
        model_names = [m.name for m in models]
        if not model_names:
            raise RuntimeError("No Ollama models installed. Pull a model (e.g. `ollama pull qwen3.5:9b`) first.")

        chosen_mode = mode if mode in ("auto", "fast", "deep", "coder") else "auto"
        think_flag = False
        if chosen_mode == "auto":
            chosen_mode, auto_model, think_flag = choose_auto_strategy(prompt, model_names, self.settings)
            effective_model = auto_model if model in ("", "auto") else model
        elif chosen_mode == "deep":
            think_flag = True
            if model in ("", "auto"):
                resolved = await self.ollama.resolve_model(self.settings.preferred_planner, role="planner")
                effective_model = resolved.name
            else:
                effective_model = model
        elif chosen_mode == "coder":
            think_flag = False
            if model in ("", "auto"):
                resolved = await self.ollama.resolve_model(self.settings.preferred_coder, role="coder")
                effective_model = resolved.name
            else:
                effective_model = model
        else:  # fast
            think_flag = False
            if model in ("", "auto"):
                resolved = await self.ollama.resolve_model(self.settings.preferred_fast, role="fast")
                effective_model = resolved.name
            else:
                effective_model = model

        if effective_model not in model_names:
            effective_model = model_names[0]

        ws_path = Path(workspace_dir).expanduser().resolve() if workspace_dir else self.settings.workspace_dir
        ws_path.mkdir(parents=True, exist_ok=True)

        run_id = uuid.uuid4().hex[:12]
        sys_prompt = build_system_prompt(chosen_mode, ws_path, model_names)
        run = TaskRun(
            run_id=run_id,
            prompt=prompt,
            mode=chosen_mode,
            requested_model=model or "auto",
            effective_model=effective_model,
            think=think_flag,
            auto_approve=auto_approve,
            workspace_dir=str(ws_path),
            status="queued",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        self.runs[run_id] = run
        logger.info(
            "[run:%s] Created mode=%s model=%s think=%s auto_approve=%s prompt=%r",
            run_id,
            chosen_mode,
            effective_model,
            think_flag,
            auto_approve,
            prompt[:80],
        )
        await self._emit(run, "user_message", {"content": prompt})
        self._persist(run)
        self._tasks[run_id] = asyncio.create_task(self._execute_loop(run))
        return run

    async def send_followup(self, run_id: str, message: str) -> TaskRun:
        run = self.runs.get(run_id)
        if not run:
            raise KeyError(f"Run {run_id} not found")
        if run.status in ("running", "waiting_approval", "queued"):
            raise RuntimeError("Run is currently active; wait for completion or cancel first.")
        run.messages.append({"role": "user", "content": message})
        run.status = "queued"
        run.error = ""
        await self._emit(run, "user_message", {"content": message})
        self._persist(run)
        self._tasks[run_id] = asyncio.create_task(self._execute_loop(run))
        return run

    async def resolve_approval(self, run_id: str, approval_id: str, approved: bool) -> bool:
        pending = self._approvals.get(approval_id)
        run = self.runs.get(run_id)
        if not pending or not run:
            return False
        pending.approved = approved
        pending.event.set()
        return True

    async def cancel_run(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if not run:
            return False
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        run.status = "cancelled"
        run.focus = "Cancelled by user"
        await self._emit(run, "status", {"status": "cancelled", "focus": run.focus})
        self._persist(run)
        return True

    async def _request_approval(self, run: TaskRun, tool_name: str, arguments: dict[str, Any]) -> bool:
        approval_id = uuid.uuid4().hex[:10]
        pending = PendingApproval(approval_id=approval_id, tool_name=tool_name, arguments=arguments)
        self._approvals[approval_id] = pending
        run.status = "waiting_approval"
        run.focus = f"Waiting approval for {tool_name}"
        await self._emit(
            run,
            "approval_required",
            {"approval_id": approval_id, "tool": tool_name, "arguments": arguments},
        )
        self._persist(run)
        try:
            await pending.event.wait()
        finally:
            self._approvals.pop(approval_id, None)
        run.status = "running"
        await self._emit(
            run,
            "approval_resolved",
            {"approval_id": approval_id, "tool": tool_name, "approved": pending.approved},
        )
        return pending.approved

    async def _run_subagent(self, parent: TaskRun, role: str, task: str, requested_model: str, depth: int) -> str:
        if depth >= 2:
            return "Subagent depth limit reached."

        if requested_model:
            model = requested_model
        elif "coder" in role.lower() or "code" in role.lower() or "dev" in role.lower():
            resolved = await self.ollama.resolve_model(self.settings.preferred_coder, role="coder")
            model = resolved.name
        else:
            resolved = await self.ollama.resolve_model(self.settings.preferred_fast, role="fast")
            model = resolved.name

        sub_id = uuid.uuid4().hex[:8]
        await self._emit(parent, "subagent_start", {"subagent_id": sub_id, "role": role, "model": model, "task": task})

        sub_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"You are a focused '{role}' subagent working under LocalGravity.\n"
                    f"Workspace: {parent.workspace_dir}\n"
                    "Complete the assigned task efficiently using tools when needed, and return a concise factual summary."
                ),
            },
            {"role": "user", "content": task},
        ]

        async def noop_focus(s: str) -> None:
            await self._emit(parent, "focus", {"status": f"[{role}] {s}"})

        async def noop_plan(steps: list[dict[str, str]]) -> None:
            pass

        async def on_artifact(artifact: dict[str, Any]) -> None:
            parent.artifacts.append(artifact)
            await self._emit(parent, "artifact", artifact)

        ctx = ToolContext(
            workspace_dir=Path(parent.workspace_dir),
            artifacts_dir=self._artifacts_dir(parent.run_id),
            command_timeout=self.settings.command_timeout,
            on_focus=noop_focus,
            on_plan=noop_plan,
            on_artifact=on_artifact,
            on_subagent=None,
        )
        schemas = get_tool_schemas(include_subagent=False)
        final_text = ""

        for _ in range(8):
            done_data: dict[str, Any] | None = None
            async for chunk in self.ollama.chat_stream(model=model, messages=sub_messages, tools=schemas, think=False):
                if chunk["type"] == "done":
                    done_data = chunk
            if not done_data:
                break
            msg = done_data["message"]
            sub_messages.append(msg)
            tool_calls = done_data["tool_calls"]
            if not tool_calls:
                final_text = msg.get("content", "")
                break
            for call in tool_calls:
                fn = call.get("function") or {}
                tname = fn.get("name", "")
                targs = fn.get("arguments") or {}
                if isinstance(targs, str):
                    try:
                        targs = json.loads(targs)
                    except Exception:
                        targs = {}
                if not parent.auto_approve and tname in MUTATING_TOOLS:
                    ok = await self._request_approval(parent, tname, targs)
                    if not ok:
                        res = {"error": "Denied by user"}
                    else:
                        res = await execute_tool(tname, targs, ctx)
                else:
                    res = await execute_tool(tname, targs, ctx)
                await self._emit(
                    parent,
                    "subagent_tool",
                    {"subagent_id": sub_id, "role": role, "tool": tname, "arguments": targs, "result": res},
                )
                sub_messages.append({"role": "tool", "content": summarize_tool_result(res)})
        else:
            final_text = "Subagent reached max steps without a final answer."

        await self._emit(parent, "subagent_done", {"subagent_id": sub_id, "role": role, "result": final_text})
        return final_text

    async def _execute_loop(self, run: TaskRun) -> None:
        started_at = time.time()
        run.status = "running"
        run.focus = f"Running {run.effective_model} ({run.mode})"
        logger.info("[run:%s] Executing with model=%s mode=%s", run.run_id, run.effective_model, run.mode)
        await self._emit(run, "status", {"status": "running", "focus": run.focus, "model": run.effective_model})
        self._persist(run)

        async def on_focus(status: str) -> None:
            run.focus = status
            await self._emit(run, "focus", {"status": status})

        async def on_plan(steps: list[dict[str, str]]) -> None:
            run.plan = steps
            await self._emit(run, "plan", {"steps": steps})

        async def on_artifact(artifact: dict[str, Any]) -> None:
            # Replace existing artifact with same filename if updated
            run.artifacts = [a for a in run.artifacts if a["filename"] != artifact["filename"]]
            run.artifacts.append(artifact)
            await self._emit(run, "artifact", artifact)

        async def on_subagent(role: str, task: str, model: str) -> str:
            return await self._run_subagent(run, role, task, model, depth=1)

        ctx = ToolContext(
            workspace_dir=Path(run.workspace_dir),
            artifacts_dir=self._artifacts_dir(run.run_id),
            command_timeout=self.settings.command_timeout,
            on_focus=on_focus,
            on_plan=on_plan,
            on_artifact=on_artifact,
            on_subagent=on_subagent,
        )
        schemas = get_tool_schemas(include_subagent=(run.mode == "deep"))
        total_tokens = 0
        tool_calls_count = 0

        try:
            for step in range(1, self.settings.max_steps + 1):
                turn_id = f"turn-{step}-{uuid.uuid4().hex[:6]}"
                done_data: dict[str, Any] | None = None
                async for chunk in self.ollama.chat_stream(
                    model=run.effective_model,
                    messages=run.messages,
                    tools=schemas,
                    think=run.think,
                ):
                    ctype = chunk["type"]
                    if ctype == "thinking":
                        await self._emit(
                            run,
                            "thinking_delta",
                            {"turn_id": turn_id, "delta": chunk["delta"]},
                            persist_event=False,
                        )
                    elif ctype == "content":
                        await self._emit(
                            run,
                            "content_delta",
                            {"turn_id": turn_id, "delta": chunk["delta"]},
                            persist_event=False,
                        )
                    elif ctype == "done":
                        done_data = chunk

                if not done_data:
                    raise RuntimeError("Empty stream from Ollama")

                msg = done_data["message"]
                thinking_text = done_data.get("thinking", "")
                meta = done_data.get("meta") or {}
                total_tokens += int(meta.get("eval_count", 0))
                run.messages.append(msg)

                if thinking_text:
                    await self._emit(run, "thinking_complete", {"turn_id": turn_id, "content": thinking_text})

                tool_calls = done_data.get("tool_calls") or []
                if not tool_calls:
                    run.final_answer = msg.get("content", "")
                    await self._emit(run, "assistant_message", {"turn_id": turn_id, "content": run.final_answer})
                    break

                if msg.get("content"):
                    await self._emit(run, "assistant_message", {"turn_id": turn_id, "content": msg["content"]})

                for call in tool_calls:
                    fn = call.get("function") or {}
                    tname = fn.get("name", "")
                    targs = fn.get("arguments") or {}
                    if isinstance(targs, str):
                        try:
                            targs = json.loads(targs)
                        except Exception:
                            targs = {"raw": targs}

                    call_id = uuid.uuid4().hex[:8]
                    tool_calls_count += 1
                    logger.info("[run:%s] Tool call %s args=%s", run.run_id, tname, json.dumps(targs, ensure_ascii=False)[:180])
                    await self._emit(run, "tool_call", {"call_id": call_id, "tool": tname, "arguments": targs})

                    t_tool = time.monotonic()
                    if not run.auto_approve and tname in MUTATING_TOOLS:
                        approved = await self._request_approval(run, tname, targs)
                        if not approved:
                            res = {"error": f"Tool '{tname}' denied by user."}
                        else:
                            res = await execute_tool(tname, targs, ctx)
                    else:
                        res = await execute_tool(tname, targs, ctx)
                    logger.info("[run:%s] Tool %s finished in %.2fs (error=%s)", run.run_id, tname, time.monotonic() - t_tool, bool(res.get("error")))

                    await self._emit(run, "tool_result", {"call_id": call_id, "tool": tname, "result": res})
                    run.messages.append({"role": "tool", "content": summarize_tool_result(res)})

                self._persist(run)
            else:
                run.final_answer = run.final_answer or "Reached maximum agent steps."

            run.status = "completed"
            run.focus = "Completed"
            run.metrics = {
                "duration_sec": round(time.time() - started_at, 2),
                "output_tokens": total_tokens,
                "tool_calls": tool_calls_count,
            }
            logger.info("[run:%s] Completed in %ss (tokens=%s, tools=%s)", run.run_id, run.metrics["duration_sec"], total_tokens, tool_calls_count)
            await self._emit(
                run,
                "done",
                {"status": "completed", "final_answer": run.final_answer, "metrics": run.metrics},
            )
            self._persist(run)

        except asyncio.CancelledError:
            run.status = "cancelled"
            run.focus = "Cancelled"
            logger.info("[run:%s] Cancelled", run.run_id)
            self._persist(run)
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            run.focus = "Failed"
            logger.exception("[run:%s] Failed: %s", run.run_id, exc)
            await self._emit(run, "error", {"error": str(exc)})
            self._persist(run)

    async def run_to_completion(
        self,
        prompt: str,
        mode: str = "fast",
        model: str = "auto",
        workspace_dir: str | None = None,
        timeout_sec: int = 180,
    ) -> TaskRun:
        """Start a run and await completion (used by webhook / programmatic API)."""
        run = await self.start_run(
            prompt=prompt,
            mode=mode,
            model=model,
            auto_approve=True,
            workspace_dir=workspace_dir,
        )
        task = self._tasks.get(run.run_id)
        if task:
            await asyncio.wait_for(task, timeout=timeout_sec)
        return run
