"""Antigravity-style tool definitions and safe execution handlers."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.internet import fetch_url, web_search

MUTATING_TOOLS = {"write_file", "edit_file", "run_command"}

IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}


def get_tool_schemas(include_subagent: bool = True) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "set_focus",
                "description": "Update the short live status shown to the user (3-7 words, e.g. 'Searching Python 3.15 release notes').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "description": "Short present-participle status phrase."}
                    },
                    "required": ["status"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_plan",
                "description": "Publish or update the ordered step-by-step plan checklist for multi-step tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "done", "failed"],
                                    },
                                },
                                "required": ["title", "status"],
                            },
                        }
                    },
                    "required": ["steps"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_artifact",
                "description": "Save a structured markdown report, plan, or summary artifact for the user to view and download.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Filename like report.md or summary.txt"},
                        "title": {"type": "string", "description": "Human-readable title"},
                        "content": {"type": "string", "description": "Full markdown or text content"},
                    },
                    "required": ["filename", "title", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the live internet for up-to-date facts, docs, news, prices, or solutions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query keywords."},
                        "max_results": {"type": "integer", "description": "Max results (1-10, default 6)."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Download and extract readable text from a public webpage URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Full http:// or https:// URL."},
                        "max_chars": {"type": "integer", "description": "Max characters to extract (default 12000)."},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files and directories inside the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path inside workspace (default '.')."}
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read lines from a file inside the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path to file."},
                        "start_line": {"type": "integer", "description": "1-based start line (default 1)."},
                        "end_line": {"type": "integer", "description": "1-based end line (default 200)."},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a file inside the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path to file."},
                        "content": {"type": "string", "description": "Full file content."},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace an exact text block in an existing workspace file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "target": {"type": "string", "description": "Exact existing substring to replace."},
                        "replacement": {"type": "string", "description": "New text."},
                    },
                    "required": ["path", "target", "replacement"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search for a text or regex pattern across files in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Text or regex pattern."},
                        "path": {"type": "string", "description": "Relative directory or file (default '.')."},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Execute a shell command inside the workspace directory (e.g., run tests, scripts, git status).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run."},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)."},
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    if include_subagent:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "delegate_subagent",
                    "description": (
                        "Delegate a focused subtask to a specialized local Ollama worker (e.g., coder model for writing/debugging code, "
                        "or researcher model for web lookup). Returns the worker's final answer."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "description": "Worker role: 'coder', 'researcher', or custom short title.",
                            },
                            "task": {"type": "string", "description": "Clear actionable instructions for the worker."},
                            "model": {
                                "type": "string",
                                "description": "Optional Ollama model name override (leave empty to auto-select).",
                            },
                        },
                        "required": ["role", "task"],
                    },
                },
            }
        )
    return schemas


def resolve_workspace_path(workspace_dir: Path, raw_path: str) -> Path:
    """Resolve a relative path safely inside workspace_dir."""
    workspace_root = workspace_dir.resolve()
    rel = (raw_path or ".").strip()
    candidate = (workspace_root / rel).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise PermissionError(f"Path '{raw_path}' escapes workspace '{workspace_root}'") from exc
    return candidate


@dataclass
class ToolContext:
    workspace_dir: Path
    artifacts_dir: Path
    command_timeout: int
    on_focus: Callable[[str], Awaitable[None]]
    on_plan: Callable[[list[dict[str, str]]], Awaitable[None]]
    on_artifact: Callable[[dict[str, Any]], Awaitable[None]]
    on_subagent: Callable[[str, str, str], Awaitable[str]] | None = None


async def execute_tool(name: str, arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Dispatch a single tool call and return a JSON-serializable result dict."""
    try:
        if name == "set_focus":
            status = str(arguments.get("status", "")).strip() or "Working..."
            await ctx.on_focus(status)
            return {"ok": True, "focus": status}

        if name == "update_plan":
            raw_steps = arguments.get("steps") or []
            cleaned = []
            for s in raw_steps:
                if isinstance(s, dict) and s.get("title"):
                    st = s.get("status", "pending")
                    if st not in ("pending", "in_progress", "done", "failed"):
                        st = "pending"
                    cleaned.append({"title": str(s["title"]), "status": st})
            await ctx.on_plan(cleaned)
            return {"ok": True, "steps": cleaned}

        if name == "create_artifact":
            filename = Path(str(arguments.get("filename", "artifact.md"))).name
            if not filename:
                filename = "artifact.md"
            title = str(arguments.get("title") or filename)
            content = str(arguments.get("content") or "")
            ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
            out_path = ctx.artifacts_dir / filename
            out_path.write_text(content, encoding="utf-8")
            artifact = {
                "filename": filename,
                "title": title,
                "content": content,
                "bytes": len(content.encode("utf-8")),
            }
            await ctx.on_artifact(artifact)
            return {"ok": True, "filename": filename, "title": title, "bytes": artifact["bytes"]}

        if name == "web_search":
            query = str(arguments.get("query", "")).strip()
            if not query:
                return {"error": "query is required"}
            max_results = int(arguments.get("max_results", 6))
            return await web_search(query=query, max_results=max_results)

        if name == "fetch_url":
            url = str(arguments.get("url", "")).strip()
            if not url:
                return {"error": "url is required"}
            max_chars = int(arguments.get("max_chars", 12000))
            return await fetch_url(url=url, max_chars=max_chars)

        if name == "list_files":
            target = resolve_workspace_path(ctx.workspace_dir, str(arguments.get("path", ".")))
            if not target.exists():
                return {"error": f"Path does not exist: {arguments.get('path', '.')}"}
            if target.is_file():
                return {"path": str(target.relative_to(ctx.workspace_dir)), "type": "file", "size": target.stat().st_size}
            entries = []
            for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if entry.name in IGNORED_DIRS:
                    continue
                entries.append(
                    {
                        "name": entry.name,
                        "type": "dir" if entry.is_dir() else "file",
                        "size": entry.stat().st_size if entry.is_file() else None,
                    }
                )
            return {"path": str(target.relative_to(ctx.workspace_dir)), "entries": entries[:200]}

        if name == "read_file":
            target = resolve_workspace_path(ctx.workspace_dir, str(arguments.get("path", "")))
            if not target.is_file():
                return {"error": f"File not found: {arguments.get('path')}"}
            start_line = max(1, int(arguments.get("start_line", 1)))
            end_line = max(start_line, int(arguments.get("end_line", start_line + 199)))
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            slice_lines = lines[start_line - 1 : end_line]
            numbered = "\n".join(f"{idx}: {line}" for idx, line in enumerate(slice_lines, start=start_line))
            return {
                "path": str(target.relative_to(ctx.workspace_dir)),
                "total_lines": len(lines),
                "start_line": start_line,
                "end_line": min(end_line, len(lines)),
                "content": numbered,
            }

        if name == "write_file":
            target = resolve_workspace_path(ctx.workspace_dir, str(arguments.get("path", "")))
            content = str(arguments.get("content", ""))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {
                "ok": True,
                "path": str(target.relative_to(ctx.workspace_dir)),
                "bytes": len(content.encode("utf-8")),
            }

        if name == "edit_file":
            target = resolve_workspace_path(ctx.workspace_dir, str(arguments.get("path", "")))
            if not target.is_file():
                return {"error": f"File not found: {arguments.get('path')}"}
            old = str(arguments.get("target", ""))
            new = str(arguments.get("replacement", ""))
            text = target.read_text(encoding="utf-8", errors="replace")
            if old not in text:
                return {"error": "Target text not found in file"}
            count = text.count(old)
            updated = text.replace(old, new)
            target.write_text(updated, encoding="utf-8")
            return {"ok": True, "path": str(target.relative_to(ctx.workspace_dir)), "replacements": count}

        if name == "search_files":
            pattern = str(arguments.get("pattern", ""))
            if not pattern:
                return {"error": "pattern is required"}
            root = resolve_workspace_path(ctx.workspace_dir, str(arguments.get("path", ".")))
            regex = re.compile(pattern, re.IGNORECASE)
            matches: list[dict[str, Any]] = []
            files_to_scan = [root] if root.is_file() else []
            if root.is_dir():
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
                    for fn in filenames:
                        files_to_scan.append(Path(dirpath) / fn)
            for fp in files_to_scan:
                try:
                    if fp.stat().st_size > 1_000_000:
                        continue
                    for idx, line in enumerate(fp.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                        if regex.search(line):
                            matches.append(
                                {
                                    "path": str(fp.relative_to(ctx.workspace_dir)),
                                    "line": idx,
                                    "text": line[:240],
                                }
                            )
                            if len(matches) >= 50:
                                return {"matches": matches, "truncated": True}
                except Exception:
                    continue
            return {"matches": matches, "truncated": False}

        if name == "run_command":
            command = str(arguments.get("command", "")).strip()
            if not command:
                return {"error": "command is required"}
            timeout = max(1, min(int(arguments.get("timeout", ctx.command_timeout)), 120))
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(ctx.workspace_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return {"error": f"Command timed out after {timeout}s", "command": command}
            stdout = stdout_b.decode("utf-8", errors="replace")[:8000]
            stderr = stderr_b.decode("utf-8", errors="replace")[:4000]
            return {
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }

        if name == "delegate_subagent":
            if not ctx.on_subagent:
                return {"error": "Subagent delegation is disabled at this depth."}
            role = str(arguments.get("role", "worker")).strip() or "worker"
            task = str(arguments.get("task", "")).strip()
            model = str(arguments.get("model", "")).strip()
            if not task:
                return {"error": "task is required"}
            result = await ctx.on_subagent(role, task, model)
            return {"ok": True, "role": role, "result": result}

        return {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def summarize_tool_result(result: dict[str, Any], max_chars: int = 8000) -> str:
    """Serialize tool output compactly so local model context stays fast."""
    raw = json.dumps(result, ensure_ascii=False)
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "...(truncated)"
