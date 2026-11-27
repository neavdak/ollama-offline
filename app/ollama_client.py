"""Async Ollama API client with model discovery, capability inspection, and tool streaming."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger("localgravity.ollama")


@dataclass(slots=True)
class ModelInfo:
    name: str
    size_bytes: int = 0
    parameter_size: str = ""
    quantization: str = ""
    family: str = ""
    context_length: int = 32768
    capabilities: list[str] = field(default_factory=list)
    supports_tools: bool = False
    supports_thinking: bool = False
    supports_vision: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_bytes / (1024**3), 2) if self.size_bytes else 0.0,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
            "family": self.family,
            "context_length": self.context_length,
            "capabilities": self.capabilities,
            "supports_tools": self.supports_tools,
            "supports_thinking": self.supports_thinking,
            "supports_vision": self.supports_vision,
        }


_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_fallback_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls emitted inline as XML/JSON if a model didn't populate tool_calls."""
    if not text:
        return []
    calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_TAG_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict) and "name" in payload:
                args = payload.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append({"function": {"name": payload["name"], "arguments": args}})
        except Exception:
            continue
    return calls


class OllamaClient:
    def __init__(self, base_url: str, keep_alive: str = "15m") -> None:
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self._model_cache: dict[str, ModelInfo] = {}

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/version")
                resp.raise_for_status()
                version = resp.json().get("version", "unknown")
                running = await self.list_running()
                return {"ok": True, "version": version, "running_models": running}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "running_models": []}

    async def list_running(self) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/ps")
                resp.raise_for_status()
                models = resp.json().get("models", [])
                return [
                    {
                        "name": m.get("name"),
                        "size_vram_gb": round(m.get("size_vram", 0) / (1024**3), 2),
                        "expires_at": m.get("expires_at"),
                    }
                    for m in models
                ]
        except Exception:
            return []

    async def inspect_model(self, name: str, size_bytes: int = 0, details: dict[str, Any] | None = None) -> ModelInfo:
        if name in self._model_cache:
            cached = self._model_cache[name]
            if size_bytes and not cached.size_bytes:
                cached.size_bytes = size_bytes
            return cached

        details = details or {}
        caps: list[str] = []
        ctx = 32768
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{self.base_url}/api/show", json={"model": name})
                if resp.status_code == 200:
                    data = resp.json()
                    caps = list(data.get("capabilities") or [])
                    model_info = data.get("model_info") or {}
                    for k, v in model_info.items():
                        if k.endswith(".context_length") and isinstance(v, int):
                            ctx = v
                            break
                    details = {**details, **(data.get("details") or {})}
        except Exception:
            pass

        info = ModelInfo(
            name=name,
            size_bytes=size_bytes,
            parameter_size=str(details.get("parameter_size", "")),
            quantization=str(details.get("quantization_level", "")),
            family=str(details.get("family", "")),
            context_length=ctx,
            capabilities=caps,
            supports_tools="tools" in caps or "qwen" in name.lower() or "llama3" in name.lower(),
            supports_thinking="thinking" in caps,
            supports_vision="vision" in caps,
        )
        self._model_cache[name] = info
        return info

    async def list_models(self, refresh: bool = False) -> list[ModelInfo]:
        if refresh:
            self._model_cache.clear()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            raw_models = resp.json().get("models", [])

        result: list[ModelInfo] = []
        for item in raw_models:
            name = item.get("name", "")
            if not name:
                continue
            info = await self.inspect_model(
                name=name,
                size_bytes=int(item.get("size") or 0),
                details=item.get("details") or {},
            )
            result.append(info)
        return result

    async def resolve_model(self, preferred: str, role: str = "general") -> ModelInfo:
        models = await self.list_models()
        if not models:
            raise RuntimeError("No Ollama models found. Run `ollama pull qwen3.5:9b` first.")

        names = {m.name: m for m in models}
        if preferred in names:
            return names[preferred]

        if role == "coder":
            for m in models:
                if "coder" in m.name.lower() or "code" in m.name.lower():
                    return m
        if role in ("planner", "fast", "general"):
            for m in models:
                if m.supports_tools and "coder" not in m.name.lower():
                    return m
            for m in models:
                if m.supports_tools:
                    return m
        return models[0]

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        think: bool | None = None,
        temperature: float = 0.3,
        num_ctx: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat completion chunks from Ollama /api/chat."""
        info = await self.inspect_model(model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
            },
        }
        if num_ctx:
            payload["options"]["num_ctx"] = num_ctx
        if tools:
            payload["tools"] = tools
        if think is not None and info.supports_thinking:
            payload["think"] = think

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        final_meta: dict[str, Any] = {}
        t_start = time.monotonic()
        first_token_logged = False

        logger.info(
            "Ollama chat_stream -> model=%s think=%s tools=%d msgs=%d",
            model,
            payload.get("think", "n/a"),
            len(tools or []),
            len(messages),
        )

        timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    err_text = body.decode("utf-8", "ignore")
                    logger.error("Ollama HTTP %s: %s", resp.status_code, err_text[:300])
                    raise RuntimeError(f"Ollama error {resp.status_code}: {err_text}")

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    msg = chunk.get("message") or {}

                    thinking_delta = msg.get("thinking") or ""
                    if thinking_delta:
                        if not first_token_logged:
                            first_token_logged = True
                            logger.info("Ollama first token (thinking) in %.2fs", time.monotonic() - t_start)
                        thinking_parts.append(thinking_delta)
                        yield {"type": "thinking", "delta": thinking_delta}

                    content_delta = msg.get("content") or ""
                    if content_delta:
                        if not first_token_logged:
                            first_token_logged = True
                            logger.info("Ollama first token (content) in %.2fs", time.monotonic() - t_start)
                        content_parts.append(content_delta)
                        yield {"type": "content", "delta": content_delta}

                    if msg.get("tool_calls"):
                        tool_calls.extend(msg["tool_calls"])

                    if chunk.get("done"):
                        eval_count = chunk.get("eval_count", 0)
                        eval_ns = chunk.get("eval_duration", 0)
                        tps = round(eval_count / (eval_ns / 1e9), 1) if eval_ns else 0.0
                        final_meta = {
                            "eval_count": eval_count,
                            "eval_duration_ns": eval_ns,
                            "prompt_eval_count": chunk.get("prompt_eval_count", 0),
                            "total_duration_ns": chunk.get("total_duration", 0),
                            "tokens_per_sec": tps,
                        }
                        logger.info(
                            "Ollama done model=%s tokens=%s (%.1f tok/s) total=%.2fs tool_calls=%d",
                            model,
                            eval_count,
                            tps,
                            time.monotonic() - t_start,
                            len(tool_calls),
                        )

        full_content = "".join(content_parts)
        full_thinking = "".join(thinking_parts)
        if not tool_calls:
            tool_calls = extract_fallback_tool_calls(full_content)

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": full_content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        yield {
            "type": "done",
            "message": assistant_message,
            "thinking": full_thinking,
            "tool_calls": tool_calls,
            "meta": final_meta,
        }
