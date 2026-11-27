"""Configuration settings for LocalGravity server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    ollama_base_url: str
    workspace_dir: Path
    data_dir: Path
    host: str
    port: int
    keep_alive: str
    max_steps: int
    preferred_planner: str
    preferred_coder: str
    preferred_fast: str
    command_timeout: int

    @classmethod
    def from_env(cls) -> "Settings":
        base_dir = Path(__file__).resolve().parent.parent
        workspace_raw = os.getenv("WORKSPACE_DIR", str(base_dir / "workspace"))
        data_raw = os.getenv("DATA_DIR", str(base_dir / "data"))
        workspace_dir = Path(workspace_raw).expanduser().resolve()
        data_dir = Path(data_raw).expanduser().resolve()
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "runs").mkdir(parents=True, exist_ok=True)
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            workspace_dir=workspace_dir,
            data_dir=data_dir,
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8080")),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "15m"),
            max_steps=int(os.getenv("MAX_AGENT_STEPS", "18")),
            preferred_planner=os.getenv("PLANNER_MODEL", "qwen3.5:9b"),
            preferred_coder=os.getenv("CODER_MODEL", "qwen2.5-coder:7b"),
            preferred_fast=os.getenv("FAST_MODEL", "qwen3.5:9b"),
            command_timeout=int(os.getenv("COMMAND_TIMEOUT", "30")),
        )


settings = Settings.from_env()
