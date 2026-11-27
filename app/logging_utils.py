"""Structured console, file, and live-streaming logging for LocalGravity."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_LOG_BUFFER: deque[dict[str, Any]] = deque(maxlen=400)
_LOG_SUBSCRIBERS: set[asyncio.Queue[dict[str, Any]]] = set()
_CONFIGURED = False


class LiveStreamHandler(logging.Handler):
    """Stores recent formatted log entries and pushes them to active SSE subscribers."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": record.created,
                "time": self.formatTime(record, "%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            _LOG_BUFFER.append(entry)
            for q in list(_LOG_SUBSCRIBERS):
                try:
                    q.put_nowait(entry)
                except asyncio.QueueFull:
                    pass
        except Exception:
            pass


def setup_logging(data_dir: Path) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("localgravity")
    if _CONFIGURED:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(logs_dir / "server.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    live_handler = LiveStreamHandler()
    live_handler.setFormatter(fmt)
    logger.addHandler(live_handler)

    _CONFIGURED = True
    return logger


def get_recent_logs(limit: int = 150) -> list[dict[str, Any]]:
    items = list(_LOG_BUFFER)
    return items[-limit:]


def subscribe_logs() -> asyncio.Queue[dict[str, Any]]:
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
    _LOG_SUBSCRIBERS.add(q)
    return q


def unsubscribe_logs(q: asyncio.Queue[dict[str, Any]]) -> None:
    _LOG_SUBSCRIBERS.discard(q)
