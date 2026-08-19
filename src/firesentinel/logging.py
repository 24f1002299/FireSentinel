"""Structured logging with no runtime dependency beyond the standard library."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from firesentinel.config import Settings


class JsonFormatter(logging.Formatter):
    """Render stable JSON records suitable for local tools and CI logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(settings: Settings) -> None:
    """Configure the FireSentinel logger once, leaving host logging untouched."""
    logger = logging.getLogger("firesentinel")
    logger.setLevel(settings.log_level)
    logger.propagate = False
    if any(
        getattr(handler, "_firesentinel_handler", False) for handler in logger.handlers
    ):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler._firesentinel_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
