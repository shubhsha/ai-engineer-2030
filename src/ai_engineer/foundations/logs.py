"""Chapter 4: one JSON object per log record, so logs can be queried."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

# Every LogRecord has these attributes; anything else came from extra=.
_STANDARD = frozenset(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Write each record as one line of JSON.

    Fields passed with ``extra=`` become top-level keys, so a record of
    a model call can be filtered by model and summed by tokens without
    parsing prose. A traceback stays inside its record, on one line.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(
            (key, value)
            for key, value in vars(record).items()
            if key not in _STANDARD
        )
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: str = "INFO", *, as_json: bool = True) -> None:
    """Configure the root logger once, at the application's entry point.

    Records go to stderr, because stdout may carry the program's output
    (or, for a stdio MCP server, its protocol). Libraries never call
    this; they only create loggers with ``logging.getLogger(__name__)``.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter()
        if as_json
        else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
