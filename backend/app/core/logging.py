"""JSON logging with automatic credential redaction."""
from __future__ import annotations

import json
import logging
import sys
import time

from app.core.security import redact_text


class RedactingJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = redact_text(record.getMessage())
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": msg,
        }
        for k in ("camera_id", "worker_id"):
            if hasattr(record, k):
                out[k] = getattr(record, k)
        if record.exc_info:
            out["exc"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(out)


class _RedactAccessLog(logging.Filter):
    """uvicorn access lines contain the full URL; hide ?token=<jwt> so logs cannot be replayed as sessions."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(redact_text(a) if isinstance(a, str) else a for a in record.args)
        return True


def setup_logging(level: str = "INFO") -> None:
    logging.getLogger("uvicorn.access").addFilter(_RedactAccessLog())
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(RedactingJsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level.upper())
