"""Structured logging helpers.

We intentionally keep this dependency-free (stdlib only). A single helper,
:func:`get_logger`, returns a logger that prints key=value pairs after the
message, which makes it grep-friendly without pulling in a third-party
structured-logging package.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional


_CONFIGURED = False


def _format_kv(**kwargs: Any) -> str:
    if not kwargs:
        return ""
    parts: list[str] = []
    for k, v in kwargs.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, default=str, ensure_ascii=False)
        parts.append(f"{k}={v}")
    return " " + " ".join(parts)


class KVLogger(logging.LoggerAdapter):
    """Logger adapter that appends key=value pairs to the formatted message."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra_kv = kwargs.pop("kv", None)
        if extra_kv:
            msg = f"{msg}{_format_kv(**extra_kv)}"
        return msg, kwargs


def configure(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Configure the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    fmt = "%(asctime)s %(levelname)s %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    _CONFIGURED = True


def get_logger(name: str) -> KVLogger:
    return KVLogger(logging.getLogger(name), {})
