from __future__ import annotations

"""Unified logging entrypoint for the service.

This module keeps business code on a small async/sync logger API while
bridging stdlib logging records from frameworks such as uvicorn and fastapi
into the same output pipeline.
"""

from datetime import datetime
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger as _logger

_VALID_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
_DEFAULT_COMPONENT = "app"
_DEFAULT_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
    "pid={process.id} | {extra[component]} | "
    "{file.name}:{function}:{line} | {message}"
)


# File names include process identity so main and worker processes never write
# into the same file during one run, and each restart creates a new file set.
def _sanitize_filename_part(value: Any, default: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = default
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    sanitized = sanitized.strip("._-")
    return sanitized or default


def _with_identity_suffix(base_filename: str, suffix_parts: list[str]) -> str:
    path = Path(base_filename)
    suffix = ".".join(part for part in suffix_parts if part)
    if path.suffix:
        return f"{path.stem}.{suffix}{path.suffix}"
    return f"{path.name}.{suffix}"


def _build_identity_suffix(
    process_role: str,
    startup_time: str,
    port: Optional[int] = None,
    model_name: Optional[str] = None,
    pid: Optional[int] = None,
) -> list[str]:
    parts = [_sanitize_filename_part(process_role, "app")]
    if model_name:
        parts.append(_sanitize_filename_part(model_name, "model"))
    if port is not None:
        parts.append(f"port-{int(port)}")
    parts.append(f"pid-{int(pid or os.getpid())}")
    parts.append(startup_time)
    return parts


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _normalize_level(value: Any, default: str = "INFO") -> str:
    level = str(value or default).strip().upper()
    if level == "WARN":
        level = "WARNING"
    if level not in _VALID_LEVELS:
        return default
    return level


def _normalize_logging_config(config: Optional[Any] = None) -> Dict[str, Any]:
    if config is None:
        raw: Dict[str, Any] = {}
    elif hasattr(config, "to_dict"):
        raw = dict(config.to_dict())
    elif isinstance(config, dict):
        raw = dict(config)
    else:
        raw = {
            "level": getattr(config, "level", "INFO"),
            "directory": getattr(config, "directory", "logs"),
            "sync_enabled": getattr(config, "sync_enabled", True),
            "sync_level": getattr(config, "sync_level", getattr(config, "level", "INFO")),
            "sync_filename": getattr(config, "sync_filename", "sync.log"),
            "console_enabled": getattr(config, "console_enabled", True),
            "rotation": getattr(config, "rotation", "100 MB"),
            "retention": getattr(config, "retention", "14 days"),
            "compression": getattr(config, "compression", None),
            "serialize": getattr(config, "serialize", False),
            "backtrace": getattr(config, "backtrace", False),
            "diagnose": getattr(config, "diagnose", False),
        }

    return {
        "level": _normalize_level(raw.get("level"), default="INFO"),
        "directory": str(raw.get("directory") or "logs"),
        "sync_enabled": _coerce_bool(raw.get("sync_enabled"), True),
        "sync_level": _normalize_level(raw.get("sync_level") or raw.get("level"), default="INFO"),
        "sync_filename": str(raw.get("sync_filename") or "sync.log"),
        "console_enabled": _coerce_bool(raw.get("console_enabled"), True),
        "rotation": str(raw.get("rotation") or "100 MB"),
        "retention": str(raw.get("retention") or "14 days"),
        "compression": raw.get("compression") or None,
        "serialize": _coerce_bool(raw.get("serialize"), False),
        "backtrace": _coerce_bool(raw.get("backtrace"), False),
        "diagnose": _coerce_bool(raw.get("diagnose"), False),
    }


class _FsyncFileSink:
    # Sync log channel uses an explicit flush + fsync sink so selected records
    # are persisted immediately instead of waiting for buffered file writes.
    def __init__(self, file_path: str):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, message: str) -> None:
        with self._lock:
            self._file.write(message)
            self._file.flush()
            os.fsync(self._file.fileno())

    def stop(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()


class _InterceptHandler(logging.Handler):
    # Third-party libraries often emit via stdlib logging. This handler forwards
    # those records into loguru so formatting and file routing stay unified.
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _logger.bind(component=record.name or _DEFAULT_COMPONENT, channel="async").opt(
            depth=depth,
            exception=record.exc_info,
        ).log(level, record.getMessage())


def _filter_channel(expected_channel: str):
    def _filter(record: Dict[str, Any]) -> bool:
        return record["extra"].get("channel", "async") == expected_channel

    return _filter


def _patch_defaults(record: Dict[str, Any]) -> None:
    record["extra"].setdefault("component", _DEFAULT_COMPONENT)
    record["extra"].setdefault("channel", "async")


def _configure_stdlib_logging(level: str) -> None:
    # Keep framework logs on the same pipeline instead of having uvicorn/fastapi
    # write with a separate formatter or destination.
    intercept = _InterceptHandler()
    logging.basicConfig(handlers=[intercept], level=getattr(logging, level, logging.INFO), force=True)

    for logger_name in (
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "fastapi",
        "asyncio",
    ):
        standard_logger = logging.getLogger(logger_name)
        standard_logger.handlers = [intercept]
        standard_logger.propagate = False
        standard_logger.setLevel(getattr(logging, level, logging.INFO))


def configure_logging(
    config: Optional[Any] = None,
    *,
    process_role: str = "app",
    port: Optional[int] = None,
    model_name: Optional[str] = None,
    startup_time: Optional[str] = None,
) -> Dict[str, Any]:
    # Reconfigure per process so each main/worker process shares the same
    # terminal formatting while sync logs optionally use a dedicated file set.
    settings = _normalize_logging_config(config)
    startup_token = startup_time or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    identity_suffix = _build_identity_suffix(
        process_role=process_role,
        startup_time=startup_token,
        port=port,
        model_name=model_name,
    )
    sync_filename = _with_identity_suffix(settings["sync_filename"], identity_suffix)

    _logger.remove()
    _logger.configure(patcher=_patch_defaults)

    if settings["console_enabled"]:
        # Console output is the shared entrypoint: async logs only appear here,
        # while sync logs are mirrored to both terminal and file.
        _logger.add(
            sys.stderr,
            level=settings["level"],
            format=_DEFAULT_FORMAT,
            backtrace=settings["backtrace"],
            diagnose=settings["diagnose"],
            enqueue=False,
        )

    if settings["sync_enabled"]:
        log_dir = Path(settings["directory"]).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        _logger.add(
            _FsyncFileSink(str(log_dir / sync_filename)),
            level=settings["sync_level"],
            format=_DEFAULT_FORMAT,
            serialize=settings["serialize"],
            backtrace=settings["backtrace"],
            diagnose=settings["diagnose"],
            enqueue=False,
            filter=_filter_channel("sync"),
        )

    _configure_stdlib_logging(settings["level"])
    settings["resolved_sync_filename"] = sync_filename if settings["sync_enabled"] else None
    settings["startup_time"] = startup_token
    settings["process_role"] = process_role
    if port is not None:
        settings["port"] = int(port)
    if model_name:
        settings["model_name"] = model_name
    return settings


def get_async_logger(component: Optional[str] = None):
    """Return the async logger used for regular service logs.

    Use this for request flow, debug output and general runtime information.
    Records on this channel go through the async file sink.
    """
    if component:
        return _logger.bind(channel="async", component=component)
    return _logger.bind(channel="async")


def get_sync_logger(component: Optional[str] = None):
    """Return the sync logger used for immediate persistence.

    Use this for startup, shutdown and fault logs that should be flushed to disk
    immediately when the sync sink is enabled.
    """
    if component:
        return _logger.bind(channel="sync", component=component)
    return _logger.bind(channel="sync")


def bind_context(**kwargs: Any):
    """Bind extra structured fields to subsequent log records."""
    return _logger.bind(**kwargs)


def shutdown_logging() -> None:
    """Flush async logging workers before process exit."""
    _logger.complete()


async_logger = get_async_logger()
sync_logger = get_sync_logger()

__all__ = [
    "async_logger",
    "sync_logger",
    "get_async_logger",
    "get_sync_logger",
    "bind_context",
    "configure_logging",
    "shutdown_logging",
]