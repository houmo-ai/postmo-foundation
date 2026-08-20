import os
import time
import json
import asyncio
import multiprocessing
import re
import uuid
import socket
import signal
from copy import deepcopy
import httpx
import uvicorn
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response, StreamingResponse, JSONResponse
from starlette.datastructures import UploadFile
from hliepython.utils.logging import configure_logging, get_async_logger, get_sync_logger
from hliepython.config import AppConfig, load_app_config
from hliepython.config.utils import resolve_model_file_path

from hliepython.model_server.engines.protocol.chat_completions import (
    ChatCompletionRequest,
)
from hliepython.model_server.engines.protocol.audio_speech import (
    AudioSpeechRequest,
)
from hliepython.model_server.engines.protocol.embeddings import (
    EmbeddingRequest,
)
from hliepython.model_server.engines.utils import split_vllm_config
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from hliepython.model_server.worker import run_worker

main_app = FastAPI(title="主进程 7800")
logger = get_async_logger("main")
sync_logger = get_sync_logger("main")

# 全局映射
MODEL_PORT_MAP: Dict[str, list[int]] = {}
PORT_MODEL_MAP: Dict[int, str] = {}
WORKERS: list[multiprocessing.Process] = []
WORKER_PROCESS_MAP: Dict[int, multiprocessing.Process] = {}
DYNAMIC_MODEL_INSTANCES: Dict[str, list[int]] = {}
APP_CONFIG = AppConfig()
RUNTIME_SETTINGS: Dict[str, Any] = {
    "main_port": 7800,
    "log_level": "info",
    "vllm_log_level": "warning",
    "model_root_dir": "/houmo/llm/models",
}
AVAILABLE_MODEL_PATHS: Dict[str, str] = {}
AVAILABLE_MODEL_ADDITIONAL_CONFIG: Dict[str, Dict[str, Any]] = {}
AVAILABLE_MODEL_BASE_NAMES: Dict[str, str] = {}
WORKER_RUNTIME_MAP: Dict[int, Dict[str, Any]] = {}
NEXT_INSTANCE_ID = 1
DEFAULT_MODEL_ARGS: Dict[str, Any] = {}

_VERSION_RE = re.compile(r"v\d+\.\d+\.\d+", re.IGNORECASE)
_XH_RE = re.compile(r"xh\s*(\d+)", re.IGNORECASE)
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bBmM]", re.IGNORECASE)
_BUILD_TIME_RE = re.compile(r"(\d{8}|\d{14})")
_VIT_SPEC_RE = re.compile(r"vit_(\d+)(?:[_x](\d+))?", re.IGNORECASE)
_MMPROJ_FAMILY_RE = re.compile(
    r"qwen2\.5-vl|qwen3-vl|qwen3\.5|qwen3\.6|gemma4[-_]26b[-_]a4b|gemma4|qwen3[-_]asr|qwen3[-_]force(?:d)?aligner|glm-ocr",
    re.IGNORECASE,
)
_OPTIONAL_MMPROJ_VL_FAMILY_RE = re.compile(r"qwen3\.5|qwen3\.6", re.IGNORECASE)


class ModelLoadManager:
    """记录服务进程内模型加载与实例运行信息。"""

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _storage_form(model_path: str, model_name: str = "") -> str:
        if model_name.lower().endswith(".gguf"):
            return "file"
        if os.path.isfile(model_path):
            return "file"
        return "directory"

    def register_instance(
        self,
        *,
        model_name: str,
        model_path: str,
        model_type: str,
        instance_cfg: Dict[str, Any],
        process: multiprocessing.Process,
        source: str,
    ) -> None:
        record = self.models.setdefault(
            model_name,
            {
                "model_name": model_name,
                "model_path": model_path,
                "model_type": model_type,
                "storage_form": self._storage_form(model_path, model_name),
                "load_count": 0,
                "instances": [],
            },
        )
        record.update(
            {
                "model_path": model_path,
                "model_type": model_type,
                "storage_form": self._storage_form(model_path, model_name),
            }
        )

        port = int(instance_cfg["port"])
        record["instances"] = [
            inst for inst in record["instances"] if int(inst.get("port", -1)) != port
        ]
        record["instances"].append(
            {
                "port": port,
                "process_id": process.pid or 0,
                "source": source,
                "vllm_config": deepcopy(instance_cfg.get("vllm_config", {})),
                "additional_config": deepcopy(instance_cfg.get("additional_config", {})),
            }
        )
        record["instances"].sort(key=lambda item: int(item.get("port", 0)))
        record["load_count"] = len(record["instances"])

    def unregister_instance(self, *, model_name: Optional[str], port: int) -> None:
        target_model = model_name
        if target_model is None:
            for name, record in self.models.items():
                if any(int(inst.get("port", -1)) == int(port) for inst in record["instances"]):
                    target_model = name
                    break

        if not target_model or target_model not in self.models:
            return

        record = self.models[target_model]
        record["instances"] = [
            inst for inst in record["instances"] if int(inst.get("port", -1)) != int(port)
        ]
        record["load_count"] = len(record["instances"])
        if record["load_count"] <= 0:
            self.models.pop(target_model, None)

    def snapshot(self) -> Dict[str, Any]:
        return {name: deepcopy(record) for name, record in sorted(self.models.items())}


MODEL_LOAD_MANAGER = ModelLoadManager()


@main_app.on_event("shutdown")
async def shutdown_main_event():
    """主进程退出时清理所有模型 worker 及其底层推理进程。"""
    _shutdown_all_workers()


def _shutdown_all_workers() -> None:
    """同步清理所有 worker，供生命周期回调和启动失败兜底共同使用。"""
    # 复制端口列表，避免 unload_worker 修改全局映射时影响遍历。
    target_ports = list(WORKER_PROCESS_MAP)
    if not target_ports:
        return

    sync_logger.info(f"主进程退出，开始清理 {len(target_ports)} 个模型 worker")
    for target_port in target_ports:
        try:
            unload_worker(port=target_port)
        except Exception:
            sync_logger.exception(f"清理模型 worker 失败: port={target_port}")
    WORKERS[:] = [proc for proc in WORKERS if proc.is_alive()]
    sync_logger.info("所有模型 worker 清理完成")


def _check_main_port_available(port: int) -> None:
    """在加载模型前确认主服务端口可绑定，避免端口冲突留下孤儿 worker。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", int(port)))
        except OSError as exc:
            raise RuntimeError(
                f"主端口 {port} 已被占用，服务不会启动模型 worker: {exc}"
            ) from exc


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0].strip()
    if media_type != "application/json":
        raise HTTPException(
            status_code=415,
            detail="Unsupported Media Type: Only 'application/json' is allowed",
        )


@main_app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    logger.warning(f"HTTP 异常: status={exc.status_code} detail={detail}")
    error_response = {
        "detail": detail,
        "message": "请求错误" if exc.status_code < 500 else "服务器错误",
        "type": "client_error" if exc.status_code < 500 else "server_error",
        "code": exc.status_code,
    }
    return JSONResponse(status_code=exc.status_code, content=error_response)


@main_app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    first_error = exc.errors()[0] if exc.errors() else {}
    error_type = first_error.get("type", "")
    message = first_error.get("msg", "请求参数校验失败")
    logger.warning(f"请求校验失败: type={error_type} message={message}")
    error_response = {
        "detail": first_error,
        "message": message,
        "type": error_type,
        "code": 400,
    }
    if error_type == "json_invalid":
        error_response["message"] = f"请求体不是合法 JSON: {message}"
        return JSONResponse(status_code=400, content=error_response)
    return JSONResponse(status_code=400, content=error_response)


@main_app.exception_handler(ValueError)
async def value_error_exception_handler(_: Request, exc: ValueError):
    logger.warning(f"值错误: {exc}")
    error_response = {
        "detail": str(exc),
        "message": "请求参数不合法",
        "type": "validation_error",
        "code": 400,
    }
    return JSONResponse(status_code=400, content=error_response)


@main_app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    sync_logger.exception(f"未处理异常: {exc}")
    error_response = {
        "detail": str(exc),
        "message": "服务器内部错误",
        "type": "internal_error",
        "code": 500,
    }
    return JSONResponse(status_code=500, content=error_response)


async def _parse_json_body(request: Request) -> Dict[str, Any]:
    try:
        req_body = await request.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"请求体不是合法 JSON: {e.msg}") from e

    if not isinstance(req_body, dict):
        raise ValueError("请求体必须是 JSON 对象")
    return req_body


def _request_media_type(raw_request: Request) -> str:
    content_type = raw_request.headers.get("content-type", "").lower()
    return content_type.split(";", maxsplit=1)[0].strip()


def _parse_form_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _parse_timestamp_granularities(value: Any) -> list[str]:
    if value is None or value == "":
        return []

    raw_items: list[Any]
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    granularities: list[str] = []
    for item in raw_items:
        if item is None:
            continue
        if isinstance(item, str):
            stripped = item.strip()
            if not stripped:
                continue
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    raw_items.extend(parsed)
                    continue
            parts = [part.strip() for part in stripped.split(",")]
        else:
            parts = [str(item).strip()]

        for part in parts:
            if part and part not in granularities:
                granularities.append(part)
    return granularities


def _should_use_forced_aligner(req_body: Dict[str, Any]) -> bool:
    granularities = _parse_timestamp_granularities(
        req_body.get("timestamp_granularities")
        or req_body.get("timestamp_granularities[]")
    )
    if not granularities:
        return False

    invalid = [item for item in granularities if item not in {"word", "segment"}]
    if invalid:
        raise ValueError(
            "timestamp_granularities 仅支持 word 或 segment: "
            f"{invalid}"
        )
    return True


def _drop_timestamp_granularity_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"timestamp_granularities", "timestamp_granularities[]"}
    }


async def _parse_form_body(raw_request: Request) -> Dict[str, Any]:
    form = await raw_request.form()
    payload: Dict[str, Any] = {}
    for key, value in form.multi_items():
        if key in payload:
            existing = payload[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                payload[key] = [existing, value]
        else:
            payload[key] = value

    server_config = payload.get("server_config")
    if isinstance(server_config, str) and server_config.strip():
        try:
            payload["server_config"] = json.loads(server_config)
        except json.JSONDecodeError:
            pass

    if "stream" in payload:
        payload["stream"] = _parse_form_bool(payload["stream"])

    return payload


def _get_served_model_names_from_instance(
    model_name: str,
    instance_cfg: Dict[str, Any],
) -> list[str]:
    try:
        _, frontend_args, api_server_args, _ = split_vllm_config(
            instance_cfg.get("vllm_config", {})
        )
    except Exception:
        frontend_args = {}
        api_server_args = {}

    configured_model_name = frontend_args.get("served_model_name")
    if configured_model_name is None:
        configured_model_name = api_server_args.get("server_model_name")
    if configured_model_name is None:
        configured_model_name = api_server_args.get("served_model_name")
    if isinstance(configured_model_name, list):
        names = [str(name) for name in configured_model_name if str(name)]
        return names or [str(model_name)]
    if configured_model_name not in (None, ""):
        return [str(configured_model_name)]
    return [str(model_name)]


def _get_served_model_name_from_instance(model_name: str, instance_cfg: Dict[str, Any]) -> str:
    return _get_served_model_names_from_instance(model_name, instance_cfg)[0]


def _find_model_by_type(model_type: str) -> tuple[Optional[str], Optional[int]]:
    for model in APP_CONFIG.models:
        if model.model_type != model_type:
            continue
        for inst in model.instances:
            inst_type = inst.model_type or model.model_type
            if inst_type != model_type:
                continue
            instance_cfg = _build_preload_instance_config(
                model_name=model.name,
                instance=inst,
                fallback_model_type=model.model_type,
            )
            return _get_served_model_name_from_instance(model.name, instance_cfg), int(inst.port)
    return None, None


def get_runtime_settings(cfg: AppConfig) -> Dict[str, Any]:
    return {
        "main_port": int(cfg.server.main_port),
        "log_level": str(cfg.server.log_level).lower(),
        "vllm_log_level": str(cfg.server.vllm_log_level).lower(),
        "model_root_dir": str(cfg.server.model_root_dir),
    }


def _is_local_hidden_entry(name: str) -> bool:
    return name.startswith(".")


def _build_mmproj_match_key(name: str) -> str:
    family_match = _MMPROJ_FAMILY_RE.search(name)
    xh_match = _XH_RE.search(name)
    param_match = _PARAM_RE.search(name)
    version_match = _VERSION_RE.search(name)
    if not family_match or not xh_match or not param_match or not version_match:
        return ""

    family = family_match.group(0).lower().replace("_", "-")
    xh = xh_match.group(0).lower()
    param = param_match.group(0).lower()
    version = version_match.group(0).lower()
    build_time_match = _BUILD_TIME_RE.search(name)

    if (
        build_time_match
        and not _OPTIONAL_MMPROJ_VL_FAMILY_RE.search(name)
    ):
        return f"{xh}_{family}_{param}_{version}_{build_time_match.group(0)}"

    return f"{xh}_{family}_{param}_{version}"


def _parse_mmproj_vit_spec(name: str) -> str:
    match = _VIT_SPEC_RE.search(name)
    if not match:
        return "unknown"
    if match.group(2):
        return f"vit_{match.group(1)}_{match.group(2)}"
    return f"vit_{match.group(1)}"


def _is_mmproj_file_name(name: str) -> bool:
    return "mmproj" in name.lower()


def _is_mmproj_main_model_name(name: str) -> bool:
    return not _is_mmproj_file_name(name) and bool(_MMPROJ_FAMILY_RE.search(name))


def _is_local_gguf_model_file(name: str) -> bool:
    lower_name = name.lower()
    return lower_name.endswith(".gguf") and not _is_mmproj_file_name(lower_name)


def _build_mmproj_model_key(model_name: str, mmproj_name: str) -> str:
    vit_spec = _parse_mmproj_vit_spec(mmproj_name)
    if vit_spec and vit_spec != "unknown":
        return f"{Path(model_name).stem}_{vit_spec}.gguf"
    return model_name


def scan_available_models(model_root_dir: str) -> Dict[str, str]:
    global AVAILABLE_MODEL_ADDITIONAL_CONFIG, AVAILABLE_MODEL_BASE_NAMES
    if not os.path.exists(model_root_dir):
        logger.warning(f"模型目录不存在: {model_root_dir}")
        AVAILABLE_MODEL_ADDITIONAL_CONFIG = {}
        AVAILABLE_MODEL_BASE_NAMES = {}
        return {}

    if not os.path.isdir(model_root_dir):
        logger.warning(f"模型目录配置不是文件夹: {model_root_dir}")
        AVAILABLE_MODEL_ADDITIONAL_CONFIG = {}
        AVAILABLE_MODEL_BASE_NAMES = {}
        return {}

    try:
        with os.scandir(model_root_dir) as entries:
            local_entries = sorted(entries, key=lambda e: e.name.lower())
    except OSError as exc:
        logger.warning(f"扫描模型目录失败: root={model_root_dir}, error={exc}")
        AVAILABLE_MODEL_ADDITIONAL_CONFIG = {}
        AVAILABLE_MODEL_BASE_NAMES = {}
        return {}

    model_paths: Dict[str, str] = {}
    model_additional_config: Dict[str, Dict[str, Any]] = {}
    model_base_names: Dict[str, str] = {}
    mmproj_main_files: list[os.DirEntry[str]] = []
    mmproj_files: list[os.DirEntry[str]] = []
    skipped_mmproj_files = 0
    for entry in local_entries:
        if _is_local_hidden_entry(entry.name):
            continue

        try:
            if entry.is_dir():
                model_paths[entry.name] = entry.path
                continue

            if not entry.is_file():
                continue
        except OSError as exc:
            logger.warning(f"读取模型目录条目失败: path={entry.path}, error={exc}")
            continue

        if not entry.name.lower().endswith(".gguf"):
            continue

        if _is_mmproj_file_name(entry.name):
            mmproj_files.append(entry)
            skipped_mmproj_files += 1
            continue

        if _is_mmproj_main_model_name(entry.name):
            mmproj_main_files.append(entry)

        if not _is_local_gguf_model_file(entry.name):
            skipped_mmproj_files += 1
            continue

        try:
            model_paths[entry.name] = str(_ensure_gguf_runtime_dir(entry.path))
        except (OSError, ValueError) as exc:
            logger.warning(f"准备 GGUF 模型运行目录失败: path={entry.path}, error={exc}")

    if skipped_mmproj_files:
        logger.info(f"跳过 mmproj GGUF 文件: root={model_root_dir}, count={skipped_mmproj_files}")

    mmproj_index: Dict[str, list[os.DirEntry[str]]] = {}
    for mmproj in mmproj_files:
        match_key = _build_mmproj_match_key(mmproj.name)
        if match_key:
            mmproj_index.setdefault(match_key, []).append(mmproj)

    paired_count = 0
    for main_model in mmproj_main_files:
        match_key = _build_mmproj_match_key(main_model.name)
        if not match_key:
            logger.warning(f"视觉主模型名称解析失败，无法匹配 mmproj: path={main_model.path}")
            continue

        matched_mmproj_files = mmproj_index.get(match_key, [])
        if not matched_mmproj_files:
            logger.warning(f"未找到匹配的 mmproj 文件: model={main_model.name}, key={match_key}")
            continue

        main_model_path = model_paths.get(main_model.name)
        if not main_model_path:
            continue

        for mmproj in sorted(matched_mmproj_files, key=lambda e: e.name.lower()):
            model_key = _build_mmproj_model_key(main_model.name, mmproj.name)
            model_paths[model_key] = main_model_path
            model_additional_config[model_key] = {"vit_model_name": mmproj.path}
            model_base_names[model_key] = main_model.name
            paired_count += 1

    if paired_count:
        logger.info(f"视觉模型 mmproj 配对完成: root={model_root_dir}, count={paired_count}")

    AVAILABLE_MODEL_ADDITIONAL_CONFIG = model_additional_config
    AVAILABLE_MODEL_BASE_NAMES = model_base_names

    return model_paths


def _ensure_gguf_runtime_dir(gguf_path: str) -> str:
    from hliepython.config.utils import resolve_gguf_runtime_dir

    return resolve_gguf_runtime_dir(Path(gguf_path))


def refresh_available_models():
    global AVAILABLE_MODEL_PATHS
    model_root_dir = str(RUNTIME_SETTINGS.get("model_root_dir", "/houmo/llm/models"))
    AVAILABLE_MODEL_PATHS = scan_available_models(model_root_dir)
    logger.info(
        f"可用模型目录扫描完成: root={model_root_dir}, count={len(AVAILABLE_MODEL_PATHS)}"
    )


def load_default_model_args() -> Dict[str, Any]:
    default_args_path = Path(__file__).resolve().parent / "default_model_args.yaml"
    if not default_args_path.exists():
        logger.warning(f"默认模型启动配置不存在: {default_args_path}")
        return {}

    with default_args_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        logger.warning(f"默认模型启动配置格式不是对象: {default_args_path}")
        return {}

    return loaded


def _resolve_additional_config_model_refs(additional_config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = deepcopy(additional_config)
    vit_model_name = resolved.get("vit_model_name")
    if isinstance(vit_model_name, str) and vit_model_name.strip():
        resolved["vit_model_name"] = resolve_model_file_path(
            vit_model_name,
            str(RUNTIME_SETTINGS.get("model_root_dir", "/houmo/llm/models")),
        )
    return resolved


def _default_dynamic_server_config(model_name: str) -> Dict[str, Any]:
    model_defaults = DEFAULT_MODEL_ARGS.get("models", {})
    if not isinstance(model_defaults, dict):
        model_defaults = {}

    base_model_name = AVAILABLE_MODEL_BASE_NAMES.get(model_name, "")
    default_cfg = (
        model_defaults.get(model_name)
        or model_defaults.get(Path(model_name).name)
        or (model_defaults.get(base_model_name) if base_model_name else None)
        or (model_defaults.get(Path(base_model_name).name) if base_model_name else None)
    )
    if default_cfg is None:
        return {}
    if not isinstance(default_cfg, dict):
        raise ValueError(f"default_model_args.yaml 中模型 {model_name} 的配置必须是对象")

    server_config = deepcopy(default_cfg)
    if "model_model_type" in server_config and "model_type" not in server_config:
        server_config["model_type"] = server_config.pop("model_model_type")
    server_config.pop("model_path", None)
    return server_config


def _merge_dynamic_server_config(
    model_name: str,
    request_server_config: Dict[str, Any],
) -> Dict[str, Any]:
    default_cfg = _default_dynamic_server_config(model_name)
    endpoint_model_type = request_server_config.get("endpoint_model_type", "llm")

    if "vllm_config" in request_server_config:
        vllm_config = deepcopy(request_server_config.get("vllm_config") or {})
    else:
        vllm_config = deepcopy(default_cfg.get("vllm_config") or {})

    if "additional_config" in request_server_config:
        additional_config = deepcopy(request_server_config.get("additional_config") or {})
    else:
        additional_config = deepcopy(default_cfg.get("additional_config") or {})

    auto_additional_config = AVAILABLE_MODEL_ADDITIONAL_CONFIG.get(model_name, {})
    for key, value in auto_additional_config.items():
        additional_config.setdefault(key, value)

    if "model_type" in default_cfg:
        model_type = default_cfg["model_type"]
    else:
        model_type = endpoint_model_type or "llm"

    merged: Dict[str, Any] = {
        "vllm_config": vllm_config,
        "additional_config": _resolve_additional_config_model_refs(additional_config),
        "model_type": model_type,
    }
    return merged


def validate_dynamic_model_name(model_name: str):
    if model_name in AVAILABLE_MODEL_PATHS:
        return

    available = sorted(AVAILABLE_MODEL_PATHS.keys())
    raise ValueError(f"模型 {model_name} 不在可动态加载列表中，可选模型: {available}")


def _normalize_server_config(
    server_config_raw: Any,
    endpoint_model_type: str = "llm",
    allow_empty: bool = False,
) -> Dict[str, Any]:
    if server_config_raw is None:
        if not allow_empty:
            raise ValueError("请求中必须包含对象类型的 server_config 字段")
        server_config_raw = {}

    if not isinstance(server_config_raw, dict):
        raise ValueError("请求中必须包含对象类型的 server_config 字段")

    has_vllm_config = "vllm_config" in server_config_raw
    vllm_config = server_config_raw.get("vllm_config") if has_vllm_config else None
    if has_vllm_config and not isinstance(vllm_config, dict):
        raise ValueError("server_config.vllm_config 必须是对象")

    has_additional_config = "additional_config" in server_config_raw
    additional_config_raw = (
        server_config_raw.get("additional_config") if has_additional_config else {}
    )
    additional_config: Dict[str, Any] = {}
    if has_additional_config and not isinstance(additional_config_raw, dict):
        raise ValueError("server_config.additional_config 必须是对象")
    if isinstance(additional_config_raw, dict):
        additional_config = dict(additional_config_raw)

    if "device_ids" not in additional_config and "device_ids" in server_config_raw:
        additional_config["device_ids"] = server_config_raw["device_ids"]

    if "device_ids" in additional_config and not isinstance(
        additional_config["device_ids"], list
    ):
        raise ValueError("server_config.additional_config.device_ids 必须是数组")

    if "enable_cacheprompt" in additional_config and not isinstance(
        additional_config["enable_cacheprompt"], str
    ):
        raise ValueError("server_config.additional_config.enable_cacheprompt 必须是字符串")

    if "model_type" in server_config_raw:
        raise ValueError("动态加载不支持 server_config.model_type；模型类型由 default_model_args.yaml 模板决定")

    if "port" in server_config_raw:
        raise ValueError(
            "动态加载不支持 server_config.port；如需指定端口和模型 ID，请在 server_config.vllm_config.api-server 中提供完整配置"
        )

    normalized: Dict[str, Any] = {"endpoint_model_type": endpoint_model_type}
    if has_vllm_config:
        normalized["vllm_config"] = vllm_config
    if has_additional_config or additional_config:
        normalized["additional_config"] = additional_config
    return normalized


def _resolve_model_type_from_path(path: str) -> Optional[str]:
    path_alias = {
        "/v1/chat/completions": "llm",
        "/v1/responses": "llm",
        "/v1/embeddings": "embedding",
        "/score": "reranker",
        "/v1/score": "reranker",
        "/rerank": "reranker",
        "/v1/rerank": "reranker",
        "/v2/rerank": "reranker",
        "/v1/images/generations": "image",
        "/v1/audio/speech": "tts",
        "/v1/audio/transcriptions": "asr",
    }
    return path_alias.get(path)


def _extract_port_from_vllm_config(vllm_config: Dict[str, Any]) -> Optional[int]:
    for section_key in ("api-server", "api_server", "api_server_args", "server"):
        section = vllm_config.get(section_key)
        if isinstance(section, dict) and section.get("port") is not None:
            return int(section["port"])
    return None


def _is_port_available(port: int) -> bool:
    if port in PORT_MODEL_MAP or port in WORKER_PROCESS_MAP:
        return False

    main_port = int(RUNTIME_SETTINGS.get("main_port", 8000))
    if int(port) == main_port:
        return False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", int(port)))
        except OSError:
            return False
    return True


def _allocate_port(start_port: int = 7900) -> int:
    port = start_port
    while not _is_port_available(port):
        port += 1
    return port


def _allocate_instance_id() -> int:
    global NEXT_INSTANCE_ID
    instance_id = NEXT_INSTANCE_ID
    NEXT_INSTANCE_ID += 1
    return instance_id


def _serialize_runtime_params(params: Dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def _register_worker_runtime(
    model_name: str,
    process: multiprocessing.Process,
    instance_cfg: Dict[str, Any],
    source: str = "dynamic",
):
    port = int(instance_cfg["port"])
    runtime = WORKER_RUNTIME_MAP.get(port, {})
    runtime.update(
        {
            "instance_id": runtime.get("instance_id") or _allocate_instance_id(),
            "params": _serialize_runtime_params(instance_cfg),
            "port": port,
            "uuid": runtime.get("uuid") or str(uuid.uuid4()),
            "process_id": process.pid or 0,
            "models": [model_name],
        }
    )
    WORKER_RUNTIME_MAP[port] = runtime
    MODEL_LOAD_MANAGER.register_instance(
        model_name=model_name,
        model_path=str(instance_cfg.get("model_path", "")),
        model_type=str(instance_cfg.get("model_type", "llm")),
        instance_cfg=instance_cfg,
        process=process,
        source=source,
    )


def _unregister_worker_runtime(port: int):
    WORKER_RUNTIME_MAP.pop(int(port), None)


def _build_model_records_for_ports(
    model_name: str,
    ports: list[int],
    created: int,
) -> list[Dict[str, Any]]:
    sorted_ports = sorted(ports)
    if not sorted_ports:
        return [
            {
                "id": model_name,
                "aliases": [],
                "tags": [],
                "object": "model",
                "owned_by": "HLIEPython",
                "created": created,
                "port": 0,
                "status": 0,
            }
        ]

    return [
        {
            "id": model_name,
            "aliases": [],
            "tags": [],
            "object": "model",
            "owned_by": "HLIEPython",
            "created": created,
            "port": port,
            "status": 1,
        }
        for port in sorted_ports
    ]


def _get_model_list_name(model_name: str, additional_config: Dict[str, Any]) -> str:
    """返回 /v1/models 使用的模型名，多模态实例以主模型和 VIT 规格组合命名。"""
    vit_model_name = additional_config.get("vit_model_name")
    if not isinstance(vit_model_name, str) or not vit_model_name.strip():
        return model_name

    return _build_mmproj_model_key(model_name, os.path.basename(vit_model_name))


def _build_model_records() -> list[Dict[str, Any]]:
    created = int(time.time())
    records = []
    configured_model_names: set[str] = set()
    emitted_model_names: set[str] = set()
    multimodal_base_names = {
        base_name
        for alias, base_name in AVAILABLE_MODEL_BASE_NAMES.items()
        if alias != base_name
    }

    for model in APP_CONFIG.models:
        active_ports = set(MODEL_PORT_MAP.get(model.name, []))
        display_ports: Dict[str, list[int]] = {}

        for instance in model.instances:
            display_name = _get_model_list_name(
                model.name,
                instance.additional_config or {},
            )
            display_ports.setdefault(display_name, [])
            if int(instance.port) in active_ports:
                display_ports[display_name].append(int(instance.port))

        if not display_ports:
            display_ports[model.name] = sorted(active_ports)

        for display_name, ports in display_ports.items():
            records.extend(
                _build_model_records_for_ports(
                    model_name=display_name,
                    ports=ports,
                    created=created,
                )
            )
            emitted_model_names.add(display_name)

        configured_model_names.add(model.name)
        configured_model_names.add(os.path.basename(os.path.normpath(model.name)))
        for instance in model.instances:
            configured_model_names.add(
                os.path.basename(os.path.normpath(instance.model_path))
            )

    for model_dir_name in sorted(AVAILABLE_MODEL_PATHS.keys()):
        if (
            model_dir_name in configured_model_names
            or model_dir_name in emitted_model_names
            or model_dir_name in multimodal_base_names
        ):
            continue

        records.extend(
            _build_model_records_for_ports(
                model_name=model_dir_name,
                ports=MODEL_PORT_MAP.get(model_dir_name, []),
                created=created,
            )
        )

    return records


def _build_processor_records() -> list[Dict[str, Any]]:
    snapshot_time = int(time.time())
    processor = [
        {
            "params": "",
            "instance_id": 0,
            "port": int(RUNTIME_SETTINGS.get("main_port", 8000)),
            "uuid": "",
            "process_id": os.getpid(),
            "start_time": snapshot_time,
            "models": [],
        }
    ]

    for port, runtime in sorted(WORKER_RUNTIME_MAP.items(), key=lambda item: item[1]["instance_id"]):
        proc = WORKER_PROCESS_MAP.get(port)
        process_id = runtime.get("process_id", 0)
        if proc is not None and proc.pid is not None:
            process_id = proc.pid

        processor.append(
            {
                "params": runtime.get("params", ""),
                "instance_id": runtime.get("instance_id", 0),
                "port": int(port),
                "uuid": runtime.get("uuid", ""),
                "process_id": process_id,
                "start_time": snapshot_time,
                "models": runtime.get("models", []),
            }
        )

    return processor


def _build_instance_config(model_name: str, server_config: Dict[str, Any]) -> Dict[str, Any]:
    server_config = _merge_dynamic_server_config(model_name, server_config)
    vllm_config = server_config.get("vllm_config", {})
    requested_port = _extract_port_from_vllm_config(vllm_config)
    if requested_port is not None:
        bound_model = PORT_MODEL_MAP.get(requested_port)
        if bound_model is not None and bound_model != model_name:
            raise ValueError(f"端口 {requested_port} 已被模型 {bound_model} 占用")
        if not _is_port_available(requested_port):
            raise ValueError(f"端口 {requested_port} 已被占用")
        port = requested_port
    else:
        start_port = int(RUNTIME_SETTINGS.get("main_port", 8000)) + 1
        port = _allocate_port(start_port=start_port)

    model_path = AVAILABLE_MODEL_PATHS.get(model_name)
    if not model_path:
        raise ValueError(f"模型 {model_name} 未在已扫描目录中找到")

    return {
        "port": int(port),
        "model_path": model_path,
        "model_type": server_config.get("model_type", "llm"),
        "vllm_config": server_config.get("vllm_config", {}),
        "additional_config": server_config.get("additional_config", {}),
    }


def _build_preload_instance_config(model_name: str, instance, fallback_model_type: str) -> Dict[str, Any]:
    return {
        "port": int(instance.port),
        "model_path": instance.model_path,
        "model_type": instance.model_type or fallback_model_type,
        "vllm_config": instance.vllm_config or {},
        "additional_config": instance.additional_config or {},
    }


def _get_instance_request_model_names(
    model_name: str,
    instance_cfg: Dict[str, Any],
) -> list[str]:
    """返回一个预加载实例可接受的所有 OpenAI API 模型名。"""
    names = {
        str(model_name),
        _get_model_list_name(
            model_name,
            instance_cfg.get("additional_config") or {},
        ),
    }
    names.update(_get_served_model_names_from_instance(model_name, instance_cfg))
    return sorted(name for name in names if name)


def normalize_port(port_value: Any) -> Optional[int]:
    if port_value is None or port_value == "":
        return None
    return int(port_value)


def wait_for_worker_ready(port: int, timeout: float = 120.0):
    deadline = time.time() + timeout
    last_error = None

    while time.time() < deadline:
        proc = WORKER_PROCESS_MAP.get(port)
        if proc is not None and not proc.is_alive():
            raise RuntimeError(f"Worker {port} 已退出，模型加载失败")

        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if response.status_code == 200:
                return
        except Exception as e:
            last_error = e
            time.sleep(0.5)

    raise RuntimeError(f"Worker {port} 启动超时: {last_error}")


def _resolve_worker_startup_timeout(instance_cfg: Dict[str, Any]) -> float:
    additional_config = instance_cfg.get("additional_config") or {}
    timeout = additional_config.get("worker_startup_timeout")
    if timeout is None:
        vllm_config = instance_cfg.get("vllm_config") or {}
        timeout = vllm_config.get("worker_startup_timeout")
    if timeout is None:
        return 120.0
    try:
        return max(float(timeout), 1.0)
    except (TypeError, ValueError):
        sync_logger.warning(f"忽略非法 worker_startup_timeout={timeout!r}")
        return 120.0


def resolve_worker_port(model_name: Optional[str] = None, port: Optional[int] = None):
    if port is not None:
        return port if port in PORT_MODEL_MAP else None

    if not model_name:
        return None

    ports = MODEL_PORT_MAP.get(model_name, [])
    if not ports:
        return None

    if len(ports) == 1:
        return ports[0]

    return "multiple"


async def _build_req_body(request: Any, raw_request: Request) -> Dict[str, Any]:
    raw_body = await _parse_json_body(raw_request)

    if request is None:
        req_body = dict(raw_body)
    elif hasattr(request, "model_dump"):
        # 请求会作为 JSON 转发给 worker，必须使用协议字段别名。OpenAI 的
        # json_schema 模型在 Python 中将 `schema` 存为 `schema_`；若不启用
        # by_alias，worker 二次校验时会认为必填的 `schema` 缺失。
        req_body = request.model_dump(mode="json", exclude_none=True, by_alias=True)
    else:
        req_body = request.dict(exclude_none=True, by_alias=True)

    if "server_config" in raw_body:
        req_body["server_config"] = raw_body["server_config"]

    # 允许通过 body.port 或 header 指定目标 worker
    body_port = raw_body.get("port") if isinstance(raw_body, dict) else None
    if body_port is None and isinstance(raw_body.get("server_config"), dict):
        body_port = raw_body["server_config"].get("port")

    req_body["port"] = normalize_port(body_port or raw_request.headers.get("X-Worker-Port"))
    return req_body


async def _build_audio_transcriptions_meta(
    raw_request: Request
) -> Dict[str, Any]:
    media_type = _request_media_type(raw_request)
    if media_type == "application/json":
        raw_body = await _parse_json_body(raw_request)
    elif media_type in {"multipart/form-data", "application/x-www-form-urlencoded"}:
        raw_body = await _parse_form_body(raw_request)
    else:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported Media Type: Only 'application/json', "
                "'multipart/form-data' and 'application/x-www-form-urlencoded' are allowed"
            ),
        )

    req_body = dict(raw_body)

    if "server_config" in raw_body:
        req_body["server_config"] = raw_body["server_config"]

    body_port = raw_body.get("port") if isinstance(raw_body, dict) else None
    if body_port is None and isinstance(raw_body.get("server_config"), dict):
        body_port = raw_body["server_config"].get("port")

    req_body["port"] = normalize_port(body_port or raw_request.headers.get("X-Worker-Port"))
    return req_body


def start_workers(config: AppConfig):
    os.environ["VLLM_LOGGING_LEVEL"] = str(
        RUNTIME_SETTINGS.get("vllm_log_level", "warning")
    )

    if not config.preload_models_on_startup:
        logger.info("已关闭启动预加载，模型将在首次请求时动态启动")
        return

    preload_models = config.models
    if not preload_models:
        logger.info("未配置预加载 models，服务仅启动主进程，模型将按请求动态加载")
        return

    for model in preload_models:
        name = model.name
        instances = model.instances
        MODEL_PORT_MAP[name] = [inst.port for inst in instances]

        for inst in instances:
            p = inst.port
            instance_cfg = _build_preload_instance_config(
                model_name=name,
                instance=inst,
                fallback_model_type=model.model_type,
            )
            request_model_names = _get_instance_request_model_names(name, instance_cfg)
            for request_model_name in request_model_names:
                MODEL_PORT_MAP.setdefault(request_model_name, [])
                if p not in MODEL_PORT_MAP[request_model_name]:
                    MODEL_PORT_MAP[request_model_name].append(p)
            PORT_MODEL_MAP[p] = name
            model_path = inst.model_path
            additional_config = inst.additional_config
            device_ids = additional_config.get("device_ids", [])
            vllm_config = inst.vllm_config
            model_type = inst.model_type or model.model_type

            proc = multiprocessing.Process(
                target=run_worker,
                args=(
                    name,
                    p,
                    model_path,
                    vllm_config,
                    additional_config,
                    model_type,
                    config.server.logging.to_dict(),
                    request_model_names,
                ),
                daemon=False,
            )
            proc.start()
            WORKERS.append(proc)
            WORKER_PROCESS_MAP[p] = proc
            _register_worker_runtime(
                model_name=name,
                process=proc,
                instance_cfg=instance_cfg,
                source="static",
            )
            sync_logger.info(
                f"预加载模型成功: model={name} port={p} device={device_ids} pid={proc.pid}"
            )


def start_dynamic_worker(model_name: str, server_config: Dict[str, Any]):
    instance_cfg = _build_instance_config(model_name, server_config)
    port = int(instance_cfg["port"])
    request_model_names = _get_instance_request_model_names(model_name, instance_cfg)

    # 同模型同端口已经存在时直接复用
    if PORT_MODEL_MAP.get(port) == model_name and port in WORKER_PROCESS_MAP:
        return instance_cfg

    proc = multiprocessing.Process(
        target=run_worker,
        args=(
            model_name,
            port,
            instance_cfg.get("model_path"),
            instance_cfg.get("vllm_config", {}),
            instance_cfg.get("additional_config", {}),
            instance_cfg.get("model_type", "llm"),
            APP_CONFIG.server.logging.to_dict(),
            request_model_names,
        ),
        daemon=False,
    )
    proc.start()

    WORKERS.append(proc)
    WORKER_PROCESS_MAP[port] = proc
    PORT_MODEL_MAP[port] = model_name
    for request_model_name in request_model_names:
        MODEL_PORT_MAP.setdefault(request_model_name, [])
        if port not in MODEL_PORT_MAP[request_model_name]:
            MODEL_PORT_MAP[request_model_name].append(port)
    DYNAMIC_MODEL_INSTANCES.setdefault(model_name, []).append(port)
    _register_worker_runtime(
        model_name=model_name,
        process=proc,
        instance_cfg=instance_cfg,
        source="dynamic",
    )

    try:
        wait_for_worker_ready(port, timeout=_resolve_worker_startup_timeout(instance_cfg))
    except Exception:
        unload_worker(port=port)
        raise

    sync_logger.info(
        f"动态启动模型成功: model={model_name} port={port} config={instance_cfg}"
    )
    return instance_cfg


def unload_worker(model_name: Optional[str] = None, port: Optional[int] = None):
    if port is not None:
        target_ports = [int(port)]
    elif model_name:
        target_ports = list(MODEL_PORT_MAP.get(model_name, []))
    else:
        raise ValueError("请提供 model 或 port 用于卸载")

    if not target_ports:
        raise ValueError("未找到可卸载的模型实例")

    unloaded = []
    for target_port in target_ports:
        proc = WORKER_PROCESS_MAP.pop(target_port, None)
        loaded_model = PORT_MODEL_MAP.pop(target_port, model_name)

        if proc is not None and proc.is_alive():
            # 先只终止 worker，让 FastAPI shutdown 事件有机会调用
            # vLLM engine.shutdown()。不能一开始就向整个进程组发送 SIGTERM，
            # 否则 EngineCore 会和 worker 同时退出，output_handler 会把正常
            # 的 EngineCore 关闭误报为 EngineDeadError。
            process_group_id = None
            if os.name == "posix" and proc.pid is not None:
                try:
                    process_group_id = os.getpgid(proc.pid)
                    if process_group_id == os.getpgrp():
                        process_group_id = None
                except OSError:
                    process_group_id = None

            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                # 优雅关闭超时后，才清理 worker 及其 vLLM 后代进程。
                if process_group_id is not None:
                    try:
                        os.killpg(process_group_id, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    proc.join(timeout=2)
                if proc.is_alive() and process_group_id is not None:
                    try:
                        os.killpg(process_group_id, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif proc.is_alive():
                    proc.kill()
                proc.join(timeout=5)

        for request_model_name in list(MODEL_PORT_MAP):
            MODEL_PORT_MAP[request_model_name] = [
                p for p in MODEL_PORT_MAP[request_model_name] if p != target_port
            ]
            if not MODEL_PORT_MAP[request_model_name]:
                MODEL_PORT_MAP.pop(request_model_name, None)

        if loaded_model and loaded_model in DYNAMIC_MODEL_INSTANCES:
            DYNAMIC_MODEL_INSTANCES[loaded_model] = [
                p for p in DYNAMIC_MODEL_INSTANCES[loaded_model] if p != target_port
            ]
            if not DYNAMIC_MODEL_INSTANCES[loaded_model]:
                DYNAMIC_MODEL_INSTANCES.pop(loaded_model, None)

        _unregister_worker_runtime(target_port)
        MODEL_LOAD_MANAGER.unregister_instance(model_name=loaded_model, port=target_port)

        unloaded.append({"model": loaded_model, "port": target_port})

    return unloaded


def _resolve_worker_url(req_body: Dict[str, Any], raw_request: Request) -> str:
    path = raw_request.url.path
    endpoint_model_type = _resolve_model_type_from_path(path) or "llm"

    model_name = req_body.get("model") or raw_request.headers.get("X-Worker-Model")
    requested_port = normalize_port(req_body.get("port") or raw_request.headers.get("X-Worker-Port"))

    if not model_name:
        raise ValueError("请求中必须包含 model 字段")

    if requested_port is not None:
        if requested_port not in PORT_MODEL_MAP:
            raise ValueError(
                f"端口 {requested_port} 未加载模型；推理接口不会按端口创建新实例，"
                "如需加载指定端口或同模型多实例，请先调用 /models/load"
            )

        bound_model = PORT_MODEL_MAP[requested_port]
        if bound_model != model_name and requested_port not in MODEL_PORT_MAP.get(model_name, []):
            raise ValueError(f"端口 {requested_port} 已绑定模型 {bound_model}，不能用于模型 {model_name}")

    worker_port = resolve_worker_port(model_name=model_name, port=requested_port)

    if worker_port == "multiple":
        raise ValueError(
            f"模型 {model_name} 存在多个实例，请通过请求体 port 或 X-Worker-Port 指定端口: {MODEL_PORT_MAP.get(model_name, [])}"
        )

    if worker_port is None:
        validate_dynamic_model_name(model_name)

        server_config_raw = req_body.get("server_config", {})
        server_config = _normalize_server_config(
            server_config_raw,
            endpoint_model_type=endpoint_model_type,
            allow_empty=True,
        )
        instance_cfg = start_dynamic_worker(model_name, server_config)
        worker_port = int(instance_cfg["port"])

    return f"http://127.0.0.1:{worker_port}{path}"


def _rewrite_response_model(content: Any, requested_model: Optional[str]) -> Any:
    """将上游响应中的模型名恢复为客户端请求使用的公开模型名。"""
    if not requested_model or not isinstance(content, dict):
        return content

    if "model" in content:
        content["model"] = requested_model

    # Responses API 的 SSE 事件将完整响应放在 response 字段中。
    nested_response = content.get("response")
    if isinstance(nested_response, dict) and "model" in nested_response:
        nested_response["model"] = requested_model
    return content


def _rewrite_sse_model_line(line: bytes, requested_model: Optional[str]) -> bytes:
    """改写单行 SSE data JSON，同时保留原始换行符。"""
    if not requested_model:
        return line

    payload_line = line.rstrip(b"\r\n")
    line_ending = line[len(payload_line):]
    if not payload_line.startswith(b"data:"):
        return line

    payload = payload_line[len(b"data:"):].strip()
    if not payload or payload == b"[DONE]":
        return line

    try:
        content = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return line

    rewritten = _rewrite_response_model(content, requested_model)
    return (
        b"data: "
        + json.dumps(rewritten, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + line_ending
    )


async def _forward_non_stream(
    url: str,
    req_body: Dict[str, Any],
    raw_request: Request,
    response_model: Optional[str] = None,
):
    path = raw_request.url.path
    async with httpx.AsyncClient(timeout=None) as client:
        upstream_task = asyncio.create_task(client.post(url, json=req_body, timeout=None))
        while not upstream_task.done():
            if await raw_request.is_disconnected():
                upstream_task.cancel()
                logger.warning(f"客户端已断开，取消转发请求: {path} -> {url}")
                return JSONResponse(status_code=499, content={"error": "客户端已断开，已取消请求"})
            await asyncio.sleep(0.05)

        resp = await upstream_task
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            content = _rewrite_response_model(resp.json(), response_model)
            return JSONResponse(content, status_code=resp.status_code)

        return StreamingResponse(
            iter([resp.content]),
            media_type=content_type or "application/octet-stream",
            status_code=resp.status_code,
        )


async def _forward_multipart(
    url: str,
    raw_request: Request,
    response_model: Optional[str] = None,
):
    form = await raw_request.form()
    data: Dict[str, str] = {}
    files: list[tuple[str, tuple[str, bytes, str]]] = []

    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            file_bytes = await value.read()
            files.append(
                (
                    key,
                    (
                        value.filename or "upload.bin",
                        file_bytes,
                        value.content_type or "application/octet-stream",
                    ),
                )
            )
        else:
            data[key] = str(value)

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(url, data=data, files=files, timeout=None)

    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        content = _rewrite_response_model(resp.json(), response_model)
        return JSONResponse(content, status_code=resp.status_code)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=content_type or "application/octet-stream",
    )


async def _forward_multipart_to_json(
    url: str,
    *,
    data: Dict[str, str],
    files: list[tuple[str, tuple[str, bytes, str]]],
) -> tuple[Dict[str, Any], int]:
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(url, data=data, files=files, timeout=None)

    try:
        content = resp.json()
    except ValueError as exc:
        raise ValueError(
            f"上游音频接口返回非 JSON 响应: status={resp.status_code} body={resp.text[:200]}"
        ) from exc

    return content, resp.status_code


async def _extract_multipart_parts(
    raw_request: Request,
) -> tuple[Dict[str, str], list[tuple[str, tuple[str, bytes, str]]]]:
    form = await raw_request.form()
    data: Dict[str, str] = {}
    files: list[tuple[str, tuple[str, bytes, str]]] = []

    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            file_bytes = await value.read()
            files.append(
                (
                    key,
                    (
                        value.filename or "upload.bin",
                        file_bytes,
                        value.content_type or "application/octet-stream",
                    ),
                )
            )
        else:
            data[key] = str(value)

    return data, files


def _multipart_files_clone(
    files: list[tuple[str, tuple[str, bytes, str]]]
) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [(key, (filename, file_bytes, content_type)) for key, (filename, file_bytes, content_type) in files]


async def _forward_multipart_with_forced_alignment(
    raw_request: Request,
    req_body: Dict[str, Any],
):
    requested_model = req_body.get("model")
    asr_req_body = _drop_timestamp_granularity_fields(req_body)
    asr_req_body["response_format"] = "json"
    asr_req_body["stream"] = False
    asr_url = _resolve_worker_url(req_body=asr_req_body, raw_request=raw_request)

    data, files = await _extract_multipart_parts(raw_request)
    asr_data = _drop_timestamp_granularity_fields(data)
    asr_data["response_format"] = "json"
    asr_data["stream"] = "false"
    asr_result, asr_status = await _forward_multipart_to_json(
        asr_url,
        data=asr_data,
        files=_multipart_files_clone(files),
    )
    if asr_status >= 400:
        return JSONResponse(asr_result, status_code=asr_status)

    transcript_text = str(asr_result.get("text") or "").strip()
    if not transcript_text:
        raise ValueError("ASR 转录结果为空，无法进行 forced alignment")

    aligner_model, aligner_port = _find_model_by_type("forced_aligner")
    if not aligner_model:
        raise ValueError("未找到 model_type=forced_aligner 的已配置模型")

    aligner_req_body = _drop_timestamp_granularity_fields(req_body)
    aligner_req_body["model"] = aligner_model
    aligner_req_body["prompt"] = transcript_text
    aligner_req_body["response_format"] = "verbose_json"
    aligner_req_body["stream"] = False
    if aligner_port is not None:
        aligner_req_body["port"] = aligner_port
    aligner_url = _resolve_worker_url(req_body=aligner_req_body, raw_request=raw_request)

    aligner_data = _drop_timestamp_granularity_fields(data)
    aligner_data["model"] = aligner_model
    aligner_data["prompt"] = transcript_text
    aligner_data["response_format"] = "verbose_json"
    aligner_data["stream"] = "false"
    if aligner_port is not None:
        aligner_data["port"] = str(aligner_port)
    aligner_result, aligner_status = await _forward_multipart_to_json(
        aligner_url,
        data=aligner_data,
        files=_multipart_files_clone(files),
    )
    if aligner_status >= 400:
        return JSONResponse(aligner_result, status_code=aligner_status)

    aligner_result.setdefault("usage", asr_result.get("usage"))
    aligner_result["text"] = aligner_result.get("text") or transcript_text
    aligner_result = _rewrite_response_model(aligner_result, requested_model)
    return JSONResponse(aligner_result, status_code=aligner_status)


async def _forward_multipart_stream(
    url: str,
    raw_request: Request,
    response_model: Optional[str] = None,
):
    form = await raw_request.form()
    data: Dict[str, str] = {}
    files: list[tuple[str, tuple[str, bytes, str]]] = []

    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            file_bytes = await value.read()
            files.append(
                (
                    key,
                    (
                        value.filename or "upload.bin",
                        file_bytes,
                        value.content_type or "application/octet-stream",
                    ),
                )
            )
        else:
            data[key] = str(value)

    async def event_generator():
        pending = b""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                url,
                data=data,
                files=files,
                timeout=None,
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    if body:
                        yield body
                    return

                async for chunk in resp.aiter_bytes():
                    if await raw_request.is_disconnected():
                        logger.warning(f"客户端断开流式连接，停止转发: {url}")
                        return
                    if chunk:
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", maxsplit=1)
                            yield _rewrite_sse_model_line(
                                line + b"\n",
                                response_model,
                            )

                if pending:
                    yield _rewrite_sse_model_line(pending, response_model)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _forward_stream(
    url: str,
    request,
    raw_request: Request,
    response_model: Optional[str] = None,
):
    async def event_generator():
        pending = b""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=request, timeout=None) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    if body:
                        yield body
                    return

                async for chunk in resp.aiter_bytes():
                    if await raw_request.is_disconnected():
                        logger.warning(f"客户端断开流式连接，停止转发: {url}")
                        return
                    if chunk:
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", maxsplit=1)
                            yield _rewrite_sse_model_line(
                                line + b"\n",
                                response_model,
                            )

                if pending:
                    yield _rewrite_sse_model_line(pending, response_model)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@main_app.get("/v1/models")
async def list_models():
    models = _build_model_records()
    return JSONResponse(
        content={
            "data": models,
        }
    )


@main_app.get("/v1/instances")
async def list_instances():
    """聚合所有 worker 的状态（hibalance 集成）。

    返回每个 worker 实例的：
    - host / port: worker 监听地址（hibalance Agent 会通过它反查 /health / /slots）
    - model: served model name
    - status / slots_idle / slots_processing / max_concurrent_requests
    """
    timeout = httpx.Timeout(connect=0.5, read=1.0, write=1.0, pool=1.0)
    instances: list[dict] = []

    # 已启动的 worker
    for port, model_name in PORT_MODEL_MAP.items():
        proc = WORKER_PROCESS_MAP.get(port)
        alive = bool(proc and proc.is_alive())
        item = {
            "host": "127.0.0.1",
            "port": port,
            "model": model_name,
            "pid": proc.pid if proc else None,
            "alive": alive,
            "status": "loading",
            "slots_idle": 0,
            "slots_processing": 0,
            "max_concurrent_requests": 0,
        }
        if alive:
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(f"http://127.0.0.1:{port}/health")
                if resp.status_code == 200:
                    body = resp.json()
                    item.update({
                        "status": body.get("status", "ok"),
                        "slots_idle": int(body.get("slots_idle", 0) or 0),
                        "slots_processing": int(body.get("slots_processing", 0) or 0),
                        "max_concurrent_requests": int(body.get("max_concurrent_requests", 0) or 0),
                    })
            except Exception:
                item["status"] = "unreachable"
        instances.append(item)

    return JSONResponse(content={"code": 0, "instances": instances})


@main_app.get("/v1/status")
async def service_status():
    return JSONResponse(
        content={
            "code": 0,
            "processor": _build_processor_records(),
            "models": MODEL_LOAD_MANAGER.snapshot(),
        }
    )


@main_app.get("/v1/model_loads")
async def model_loads():
    return JSONResponse(content={"code": 0, "models": MODEL_LOAD_MANAGER.snapshot()})


@main_app.post("/models/load")
async def load_model(request: Request):
    req_body = await _parse_json_body(request)
    model_name = req_body.get("model")
    if not model_name:
        raise ValueError("请求中必须包含 model 字段")

    validate_dynamic_model_name(model_name)

    raw_server_config = req_body.get("server_config")
    server_config = _normalize_server_config(
        raw_server_config,
        endpoint_model_type="llm",
        allow_empty=True,
    )
    instance_cfg = start_dynamic_worker(model_name, server_config)
    worker_port = int(instance_cfg["port"])

    return JSONResponse(
        content={
            "message": f"模型 {model_name} 加载成功",
            "port": worker_port,
            "config": instance_cfg,
            "model_load": MODEL_LOAD_MANAGER.snapshot().get(model_name),
        }
    )


@main_app.post("/models/unload")
async def unload_model(request: Request):
    req_body = await _parse_json_body(request)
    model = req_body.get("model")
    port = req_body.get("port")

    if model in (None, "") and port in (None, ""):
        raise ValueError("请求中必须包含 model 或 port 字段")

    if port is not None:
        bound_model = PORT_MODEL_MAP.get(int(port))
        if model not in (None, "") and bound_model != model:
            raise ValueError(f"端口 {port} 未绑定模型 {model}，当前绑定模型: {bound_model}")
        unloaded = unload_worker(port=int(port))
    else:
        unloaded = unload_worker(model_name=str(model))
    return JSONResponse(
        content={
            "message": "模型卸载成功",
            "unloaded": unloaded,
            "models": MODEL_LOAD_MANAGER.snapshot(),
        }
    )

@main_app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    req_body = await _build_req_body(request=request, raw_request=raw_request)
    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)
    requested_model = req_body.get("model")

    stream = bool(req_body.get("stream", False))
    if stream:
        return await _forward_stream(
            worker_url,
            request=req_body,
            raw_request=raw_request,
            response_model=requested_model,
        )

    return await _forward_non_stream(
        worker_url,
        req_body=req_body,
        raw_request=raw_request,
        response_model=requested_model,
    )


@main_app.post("/v1/responses", dependencies=[Depends(validate_json_request)])
async def create_responses(request: ResponsesRequest, raw_request: Request):
    req_body = await _build_req_body(request=request, raw_request=raw_request)
    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)
    requested_model = req_body.get("model")

    stream = bool(req_body.get("stream", False))
    if stream:
        return await _forward_stream(
            worker_url,
            request=req_body,
            raw_request=raw_request,
            response_model=requested_model,
        )

    return await _forward_non_stream(
        worker_url,
        req_body=req_body,
        raw_request=raw_request,
        response_model=requested_model,
    )


@main_app.post("/v1/embeddings", dependencies=[Depends(validate_json_request)])
async def create_embeddings(request: EmbeddingRequest, raw_request: Request):
    req_body = await _build_req_body(request=request, raw_request=raw_request)
    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)
    return await _forward_non_stream(
        worker_url,
        req_body=req_body,
        raw_request=raw_request,
        response_model=req_body.get("model"),
    )


@main_app.post("/score", dependencies=[Depends(validate_json_request)])
@main_app.post("/v1/score", dependencies=[Depends(validate_json_request)])
async def create_score(raw_request: Request):
    req_body = await _build_req_body(request=None, raw_request=raw_request)
    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)
    return await _forward_non_stream(
        worker_url,
        req_body=req_body,
        raw_request=raw_request,
        response_model=req_body.get("model"),
    )


@main_app.post("/rerank", dependencies=[Depends(validate_json_request)])
@main_app.post("/v1/rerank", dependencies=[Depends(validate_json_request)])
@main_app.post("/v2/rerank", dependencies=[Depends(validate_json_request)])
async def create_rerank(raw_request: Request):
    req_body = await _build_req_body(request=None, raw_request=raw_request)
    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)
    return await _forward_non_stream(
        worker_url,
        req_body=req_body,
        raw_request=raw_request,
        response_model=req_body.get("model"),
    )


@main_app.post("/v1/images/generations", dependencies=[Depends(validate_json_request)])
async def create_image_generation(raw_request: Request):
    req_body = await _build_req_body(request=None, raw_request=raw_request)
    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)
    return await _forward_non_stream(
        worker_url,
        req_body=req_body,
        raw_request=raw_request,
        response_model=req_body.get("model"),
    )


@main_app.post("/v1/audio/speech", dependencies=[Depends(validate_json_request)])
async def create_audio_speech(request: AudioSpeechRequest, raw_request: Request):
    req_body = await _build_req_body(request=request, raw_request=raw_request)
    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)
    return await _forward_non_stream(
        worker_url,
        req_body=req_body,
        raw_request=raw_request,
        response_model=req_body.get("model"),
    )


@main_app.post("/v1/audio/transcriptions")
async def create_audio_transcriptions(raw_request: Request):
    media_type = _request_media_type(raw_request)

    if media_type == "application/json":
        req_body = await _build_req_body(request=None, raw_request=raw_request)
        if _should_use_forced_aligner(req_body):
            raise ValueError("timestamp_granularities 对齐模式当前仅支持 multipart/form-data 请求")
        worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)

        stream = bool(req_body.get("stream", False))
        if stream:
            return await _forward_stream(
                worker_url,
                request=req_body,
                raw_request=raw_request,
                response_model=req_body.get("model"),
            )
        return await _forward_non_stream(
            worker_url,
            req_body=req_body,
            raw_request=raw_request,
            response_model=req_body.get("model"),
        )

    req_body = await _build_audio_transcriptions_meta(raw_request=raw_request)
    if _should_use_forced_aligner(req_body):
        stream = bool(req_body.get("stream", False))
        if stream:
            raise ValueError("timestamp_granularities 对齐模式不支持 stream=true")
        return await _forward_multipart_with_forced_alignment(
            raw_request=raw_request,
            req_body=req_body,
        )

    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)

    stream = bool(req_body.get("stream", False))
    if stream:
        return await _forward_multipart_stream(
            worker_url,
            raw_request=raw_request,
            response_model=req_body.get("model"),
        )
    return await _forward_multipart(
        worker_url,
        raw_request=raw_request,
        response_model=req_body.get("model"),
    )


def main():
    global APP_CONFIG, RUNTIME_SETTINGS, logger, sync_logger, DEFAULT_MODEL_ARGS
    APP_CONFIG = load_app_config()
    logging_settings = configure_logging(
        APP_CONFIG.server.logging,
        process_role="main",
        port=int(APP_CONFIG.server.main_port),
    )
    logger = get_async_logger(f"main:{APP_CONFIG.server.main_port}")
    sync_logger = get_sync_logger(f"main:{APP_CONFIG.server.main_port}")
    RUNTIME_SETTINGS = get_runtime_settings(APP_CONFIG)
    DEFAULT_MODEL_ARGS = load_default_model_args()
    refresh_available_models()

    logger.debug(
        f"加载配置:\n{json.dumps(APP_CONFIG.to_dict(), ensure_ascii=False, indent=2)}"
    )
    sync_logger.info(
        "运行时配置: "
        f"main_port={RUNTIME_SETTINGS['main_port']}, "
        f"log_level={RUNTIME_SETTINGS['log_level']}, "
        f"vllm_log_level={RUNTIME_SETTINGS['vllm_log_level']}"
    )
    sync_logger.info(
        "日志配置: "
        f"log_dir={APP_CONFIG.server.logging.directory}, "
        f"sync_enabled={APP_CONFIG.server.logging.sync_enabled}, "
        f"sync_level={APP_CONFIG.server.logging.sync_level}, "
        f"sync_file={logging_settings.get('resolved_sync_filename')}, "
        f"rotation={APP_CONFIG.server.logging.rotation}, "
        f"retention={APP_CONFIG.server.logging.retention}"
    )

    main_port = int(RUNTIME_SETTINGS["main_port"])
    _check_main_port_available(main_port)
    try:
        start_workers(APP_CONFIG)
        sync_logger.info(f"主进程启动: http://127.0.0.1:{main_port}")
        uvicorn.run(
            main_app,
            host="0.0.0.0",
            port=main_port,
            log_level=RUNTIME_SETTINGS["log_level"],
        )
    except Exception:
        sync_logger.exception("主服务启动或运行失败")
        raise
    finally:
        # Uvicorn 可能因端口占用、启动异常或外部停止而直接返回；不能只
        # 依赖 FastAPI shutdown 事件，否则已启动的模型 worker 会继续运行。
        _shutdown_all_workers()


if __name__ == "__main__":
    main()
