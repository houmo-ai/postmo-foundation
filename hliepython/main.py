import os
import time
import json
import asyncio
import multiprocessing
import uuid
import httpx
import uvicorn
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response, StreamingResponse, JSONResponse
from starlette.datastructures import UploadFile
from hliepython.utils.logging import configure_logging, get_async_logger, get_sync_logger
from hliepython.config import AppConfig, load_app_config

from hliepython.model_server.engines.protocol.chat_completions import (
    ChatCompletionRequest,
)
from hliepython.model_server.engines.protocol.audio_speech import (
    AudioSpeechRequest,
)
from hliepython.model_server.engines.protocol.embeddings import (
    EmbeddingRequest,
)
from hliepython.model_server.worker import run_worker

main_app = FastAPI(title="主进程 8000")
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
    "main_port": 8000,
    "log_level": "info",
    "vllm_log_level": "warning",
    "model_root_dir": "/houmo/llm/models",
}
AVAILABLE_MODEL_PATHS: Dict[str, str] = {}
WORKER_RUNTIME_MAP: Dict[int, Dict[str, Any]] = {}
NEXT_INSTANCE_ID = 1


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


async def _parse_form_body(raw_request: Request) -> Dict[str, Any]:
    form = await raw_request.form()
    payload: Dict[str, Any] = {}
    for key, value in form.multi_items():
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


def get_runtime_settings(cfg: AppConfig) -> Dict[str, Any]:
    return {
        "main_port": int(cfg.server.main_port),
        "log_level": str(cfg.server.log_level).lower(),
        "vllm_log_level": str(cfg.server.vllm_log_level).lower(),
        "model_root_dir": str(cfg.server.model_root_dir),
    }


def scan_available_models(model_root_dir: str) -> Dict[str, str]:
    if not os.path.exists(model_root_dir):
        logger.warning(f"模型目录不存在: {model_root_dir}")
        return {}

    if not os.path.isdir(model_root_dir):
        logger.warning(f"模型目录配置不是文件夹: {model_root_dir}")
        return {}

    model_paths: Dict[str, str] = {}
    for entry in sorted(os.scandir(model_root_dir), key=lambda e: e.name):
        if entry.is_dir() and not entry.name.startswith("."):
            model_paths[entry.name] = entry.path

    return model_paths


def refresh_available_models():
    global AVAILABLE_MODEL_PATHS
    model_root_dir = str(RUNTIME_SETTINGS.get("model_root_dir", "/houmo/llm/models"))
    AVAILABLE_MODEL_PATHS = scan_available_models(model_root_dir)
    logger.info(
        f"可用模型目录扫描完成: root={model_root_dir}, count={len(AVAILABLE_MODEL_PATHS)}"
    )


def validate_dynamic_model_name(model_name: str):
    if model_name in AVAILABLE_MODEL_PATHS:
        return

    available = sorted(AVAILABLE_MODEL_PATHS.keys())
    raise ValueError(f"模型 {model_name} 不在可动态加载列表中，可选模型: {available}")


def _normalize_server_config(
    server_config_raw: Any,
    endpoint_model_type: str = "text",
    allow_empty: bool = False,
) -> Dict[str, Any]:
    if server_config_raw is None:
        if not allow_empty:
            raise ValueError("请求中必须包含对象类型的 server_config 字段")
        server_config_raw = {}

    if not isinstance(server_config_raw, dict):
        raise ValueError("请求中必须包含对象类型的 server_config 字段")

    vllm_config = server_config_raw.get("vllm_config", {})
    if not isinstance(vllm_config, dict):
        raise ValueError("server_config.vllm_config 必须是对象")

    additional_config = server_config_raw.get("additional_config", {})
    if not isinstance(additional_config, dict):
        raise ValueError("server_config.additional_config 必须是对象")

    if "device_ids" not in additional_config and "device_ids" in server_config_raw:
        additional_config = dict(additional_config)
        additional_config["device_ids"] = server_config_raw["device_ids"]

    device_ids = additional_config.get("device_ids", [0])
    if not isinstance(device_ids, list):
        raise ValueError("server_config.additional_config.device_ids 必须是数组")
    additional_config = dict(additional_config)
    additional_config["device_ids"] = device_ids

    if "enable_cacheprompt" in additional_config and not isinstance(
        additional_config["enable_cacheprompt"], str
    ):
        raise ValueError("server_config.additional_config.enable_cacheprompt 必须是字符串")
    additional_config.setdefault("enable_cacheprompt", "")

    model_type = server_config_raw.get("model_type")
    if model_type is not None and not isinstance(model_type, str):
        raise ValueError("server_config.model_type 必须是字符串")

    port = server_config_raw.get("port")

    normalized: Dict[str, Any] = {
        "vllm_config": vllm_config,
        "additional_config": additional_config,
        "model_type": model_type or endpoint_model_type,
    }
    if port is not None:
        normalized["port"] = int(port)
    return normalized


def _resolve_model_type_from_path(path: str) -> Optional[str]:
    path_alias = {
        "/v1/chat/completions": "text",
        "/v1/embeddings": "embedding",
        "/v1/images/generations": "image",
        "/v1/audio/speech": "tts",
        "/v1/audio/transcriptions": "asr",
    }
    return path_alias.get(path)


def _allocate_port(start_port: int = 7900) -> int:
    port = start_port
    while port in PORT_MODEL_MAP:
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


def _build_model_records() -> list[Dict[str, Any]]:
    created = int(time.time())
    records = []
    configured_dir_names: set[str] = set()

    for model in APP_CONFIG.models:
        for instance in model.instances:
            configured_dir_names.add(os.path.basename(os.path.normpath(instance.model_path)))

        records.extend(
            _build_model_records_for_ports(
                model_name=model.name,
                ports=MODEL_PORT_MAP.get(model.name, []),
                created=created,
            )
        )

    for model_dir_name in sorted(AVAILABLE_MODEL_PATHS.keys()):
        if model_dir_name in configured_dir_names:
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
    requested_port = server_config.get("port")
    if requested_port is not None:
        port = int(requested_port)
        bound_model = PORT_MODEL_MAP.get(port)
        if bound_model is not None and bound_model != model_name:
            raise ValueError(f"端口 {port} 已被模型 {bound_model} 占用")
    else:
        start_port = int(RUNTIME_SETTINGS.get("main_port", 8000)) + 1
        port = _allocate_port(start_port=start_port)

    model_path = AVAILABLE_MODEL_PATHS.get(model_name)
    if not model_path:
        raise ValueError(f"模型 {model_name} 未在已扫描目录中找到")

    return {
        "port": int(port),
        "model_path": model_path,
        "model_type": server_config.get("model_type", "text"),
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
        req_body = request.model_dump(exclude_none=True)
    else:
        req_body = request.dict(exclude_none=True)

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
                ),
                daemon=False,
            )
            proc.start()
            WORKERS.append(proc)
            WORKER_PROCESS_MAP[p] = proc
            _register_worker_runtime(
                model_name=name,
                process=proc,
                instance_cfg=_build_preload_instance_config(
                    model_name=name,
                    instance=inst,
                    fallback_model_type=model.model_type,
                ),
            )
            sync_logger.info(
                f"预加载模型成功: model={name} port={p} device={device_ids} pid={proc.pid}"
            )


def start_dynamic_worker(model_name: str, server_config: Dict[str, Any]):
    instance_cfg = _build_instance_config(model_name, server_config)
    port = int(instance_cfg["port"])

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
            instance_cfg.get("model_type", "text"),
            APP_CONFIG.server.logging.to_dict(),
        ),
        daemon=False,
    )
    proc.start()

    WORKERS.append(proc)
    WORKER_PROCESS_MAP[port] = proc
    PORT_MODEL_MAP[port] = model_name
    MODEL_PORT_MAP.setdefault(model_name, []).append(port)
    DYNAMIC_MODEL_INSTANCES.setdefault(model_name, []).append(port)
    _register_worker_runtime(model_name=model_name, process=proc, instance_cfg=instance_cfg)

    try:
        wait_for_worker_ready(port)
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
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)

        if loaded_model and loaded_model in MODEL_PORT_MAP:
            MODEL_PORT_MAP[loaded_model] = [
                p for p in MODEL_PORT_MAP[loaded_model] if p != target_port
            ]
            if not MODEL_PORT_MAP[loaded_model]:
                MODEL_PORT_MAP.pop(loaded_model, None)

        if loaded_model and loaded_model in DYNAMIC_MODEL_INSTANCES:
            DYNAMIC_MODEL_INSTANCES[loaded_model] = [
                p for p in DYNAMIC_MODEL_INSTANCES[loaded_model] if p != target_port
            ]
            if not DYNAMIC_MODEL_INSTANCES[loaded_model]:
                DYNAMIC_MODEL_INSTANCES.pop(loaded_model, None)

        _unregister_worker_runtime(target_port)

        unloaded.append({"model": loaded_model, "port": target_port})

    return unloaded


def _resolve_worker_url(req_body: Dict[str, Any], raw_request: Request) -> str:
    path = raw_request.url.path
    endpoint_model_type = _resolve_model_type_from_path(path) or "text"

    model_name = req_body.get("model") or raw_request.headers.get("X-Worker-Model")
    port = normalize_port(req_body.get("port") or raw_request.headers.get("X-Worker-Port"))

    if not model_name:
        raise ValueError("请求中必须包含 model 字段")

    if port is not None and port in PORT_MODEL_MAP:
        bound_model = PORT_MODEL_MAP[port]
        if bound_model != model_name:
            raise ValueError(f"端口 {port} 已绑定模型 {bound_model}，不能用于模型 {model_name}")

    worker_port = resolve_worker_port(model_name=model_name, port=port)

    if worker_port == "multiple":
        raise ValueError(
            f"模型 {model_name} 存在多个实例，请通过 server_config.port 或 X-Worker-Port 指定端口: {MODEL_PORT_MAP.get(model_name, [])}"
        )

    if worker_port is None:
        validate_dynamic_model_name(model_name)

        server_config_raw = req_body.get("server_config")
        if server_config_raw is None:
            raise ValueError(
                "模型未加载。请先调用 POST /hlie/v1/models 加载，或在本次请求中提供 server_config 进行动态拉起"
            )

        server_config = _normalize_server_config(
            server_config_raw,
            endpoint_model_type=endpoint_model_type,
            allow_empty=False,
        )
        instance_cfg = start_dynamic_worker(model_name, server_config)
        worker_port = int(instance_cfg["port"])

    return f"http://127.0.0.1:{worker_port}{path}"


async def _forward_non_stream(url: str, req_body: Dict[str, Any], raw_request: Request):
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
            return JSONResponse(resp.json(), status_code=resp.status_code)

        return StreamingResponse(
            iter([resp.content]),
            media_type=content_type or "application/octet-stream",
            status_code=resp.status_code,
        )


async def _forward_multipart(url: str, raw_request: Request):
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
        return JSONResponse(resp.json(), status_code=resp.status_code)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=content_type or "application/octet-stream",
    )


async def _forward_multipart_stream(url: str, raw_request: Request):
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
                        yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _forward_stream(url: str, request, raw_request: Request):
    async def event_generator():
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
                        yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@main_app.get("/v1/models")
@main_app.get("/hlie/v1/models")
async def list_models():
    models = _build_model_records()
    return JSONResponse(
        content={
            "data": models,
        }
    )


@main_app.get("/hlie/v1/status")
async def service_status():
    return JSONResponse(content={"code": 0, "processor": _build_processor_records()})


@main_app.post("/hlie/v1/models")
async def load_model(request: Request):
    req_body = await _parse_json_body(request)
    model_name = req_body.get("model_name")
    if not model_name:
        raise ValueError("请求中必须包含 model_name 字段")

    validate_dynamic_model_name(model_name)

    raw_server_config = req_body.get("server_config")
    server_config = _normalize_server_config(
        raw_server_config,
        endpoint_model_type="text",
        allow_empty=True,
    )
    instance_cfg = start_dynamic_worker(model_name, server_config)
    worker_port = int(instance_cfg["port"])

    return JSONResponse(
        content={
            "message": f"模型 {model_name} 加载成功",
            "port": worker_port,
            "config": instance_cfg,
        }
    )


@main_app.delete("/hlie/v1/models/{model_name:path}")
async def unload_model(model_name: str):
    unloaded = unload_worker(model_name=model_name)
    return JSONResponse(content={"message": "模型卸载成功", "unloaded": unloaded})


@main_app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    req_body = await _build_req_body(request=request, raw_request=raw_request)
    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)

    stream = bool(req_body.get("stream", False))
    if stream:
        return await _forward_stream(worker_url, request=req_body, raw_request=raw_request)

    return await _forward_non_stream(worker_url, req_body=req_body, raw_request=raw_request)


@main_app.post("/v1/embeddings", dependencies=[Depends(validate_json_request)])
async def create_embeddings(request: EmbeddingRequest, raw_request: Request):
    req_body = await _build_req_body(request=request, raw_request=raw_request)
    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)
    return await _forward_non_stream(worker_url, req_body=req_body, raw_request=raw_request)


@main_app.post("/v1/images/generations", dependencies=[Depends(validate_json_request)])
async def create_image_generation(raw_request: Request):
    req_body = await _build_req_body(request=None, raw_request=raw_request)
    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)
    return await _forward_non_stream(worker_url, req_body=req_body, raw_request=raw_request)


@main_app.post("/v1/audio/speech", dependencies=[Depends(validate_json_request)])
async def create_audio_speech(request: AudioSpeechRequest, raw_request: Request):
    req_body = await _build_req_body(request=request, raw_request=raw_request)
    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)
    return await _forward_non_stream(worker_url, req_body=req_body, raw_request=raw_request)


@main_app.post("/v1/audio/transcriptions")
async def create_audio_transcriptions(raw_request: Request):
    media_type = _request_media_type(raw_request)

    if media_type == "application/json":
        req_body = await _build_req_body(request=None, raw_request=raw_request)
        worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)

        stream = bool(req_body.get("stream", False))
        if stream:
            return await _forward_stream(worker_url, request=req_body, raw_request=raw_request)
        return await _forward_non_stream(worker_url, req_body=req_body, raw_request=raw_request)

    req_body = await _build_audio_transcriptions_meta(raw_request=raw_request)
    worker_url = _resolve_worker_url(req_body=req_body, raw_request=raw_request)

    stream = bool(req_body.get("stream", False))
    if stream:
        return await _forward_multipart_stream(worker_url, raw_request=raw_request)
    return await _forward_multipart(worker_url, raw_request=raw_request)


def main():
    global APP_CONFIG, RUNTIME_SETTINGS, logger, sync_logger
    APP_CONFIG = load_app_config()
    logging_settings = configure_logging(
        APP_CONFIG.server.logging,
        process_role="main",
        port=int(APP_CONFIG.server.main_port),
    )
    logger = get_async_logger(f"main:{APP_CONFIG.server.main_port}")
    sync_logger = get_sync_logger(f"main:{APP_CONFIG.server.main_port}")
    RUNTIME_SETTINGS = get_runtime_settings(APP_CONFIG)
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

    start_workers(APP_CONFIG)
    sync_logger.info(f"主进程启动: http://127.0.0.1:{RUNTIME_SETTINGS['main_port']}")
    uvicorn.run(
        main_app,
        host="0.0.0.0",
        port=RUNTIME_SETTINGS["main_port"],
        log_level=RUNTIME_SETTINGS["log_level"],
    )


if __name__ == "__main__":
    main()
