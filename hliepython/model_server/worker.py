import os
import json
import asyncio
import base64
import importlib
from http import HTTPStatus
from collections.abc import AsyncGenerator
from inspect import isawaitable
from hliepython.utils.logging import configure_logging, get_async_logger, get_sync_logger
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile
from typing import Dict, Any, Optional
from .inference_adapter import InferenceEngineFactory, InferenceEngine

from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranslationResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
    StreamingResponsesResponse,
)

from hliepython.model_server.engines.protocol.audio_speech import (
    AudioSpeechRequest,
)
from hliepython.model_server.engines.protocol.chat_completions import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    # ErrorResponse
)
from hliepython.model_server.engines.protocol.embeddings import (
    EmbeddingRequest,
)
from hliepython.model_server.engines.utils import split_vllm_config
from hliepython.model_server.slot_tracker import SlotTracker

# 初始化子进程服务
app = FastAPI()
logger = get_async_logger("worker")
sync_logger = get_sync_logger("worker")
API_KEYS: set[str] = set()


def _reconfigure_app_for_api_server(api_server_args: dict[str, Any]) -> None:
    """把 vLLM api-server 层参数映射到当前 worker FastAPI app。"""
    global API_KEYS
    if not api_server_args:
        return

    if api_server_args.get("disable_fastapi_docs"):
        app.docs_url = None
        app.redoc_url = None
        app.openapi_url = None

    root_path = api_server_args.get("root_path")
    if isinstance(root_path, str):
        app.root_path = root_path

    api_key = api_server_args.get("api_key")
    if isinstance(api_key, str) and api_key:
        API_KEYS = {api_key}
    elif isinstance(api_key, list):
        API_KEYS = {str(key) for key in api_key if str(key)}

    allowed_origins = api_server_args.get("allowed_origins")
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=api_server_args.get("allowed_methods") or ["*"],
            allow_headers=api_server_args.get("allowed_headers") or ["*"],
        )


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if not API_KEYS or request.url.path == "/health":
        return await call_next(request)

    auth = request.headers.get("authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else auth
    if token not in API_KEYS:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)

# 子进程绑定的模型和端口
MODEL_NAME = ""
ALLOWED_MODEL_NAMES: set[str] = set()
PORT = 0

# 子进程实例配置
INSTANCE_CONFIG: Dict[str, Any] = {}

# 推理引擎实例
engine: Optional[InferenceEngine] = None

# 槽位跟踪器（hibalance 集成）
SLOT_TRACKER: Optional[SlotTracker] = None

# vLLM engine max-num-seqs（vLLM 的并发上限）
VLLM_MAX_NUM_SEQS: int = 1

# 路由注册表
routes: Dict[str, str] = {
    "/v1/chat/completions": "create_chat_completions",
    "/v1/responses": "create_responses",
    "/v1/embeddings": "create_embeddings",
    "/score": "create_score",
    "/v1/score": "create_score",
    "/rerank": "create_rerank",
    "/v1/rerank": "create_rerank",
    "/v2/rerank": "create_rerank",
    "/v1/images/generations": "create_images_generations",
    "/v1/audio/speech": "create_audio_speech",
    "/v1/audio/transcriptions": "create_audio_transcriptions",
}

@app.on_event("startup")
async def startup_event():
    """启动时初始化推理引擎"""
    global engine, SLOT_TRACKER, VLLM_MAX_NUM_SEQS
    INSTANCE_CONFIG["model_name"] = MODEL_NAME
    engine = InferenceEngineFactory.get_engine(INSTANCE_CONFIG)

    # 读取 vllm_config.engine.max_num_seqs 作为 slot 上限
    vllm_cfg = INSTANCE_CONFIG.get("vllm_config") or {}
    engine_cfg = vllm_cfg.get("engine") or {}
    max_seqs = engine_cfg.get("max_num_seqs")
    if not isinstance(max_seqs, int) or max_seqs < 1:
        max_seqs = 1
    VLLM_MAX_NUM_SEQS = max_seqs
    SLOT_TRACKER = SlotTracker(max_slots=max_seqs)
    sync_logger.info(
        f"Worker {MODEL_NAME}:{PORT} 推理引擎初始化完成 "
        f"(max_slots={max_seqs})"
    )

@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理资源"""
    global engine, SLOT_TRACKER
    if engine:
        try:
            engine = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            sync_logger.info(f"Worker {MODEL_NAME}:{PORT} 推理引擎关闭")
        finally:
            engine = None
    if SLOT_TRACKER:
        SLOT_TRACKER.shutdown()
        SLOT_TRACKER = None


@app.get("/health")
async def health_check():
    """兼容 hibalance Agent 的健康检查端点。

    关键字段：
    - status: 文本"ok"/"busy"
    - slots_idle: 空闲槽位数
    - slots_processing: 正在执行的槽位数
    - model: served model name
    - max_concurrent_requests: 槽位上限
    """
    if SLOT_TRACKER is None:
        return {
            "status": "loading",
            "model": MODEL_NAME,
            "allowed_models": sorted(ALLOWED_MODEL_NAMES),
            "port": PORT,
            "engine_loaded": engine is not None,
            "slots_idle": 0,
            "slots_processing": 0,
            "max_concurrent_requests": 0,
        }
    base = SLOT_TRACKER.aggregated_status()
    base.update({
        "model": MODEL_NAME,
        "allowed_models": sorted(ALLOWED_MODEL_NAMES),
        "port": PORT,
        "engine_loaded": engine is not None,
        "max_concurrent_requests": SLOT_TRACKER.max_slots,
    })
    return base


@app.get("/slots")
async def slots_info():
    """hi.cpp 兼容的 /slots 端点（hibalance Agent 用来累计 slot 数）。"""
    if SLOT_TRACKER is None:
        return []
    return SLOT_TRACKER.to_slots_array()


@app.websocket("/ws/audio/transcriptions")
async def websocket_audio_transcriptions(websocket: WebSocket):
    """模拟客户端与服务端的实时语音通信（ASR）。

    支持两种输入：
    1) 二进制帧：直接发送一段完整音频 bytes（如 wav/mp3）。
    2) 文本 JSON：{"type":"audio_base64","data":"..."}

    可选控制消息：
    - {"type":"config","language":"zh","prompt":"..."}
    - {"type":"ping"}
    - {"type":"stop"}
    """
    global engine

    await websocket.accept()

    if engine is None:
        await websocket.send_json({"type": "error", "message": "推理引擎未初始化"})
        await websocket.close(code=1011)
        return

    if not hasattr(engine, "websocket_transcribe_bytes"):
        await websocket.send_json({"type": "error", "message": "当前模型不支持 WebSocket ASR"})
        await websocket.close(code=1003)
        return

    language = "zh"
    prompt = ""
    return_mode = "delta"  # full | delta | chunk_only
    chunk_index = 0
    audio_buffer = bytearray()
    last_full_text = ""
    delta_display_text = ""
    chunk_only_full_text = ""
    chunk_only_audio_buffer = bytearray()
    chunk_only_force_cumulative = False

    def _longest_common_prefix_len(a: str, b: str) -> int:
        max_len = min(len(a), len(b))
        idx = 0
        while idx < max_len and a[idx] == b[idx]:
            idx += 1
        return idx

    def _suffix_prefix_overlap_len(previous_text: str, current_text: str) -> int:
        max_len = min(len(previous_text), len(current_text), 512)
        for size in range(max_len, 0, -1):
            if previous_text[-size:] == current_text[:size]:
                return size
        return 0

    def _delta_append_only(display_text: str, current_full_text: str) -> tuple[str, str]:
        if not display_text:
            return current_full_text, current_full_text

        if current_full_text.startswith(display_text):
            delta = current_full_text[len(display_text):]
            return delta, current_full_text

        # Prefer a suffix/prefix overlap to keep streamed chunks readable.
        overlap = _suffix_prefix_overlap_len(display_text, current_full_text)
        if overlap > 0:
            delta = current_full_text[overlap:]
            return delta, display_text + delta

        # Fallback: if there is still a meaningful shared prefix, append from
        # that point; otherwise skip this frame to avoid confusing rewinds.
        lcp = _longest_common_prefix_len(display_text, current_full_text)
        min_anchor = min(16, len(display_text), len(current_full_text))
        if lcp >= min_anchor:
            delta = current_full_text[lcp:]
            return delta, display_text + delta

        return "", display_text

    async def _transcribe(audio_bytes: bytes) -> str:
        transcribe_fn = getattr(engine, "websocket_transcribe_bytes")
        result = transcribe_fn(audio_bytes, language=language, prompt=prompt)
        if isawaitable(result):
            result = await result
        return str(result)

    async def _handle_chunk(audio_piece: bytes) -> tuple[str, str]:
        nonlocal chunk_only_full_text, last_full_text, delta_display_text, chunk_only_force_cumulative

        if return_mode == "chunk_only":
            chunk_only_audio_buffer.extend(audio_piece)
            if chunk_only_force_cumulative:
                full_text = await _transcribe(bytes(chunk_only_audio_buffer))
                chunk_text, chunk_only_full_text = _delta_append_only(
                    chunk_only_full_text,
                    full_text,
                )
                return chunk_text, chunk_only_full_text

            try:
                chunk_text = await _transcribe(audio_piece)
                chunk_only_full_text += chunk_text
            except ValueError:
                # Compressed formats (e.g. mp3) often cannot be decoded from
                # arbitrary byte slices. Fall back to cumulative decode and
                # emit only newly added text.
                chunk_only_force_cumulative = True
                full_text = await _transcribe(bytes(chunk_only_audio_buffer))
                chunk_text, chunk_only_full_text = _delta_append_only(
                    chunk_only_full_text,
                    full_text,
                )
            return chunk_text, chunk_only_full_text

        audio_buffer.extend(audio_piece)
        full_text = await _transcribe(bytes(audio_buffer))
        if return_mode == "delta":
            out_text, delta_display_text = _delta_append_only(delta_display_text, full_text)
        else:
            out_text = full_text
        last_full_text = full_text
        return out_text, full_text

    await websocket.send_json(
        {
            "type": "ready",
            "model": MODEL_NAME,
            "port": PORT,
            "message": "WebSocket ASR 已连接",
        }
    )

    try:
        while True:
            frame = await websocket.receive()

            if frame.get("type") == "websocket.disconnect":
                break

            text_payload = frame.get("text")
            bytes_payload = frame.get("bytes")

            if text_payload is not None:
                try:
                    msg = json.loads(text_payload)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "无效 JSON 消息"})
                    continue

                msg_type = msg.get("type")
                if msg_type == "config":
                    nonlocal_reset = False
                    language = msg.get("language", language)
                    prompt = msg.get("prompt", prompt)
                    mode = str(msg.get("return_mode", return_mode)).lower()
                    if mode in {"delta", "full", "chunk_only"}:
                        return_mode = mode
                    if msg.get("reset", False):
                        nonlocal_reset = True
                        audio_buffer.clear()
                        chunk_index = 0
                        last_full_text = ""
                        delta_display_text = ""
                        chunk_only_full_text = ""
                        chunk_only_audio_buffer.clear()
                        chunk_only_force_cumulative = False
                    await websocket.send_json(
                        {
                            "type": "ack",
                            "message": "配置已更新",
                            "language": language,
                            "return_mode": return_mode,
                            "reset": nonlocal_reset,
                        }
                    )
                    continue

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                if msg_type == "stop":
                    # 在 stop 时返回一次最终结果，便于客户端收敛文本。
                    if return_mode == "chunk_only":
                        final_text = chunk_only_full_text
                        if not final_text and chunk_only_audio_buffer:
                            final_text = await _transcribe(bytes(chunk_only_audio_buffer))
                    else:
                        final_text = last_full_text
                        if not final_text and audio_buffer:
                            final_text = await _transcribe(bytes(audio_buffer))

                    if return_mode == "delta":
                        final_payload, delta_display_text = _delta_append_only(
                            delta_display_text,
                            final_text,
                        )
                    else:
                        final_payload = final_text

                    await websocket.send_json(
                        {
                            "type": "done",
                            "text": final_payload,
                            "full_text": final_text,
                            "return_mode": return_mode,
                            "final": True,
                        }
                    )
                    await websocket.close(code=1000)
                    break

                if msg_type == "reset":
                    audio_buffer.clear()
                    chunk_index = 0
                    last_full_text = ""
                    delta_display_text = ""
                    chunk_only_full_text = ""
                    chunk_only_audio_buffer.clear()
                    chunk_only_force_cumulative = False
                    await websocket.send_json({"type": "ack", "message": "音频缓冲已清空"})
                    continue

                if msg_type == "audio_base64":
                    try:
                        audio_bytes = base64.b64decode(msg.get("data", ""))
                    except Exception:
                        await websocket.send_json({"type": "error", "message": "audio_base64 解码失败"})
                        continue

                    if not audio_bytes:
                        await websocket.send_json({"type": "error", "message": "空音频分片"})
                        continue

                    chunk_index += 1
                    out_text, full_text = await _handle_chunk(audio_bytes)

                    await websocket.send_json(
                        {
                            "type": "transcript",
                            "text": out_text,
                            "full_text": full_text,
                            "return_mode": return_mode,
                            "chunk_index": chunk_index,
                            "final": False,
                        }
                    )
                    continue

                await websocket.send_json({"type": "error", "message": f"不支持的消息类型: {msg_type}"})
                continue

            if bytes_payload is not None:
                if not bytes_payload:
                    await websocket.send_json({"type": "error", "message": "空音频分片"})
                    continue

                chunk_index += 1
                out_text, full_text = await _handle_chunk(bytes_payload)

                await websocket.send_json(
                    {
                        "type": "transcript",
                        "text": out_text,
                        "full_text": full_text,
                        "return_mode": return_mode,
                        "chunk_index": chunk_index,
                        "final": False,
                    }
                )
                continue

    except WebSocketDisconnect:
        logger.info(f"Worker {MODEL_NAME}:{PORT} WebSocket 客户端断开")
    except Exception as e:
        logger.exception(f"Worker {MODEL_NAME}:{PORT} WebSocket ASR 处理失败: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            await websocket.close(code=1011)


async def _parse_request_payload(raw_request: Request, method_name: str) -> Dict[str, Any]:
    logger.info(f"method={method_name} content-type={raw_request.headers.get('content-type')}")
    if method_name in {"create_audio_speech", "create_audio_transcriptions"}:
        content_type = raw_request.headers.get("content-type", "").lower()
        media_type = content_type.split(";", maxsplit=1)[0].strip()

        if media_type == "application/json":
            try:
                return await raw_request.json()
            except json.JSONDecodeError as exc:
                raise ValueError(f"请求体不是合法 JSON: {exc.msg}") from exc

        form = await raw_request.form()
        payload: Dict[str, Any] = {}
        for key, value in form.multi_items():
            payload[key] = value

        # 允许通过 multipart 表单中的 server_config 字段传递 JSON 字符串。
        server_config = payload.get("server_config")
        if isinstance(server_config, str) and server_config.strip():
            try:
                payload["server_config"] = json.loads(server_config)
            except json.JSONDecodeError:
                pass

        return payload

    try:
        return await raw_request.json()
    except json.JSONDecodeError as exc:
        raise ValueError(f"请求体不是合法 JSON: {exc.msg}") from exc


def _to_response(result: Any) -> Response:
    if isinstance(result, (Response, StreamingResponse)):
        return result

    if isinstance(result, AsyncGenerator):
        return StreamingResponse(content=result, media_type="text/event-stream")

    status_code = HTTPStatus.OK
    content: Any = result

    if hasattr(result, "model_dump"):
        content = result.model_dump(mode="json")

    if isinstance(content, dict):
        code = content.get("code")
        if isinstance(code, int) and 400 <= code <= 599:
            status_code = code

    return JSONResponse(content=jsonable_encoder(content), status_code=int(status_code))


async def _convert_responses_stream_to_sse_events(
    generator: AsyncGenerator[StreamingResponsesResponse, None],
) -> AsyncGenerator[str, None]:
    """Convert vLLM Responses API stream objects to SSE events."""
    async for event in generator:
        event_type = getattr(event, "type", "unknown")
        event_data = event.model_dump_json(indent=None, by_alias=True)
        yield f"event: {event_type}\ndata: {event_data}\n\n"


def create_route_handler(method_name: str):
    """动态创建路由处理器"""
    async def handler(raw_request: Request):
        global MODEL_NAME, PORT, engine

        # 槽位准入：等待空闲槽位
        if SLOT_TRACKER is None:
            return JSONResponse(status_code=503, content={"error": "Worker 尚未就绪"})
        slot_id = await SLOT_TRACKER.acquire()
        try:
            req_data = await _parse_request_payload(raw_request, method_name)
            request_model = str(req_data.get("model", ""))

            # 安全校验：端口必须匹配模型
            if request_model not in ALLOWED_MODEL_NAMES:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"端口{PORT}只允许模型 {sorted(ALLOWED_MODEL_NAMES)}，不允许 {request_model}"}
                )

            if not engine:
                return JSONResponse(status_code=500, content={"error": "推理引擎未初始化"})

            # 调用对应的推理方法，兼容同步/异步实现
            method = getattr(engine, method_name, None)
            if method is None:
                return JSONResponse(status_code=400, content={"error": f"不支持的方法: {method_name}"})

            method_input: Any = req_data
            if method_name == "create_chat_completions":
                method_input = ChatCompletionRequest(**req_data)
            elif method_name == "create_responses":
                method_input = ResponsesRequest.model_validate(req_data)
            elif method_name == "create_embeddings":
                method_input = EmbeddingRequest(**req_data)
            elif method_name == "create_score":
                from pydantic import TypeAdapter

                scoring_protocol = importlib.import_module("vllm.entrypoints.pooling.scoring.protocol")
                ScoreRequest = getattr(scoring_protocol, "ScoreRequest")
                method_input = TypeAdapter(ScoreRequest).validate_python(req_data)
            elif method_name == "create_rerank":
                scoring_protocol = importlib.import_module("vllm.entrypoints.pooling.scoring.protocol")
                RerankRequest = getattr(scoring_protocol, "RerankRequest")
                method_input = RerankRequest(**req_data)
            elif method_name == "create_audio_speech":
                method_input = AudioSpeechRequest(**req_data)
            elif method_name == "create_audio_transcriptions":
                method_input = TranscriptionRequest(**req_data)

            result = method(method_input, raw_request)
            if isawaitable(result):
                infer_task = asyncio.ensure_future(result)
                while not infer_task.done():
                    if await raw_request.is_disconnected():
                        infer_task.cancel()
                        logger.warning(f"Worker {MODEL_NAME}:{PORT} 检测到客户端断开，已取消推理任务")
                        return JSONResponse(status_code=499, content={"error": "客户端已断开，推理已取消"})
                    await asyncio.sleep(0.05)
                result = await infer_task

            if isinstance(result, ErrorResponse):
                return JSONResponse(
                    content=result.model_dump(mode="json", by_alias=True),
                    status_code=result.error.code,
                )

            if isinstance(result, ResponsesResponse):
                return JSONResponse(content=result.model_dump(mode="json", by_alias=True))

            if method_name == "create_responses" and isinstance(result, AsyncGenerator):
                return StreamingResponse(
                    content=_convert_responses_stream_to_sse_events(result),
                    media_type="text/event-stream",
                )

            return _to_response(result)

        except NotImplementedError as e:
            import traceback
            logger.error(f"Worker {MODEL_NAME}:{PORT} 未实现: {e}\n{traceback.format_exc()}")
            return JSONResponse(status_code=501, content={"error": str(e)})
        except ValueError as e:
            logger.warning(f"Worker {MODEL_NAME}:{PORT} 请求参数错误: {e}")
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            import traceback
            logger.error(f"Worker {MODEL_NAME}:{PORT} 请求处理失败: {e}\n{traceback.format_exc()}")
            return JSONResponse(status_code=500, content={"error": str(e)})
        finally:
            # 释放 slot
            SLOT_TRACKER.release(slot_id)

    return handler

# 动态注册路由
for endpoint, method_name in routes.items():
    app.add_api_route(endpoint, create_route_handler(method_name), methods=["POST"])
    logger.info(f"📝 注册路由: {endpoint} -> {method_name}")

def run_worker(
    model_name: str,
    port: int,
    model_path=None,
    vllm_config=None,
    additional_config=None,
    model_type="llm",
    logging_config=None,
):
    import uvicorn
    global MODEL_NAME, ALLOWED_MODEL_NAMES, PORT, INSTANCE_CONFIG, logger, sync_logger

    try:
        _, _, api_server_args, unknown_args = split_vllm_config(vllm_config or {})
    except Exception as exc:
        api_server_args = {}
        unknown_args = {}

    configured_model_name = api_server_args.get("server_model_name")
    if configured_model_name is None:
        configured_model_name = api_server_args.get("served_model_name")
    if isinstance(configured_model_name, list):
        configured_model_names = [str(name) for name in configured_model_name if str(name)]
    elif configured_model_name not in (None, ""):
        configured_model_names = [str(configured_model_name)]
    else:
        configured_model_names = []

    served_model_name = configured_model_names[0] if configured_model_names else str(model_name)
    allowed_model_names = set(configured_model_names or [str(model_name)])
    allowed_model_names.add(served_model_name)

    configure_logging(
        logging_config,
        process_role="worker",
        port=int(port),
        model_name=served_model_name,
    )
    MODEL_NAME = served_model_name
    ALLOWED_MODEL_NAMES = allowed_model_names
    PORT = port
    logger = get_async_logger(f"worker:{served_model_name}:{port}")
    sync_logger = get_sync_logger(f"worker:{served_model_name}:{port}")
    additional_config = additional_config or {}
    device_ids = additional_config.get("device_ids", [])
    INSTANCE_CONFIG = {
        "model_name": served_model_name,
        "request_model_names": sorted(ALLOWED_MODEL_NAMES),
        "device_ids": device_ids or [],
        "model_path": model_path,
        "model_type": model_type,
        "vllm_config": vllm_config or {},
        "vllm_extra_config": vllm_config or {},
        "additional_config": additional_config,
        "extra_config": additional_config,
    }

    # # 为该子进程设置 GPU 设备
    # if device_ids is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
    #     os.environ["NVIDIA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)

    sync_logger.info(
        f"启动子进程, 模型={MODEL_NAME} 允许模型={sorted(ALLOWED_MODEL_NAMES)} 端口={PORT} 设备={device_ids} 路径={model_path}"
    )

    # 通用配置可根据 model_path/additional_config 动态设置
    # 这里可扩展为加载本地模型路径或专用调参
    if unknown_args:
        sync_logger.warning(f"vLLM 未识别参数: {unknown_args}")
    if api_server_args:
        sync_logger.info(f"当前 worker server 应用 vLLM api-server 层参数: {api_server_args}")
    _reconfigure_app_for_api_server(api_server_args)

    configured_port = api_server_args.get("port")
    if configured_port is not None and int(configured_port) != int(port):
        sync_logger.warning(
            f"忽略 vLLM api-server.port={configured_port}，当前子进程端口由模型实例 port={port} 决定"
        )

    host = api_server_args.get("host", "0.0.0.0")
    uvicorn_log_level = api_server_args.get("uvicorn_log_level", "warning")
    run_kwargs = {
        "host": host,
        "port": int(port),
        "log_level": uvicorn_log_level,
        "ssl_keyfile": api_server_args.get("ssl_keyfile"),
        "ssl_certfile": api_server_args.get("ssl_certfile"),
        "ssl_ca_certs": api_server_args.get("ssl_ca_certs"),
        "ssl_cert_reqs": api_server_args.get("ssl_cert_reqs"),
    }
    uvicorn.run(app, **{key: value for key, value in run_kwargs.items() if value is not None})