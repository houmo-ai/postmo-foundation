import os
import json
import asyncio
import base64
import importlib
import io
import wave
import uuid
import gc
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

    # vLLM Responses PDF OCR 需要从 raw_request.app.state 获取同一实例的
    # OpenAIServingChat，用于将 PDF 页面裁剪区域重新渲染为图像输入。
    # hliepython worker 自己承载 HTTP 请求，因此这里需要显式注册该对象。
    openai_serving_chat = getattr(engine, "openai_serving_chat", None)
    if openai_serving_chat is not None:
        app.state.openai_serving_chat = openai_serving_chat

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
    current_engine = engine
    engine = None
    if current_engine:
        try:
            # 当前 vLLM shutdown() 会先关闭 EngineCore，再取消 output_handler。
            # 两者之间存在竞争窗口，output_handler 可能读到 EngineCore 断开并
            # 将正常卸载记录为 EngineDeadError。这里先停止输出任务再关核心。
            backend_engine = getattr(current_engine, "engine", current_engine)
            output_handler = getattr(backend_engine, "output_handler", None)
            if isinstance(output_handler, asyncio.Task):
                output_handler.cancel()
                try:
                    await output_handler
                except asyncio.CancelledError:
                    pass
                setattr(backend_engine, "output_handler", None)

            shutdown_fn = getattr(current_engine, "shutdown", None)
            if not callable(shutdown_fn):
                shutdown_fn = getattr(backend_engine, "shutdown", None)
            if callable(shutdown_fn):
                result = shutdown_fn()
                if isawaitable(result):
                    await result
            sync_logger.info(f"Worker {MODEL_NAME}:{PORT} 推理引擎关闭")
        except Exception:
            sync_logger.exception(f"Worker {MODEL_NAME}:{PORT} 推理引擎关闭失败")
        finally:
            del current_engine

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception:
        sync_logger.exception(f"Worker {MODEL_NAME}:{PORT} CUDA 缓存清理失败")

    if SLOT_TRACKER is not None:
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


def _pcm16le_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw mono PCM16 little-endian audio as a WAV container."""
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()


def _pcm16le_rms(pcm_bytes: bytes) -> float:
    """Return RMS amplitude for PCM16LE bytes without adding heavy dependencies."""
    if len(pcm_bytes) < 2:
        return 0.0

    sample_count = len(pcm_bytes) // 2
    total = 0
    view = memoryview(pcm_bytes[: sample_count * 2])
    for idx in range(0, len(view), 2):
        sample = int.from_bytes(view[idx: idx + 2], byteorder="little", signed=True)
        total += sample * sample
    return (total / sample_count) ** 0.5


@app.websocket("/")
@app.websocket("/v1/realtime")
@app.websocket("/ws/v1/realtime")
async def websocket_openai_realtime_transcriptions(websocket: WebSocket):
    """OpenAI Realtime transcription compatible WebSocket endpoint.

    Supported client events:
    - session.update
    - input_audio_buffer.append (base64 PCM16LE mono)
    - input_audio_buffer.commit
    - ping
    """
    global engine

    await websocket.accept()

    if engine is None:
        await websocket.send_json({
            "type": "error",
            "error": {"type": "server_error", "message": "推理引擎未初始化"},
        })
        await websocket.close(code=1011)
        return

    if not (
        hasattr(engine, "websocket_transcribe_stream_with_usage")
        or hasattr(engine, "websocket_transcribe_stream")
    ):
        await websocket.send_json({
            "type": "error",
            "error": {"type": "unsupported_model", "message": "当前模型不支持实时 ASR"},
        })
        await websocket.close(code=1003)
        return

    session_id = f"sess_{PORT}_{id(websocket)}"
    item_index = 0
    sample_rate = 24000
    language = ""
    prompt = ""
    hotwords: str | list[str] | None = None
    response_format = "json"
    include: list[str] = []
    turn_detection: dict[str, Any] | None = {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
        "max_turn_duration_ms": 3000,
    }
    audio_buffer = bytearray()
    speech_started = False
    speech_stopped = False
    silence_ms = 0.0
    turn_ms = 0.0
    committed = False

    async def _send_error(message: str, err_type: str = "invalid_request_error") -> None:
        await websocket.send_json({
            "type": "error",
            "event_id": f"evt_{uuid.uuid4().hex}",
            "error": {"type": err_type, "message": message},
        })

    def _session_payload() -> dict[str, Any]:
        return {
            "id": session_id,
            "object": "realtime.transcription_session",
            "type": "transcription",
            "model": MODEL_NAME,
            "include": include,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": sample_rate},
                    "transcription": {
                        "model": MODEL_NAME,
                        "language": language,
                        "prompt": prompt,
                        "hotwords": hotwords,
                        "response_format": response_format,
                    },
                    "turn_detection": turn_detection,
                }
            },
        }

    async def _transcribe_buffer(
        final_audio: bytes,
    ) -> AsyncGenerator[
        tuple[str, list[dict[str, Any]] | None, dict[str, Any] | None],
        None,
    ]:
        wav_audio = _pcm16le_to_wav_bytes(final_audio, sample_rate)
        include_logprobs = "item.input_audio_transcription.logprobs" in include
        transcribe_with_usage_fn = getattr(
            engine,
            "websocket_transcribe_stream_with_usage",
            None,
        )
        if transcribe_with_usage_fn is not None:
            async for result in transcribe_with_usage_fn(
                wav_audio,
                language=language or None,
                prompt=prompt,
                hotwords=hotwords,
                include_logprobs=include_logprobs,
            ):
                yield result.delta, result.logprobs, result.usage
            return

        transcribe_fn = getattr(engine, "websocket_transcribe_stream")
        async for delta in transcribe_fn(
            wav_audio,
            language=language or None,
            prompt=prompt,
            hotwords=hotwords,
        ):
            if delta:
                yield str(delta), None, None
        yield "", None, {
            "type": "duration",
            "seconds": len(final_audio) / (sample_rate * 2),
        }

    async def _commit_turn() -> None:
        nonlocal audio_buffer, item_index, speech_started, speech_stopped, silence_ms, turn_ms, committed

        if committed:
            return
        committed = True

        if not audio_buffer:
            await _send_error("音频缓冲为空")
            return

        item_index += 1
        item_id = f"item_{item_index}"
        content_index = 0

        vad_enabled = (
            isinstance(turn_detection, dict)
            and turn_detection.get("type") == "server_vad"
        )
        if vad_enabled and speech_started and not speech_stopped:
            speech_stopped = True
            await websocket.send_json({
                "type": "input_audio_buffer.speech_stopped",
                "event_id": f"evt_{uuid.uuid4().hex}",
                "item_id": item_id,
                "audio_end_ms": int(turn_ms),
            })

        await websocket.send_json({
            "type": "input_audio_buffer.committed",
            "event_id": f"evt_{uuid.uuid4().hex}",
            "item_id": item_id,
        })

        try:
            transcript_parts: list[str] = []
            usage: dict[str, Any] | None = None
            completed_logprobs: list[dict[str, Any]] | None = None
            async for delta, result_logprobs, result_usage in _transcribe_buffer(
                bytes(audio_buffer)
            ):
                if result_usage is not None:
                    usage = result_usage
                    completed_logprobs = result_logprobs
                if delta:
                    transcript_parts.append(delta)
                    delta_event: dict[str, Any] = {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "event_id": f"evt_{uuid.uuid4().hex}",
                        "item_id": item_id,
                        "content_index": content_index,
                        "delta": delta,
                    }
                    if result_logprobs is not None:
                        delta_event["logprobs"] = result_logprobs
                    await websocket.send_json(delta_event)
            transcript = "".join(transcript_parts).strip()
        except Exception as exc:
            logger.exception(f"Worker {MODEL_NAME}:{PORT} OpenAI Realtime ASR 转写失败: {exc}")
            await websocket.send_json({
                "type": "conversation.item.input_audio_transcription.failed",
                "event_id": f"evt_{uuid.uuid4().hex}",
                "item_id": item_id,
                "content_index": content_index,
                "error": {"type": "server_error", "message": str(exc)},
            })
        else:
            completed_event: dict[str, Any] = {
                "type": "conversation.item.input_audio_transcription.completed",
                "event_id": f"evt_{uuid.uuid4().hex}",
                "item_id": item_id,
                "content_index": content_index,
                "transcript": transcript,
                "usage": usage,
            }
            if completed_logprobs is not None:
                completed_event["logprobs"] = completed_logprobs
            await websocket.send_json(completed_event)
        finally:
            audio_buffer = bytearray()
            speech_started = False
            speech_stopped = False
            silence_ms = 0.0
            turn_ms = 0.0

    await websocket.send_json({
        "type": "session.created",
        "event_id": f"evt_{uuid.uuid4().hex}",
        "session": _session_payload(),
    })

    try:
        while True:
            frame = await websocket.receive()
            if frame.get("type") == "websocket.disconnect":
                break

            text_payload = frame.get("text")
            if text_payload is None:
                await _send_error("仅支持 JSON 文本事件")
                continue

            try:
                msg = json.loads(text_payload)
            except json.JSONDecodeError:
                await _send_error("无效 JSON 消息")
                continue

            msg_type = msg.get("type")
            if msg_type == "session.update":
                session = msg.get("session") or {}
                requested_include = session.get("include")
                if requested_include is not None:
                    if not isinstance(requested_include, list) or not all(
                        isinstance(value, str) for value in requested_include
                    ):
                        await _send_error("session.include 必须是字符串数组")
                        continue
                    unsupported_include = set(requested_include) - {
                        "item.input_audio_transcription.logprobs"
                    }
                    if unsupported_include:
                        await _send_error(
                            "不支持的 session.include 值: "
                            + ", ".join(sorted(unsupported_include))
                        )
                        continue
                    include = list(dict.fromkeys(requested_include))
                audio_input = ((session.get("audio") or {}).get("input") or {})
                fmt = audio_input.get("format") or {}
                transcription = audio_input.get("transcription") or {}

                if isinstance(fmt, dict) and fmt.get("rate"):
                    sample_rate = int(fmt["rate"])
                language = str(transcription.get("language") or language or "")
                prompt = str(transcription.get("prompt") or prompt or "")
                requested_hotwords = transcription.get("hotwords", hotwords)
                if requested_hotwords is not None and not (
                    isinstance(requested_hotwords, str)
                    or (
                        isinstance(requested_hotwords, list)
                        and all(isinstance(value, str) for value in requested_hotwords)
                    )
                ):
                    await _send_error("transcription.hotwords 必须是字符串或字符串数组")
                    continue
                hotwords = requested_hotwords
                response_format = str(transcription.get("response_format") or response_format)
                new_turn_detection = audio_input.get("turn_detection")
                turn_detection = new_turn_detection if isinstance(new_turn_detection, dict) else None

                await websocket.send_json({
                    "type": "session.updated",
                    "event_id": f"evt_{uuid.uuid4().hex}",
                    "session": _session_payload(),
                })
                continue

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type == "input_audio_buffer.append":
                audio_b64 = msg.get("audio")
                if not isinstance(audio_b64, str) or not audio_b64:
                    await _send_error("input_audio_buffer.append 缺少 audio")
                    continue

                try:
                    chunk = base64.b64decode(audio_b64)
                except Exception:
                    await _send_error("audio base64 解码失败")
                    continue

                if len(chunk) % 2 != 0:
                    chunk = chunk[:-1]
                if not chunk:
                    continue

                if committed:
                    committed = False
                audio_buffer.extend(chunk)

                chunk_ms = len(chunk) / 2 / max(sample_rate, 1) * 1000.0
                turn_ms += chunk_ms

                vad_enabled = isinstance(turn_detection, dict) and turn_detection.get("type") == "server_vad"
                if vad_enabled:
                    threshold = float(turn_detection.get("threshold", 0.5))
                    silence_duration_ms = int(turn_detection.get("silence_duration_ms", 500))
                    max_turn_duration_ms = int(turn_detection.get("max_turn_duration_ms", 3000))
                    # OpenAI threshold is normalized; use a conservative PCM16 RMS mapping.
                    rms_threshold = max(100.0, min(3000.0, threshold * 2000.0))
                    is_silence = _pcm16le_rms(chunk) < rms_threshold

                    if not speech_started and not is_silence:
                        speech_started = True
                        await websocket.send_json({
                            "type": "input_audio_buffer.speech_started",
                            "event_id": f"evt_{uuid.uuid4().hex}",
                            "item_id": f"item_{item_index + 1}",
                            "audio_start_ms": max(0, int(turn_ms - chunk_ms)),
                        })

                    if speech_started:
                        silence_ms = silence_ms + chunk_ms if is_silence else 0.0
                        if silence_ms >= silence_duration_ms or turn_ms >= max_turn_duration_ms:
                            await _commit_turn()
                continue

            if msg_type == "input_audio_buffer.commit":
                await _commit_turn()
                continue

            if msg_type == "input_audio_buffer.clear":
                audio_buffer.clear()
                speech_started = False
                speech_stopped = False
                silence_ms = 0.0
                turn_ms = 0.0
                committed = False
                await websocket.send_json({"type": "input_audio_buffer.cleared"})
                continue

            await _send_error(f"不支持的事件类型: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"Worker {MODEL_NAME}:{PORT} OpenAI Realtime WebSocket 客户端断开")
    except Exception as e:
        logger.exception(f"Worker {MODEL_NAME}:{PORT} OpenAI Realtime WebSocket 处理失败: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": {"type": "server_error", "message": str(e)},
            })
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

            # API 层允许同一实例使用展示别名；vLLM serving 只识别初始化时
            # 注册的 served model name，因此校验通过后统一改为该规范名称。
            req_data["model"] = MODEL_NAME

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
    request_model_names=None,
):
    global MODEL_NAME, ALLOWED_MODEL_NAMES, PORT, INSTANCE_CONFIG, logger, sync_logger

    # vLLM 会在 worker 内启动 EngineCore 等后代进程。为每个 worker
    # 建立独立进程组，主进程强制卸载时可以避免后代进程脱离并被 PID 1 接管。
    if os.name == "posix":
        try:
            os.setsid()
        except OSError:
            get_sync_logger("worker").warning(
                "无法为 worker 创建独立 session，卸载时将退化为单进程清理"
            )

    import uvicorn

    try:
        _, frontend_args, api_server_args, unknown_args = split_vllm_config(vllm_config or {})
    except Exception as exc:
        frontend_args = {}
        api_server_args = {}
        unknown_args = {}

    configured_model_name = frontend_args.get("served_model_name")
    if configured_model_name is None:
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
    allowed_model_names.update(str(name) for name in (request_model_names or []) if str(name))
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