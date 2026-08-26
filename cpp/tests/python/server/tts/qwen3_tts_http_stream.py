#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load Qwen3-TTS and verify OpenAI-compatible HTTP audio streaming.

Base mode deliberately omits prompt_text/prompt_wav_path by default so the
server must select samples/zero_shot_prompt.wav as its default Clone profile.
CustomVoice mode sends a fixed voice name and verifies that no Clone profile is
selected. Both modes assert that multiple non-empty PCM chunks arrive over a
measurable period, then write raw PCM and a WAV file for listening.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:7901"
DEFAULT_TEXT = (
    "你好，这是Qwen3-TTS的HTTP流式测试。"
    "虽然音频会分成多个数据块返回，但整个段落只进行一次语音推理。"
    "请重点听句子衔接处的音色、语速和语气是否自然连贯。"
)
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "qwen3_tts_http_stream"


@dataclass(frozen=True)
class StreamResult:
    chunk_count: int
    byte_count: int
    first_chunk_seconds: float
    arrival_span_seconds: float
    elapsed_seconds: float
    sample_rate: int
    channels: int
    bits_per_sample: int


def configure_utf8_console() -> None:
    """Keep Chinese output readable in Windows PowerShell and cmd.exe."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except (AttributeError, OSError):
            pass

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass


def log(level: str, message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} | {level:<7} | {message}", flush=True)


def endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def model_id_from_path(model_path: str) -> str:
    filename = model_path.replace("\\", "/").rsplit("/", 1)[-1]
    return filename[:-5] if filename.lower().endswith(".gguf") else filename


def require_success(response: requests.Response, operation: str) -> None:
    if 200 <= response.status_code < 300:
        return
    body = response.text[:2_000]
    raise AssertionError(
        f"{operation} failed: HTTP {response.status_code}: {body}"
    )


def response_json(response: requests.Response, operation: str) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError as exc:
        raise AssertionError(
            f"{operation} returned non-JSON content: {response.text[:2_000]}"
        ) from exc
    if not isinstance(value, dict):
        raise AssertionError(f"{operation} returned a non-object JSON response")
    return value


def check_health(
    session: requests.Session,
    base_url: str,
    connect_timeout: float,
) -> None:
    response = session.get(
        endpoint(base_url, "/health"), timeout=connect_timeout
    )
    require_success(response, "health check")
    log("PASS", f"Service health check: HTTP {response.status_code}")


def load_model(
    session: requests.Session,
    base_url: str,
    model_path: str,
    variant: str,
    devices: str,
    connect_timeout: float,
    inference_timeout: float,
) -> None:
    log("INFO", f"Loading Qwen3-TTS {variant}: {model_path}")
    response = session.post(
        endpoint(base_url, "/models/load"),
        json={
            "model_type": "qwen3_tts",
            "model_path": model_path,
            "devices": devices,
        },
        timeout=(connect_timeout, inference_timeout),
    )
    require_success(response, "model load")
    body = response_json(response, "model load")
    if body.get("code", 0) != 0:
        raise AssertionError(f"model load returned code={body.get('code')}: {body}")
    log("PASS", f"Qwen3-TTS {variant} model loaded")


def require_model_instance(
    session: requests.Session,
    base_url: str,
    model_id: str,
    connect_timeout: float,
) -> int:
    """Verify the concrete model ID is registered and return its worker port."""
    response = session.get(
        endpoint(base_url, "/hlie/v1/status"), timeout=connect_timeout
    )
    require_success(response, "model instance discovery")
    body = response_json(response, "model instance discovery")
    processors = body.get("processor", [])
    if not isinstance(processors, list):
        raise AssertionError("status response contains an invalid processor list")

    normalized_id = model_id.lower()
    matches: list[int] = []
    for processor in processors:
        if not isinstance(processor, dict):
            continue
        models = processor.get("models", [])
        if not isinstance(models, list) or not any(
            isinstance(value, str) and value.lower() == normalized_id
            for value in models
        ):
            continue
        port = processor.get("port")
        if isinstance(port, int) and port > 0:
            matches.append(port)

    if not matches:
        raise AssertionError(
            f"loaded model {model_id!r} was not found in /hlie/v1/status"
        )
    if len(set(matches)) != 1:
        raise AssertionError(
            f"loaded model {model_id!r} is ambiguous across ports {matches}"
        )
    port = matches[0]
    log("PASS", f"Resolved model instance: {model_id} -> worker port {port}")
    return port


def unload_model(
    session: requests.Session,
    base_url: str,
    connect_timeout: float,
    inference_timeout: float,
) -> None:
    response = session.post(
        endpoint(base_url, "/models/unload"),
        json={"model_type": "qwen3_tts"},
        timeout=(connect_timeout, inference_timeout),
    )
    require_success(response, "model unload")
    log("PASS", "Qwen3-TTS model unloaded")


def require_header(
    response: requests.Response,
    name: str,
    expected: str,
) -> str:
    actual = response.headers.get(name, "")
    if actual.lower() != expected.lower():
        raise AssertionError(
            f"unexpected {name}: expected {expected!r}, received {actual!r}"
        )
    return actual


def positive_int_header(response: requests.Response, name: str) -> int:
    raw_value = response.headers.get(name, "")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise AssertionError(f"invalid {name}: {raw_value!r}") from exc
    if value <= 0:
        raise AssertionError(f"{name} must be positive, received {value}")
    return value


def trailer_value(response: requests.Response, name: str) -> str | None:
    """Return a trailer when the requests/urllib3 version exposes it."""
    value = response.headers.get(name)
    if value:
        return value
    raw_headers = getattr(response.raw, "headers", None)
    if raw_headers is None:
        return None
    return raw_headers.get(name)


def verify_stream(
    session: requests.Session,
    args: argparse.Namespace,
    pcm_path: Path,
    speech_base_url: str,
    model_id: str,
) -> StreamResult:
    payload: dict[str, Any] = {
        "model": model_id,
        "input": args.text,
        "language": args.language,
        "stream": True,
        "response_format": "pcm",
        "seed": args.seed,
        "max_frames": args.max_frames,
    }
    if args.variant == "base":
        if args.reference_wav:
            payload["prompt_wav_path"] = args.reference_wav
            payload["prompt_text"] = args.reference_text
    else:
        payload["voice"] = args.voice

    pcm_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log(
        "INFO",
        f"Starting stream=true synthesis: {endpoint(speech_base_url, '/v1/audio/speech')}",
    )
    with session.post(
        endpoint(speech_base_url, "/v1/audio/speech"),
        json=payload,
        headers={"Accept": "audio/pcm", "Accept-Encoding": "identity"},
        stream=True,
        timeout=(args.connect_timeout, args.inference_timeout),
    ) as response:
        require_success(response, "streaming synthesis")
        content_type = response.headers.get("Content-Type", "")
        if not content_type.lower().startswith("audio/pcm"):
            raise AssertionError(f"expected audio/pcm, received {content_type!r}")
        if "chunked" not in response.headers.get("Transfer-Encoding", "").lower():
            raise AssertionError(
                "streaming response is missing Transfer-Encoding: chunked"
            )
        require_header(response, "X-HLIE-TTS-Backend", "qwen3_tts")
        require_header(response, "X-Qwen3-TTS-Stream", "true")
        require_header(response, "X-Qwen3-TTS-Variant", args.variant)
        require_header(response, "X-Audio-Format", "s16le")
        reference_profile = response.headers.get(
            "X-Qwen3-TTS-Reference-Profile"
        )
        if args.variant == "base" and not args.reference_wav:
            require_header(
                response,
                "X-Qwen3-TTS-Reference-Profile",
                "default-zero-shot",
            )
        elif args.variant == "custom_voice" and reference_profile is not None:
            raise AssertionError(
                "CustomVoice response unexpectedly selected Clone profile "
                f"{reference_profile!r}"
            )

        sample_rate = positive_int_header(response, "X-Audio-Sample-Rate")
        channels = positive_int_header(response, "X-Audio-Channels")
        bits_per_sample = positive_int_header(
            response, "X-Audio-Bits-Per-Sample"
        )
        if channels != 1 or bits_per_sample != 16:
            raise AssertionError(
                "test only accepts mono PCM16LE; received "
                f"channels={channels}, bits={bits_per_sample}"
            )

        arrival_times: list[float] = []
        byte_count = 0
        with pcm_path.open("wb") as output:
            # chunk_size=None preserves HTTP chunk boundaries for chunked
            # responses instead of imposing an artificial client-side size.
            for chunk in response.iter_content(chunk_size=None):
                if not chunk:
                    continue
                now = time.monotonic()
                arrival_times.append(now)
                byte_count += len(chunk)
                output.write(chunk)
                log(
                    "CHUNK",
                    f"index={len(arrival_times)}, bytes={len(chunk)}, "
                    f"total={byte_count}, elapsed={now - started:.3f}s",
                )

        completed = time.monotonic()
        if not arrival_times:
            raise AssertionError("stream completed without audio data")
        if len(arrival_times) < args.min_chunks:
            raise AssertionError(
                f"expected at least {args.min_chunks} HTTP audio chunks, "
                f"received {len(arrival_times)}"
            )
        arrival_span = arrival_times[-1] - arrival_times[0]
        minimum_span = args.min_stream_span_ms / 1_000.0
        if arrival_span < minimum_span:
            raise AssertionError(
                "audio chunks arrived too close together to demonstrate "
                f"progressive delivery: {arrival_span * 1_000.0:.3f} ms "
                f"< {args.min_stream_span_ms:.3f} ms"
            )
        if byte_count % (channels * bits_per_sample // 8) != 0:
            raise AssertionError(
                f"PCM byte count {byte_count} does not contain complete samples"
            )

        finish_reason = trailer_value(response, "X-Qwen3-TTS-Finish-Reason")
        reached_eos = trailer_value(response, "X-Qwen3-TTS-Reached-EOS")
        trailer_samples = trailer_value(response, "X-Qwen3-TTS-Samples")
        if finish_reason is not None and finish_reason.lower() != "stop":
            stream_error = trailer_value(response, "X-Qwen3-TTS-Error")
            raise AssertionError(
                f"stream finish reason is {finish_reason!r}: {stream_error or ''}"
            )
        if reached_eos is not None and reached_eos.lower() != "true":
            raise AssertionError("stream did not reach Qwen3-TTS EOS")
        if trailer_samples is not None:
            expected_samples = byte_count // (channels * bits_per_sample // 8)
            if int(trailer_samples) != expected_samples:
                raise AssertionError(
                    f"sample trailer mismatch: expected {expected_samples}, "
                    f"received {trailer_samples}"
                )
        else:
            log(
                "INFO",
                "requests/urllib3 did not expose HTTP trailers; "
                "body and initial-header assertions still passed",
            )

    return StreamResult(
        chunk_count=len(arrival_times),
        byte_count=byte_count,
        first_chunk_seconds=arrival_times[0] - started,
        arrival_span_seconds=arrival_span,
        elapsed_seconds=completed - started,
        sample_rate=sample_rate,
        channels=channels,
        bits_per_sample=bits_per_sample,
    )


def write_wav(pcm_path: Path, wav_path: Path, result: StreamResult) -> None:
    with pcm_path.open("rb") as source, wave.open(str(wav_path), "wb") as output:
        output.setnchannels(result.channels)
        output.setsampwidth(result.bits_per_sample // 8)
        output.setframerate(result.sample_rate)
        output.writeframes(source.read())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "QWEN3_TTS_MODEL",
            os.environ.get("QWEN3_TTS_BASE_MODEL", ""),
        ),
        help=(
            "Qwen3-TTS GGUF path; defaults to QWEN3_TTS_MODEL, then the "
            "legacy QWEN3_TTS_BASE_MODEL"
        ),
    )
    parser.add_argument(
        "--variant",
        choices=("base", "custom_voice"),
        default="base",
        help="loaded Qwen3-TTS model variant",
    )
    parser.add_argument(
        "--model",
        help=(
            "model ID sent to /v1/audio/speech; defaults to the GGUF filename "
            "without its extension"
        ),
    )
    parser.add_argument(
        "--voice",
        default="vivian",
        help="fixed CustomVoice speaker; ignored for Base",
    )
    parser.add_argument("--devices", default="0")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--inference-timeout", type=float, default=900.0)
    parser.add_argument("--min-chunks", type=int, default=2)
    parser.add_argument("--min-stream-span-ms", type=float, default=10.0)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--reference-wav",
        help="optional explicit Clone reference; omit to test the default profile",
    )
    parser.add_argument("--reference-text")
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="use an already loaded Qwen3-TTS model",
    )
    parser.add_argument(
        "--unload",
        action="store_true",
        help="unload Qwen3-TTS after the test",
    )
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="honor HTTP_PROXY/HTTPS_PROXY; disabled by default to avoid buffering",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.skip_load and not args.model_path:
        parser.error("--model-path or QWEN3_TTS_BASE_MODEL is required")
    if bool(args.reference_wav) != bool(args.reference_text):
        parser.error("--reference-wav and --reference-text must be provided together")
    if args.variant == "custom_voice" and args.reference_wav:
        parser.error(
            "--reference-wav/--reference-text are only valid with --variant base"
        )
    if args.variant == "custom_voice" and not args.voice.strip():
        parser.error("--voice cannot be empty with --variant custom_voice")
    if args.skip_load and not args.model and not args.model_path:
        parser.error(
            "--skip-load requires --model or --model-path so the worker "
            "instance can be discovered"
        )
    if args.min_chunks < 2:
        parser.error("--min-chunks must be at least 2")
    if args.min_stream_span_ms < 0:
        parser.error("--min-stream-span-ms cannot be negative")


def main() -> int:
    configure_utf8_console()
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    model_id = args.model or model_id_from_path(args.model_path)

    session = requests.Session()
    session.trust_env = args.use_env_proxy
    output_stem = (
        "qwen3_tts_stream"
        if args.variant == "base"
        else "qwen3_tts_custom_voice_stream"
    )
    pcm_path = args.output_dir / f"{output_stem}.pcm"
    wav_path = args.output_dir / f"{output_stem}.wav"

    try:
        check_health(session, args.base_url, args.connect_timeout)
        if not args.skip_load:
            load_model(
                session,
                args.base_url,
                args.model_path,
                args.variant,
                args.devices,
                args.connect_timeout,
                args.inference_timeout,
            )
        worker_port = require_model_instance(
            session,
            args.base_url,
            model_id,
            args.connect_timeout,
        )
        speech_base_url = args.base_url
        log(
            "INFO",
            "Sending speech request through primary endpoint "
            f"{args.base_url} (registered worker port {worker_port})",
        )
        result = verify_stream(
            session,
            args,
            pcm_path,
            speech_base_url,
            model_id,
        )
        write_wav(pcm_path, wav_path, result)
        log(
            "PASS",
            "Progressive HTTP audio verified: "
            f"chunks={result.chunk_count}, bytes={result.byte_count}, "
            f"first_chunk={result.first_chunk_seconds:.3f}s, "
            f"arrival_span={result.arrival_span_seconds:.3f}s, "
            f"total={result.elapsed_seconds:.3f}s",
        )
        log("PASS", f"Raw PCM: {pcm_path}")
        log("PASS", f"Playable WAV: {wav_path}")
        if args.unload:
            unload_model(
                session,
                args.base_url,
                args.connect_timeout,
                args.inference_timeout,
            )
        return 0
    except (AssertionError, OSError, requests.RequestException, ValueError) as exc:
        log("FAIL", str(exc))
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
