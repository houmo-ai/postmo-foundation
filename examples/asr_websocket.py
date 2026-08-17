#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OpenAI-compatible HLIECpp ASR WebSocket smoke test.

Protocol used by HLIECpp:
    1. Server sends ``session.created`` after the WebSocket opens.
    2. Client sends ``session.update`` with transcription audio settings.
    3. Client streams base64-encoded little-endian PCM16 chunks with
       ``input_audio_buffer.append``.
        4. Server VAD sends ``input_audio_buffer.committed``, followed by zero or
             more incremental transcription ``delta`` events and one ``completed``
             event. With ``--manual-commit``, the client sends
       ``input_audio_buffer.commit`` after upload instead.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_URI = "ws://127.0.0.1:7803/v1/realtime"
DEFAULT_AUDIO_FILE = "./long_audio.mp3"
TARGET_SAMPLE_RATE = 24_000
CHUNK_DURATION_MS = 250
TRAILING_SILENCE_MS = 750
RECEIVE_TIMEOUT_SECONDS = 2.0
FINAL_TIMEOUT_SECONDS = 1800.0


def log(level: str, message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} | {level:<7} | {message}", flush=True)


def read_wav_pcm(path: Path) -> tuple[np.ndarray, int]:
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"Unsupported audio file, expected RIFF/WAVE: {path}")

    audio_format = None
    channels = None
    sample_rate = None
    bits_per_sample = None
    frames = None

    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(data):
            raise ValueError(f"Invalid WAV chunk size in {path}")

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError("Invalid WAV fmt chunk")
            audio_format, channels, sample_rate, _, _, bits_per_sample = struct.unpack_from(
                "<HHIIHH", data, chunk_start
            )
        elif chunk_id == b"data":
            frames = data[chunk_start:chunk_end]

        offset = chunk_end + (chunk_size % 2)

    if audio_format is None or channels is None or sample_rate is None or bits_per_sample is None or frames is None:
        raise ValueError(f"Missing WAV fmt/data chunk: {path}")

    sample_width = bits_per_sample // 8
    if audio_format == 1 and sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif audio_format == 1 and sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif audio_format == 1 and sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif audio_format == 3 and sample_width == 4:
        samples = np.frombuffer(frames, dtype="<f4").astype(np.float32)
    elif audio_format == 3 and sample_width == 8:
        samples = np.frombuffer(frames, dtype="<f8").astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV format={audio_format}, sample_width={sample_width} bytes")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return samples.astype(np.float32, copy=False), sample_rate


def decode_audio_ffmpeg(path: Path, target_rate: int) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"Audio format {path.suffix or '<unknown>'} requires ffmpeg, but ffmpeg was not found in PATH"
        )

    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(target_rate),
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Failed to decode audio file {path}: {error or 'ffmpeg exited with an error'}")

    samples = np.frombuffer(result.stdout, dtype="<f4")
    if samples.size == 0:
        raise ValueError(f"Decoded audio is empty: {path}")
    return samples.astype(np.float32, copy=False)


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)

    output_size = max(1, math.ceil(samples.size * target_rate / source_rate))
    source_x = np.arange(samples.size, dtype=np.float64)
    target_x = np.arange(output_size, dtype=np.float64) * source_rate / target_rate
    return np.interp(target_x, source_x, samples).astype(np.float32)


def load_audio(path: Path, target_rate: int) -> np.ndarray:
    log("INFO", f"Loading audio: {path}")
    if path.suffix.lower() in {".wav", ".wave"}:
        samples, sample_rate = read_wav_pcm(path)
        log("INFO", f"Original audio: sample_rate={sample_rate}, samples={samples.size}")
        samples = resample_linear(samples, sample_rate, target_rate)
    else:
        samples = decode_audio_ffmpeg(path, target_rate)
        log("INFO", f"Decoded audio with ffmpeg: samples={samples.size}")

    samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)

    duration = samples.size / target_rate
    log("SUCCESS", f"Audio ready: sample_rate={target_rate}, samples={samples.size}, duration={duration:.2f}s")
    return samples


def pcm16_base64(samples: np.ndarray) -> str:
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


async def send_json(websocket: Any, event: dict[str, Any]) -> None:
    await websocket.send(json.dumps(event, separators=(",", ":")))


async def recv_json(websocket: Any, timeout: float | None = None) -> dict[str, Any]:
    raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
    event = json.loads(raw)
    if not isinstance(event, dict):
        raise RuntimeError(f"Expected JSON object event, got: {event!r}")
    return event


async def configure_session(
    websocket: Any,
    sample_rate: int,
    model: str,
    prompt: str,
    hotwords: list[str],
    language: str,
    response_format: str,
    vad_enabled: bool,
    vad_threshold: float,
    silence_duration_ms: int,
    max_turn_duration_ms: int,
    include_logprobs: bool,
) -> None:
    turn_detection = None
    if vad_enabled:
        turn_detection = {
            "type": "server_vad",
            "threshold": vad_threshold,
            "prefix_padding_ms": 300,
            "silence_duration_ms": silence_duration_ms,
            "max_turn_duration_ms": max_turn_duration_ms,
        }

    transcription: dict[str, Any] = {}
    if model:
        transcription["model"] = model
    if prompt:
        transcription["prompt"] = prompt
    if hotwords:
        transcription["hotwords"] = hotwords
    if language:
        transcription["language"] = language
    if response_format:
        transcription["response_format"] = response_format

    await send_json(
        websocket,
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "include": (
                    ["item.input_audio_transcription.logprobs"]
                    if include_logprobs
                    else []
                ),
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": sample_rate},
                        "transcription": transcription,
                        "turn_detection": turn_detection,
                    }
                },
            },
        },
    )

    while True:
        event = await recv_json(websocket, timeout=RECEIVE_TIMEOUT_SECONDS)
        event_type = event.get("type")
        log("INFO", f"Session event: {event_type}")
        if event_type in {"session.updated", "transcription_session.updated"}:
            return
        if event_type == "error":
            raise RuntimeError(event.get("error", {}).get("message", "session.update failed"))


async def stream_audio(
    websocket: Any,
    samples: np.ndarray,
    sample_rate: int,
    chunk_duration_ms: int,
    trailing_silence_ms: int,
    realtime: bool,
    log_every_chunks: int,
) -> None:
    chunk_size = max(1, sample_rate * chunk_duration_ms // 1000)
    total_chunks = math.ceil(samples.size / chunk_size)
    log("INFO", f"Streaming audio: chunk_ms={chunk_duration_ms}, chunks={total_chunks}")

    started = time.time()
    for index, start in enumerate(range(0, samples.size, chunk_size), start=1):
        chunk = samples[start : start + chunk_size]
        await send_json(
            websocket,
            {
                "type": "input_audio_buffer.append",
                "audio": pcm16_base64(chunk),
            },
        )
        if index == 1 or index == total_chunks or (log_every_chunks > 0 and index % log_every_chunks == 0):
            log("INFO", f"Sent chunk {index}/{total_chunks}: samples={chunk.size}")
        if realtime:
            await asyncio.sleep(chunk.size / sample_rate)

    if trailing_silence_ms > 0:
        silence_samples = sample_rate * trailing_silence_ms // 1000
        silence = np.zeros(silence_samples, dtype=np.float32)
        for start in range(0, silence.size, chunk_size):
            chunk = silence[start : start + chunk_size]
            await send_json(
                websocket,
                {
                    "type": "input_audio_buffer.append",
                    "audio": pcm16_base64(chunk),
                },
            )
            if realtime:
                await asyncio.sleep(chunk.size / sample_rate)

    log("SUCCESS", f"Audio upload finished in {time.time() - started:.2f}s")


def is_connection_closed_error(exc: BaseException) -> bool:
    name = exc.__class__.__name__.lower()
    return "connectionclosed" in name or "connection closed" in str(exc).lower()


async def receive_transcripts(websocket: Any, sender_done: asyncio.Event, final_timeout: float) -> str:
    committed_turns = 0
    completed_turns = 0
    transcripts: list[str] = []
    delta_parts_by_item: dict[str, list[str]] = {}
    deadline = time.monotonic() + final_timeout

    while True:
        remaining = max(0.1, min(RECEIVE_TIMEOUT_SECONDS, deadline - time.monotonic()))
        try:
            event = await recv_json(websocket, timeout=remaining)
        except asyncio.TimeoutError:
            if time.monotonic() >= deadline:
                log("WARN", "Timed out waiting for transcription completion; returning partial result")
                break
            if sender_done.is_set() and committed_turns > 0 and completed_turns >= committed_turns:
                break
            continue
        except Exception as exc:
            if is_connection_closed_error(exc):
                log("WARN", f"WebSocket closed while receiving; returning partial result: {exc}")
                break
            raise

        event_type = event.get("type")

        if event_type == "session.created":
            log("INFO", "Session created")
        elif event_type == "session.updated":
            log("INFO", "Session updated")
        elif event_type == "input_audio_buffer.speech_started":
            log("INFO", f"Speech started: {event.get('item_id')}")
        elif event_type == "input_audio_buffer.speech_stopped":
            log("INFO", f"Speech stopped: {event.get('item_id')}")
        elif event_type == "input_audio_buffer.committed":
            committed_turns += 1
            log("INFO", f"Committed turn {committed_turns}: {event.get('item_id')}")
        elif event_type == "conversation.item.input_audio_transcription.delta":
            item_id = str(event.get("item_id", ""))
            delta = str(event.get("delta", ""))
            delta_parts_by_item.setdefault(item_id, []).append(delta)
            print(delta, end="", flush=True)
            if event.get("logprobs") is not None:
                log("DEBUG", f"Delta logprobs: {event['logprobs']}")
        elif event_type == "conversation.item.input_audio_transcription.completed":
            item_id = str(event.get("item_id", ""))
            transcript = event.get("transcript", "").strip()
            streamed_transcript = "".join(delta_parts_by_item.pop(item_id, []))
            if streamed_transcript.strip() != transcript:
                log(
                    "WARN",
                    "Incremental deltas differ from completed transcript; "
                    "using completed transcript as the authoritative result",
                )
            transcripts.append(transcript)
            completed_turns += 1
            log("SUCCESS", f"\nCompleted turn {completed_turns}/{committed_turns}: {transcript}")
            if event.get("logprobs") is not None:
                log("INFO", f"Completed logprobs: {event['logprobs']}")
        elif event_type == "conversation.item.input_audio_transcription.failed":
            completed_turns += 1
            error = event.get("error", {})
            raise RuntimeError(error.get("message", "Transcription failed"))
        elif event_type == "error":
            error = event.get("error", {})
            raise RuntimeError(error.get("message", "WebSocket error"))
        else:
            log("DEBUG", f"Ignoring event: {event_type}")

    return " ".join(text for text in transcripts if text)


async def run(args: argparse.Namespace) -> str:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install test requirements with `pip install -r src/tests/requirements.txt`") from exc

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    samples = load_audio(audio_path, args.sample_rate)
    log("INFO", f"Connecting WebSocket: {args.uri}")

    async with websockets.connect(
        args.uri,
        max_size=20 * 1024 * 1024,
        ping_interval=None if args.ping_interval <= 0 else args.ping_interval,
        ping_timeout=None if args.ping_timeout <= 0 else args.ping_timeout,
    ) as websocket:
        created = await recv_json(websocket, timeout=RECEIVE_TIMEOUT_SECONDS)
        log("INFO", f"Session event: {created.get('type')}")
        if created.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got: {created}")

        await configure_session(
            websocket,
            sample_rate=args.sample_rate,
            model=args.model,
            prompt=args.prompt,
            hotwords=args.hotwords,
            language=args.language,
            response_format=args.response_format,
            vad_enabled=not args.manual_commit,
            vad_threshold=args.vad_threshold,
            silence_duration_ms=args.silence_duration_ms,
            max_turn_duration_ms=args.max_turn_duration_ms,
            include_logprobs=args.logprobs,
        )

        sender_done = asyncio.Event()
        receiver = asyncio.create_task(receive_transcripts(websocket, sender_done, args.final_timeout))
        stream_error: BaseException | None = None
        try:
            try:
                await stream_audio(
                    websocket,
                    samples,
                    sample_rate=args.sample_rate,
                    chunk_duration_ms=args.chunk_ms,
                    trailing_silence_ms=0 if args.manual_commit else args.trailing_silence_ms,
                    realtime=not args.no_realtime,
                    log_every_chunks=args.log_every_chunks,
                )
                if args.manual_commit:
                    await send_json(websocket, {"type": "input_audio_buffer.commit"})
            except Exception as exc:
                stream_error = exc
                if is_connection_closed_error(exc):
                    log("WARN", f"WebSocket closed while sending audio; stopping gracefully: {exc}")
                else:
                    raise
        finally:
            sender_done.set()

        try:
            transcript = await receiver
        except Exception as exc:
            if stream_error is not None and is_connection_closed_error(exc):
                log("WARN", f"Receiver stopped after send failure: {exc}")
                transcript = ""
            else:
                raise

        if stream_error is not None and not transcript:
            return ""

        log("SUCCESS", f"ASR Result: {transcript}")
        return transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=DEFAULT_URI, help=f"WebSocket URI, default: {DEFAULT_URI}")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO_FILE), help="Path to an audio file (WAV, MP3, etc.)")
    parser.add_argument("--sample-rate", type=int, default=TARGET_SAMPLE_RATE, help="Input sample rate declared to server")
    parser.add_argument("--model", default="", help="Optional transcription model; overrides model in the WebSocket URI")
    parser.add_argument("--prompt", default="", help="Optional transcription prompt")
    parser.add_argument(
        "--hotwords",
        nargs="*",
        default=[],
        help="Contextual hotwords, for example: --hotwords HLIE Houmo Qwen",
    )
    parser.add_argument("--language", default="", help="Optional transcription language hint")
    parser.add_argument("--response-format", default="json", help="Transcription response format")
    parser.add_argument("--logprobs", action="store_true", help="Include token log probabilities in Realtime transcription events")
    parser.add_argument("--chunk-ms", type=int, default=CHUNK_DURATION_MS, help="Audio chunk duration in milliseconds")
    parser.add_argument("--trailing-silence-ms", type=int, default=TRAILING_SILENCE_MS, help="Silence sent to trigger VAD")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="OpenAI server_vad threshold")
    parser.add_argument("--silence-duration-ms", type=int, default=500, help="OpenAI server_vad silence duration")
    parser.add_argument("--max-turn-duration-ms", type=int, default=3000, help="HLIECpp max VAD turn duration")
    parser.add_argument("--manual-commit", action="store_true", help="Disable server VAD and send input_audio_buffer.commit")
    parser.add_argument("--no-realtime", action="store_true", help="Send chunks without sleeping between them")
    parser.add_argument("--final-timeout", type=float, default=FINAL_TIMEOUT_SECONDS, help="Seconds to wait for final transcripts")
    parser.add_argument("--ping-interval", type=float, default=0.0, help="WebSocket ping interval seconds; 0 disables client pings")
    parser.add_argument("--ping-timeout", type=float, default=0.0, help="WebSocket ping timeout seconds; 0 disables ping timeout")
    parser.add_argument("--log-every-chunks", type=int, default=20, help="Log one sent-chunk line every N chunks; 0 disables periodic chunk logs")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        log("WARN", "Interrupted")
    except Exception as exc:
        log("ERROR", str(exc))
