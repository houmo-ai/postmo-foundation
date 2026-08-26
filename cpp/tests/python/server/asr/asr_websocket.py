#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OpenAI-compatible HLIECpp ASR WebSocket smoke test.

Protocol used by HLIECpp:
    1. Server sends ``session.created`` after the WebSocket opens.
    2. Client sends ``session.update`` with transcription audio settings.
    3. Client streams base64-encoded little-endian PCM16 chunks with
       ``input_audio_buffer.append``.
    4. Server VAD sends ``input_audio_buffer.committed`` and transcription
       delta/completed events. With ``--manual-commit``, the client sends
       ``input_audio_buffer.commit`` after upload instead.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import math
import struct
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from asr_stream_assertions import TranscriptionStreamAssertions


DEFAULT_URI = "ws://127.0.0.1:7999"
DEFAULT_AUDIO_FILE = "E:/test_data/audio/long_2.wav"
TARGET_SAMPLE_RATE = 24_000
CHUNK_DURATION_MS = 250
TRAILING_SILENCE_MS = 750
DEFAULT_VAD_SILENCE_DURATION_MS = 2000
VAD_TRAILING_MARGIN_MS = 200
RECEIVE_TIMEOUT_SECONDS = 2.0
FINAL_TIMEOUT_SECONDS = 1800.0


def configure_utf8_console() -> None:
    """Keep Chinese transcripts readable in Windows PowerShell and cmd.exe."""
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


def normalize_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "Z"))
    )


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference = normalize_transcript(reference)
    hypothesis = normalize_transcript(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0

    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, start=1):
            substitution = previous[hypothesis_index - 1] + (
                reference_character != hypothesis_character
            )
            current.append(min(
                previous[hypothesis_index] + 1,
                current[hypothesis_index - 1] + 1,
                substitution,
            ))
        previous = current
    return previous[-1] / len(reference)


def effective_vad_trailing_silence_ms(
    requested_ms: int,
    silence_duration_ms: int,
) -> int:
    """Ensure a VAD smoke test sends enough silence to close its final turn."""
    return max(requested_ms, silence_duration_ms + VAD_TRAILING_MARGIN_MS)


def assert_vad_quality(
    manual_transcript: str,
    vad_transcript: str,
    max_cer: float,
    min_length_ratio: float,
    max_length_ratio: float,
) -> None:
    manual_normalized = normalize_transcript(manual_transcript)
    vad_normalized = normalize_transcript(vad_transcript)
    if not manual_normalized:
        raise AssertionError("Manual-commit baseline transcript is empty")

    cer = character_error_rate(manual_transcript, vad_transcript)
    length_ratio = len(vad_normalized) / len(manual_normalized)
    log("INFO", f"VAD quality: CER={cer:.4f}, length_ratio={length_ratio:.4f}")
    if cer > max_cer:
        raise AssertionError(f"VAD CER {cer:.4f} exceeds limit {max_cer:.4f}")
    if not min_length_ratio <= length_ratio <= max_length_ratio:
        raise AssertionError(
            f"VAD transcript length ratio {length_ratio:.4f} is outside "
            f"[{min_length_ratio:.4f}, {max_length_ratio:.4f}]"
        )


def read_wav_pcm(path: Path) -> tuple[np.ndarray, int]:
    audio_format = None
    channels = None
    sample_rate = None
    bits_per_sample = None
    data_offset = None
    data_size = None

    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"Unsupported audio file, expected RIFF/WAVE: {path}")

        while True:
            chunk_header = stream.read(8)
            if not chunk_header:
                break
            if len(chunk_header) != 8:
                raise ValueError(f"Truncated WAV chunk header in {path}")

            chunk_id = chunk_header[:4]
            chunk_size = struct.unpack_from("<I", chunk_header, 4)[0]
            chunk_start = stream.tell()
            if chunk_start + chunk_size > path.stat().st_size:
                raise ValueError(f"Invalid WAV chunk size in {path}")

            if chunk_id == b"fmt ":
                fmt = stream.read(chunk_size)
                if len(fmt) < 16:
                    raise ValueError("Invalid WAV fmt chunk")
                audio_format, channels, sample_rate, _, _, bits_per_sample = struct.unpack_from(
                    "<HHIIHH", fmt, 0
                )
                if audio_format == 0xFFFE:
                    if len(fmt) < 40:
                        raise ValueError("Invalid WAVE_FORMAT_EXTENSIBLE fmt chunk")
                    # The first two bytes of the extensible SubFormat GUID contain
                    # the underlying WAVE_FORMAT_PCM or WAVE_FORMAT_IEEE_FLOAT tag.
                    audio_format = struct.unpack_from("<H", fmt, 24)[0]
            elif chunk_id == b"data":
                data_offset = chunk_start
                data_size = chunk_size
                stream.seek(chunk_size, 1)
            else:
                stream.seek(chunk_size, 1)

            if chunk_size % 2:
                stream.seek(1, 1)

    if (
        audio_format is None
        or channels is None
        or sample_rate is None
        or bits_per_sample is None
        or data_offset is None
        or data_size is None
    ):
        raise ValueError(f"Missing WAV fmt/data chunk: {path}")

    sample_width = bits_per_sample // 8
    if audio_format == 1 and sample_width == 1:
        dtype = np.dtype(np.uint8)
        offset_value = 128.0
        scale = 128.0
    elif audio_format == 1 and sample_width == 2:
        dtype = np.dtype("<i2")
        offset_value = 0.0
        scale = 32768.0
    elif audio_format == 1 and sample_width == 4:
        dtype = np.dtype("<i4")
        offset_value = 0.0
        scale = 2147483648.0
    elif audio_format == 3 and sample_width == 4:
        dtype = np.dtype("<f4")
        offset_value = 0.0
        scale = 1.0
    elif audio_format == 3 and sample_width == 8:
        dtype = np.dtype("<f8")
        offset_value = 0.0
        scale = 1.0
    else:
        raise ValueError(f"Unsupported WAV format={audio_format}, sample_width={sample_width} bytes")

    if data_size % sample_width != 0:
        raise ValueError(f"WAV data chunk does not contain complete {sample_width}-byte samples")
    sample_count = data_size // sample_width
    if sample_count % channels != 0:
        raise ValueError("WAV data chunk does not contain complete channel frames")

    frame_count = sample_count // channels
    mapped = np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=(sample_count,))
    samples = np.empty(frame_count, dtype=np.float32)
    chunk_frames = 262_144
    for frame_start in range(0, frame_count, chunk_frames):
        frame_end = min(frame_count, frame_start + chunk_frames)
        sample_start = frame_start * channels
        sample_end = frame_end * channels
        block = np.asarray(mapped[sample_start:sample_end]).astype(np.float32)
        if offset_value:
            block -= offset_value
        if scale != 1.0:
            block /= scale
        if channels > 1:
            samples[frame_start:frame_end] = block.reshape(-1, channels).mean(
                axis=1, dtype=np.float32
            )
        else:
            samples[frame_start:frame_end] = block

    return samples, sample_rate


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)

    output_size = max(1, math.ceil(samples.size * target_rate / source_rate))
    source_x = np.arange(samples.size, dtype=np.float64)
    target_x = np.arange(output_size, dtype=np.float64) * source_rate / target_rate
    return np.interp(target_x, source_x, samples).astype(np.float32)


def load_audio(path: Path, target_rate: int) -> np.ndarray:
    log("INFO", f"Loading audio: {path}")
    samples, sample_rate = read_wav_pcm(path)
    log("INFO", f"Original audio: sample_rate={sample_rate}, samples={samples.size}")

    samples = resample_linear(samples, sample_rate, target_rate)
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)

    duration = samples.size / target_rate
    log("SUCCESS", f"Audio ready: sample_rate={target_rate}, samples={samples.size}, duration={duration:.2f}s")
    return samples


def limit_pcm16_audio_bytes(samples: np.ndarray, audio_bytes: int) -> np.ndarray:
    if audio_bytes <= 0:
        return samples
    if audio_bytes % 2 != 0:
        raise ValueError(f"PCM16 audio byte limit must be even, got {audio_bytes}")

    sample_count = audio_bytes // 2
    if samples.size < sample_count:
        raise ValueError(
            f"Audio only contains {samples.size * 2} PCM16 bytes, fewer than requested {audio_bytes}"
        )
    limited = samples[:sample_count]
    log("INFO", f"Limited audio payload to {audio_bytes} PCM16 bytes ({sample_count} samples)")
    return limited


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
    language: str,
    response_format: str,
    vad_enabled: bool,
    vad_threshold: float,
    silence_duration_ms: int,
    max_turn_duration_ms: int,
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

    transcription: dict[str, str] = {}
    if model:
        transcription["model"] = model
    if prompt:
        transcription["prompt"] = prompt
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


async def receive_transcripts(
    websocket: Any,
    sender_done: asyncio.Event,
    final_timeout: float,
    assert_streaming: bool = False,
) -> str:
    committed_turns = 0
    completed_turns = 0
    transcripts: list[str] = []
    stream_assertions = TranscriptionStreamAssertions() if assert_streaming else None
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
            if stream_assertions is not None:
                stream_assertions.consume(event)
            print(event.get("delta", ""), end="", flush=True)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            if stream_assertions is not None:
                stream_assertions.consume(event)
            transcript = event.get("transcript", "")
            transcripts.append(transcript.strip())
            completed_turns += 1
            log("SUCCESS", f"\nCompleted turn {completed_turns}/{committed_turns}: {transcript}")
        elif event_type == "conversation.item.input_audio_transcription.failed":
            completed_turns += 1
            error = event.get("error", {})
            raise RuntimeError(error.get("message", "Transcription failed"))
        elif event_type == "error":
            error = event.get("error", {})
            raise RuntimeError(error.get("message", "WebSocket error"))
        else:
            log("DEBUG", f"Ignoring event: {event_type}")

    if stream_assertions is not None:
        stream_assertions.finish()
        if committed_turns == 0:
            raise AssertionError("No input_audio_buffer.committed event received")
        if completed_turns != committed_turns:
            raise AssertionError(
                f"Committed/completed turn count mismatch: committed={committed_turns} completed={completed_turns}"
            )

    return " ".join(text for text in transcripts if text)


async def run(args: argparse.Namespace) -> str:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install test requirements with `pip install -r tests/python/server/requirements.txt`") from exc

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    samples = load_audio(audio_path, args.sample_rate)
    samples = limit_pcm16_audio_bytes(samples, args.audio_bytes)
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
            language=args.language,
            response_format=args.response_format,
            vad_enabled=not args.manual_commit,
            vad_threshold=args.vad_threshold,
            silence_duration_ms=args.silence_duration_ms,
            max_turn_duration_ms=args.max_turn_duration_ms,
        )

        sender_done = asyncio.Event()
        receiver = asyncio.create_task(
            receive_transcripts(
                websocket,
                sender_done,
                args.final_timeout,
                assert_streaming=args.assert_streaming,
            )
        )
        stream_error: BaseException | None = None
        try:
            try:
                trailing_silence_ms = 0
                if not args.manual_commit:
                    trailing_silence_ms = effective_vad_trailing_silence_ms(
                        args.trailing_silence_ms,
                        args.silence_duration_ms,
                    )
                    if trailing_silence_ms != args.trailing_silence_ms:
                        log(
                            "INFO",
                            "Raised trailing silence to "
                            f"{trailing_silence_ms} ms so the final VAD turn can close",
                        )
                await stream_audio(
                    websocket,
                    samples,
                    sample_rate=args.sample_rate,
                    chunk_duration_ms=args.chunk_ms,
                    trailing_silence_ms=trailing_silence_ms,
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
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO_FILE), help="Path to a PCM WAV file")
    parser.add_argument(
        "--audio-bytes",
        type=int,
        default=0,
        help="Limit the uploaded PCM16 payload to this many bytes; 0 sends the full audio",
    )
    parser.add_argument("--sample-rate", type=int, default=TARGET_SAMPLE_RATE, help="Input sample rate declared to server")
    parser.add_argument("--model", default="", help="Optional transcription model; overrides model in the WebSocket URI")
    parser.add_argument("--prompt", default="", help="Optional transcription prompt")
    parser.add_argument("--language", default="", help="Optional transcription language hint")
    parser.add_argument("--response-format", default="json", help="Transcription response format")
    parser.add_argument("--chunk-ms", type=int, default=CHUNK_DURATION_MS, help="Audio chunk duration in milliseconds")
    parser.add_argument("--trailing-silence-ms", type=int, default=TRAILING_SILENCE_MS, help="Silence sent to trigger VAD")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="OpenAI server_vad threshold")
    parser.add_argument(
        "--silence-duration-ms",
        type=int,
        default=DEFAULT_VAD_SILENCE_DURATION_MS,
        help="OpenAI server_vad silence duration; 2000 ms is the stable Qwen3-ASR default",
    )
    parser.add_argument("--max-turn-duration-ms", type=int, default=30000, help="HLIECpp max VAD turn duration")
    parser.add_argument("--manual-commit", action="store_true", help="Disable server VAD and send input_audio_buffer.commit")
    parser.add_argument("--no-realtime", action="store_true", help="Send chunks without sleeping between them")
    parser.add_argument("--final-timeout", type=float, default=FINAL_TIMEOUT_SECONDS, help="Seconds to wait for final transcripts")
    parser.add_argument("--ping-interval", type=float, default=0.0, help="WebSocket ping interval seconds; 0 disables client pings")
    parser.add_argument("--ping-timeout", type=float, default=0.0, help="WebSocket ping timeout seconds; 0 disables ping timeout")
    parser.add_argument("--log-every-chunks", type=int, default=20, help="Log one sent-chunk line every N chunks; 0 disables periodic chunk logs")
    parser.add_argument(
        "--assert-streaming",
        action="store_true",
        help="Require delta events, exact delta/completed equality, and no leaked Qwen ASR markers",
    )
    parser.add_argument(
        "--compare-manual-baseline",
        action="store_true",
        help="Run manual-commit and VAD modes, then enforce transcript quality thresholds",
    )
    parser.add_argument("--max-vad-cer", type=float, default=0.08, help="Maximum VAD CER versus manual baseline")
    parser.add_argument("--min-vad-length-ratio", type=float, default=0.85, help="Minimum VAD/manual text length ratio")
    parser.add_argument("--max-vad-length-ratio", type=float, default=1.15, help="Maximum VAD/manual text length ratio")
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    if not args.compare_manual_baseline:
        await run(args)
        return

    manual_args = copy.copy(args)
    manual_args.manual_commit = True
    manual_args.compare_manual_baseline = False
    log("INFO", "Running manual-commit ASR baseline")
    manual_transcript = await run(manual_args)

    vad_args = copy.copy(args)
    vad_args.manual_commit = False
    vad_args.compare_manual_baseline = False
    log("INFO", "Running server VAD ASR comparison")
    vad_transcript = await run(vad_args)

    assert_vad_quality(
        manual_transcript,
        vad_transcript,
        args.max_vad_cer,
        args.min_vad_length_ratio,
        args.max_vad_length_ratio,
    )
    log("SUCCESS", "VAD transcript quality matches the manual-commit baseline")


if __name__ == "__main__":
    configure_utf8_console()
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        log("WARN", "Interrupted")
    except Exception as exc:
        log("ERROR", str(exc))
        sys.exit(1)
