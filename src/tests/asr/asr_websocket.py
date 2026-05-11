#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Whisper.cpp / miniaudio Compatible WebSocket Audio Stream Client

功能:
    1. 读取 wav 音频
    2. 转 mono
    3. 重采样到 16k
    4. 转 float32 PCM [-1,1]
    5. websocket 流式发送

目标:
    与 whisper.cpp 中 miniaudio 的行为尽可能一致

数据格式:
    float32 PCM
    16000Hz
    mono

发送协议:
    binary:
        float32 PCM bytes

    text:
        EOS
"""

import asyncio
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets
from loguru import logger
from scipy import signal

# =============================================================================
# Config
# =============================================================================

TARGET_SAMPLE_RATE = 16000

# 1000ms chunk
CHUNK_DURATION_MS = 1000

# websocket uri
WS_URI = "ws://10.65.32.191:7999"

# audio file
AUDIO_FILE = "D:/workspace/wtt/houmo/llm/hi.cpp/samples/zero_shot_prompt.wav"

# =============================================================================
# Logger
# =============================================================================

logger.remove()

logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "{message}"
    ),
)

logger.add(
    "asr_websocket_test.log",
    level="DEBUG",
    rotation="10 MB",
    retention=5,
    encoding="utf-8",
)


# =============================================================================
# Audio Process
# =============================================================================

def load_audio(file_path: str) -> np.ndarray:
    """
    加载音频并转换成:

        float32
        mono
        16000Hz
        [-1,1]

    尽量模拟 miniaudio 行为
    """

    logger.info(f"Loading audio: {file_path}")

    # soundfile:
    #   int16 wav -> int16
    #   float wav -> float
    data, sample_rate = sf.read(file_path)

    logger.info(
        f"Original audio info: "
        f"sample_rate={sample_rate}, "
        f"shape={data.shape}, "
        f"dtype={data.dtype}"
    )

    # =========================================================================
    # Convert to mono
    # =========================================================================

    if data.ndim > 1:
        logger.info(
            f"Convert multi-channel -> mono "
            f"(channels={data.shape[1]})"
        )

        # IMPORTANT:
        # whisper.cpp/miniaudio:
        #   left + right
        #
        # NOT:
        #   (left + right) / 2
        #
        # 这里使用 sum 更接近 whisper.cpp
        data = np.sum(data, axis=1)

    # =========================================================================
    # Resample to 16k
    # =========================================================================

    if sample_rate != TARGET_SAMPLE_RATE:
        logger.info(
            f"Resample audio: "
            f"{sample_rate}Hz -> {TARGET_SAMPLE_RATE}Hz"
        )

        num_samples = int(
            len(data)
            * TARGET_SAMPLE_RATE
            / sample_rate
        )

        data = signal.resample(
            data,
            num_samples,
        )

        sample_rate = TARGET_SAMPLE_RATE

    # =========================================================================
    # Convert to float32
    # =========================================================================

    if data.dtype == np.int16:

        logger.info("Convert int16 -> float32")

        data = data.astype(np.float32) / 32768.0

    elif data.dtype == np.int32:

        logger.info("Convert int32 -> float32")

        data = data.astype(np.float32) / 2147483648.0

    elif data.dtype != np.float32:

        logger.info(f"Convert {data.dtype} -> float32")

        data = data.astype(np.float32)

    # =========================================================================
    # Clip
    # =========================================================================

    # 防止 stereo 相加后超过 [-1,1]
    data = np.clip(data, -1.0, 1.0)

    duration = len(data) / TARGET_SAMPLE_RATE

    logger.success(
        f"Audio ready: "
        f"samples={len(data)}, "
        f"duration={duration:.2f}s, "
        f"dtype={data.dtype}"
    )

    return data


# =============================================================================
# WebSocket Stream
# =============================================================================

async def stream_audio(
        websocket,
        pcmf32: np.ndarray,
):
    """
    websocket 流式发送 float32 PCM
    """

    chunk_size = int(
        TARGET_SAMPLE_RATE
        * CHUNK_DURATION_MS
        / 1000
    )

    total_chunks = (
                           len(pcmf32)
                           + chunk_size
                           - 1
                   ) // chunk_size

    logger.info(
        f"Start streaming: "
        f"chunk_duration={CHUNK_DURATION_MS}ms, "
        f"chunk_size={chunk_size}, "
        f"total_chunks={total_chunks}"
    )

    start_time = time.time()

    for chunk_index, start in enumerate(
            range(0, len(pcmf32), chunk_size),
            start=1,
    ):
        end = start + chunk_size

        chunk = pcmf32[start:end]

        # binary float32 pcm
        binary = chunk.astype(np.float32).tobytes()

        await websocket.send(binary)

        chunk_duration = (
                len(chunk)
                / TARGET_SAMPLE_RATE
        )

        logger.info(
            f"[Chunk {chunk_index}/{total_chunks}] "
            f"samples={len(chunk)}, "
            f"bytes={len(binary)}, "
            f"duration={chunk_duration:.3f}s"
        )

        # 模拟真实实时发送
        await asyncio.sleep(chunk_duration)

    elapsed = time.time() - start_time

    logger.success(
        f"Audio stream completed (elapsed={elapsed:.2f}s)"
    )


# =============================================================================
# Main
# =============================================================================

async def simulate_frontend_stream(
        file_path: str,
        uri: str,
):
    if not Path(file_path).exists():
        logger.error(
            f"Audio file not found: {file_path}"
        )

        return

    # load audio
    pcmf32 = load_audio(file_path)

    logger.info(f"Connecting websocket: {uri}")

    try:

        async with websockets.connect(
                uri,
                max_size=20 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
        ) as websocket:

            logger.success("WebSocket connected")

            # stream audio
            await stream_audio(
                websocket,
                pcmf32,
            )

            # EOS
            logger.info("Sending EOS")

            await websocket.send("EOS")

            # wait result
            logger.info("Waiting ASR result...")

            result = await asyncio.wait_for(
                websocket.recv(),
                timeout=30.0,
            )

            logger.success(
                f"ASR Result:\n{result}"
            )

    except asyncio.TimeoutError:

        logger.error(
            "Timeout waiting ASR result"
        )

    except websockets.ConnectionClosed as e:

        logger.error(
            f"WebSocket closed: "
            f"code={e.code}, "
            f"reason={e.reason}"
        )

    except Exception as e:

        logger.exception(
            f"Unexpected error: {e}"
        )

    finally:

        logger.info("Test finished")


# =============================================================================
# Entry
# =============================================================================

if __name__ == "__main__":
    asyncio.run(
        simulate_frontend_stream(
            AUDIO_FILE,
            WS_URI,
        )
    )
